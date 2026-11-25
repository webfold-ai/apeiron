//! Attention: the per-layer key/value cache and one causal query head (port-map row 5).

use super::ONE;
use crate::engine::mixer::f32_carrier::q32_to_f32;
use crate::engine::mixer::{carrier_from_env, Carrier};
use crate::engine::num::fp::F32;
use crate::engine::num::value::{div_round_nearest, round_shift_32};
use crate::engine::ops::matvec::row_dot_fixed;
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_rsqrt::fixed_rsqrt;
use std::cell::Cell;
use std::sync::Arc;

#[path = "attn/cert.rs"]
mod cert;
#[path = "attn/f32_core.rs"]
pub mod f32_core;
#[path = "attn/i8.rs"]
pub(super) mod i8;
#[path = "attn/kvplane.rs"]
pub mod kvplane;

pub use self::cert::CertStats;
use self::kvplane::{KvFileError, KvMap, KvPlane};

/// The numerical carrier of the key/value cache: the dial `APEIRON_KV_CARRIER` selects.
///
/// [`KvCarrier::Q32`] is the exact fixed-point oracle and the only carrier the block certificate
/// reads; [`KvCarrier::F32`] is the measured decode default; [`KvCarrier::I8`] stores one signed
/// code per component with one binary32 scale per `(kv_head, position, K|V)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCarrier {
    /// Q32.32 fixed-point planes.
    Q32,
    /// Native binary32 planes.
    F32,
    /// Signed 8-bit codes with one binary32 scale per row.
    I8,
}

/// The cache carrier from `APEIRON_KV_CARRIER` (`q32` | `f32` | `i8`), falling back to
/// [`carrier_from_env`] when unset or unrecognised.
///
/// `APEIRON_F32_CARRIER=0` still selects Q32 and every other value F32, so a run that sets only
/// the older dial keeps its carrier. Parsed once per process.
///
/// # Complexity
///
/// `O(1)` after the first call; one `OnceLock` and one environment lookup.
pub(crate) fn kv_carrier_from_env() -> KvCarrier {
    static CARRIER: std::sync::OnceLock<KvCarrier> = std::sync::OnceLock::new();
    *CARRIER.get_or_init(|| match std::env::var("APEIRON_KV_CARRIER").as_deref() {
        Ok(v) if v.eq_ignore_ascii_case("i8") => KvCarrier::I8,
        Ok(v) if v.eq_ignore_ascii_case("f32") => KvCarrier::F32,
        Ok(v) if v.eq_ignore_ascii_case("q32") => KvCarrier::Q32,
        _ => match carrier_from_env() {
            Carrier::Q32 => KvCarrier::Q32,
            Carrier::F32 => KvCarrier::F32,
        },
    })
}

// Test seam for `APEIRON_EXACT_SCAN`: a per-thread override of the flag, `None` falling through
// to the process value. The process parse lives in `src/engine/step/cols.rs` where the flag is
// consumed; the seam is here because `src/engine/tests.rs` cannot name the private
// `engine::step::cols` module.
thread_local! {
    static EXACT_SCAN_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// The thread's forced exact-scan value, or `None` to fall through to the process flag.
pub(crate) fn exact_scan_override() -> Option<bool> {
    EXACT_SCAN_OVERRIDE.with(Cell::get)
}

/// Force the exact-scan flag on this thread, or clear it with `None` (the test seam).
#[cfg(test)]
pub(crate) fn set_exact_scan_override(value: Option<bool>) {
    EXACT_SCAN_OVERRIDE.with(|cell| cell.set(value));
}

/// A refused fixed-point attention read (port-map row 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnError {
    /// A geometry parameter is zero, or `n_head` is not a multiple of `n_head_kv`.
    BadGeometry,
    /// `head_dim` left the `fixed_rsqrt` window, so the `1 / sqrt(head_dim)` score scale is
    /// undefined.
    ScaleOutOfRange,
    /// A layer index is at or past the configured layer count.
    LayerOutOfRange {
        /// Layer index supplied.
        layer: usize,
        /// Configured layer count.
        n_layer: usize,
    },
    /// A query-head index is at or past `n_head`.
    HeadOutOfRange {
        /// Query-head index supplied.
        head: usize,
        /// Configured query-head count.
        n_head: usize,
    },
    /// The key/value row block, the query or the output is not the required width.
    RowLength {
        /// Required width.
        expected: usize,
        /// Width supplied.
        got: usize,
    },
    /// The attention position is at or past the number of positions appended for the layer.
    PositionOutOfRange {
        /// Position supplied.
        pos: usize,
        /// Positions appended for the layer.
        len: usize,
    },
    /// The layer already holds `cap` positions, so no further append is admitted.
    CapacityExceeded {
        /// Configured position capacity.
        cap: usize,
    },
    /// A [`fixed_exp`] call in the score-shifted softmax refused. The shift is `<= 0` and its only
    /// reachable window edge `-16*ONE` is admitted, so with the underflow saturation this variant is
    /// retained but unreachable.
    ExpOutOfRange,
    /// An `i128` accumulation or an `i64` narrowing in the score, softmax or output left its range.
    Overflow,
    /// The softmax total is zero, so the probability normalisation is undefined.
    ZeroSum,
}

/// Typed storage for key and value planes: Q32.32 fixed-point `i64`, binary32 `f32`, or signed
/// 8-bit codes with a binary32 scale per row.
///
/// Each plane is a [`KvPlane`]: an owned vector, or a disjoint range of one writable cache file
/// ([`KvState::with_file`]). The read and write paths are identical for both backings.
#[derive(Debug, Clone)]
pub enum KvStorage {
    /// Fixed-point Q32.32 key and value planes.
    Q32 {
        /// Position-major key planes, one per layer.
        k: Vec<KvPlane<i64>>,
        /// Position-major value planes, one per layer.
        v: Vec<KvPlane<i64>>,
    },
    /// Native binary32 floating-point key and value planes.
    F32 {
        /// Position-major key planes, one per layer.
        k: Vec<KvPlane<F32>>,
        /// Position-major value planes, one per layer.
        v: Vec<KvPlane<F32>>,
    },
    /// Signed 8-bit key and value codes, KV-head-major, with one binary32 scale per
    /// `(kv_head, position)` in `ks`/`vs`. Component `i` of position `s` of KV head `kvh` is at
    /// `((kvh · cap) + s) · head_dim + i`: one KV head's positions are contiguous, so the group
    /// read walks one row run with `stride = head_dim`.
    I8 {
        /// KV-head-major key code planes, one per layer.
        k: Vec<KvPlane<i8>>,
        /// KV-head-major value code planes, one per layer.
        v: Vec<KvPlane<i8>>,
        /// Key scale of `(kv_head, position)`, `n_head_kv · cap` per layer.
        ks: Vec<KvPlane<F32>>,
        /// Value scale of `(kv_head, position)`, `n_head_kv · cap` per layer.
        vs: Vec<KvPlane<F32>>,
    },
}

/// Allocate the zeroed native binary32 key and value planes for `n_layer` layers of `plane` values.
///
/// This is the carrier's binary32 allocation and nothing else: the geometry guards, the
/// `fixed_rsqrt` window and the checked plane product are decided by [`KvState::with_carrier`]
/// before it is called, and it cannot refuse. It is a named function so that the extraction
/// boundary can be the allocation rather than the whole constructor — the constructor's guards and
/// its Q32.32 allocation stay translated.
///
/// # Complexity
///
/// `O(n_layer · plane)` time and space in `f32` words: one zeroed `plane`-value plane, `n_layer`
/// clones of it into the key planes and the move of the blank into the value plane. The count is
/// two `vec![blank; n_layer]` repetitions over a `plane`-element element, so it closes at
/// `2 · n_layer · plane` writes.
fn alloc_f32_storage(plane: usize, n_layer: usize) -> KvStorage {
    let blank = KvPlane::owned(vec![F32::ZERO; plane]);
    let k = vec![blank.clone(); n_layer];
    let v = vec![blank; n_layer];
    KvStorage::F32 { k, v }
}

/// Allocate the zeroed int8 code planes and their per-row scales.
///
/// `plane` is the code-plane length (`n_head_kv · cap · head_dim`), `rows` the scale-plane length
/// (`n_head_kv · cap`) and `n_layer` the layer count. The codes start at zero and every scale at
/// one: a partly written position therefore reads as a zero row rather than as a row scaled by
/// zero, which keeps a truncated append's neighbour positions finite.
///
/// # Complexity
///
/// `O(n_layer · (plane + 2 · rows))` time and space in the unions of the two element types: one
/// zeroed code plane and one unit scale plane cloned `n_layer` times each.
fn alloc_i8_storage(plane: usize, rows: usize, n_layer: usize) -> KvStorage {
    let blank_k = KvPlane::owned(vec![0i8; plane]);
    let blank_s = KvPlane::owned(vec![F32::ONE; rows]);
    KvStorage::I8 {
        k: vec![blank_k.clone(); n_layer],
        v: vec![blank_k; n_layer],
        ks: vec![blank_s.clone(); n_layer],
        vs: vec![blank_s; n_layer],
    }
}

/// A refused file-backed key/value cache ([`KvState::with_file`]).
#[derive(Debug)]
pub enum KvInitError {
    /// The geometry the cache was asked for is not a cache geometry.
    Geometry(AttnError),
    /// The cache file could not be created, allocated or mapped.
    File(KvFileError),
}

impl core::fmt::Display for KvInitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Geometry(e) => write!(f, "kv geometry: {e:?}"),
            Self::File(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for KvInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Geometry(_) => None,
            Self::File(e) => Some(e),
        }
    }
}

impl From<AttnError> for KvInitError {
    fn from(e: AttnError) -> Self {
        Self::Geometry(e)
    }
}

impl From<KvFileError> for KvInitError {
    fn from(e: KvFileError) -> Self {
        Self::File(e)
    }
}

/// The geometry guard chain shared by the owned and file-backed constructors: the zero and
/// divisibility guards, the `fixed_rsqrt` window, and the checked per-layer plane length. Returns
/// the plane length in elements and the load-time score scale.
///
/// # Errors
///
/// [`AttnError::BadGeometry`], [`AttnError::ScaleOutOfRange`] and [`AttnError::Overflow`], in that
/// order, exactly as [`KvState::with_carrier`] has always refused.
///
/// # Complexity
///
/// `O(1)` checked arithmetic.
fn checked_geometry(
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    cap: usize,
) -> Result<(usize, i64), AttnError> {
    if n_layer == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0 || cap == 0 {
        return Err(AttnError::BadGeometry);
    }
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(AttnError::BadGeometry);
    }
    // `head_dim` widened after an explicit range check: `i64::try_from` emits an unmodeled
    // external the fail-closed extraction audit rejects. The check is exact on 64-bit, where
    // `head_dim as i64` is negative exactly when `head_dim > i64::MAX`.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "the `hd_i64 < 0` check immediately above the cast is the range guard; the \
                  cast is exact for every value it admits"
    )]
    let hd_signed = head_dim as i64;
    if hd_signed < 0 {
        return Err(AttnError::ScaleOutOfRange);
    }
    // `checked_shl` is unmodeled; `head_dim · 2^FIXED_FRAC_BITS` is the same radicand and
    // overflows `i64` on exactly the inputs the shift rejects.
    let radicand = hd_signed
        .checked_mul(ONE)
        .ok_or(AttnError::ScaleOutOfRange)?;
    let scale = fixed_rsqrt(radicand).map_err(|_| AttnError::ScaleOutOfRange)?;
    // `Option::and_then` is unmodeled; the nested match is the same checked product.
    let plane = match n_head_kv.checked_mul(cap) {
        Some(x) => match x.checked_mul(head_dim) {
            Some(plane) => plane,
            None => return Err(AttnError::Overflow),
        },
        None => return Err(AttnError::Overflow),
    };
    Ok((plane, scale))
}

/// The byte offsets of the `(key, value)` planes of `layer` in a cache file whose planes are each
/// `plane_bytes` wide, key then value per layer.
fn plane_offsets(plane_bytes: usize, layer: usize) -> Option<(usize, usize)> {
    let k = plane_bytes.checked_mul(layer.checked_mul(2)?)?;
    let v = k.checked_add(plane_bytes)?;
    Some((k, v))
}

/// The single-token key/value cache of port-map row 5: one flat Q32.32 or binary32 plane per layer.
///
/// Keys and values are stored position-major, so position `p` carries all `n_head_kv` rows
/// contiguously and the row for `(p, kv_head)` is the `head_dim` values at
/// `(p * n_head_kv + kv_head) * head_dim`. One KV head serves `n_head / n_head_kv` query heads
/// (GQA). [`KvState::append`] writes a whole token's KV rows at the next position of a layer and
/// returns that position; [`KvState::attn_head`] reads one query head over the causal prefix
/// `0..=pos`.
///
/// The planes are owned vectors ([`KvState::new`]) or ranges of one writable cache file
/// ([`KvState::with_file`]). The file backing changes where the words live, not the words: every
/// read and write below is shared, and a clone materialises the planes into owned memory so two
/// writable states never share the file.
#[derive(Debug)]
pub struct KvState {
    carrier: KvCarrier,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    cap: usize,
    scale: i64,
    occupied: Vec<usize>,
    storage: KvStorage,
    /// The cache file shared by every mapped plane, or `None` for an owned state.
    map: Option<Arc<KvMap>>,
    /// The exact-on-certificate block summaries, or `None` for the full-scan read.
    blocks: Option<cert::KvBlockIndex>,
}

impl Clone for KvState {
    /// A cloned state owns its planes.
    ///
    /// A mapped plane is copied into an owned vector, so the clone and the original never share a
    /// writable mapping and the clone cannot overwrite the file the original reads. The prefix
    /// cache is the only clone site; the serving path never clones a file-backed state.
    fn clone(&self) -> Self {
        Self {
            carrier: self.carrier,
            n_layer: self.n_layer,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            head_dim: self.head_dim,
            cap: self.cap,
            scale: self.scale,
            occupied: self.occupied.clone(),
            storage: self.storage.clone(),
            map: None,
            blocks: self.blocks.clone(),
        }
    }
}

/// The Q32.32 position-major value of `(s, kvh, i)` in a layer's key or value plane.
///
/// The checked index product is the one the pinned Charon preset keeps as `checked.*` (it cannot
/// prove the bound), so the real `(s · n_head_kv + kvh) · head_dim + i` arithmetic appears here as
/// checked steps. The guards in [`KvState::attn_head`] admit only in-range positions.
fn plane_at(kv: usize, kvh: usize, hd: usize, s: usize, i: usize) -> Result<usize, AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    base.checked_add(i).ok_or(AttnError::Overflow)
}

/// Fold one key into the fused state: `den += exp_word` and `num[comp] += exp_word · v[comp]` for
/// each `comp < hd`, every `i128` step checked. Stops at the first failure, which is returned; the
/// partial update is discarded by the caller. Time `O(hd)` checked multiply-adds, space `O(1)`,
/// no allocation.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_key(
    score: i64,
    m: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    key: usize,
    den: &mut i128,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let exp_word = exp_at(score, m)?;
    let next = den
        .checked_add(i128::from(exp_word))
        .ok_or(AttnError::Overflow)?;
    *den = next;
    let mut comp = 0usize;
    let mut res: Result<(), AttnError> = Ok(());
    while comp < hd && res.is_ok() {
        res = fold_comp(exp_word, vplane, kv, kvh, hd, key, comp, num);
        comp += 1;
    }
    res
}

/// The key loop of the fused fold: [`fold_key`] for each `scores[key]`, in index order, stopping at
/// the first error. Time `O(count · hd)` checked operations, space `O(1)`.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_keys(
    scores: &[i64],
    m: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    den: &mut i128,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let count = scores.len();
    let mut key = 0usize;
    let mut res: Result<(), AttnError> = Ok(());
    while key < count && res.is_ok() {
        res = fold_key(scores[key], m, vplane, kv, kvh, hd, key, den, num);
        key += 1;
    }
    res
}

/// One component of [`fold_key`]: `num[comp] += exp_word · v[plane_at(comp)]`, checked. Straight
/// line, no loop; `O(1)` checked operations.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_comp(
    exp_word: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    key: usize,
    comp: usize,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let at = plane_at(kv, kvh, hd, key, comp)?;
    let vb = *vplane.get(at).ok_or(AttnError::Overflow)?;
    let term = i128::from(exp_word)
        .checked_mul(i128::from(vb))
        .ok_or(AttnError::Overflow)?;
    let next = num[comp].checked_add(term).ok_or(AttnError::Overflow)?;
    num[comp] = next;
    Ok(())
}

/// The checked `[base, base + head_dim)` bounds of the `head_dim`-wide row of key `s` in a
/// position-major plane: the same three `checked.*` index steps as [`plane_at`], with the row end
/// as one more. Test-only: the index oracle the hoisted `first + s·stride` base in
/// [`KvState::attn_head`] is pinned against, so the checked chain and the product stay the same
/// integer.
#[cfg(test)]
fn row_bounds(kv: usize, kvh: usize, hd: usize, s: usize) -> Result<(usize, usize), AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    let end = base.checked_add(hd).ok_or(AttnError::Overflow)?;
    Ok((base, end))
}

/// One Q32.32 causal key score: `row_dot_fixed` of `q` against key `s`, then the load-time `scale`
/// epilogue. The real `row_dot_fixed`, `checked_mul` and `round_shift_32` bodies are called.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn score_at(
    kplane: &[i64],
    q: &[i64],
    scale: i64,
    kv: usize,
    kvh: usize,
    hd: usize,
    s: usize,
) -> Result<i64, AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    let end = base.checked_add(hd).ok_or(AttnError::Overflow)?;
    let raw = row_dot_fixed(&kplane[base..end], q).map_err(|_| AttnError::Overflow)?;
    let prod = i128::from(raw)
        .checked_mul(i128::from(scale))
        .ok_or(AttnError::Overflow)?;
    let rounded = round_shift_32(prod);
    if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(rounded as i64)
}

/// One softmax numerator: the exact maximum shift, the `fixed_exp` window with the documented
/// underflow saturation below `-16·ONE`, and no clamping.
fn exp_at(score: i64, mx: i64) -> Result<i64, AttnError> {
    let shifted = score.checked_sub(mx).ok_or(AttnError::Overflow)?;
    if shifted < -16 * ONE {
        Ok(0)
    } else {
        fixed_exp(shifted).map_err(|_| AttnError::ExpOutOfRange)
    }
}

/// One Q32.32 softmax weight: the round-to-nearest division `round(e·2^32 / total)`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn weight_at(e: i64, total: i128) -> Result<i64, AttnError> {
    let scaled = i128::from(e)
        .checked_mul(i128::from(ONE))
        .ok_or(AttnError::Overflow)?;
    let p = div_round_nearest(scaled, total);
    if p > i128::from(i64::MAX) || p < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(p as i64)
}

/// The round-up `u64` reciprocal `floor(2^64 / total) + 1` of [`weight_at_recip`], or `None` when
/// `total` leaves the window `2 ..= u64::MAX/2` the two-operand product bound needs.
///
/// `floor(2^64/total) + 1` is the classical round-up multiplier: `n2 · recip / 2^64` is then
/// `floor(n2/total)` or one greater, which the single subtract fixup in [`weight_at_recip`] removes.
/// The truncated `floor(2^64/total)` reciprocal is one *below* `floor(n2/total)` whenever
/// `2^64 mod total != 0` and the product lands on the low side (for example `total = 3`, `e = 2`),
/// and a subtract-only fixup cannot repair that.
///
/// # Complexity
///
/// O(1) time and space: one `u128` division, one add and two comparisons.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the `total >= 2` guard makes the quotient `<= 2^63`, so both casts are exact; \
              `i64::try_from`/`u64::try_from` would emit unmodeled externals the fail-closed \
              extraction audit rejects"
)]
fn recip_u64(total: i128) -> Option<u64> {
    if total < 2 || total > i128::from(u64::MAX / 2) {
        return None;
    }
    let base = ((1u128 << 64) / (total as u128)) as u64;
    base.checked_add(1)
}

/// One softmax weight through the `u64` round-up reciprocal: the same
/// `div_round_nearest(e·2^32, total)` value as [`weight_at`], with the division replaced by one
/// `u128` multiply and shift.
///
/// The fast path applies exactly when `e >= 0`, `e·2^32 + total/2` fits `u64`, and `total` fits the
/// `recip_u64` window (guaranteed by the caller's `Some`); every other operand — a negative `e`, an
/// `i128` overflow, or a numerator above `u64::MAX` — delegates to the literal i128 [`weight_at`].
///
/// # Complexity
///
/// O(1) time and space: one `u128` multiply, one shift and one fixup comparison.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "every `as` is guarded: `e >= 0` fixes the sign of `n`, `n <= i128::from(u64::MAX)` \
              and the caller's `total <= u64::MAX/2` fix the widths, and the `i64` narrowing is \
              range-checked against i128::from(i64::MAX)/i64::MIN immediately before it; the \
              `try_from` forms emit unmodeled externals the fail-closed extraction audit rejects"
)]
fn weight_at_recip(e: i64, total: i128, recip: u64) -> Result<i64, AttnError> {
    let scaled = i128::from(e).checked_mul(i128::from(ONE));
    let narrowed = match scaled {
        Some(s) => match s.checked_add(total / 2) {
            Some(n) => {
                if e >= 0 {
                    if n <= i128::from(u64::MAX) {
                        Some(n as u64)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None,
        },
        None => None,
    };
    match narrowed {
        Some(n2) => {
            let d = total as u64;
            let q = ((u128::from(n2) * u128::from(recip)) >> 64) as u64;
            let q = if u128::from(q) * u128::from(d) > u128::from(n2) {
                q - 1
            } else {
                q
            };
            let p = i128::from(q);
            if p > i128::from(i64::MAX) || p < i128::from(i64::MIN) {
                return Err(AttnError::Overflow);
            }
            Ok(p as i64)
        }
        None => weight_at(e, total),
    }
}

/// One `p_s · v_s` term of the output component `i`. Test-only: the key-outer production output
/// pass folds the same terms into all components at once, and this per-term oracle keeps the
/// old per-component arithmetic exercised by the output-pin test.
#[cfg(test)]
fn value_term(
    vplane: &[i64],
    weights: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    s: usize,
    i: usize,
) -> Result<i128, AttnError> {
    let at = plane_at(kv, kvh, hd, s, i)?;
    let vb = *vplane.get(at).ok_or(AttnError::Overflow)?;
    i128::from(weights[s])
        .checked_mul(i128::from(vb))
        .ok_or(AttnError::Overflow)
}

/// The single-component output accumulation: the checked `i128` sum over the causal prefix, then
/// one `round_shift_32` epilogue. The position loop carries a `bool` failure flag so no early
/// return appears inside a loop, the extractable form the pinned fragment admits.
///
/// Test-only oracle for the per-component scan the key-outer production pass replaces.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn value_component(
    vplane: &[i64],
    weights: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    pos: usize,
    i: usize,
) -> Result<i64, AttnError> {
    let mut acc: i128 = 0;
    let mut s = 0usize;
    let mut ok = true;
    let mut err: Option<AttnError> = None;
    while s <= pos && ok {
        match value_term(vplane, weights, kv, kvh, hd, s, i) {
            Ok(term) => {
                if let Some(a) = acc.checked_add(term) {
                    acc = a;
                } else {
                    err = Some(AttnError::Overflow);
                    ok = false;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        s += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let rounded = round_shift_32(acc);
    if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(rounded as i64)
}

/// The fused online-softmax state of one query head: the fixed max-shift score and the exact
/// `i128` denominator and numerator sums of the causal prefix at that shift.
///
/// `den = Σ_{j≤pos} e_j` and `num_i = Σ_{j≤pos} e_j·v_{j,i}`, with `e_j = exp_at(score_j, m)`.
/// Both sums are exact checked `i128` additions, so they are independent of the order and of the
/// tile grouping of the keys (integer addition is associative and commutative). In real reading
/// `num_i/den` is `Σ_j softmax(score)_j·v_{j,i}` — the common shift cancels — the dense softmax
/// attention `Numeric.EngineAttnRefinement.denseAttnReal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttnFusedState {
    /// The fixed max-shift score `m`.
    pub m: i64,
    /// The exact softmax denominator `Σ_j e_j` (`> 0` for a state that was returned).
    pub den: i128,
    /// The exact numerator `Σ_j e_j·v_{j,i}`, `head_dim` wide.
    pub num: Vec<i128>,
}

/// Round-to-nearest `num / den` (`den > 0`) for a signed numerator: the sign-symmetric companion
/// of [`round_shift_32`], ties away from zero. The error against the exact quotient is below
/// `1/2 + 1/(2·den)`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn round_div_signed(num: i128, den: i128) -> Result<i64, AttnError> {
    let q = if num >= 0 {
        (num + den / 2) / den
    } else {
        // `(num - den/2) / den` rounds the magnitude to nearest without negating `num`; the plain
        // form stays total at `i128::MIN`.
        (num - den / 2) / den
    };
    if q > i128::from(i64::MAX) || q < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(q as i64)
}

impl KvState {
    /// The shared guard chain of the fused read, in the order [`KvState::attn_head`] checks it
    /// (layer, query head, query width, position); returns the GQA KV head `q_head` reads.
    fn fused_kvh(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != self.head_dim {
            return Err(AttnError::RowLength {
                expected: self.head_dim,
                got: q.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        Ok(q_head / (self.n_head / self.n_head_kv))
    }

    /// The score prefix `0..=pos` of the fused read: one [`score_at`] per causal key, the same
    /// scores [`KvState::attn_head`] computes. Returns the GQA KV head and the score list.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds, `O(pos + 1)` space for the
    /// score list: `O(pos + 1)` calls into `O(head_dim)` each, accumulated once per key.
    fn score_prefix(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<(usize, Vec<i64>), AttnError> {
        let kvh = self.fused_kvh(layer, q_head, q, pos)?;
        let hd = self.head_dim;
        let kv = self.n_head_kv;
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let kplane = match &self.storage {
            KvStorage::Q32 { k, .. } => &k[layer],
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        };
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match score_at(kplane, q, self.scale, kv, kvh, hd, s) {
                Ok(score) => scores.push(score),
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok((kvh, scores))
    }

    /// Fold the precomputed score prefix into a fused `(m, den, num)` state at the fixed shift
    /// `m`, one key at a time in index order. `tile` has no effect: the `i128` sums are the same
    /// for every value, including `0`, which `attn_fused_state` and `attn_head_fused` refuse with
    /// [`AttnError::BadGeometry`] before this call.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds and one [`fixed_exp`] per
    /// key, `O(head_dim)` space for `num` plus `O(1)` for the key cursor; the op count is
    /// independent of `tile`.
    fn fused_fold(
        &self,
        layer: usize,
        kvh: usize,
        scores: &[i64],
        m: i64,
        _tile: usize,
    ) -> Result<AttnFusedState, AttnError> {
        let hd = self.head_dim;
        let kv = self.n_head_kv;
        let vplane = match &self.storage {
            KvStorage::Q32 { v, .. } => &v[layer],
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        };
        let mut den: i128 = 0;
        let mut num: Vec<i128> = vec![0i128; hd];
        fold_keys(scores, m, &vplane[..], kv, kvh, hd, &mut den, &mut num[..])?;
        if den <= 0 {
            return Err(AttnError::ZeroSum);
        }
        Ok(AttnFusedState { m, den, num })
    }

    /// The exact maximum score over the causal prefix `0..=pos`: pass 1 of the fused read, the
    /// shift the fold fixes for every key. The prefix is nonempty (`pos < occupied`), so the
    /// result is a real score.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain and its `AttnError::Overflow` on a score epilogue.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time, `O(pos + 1)` space for the score prefix.
    #[must_use = "the max shift is the fold's fixed `m`"]
    pub fn attn_max(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<i64, AttnError> {
        let (_kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s < scores.len() {
            if scores[s] > mx {
                mx = scores[s];
            }
            s += 1;
        }
        Ok(mx)
    }

    /// The fused `(m, den, num)` state of the causal prefix at the fixed shift `m`, folded key by
    /// key. `m` is caller-supplied: the online read passes the exact maximum from
    /// [`KvState::attn_max`], so every key is shifted by one common value and the exact `i128`
    /// sums do not depend on `tile`.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain; [`AttnError::BadGeometry`] when `tile == 0`;
    /// [`AttnError::ZeroSum`] when `den == 0` (every shift below the `fixed_exp` window);
    /// [`AttnError::Overflow`] when an `i128` accumulation leaves range.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time, `O(head_dim)` space beyond the `O(pos + 1)` score prefix;
    /// independent of `tile`.
    pub fn attn_fused_state(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        m: i64,
        tile: usize,
    ) -> Result<AttnFusedState, AttnError> {
        if tile == 0 {
            return Err(AttnError::BadGeometry);
        }
        let (kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        self.fused_fold(layer, kvh, &scores, m, tile)
    }

    /// The fused online attention read of one query head at `pos`, the exact-scan companion of
    /// [`KvState::attn_head`]: the exact maximum score over the causal prefix, the exact
    /// `(m, den, num)` fold in `tile`-key groups, and one normalisation `round(num_i / den)` per
    /// output component. The per-probability path of [`KvState::attn_head`] is unchanged and
    /// remains the oracle.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain; [`AttnError::BadGeometry`] when `tile == 0`;
    /// [`AttnError::ZeroSum`] when the fold's denominator is zero (unreachable at the exact
    /// maximum, retained for a caller-supplied `m`); [`AttnError::Overflow`] when an `i128`
    /// accumulation or an `i64` narrowing leaves range.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds and one [`fixed_exp`] per
    /// key, `O(pos + 1)` space for the score prefix and `O(head_dim)` for the fused state; the
    /// per-position weight vector and its `O(pos + 1)` divisions are removed.
    pub fn attn_head_fused(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        tile: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        if tile == 0 {
            return Err(AttnError::BadGeometry);
        }
        if self.carrier == KvCarrier::I8 {
            // The int8 carrier has one read and no tile; the per-head call is its one-slot case.
            return self.attn_head(layer, q_head, q, pos, out);
        }
        if self.carrier == KvCarrier::F32 {
            return self.attn_head_f32(layer, q_head, q, pos, out);
        }
        if out.len() != self.head_dim {
            return Err(AttnError::RowLength {
                expected: self.head_dim,
                got: out.len(),
            });
        }
        let (kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        // Pass 1 of the online read: the exact global maximum, the one shift every tile folds
        // under. A per-tile maximum would rescale the running sums and make the result depend on
        // the tile boundaries.
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s < scores.len() {
            if scores[s] > mx {
                mx = scores[s];
            }
            s += 1;
        }
        let state = self.fused_fold(layer, kvh, &scores, mx, tile)?;
        let mut i = 0usize;
        while i < self.head_dim {
            out[i] = round_div_signed(state.num[i], state.den)?;
            i += 1;
        }
        Ok(())
    }

    /// Allocate a zeroed cache for `n_layer` layers, `n_head` query heads, `n_head_kv` KV heads,
    /// `head_dim` values per head and at most `cap` positions per layer, resolving the carrier
    /// from the process environment (`carrier_from_env()`).
    ///
    /// The load-time score scale `fixed_rsqrt(head_dim)` is computed once here, from the same
    /// integer kernel the executing path uses; `head_dim` is admitted in the `fixed_rsqrt` window.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::BadGeometry`] for a zero parameter or `n_head` not a multiple of
    /// `n_head_kv`, [`AttnError::ScaleOutOfRange`] when `head_dim` leaves the `fixed_rsqrt` window,
    /// and [`AttnError::Overflow`] when the per-layer plane length leaves `usize`.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv · cap · head_dim)` time and space: one zeroed plane per layer.
    pub fn new(
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
    ) -> Result<KvState, AttnError> {
        Self::with_carrier(
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            kv_carrier_from_env(),
        )
    }

    /// Allocate a zeroed cache with an explicitly specified [`KvCarrier`].
    ///
    /// Under [`KvCarrier::Q32`], allocates Q32.32 `i64` planes. Under [`KvCarrier::F32`],
    /// allocates native binary32 `f32` planes (50% memory footprint, zero per-call float
    /// conversion). Under [`KvCarrier::I8`], allocates signed code planes and one binary32 scale
    /// per `(kv_head, position)`.
    ///
    /// # Errors
    ///
    /// Same geometry and range refusals as [`KvState::new`]; [`KvCarrier::I8`] adds the per-row
    /// scale plane to the checked product.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv · cap · head_dim)` time and space for the Q32.32 and binary32
    /// carriers, and `O(n_layer · n_head_kv · cap · (head_dim + 2))` for the int8 carrier.
    pub(crate) fn with_carrier(
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
        carrier: KvCarrier,
    ) -> Result<KvState, AttnError> {
        let (plane, scale) = checked_geometry(n_layer, n_head, n_head_kv, head_dim, cap)?;
        let storage = match carrier {
            KvCarrier::Q32 => {
                let blank = KvPlane::owned(vec![0i64; plane]);
                let k = vec![blank.clone(); n_layer];
                let v = vec![blank; n_layer];
                KvStorage::Q32 { k, v }
            }
            KvCarrier::F32 => alloc_f32_storage(plane, n_layer),
            KvCarrier::I8 => {
                let rows = n_head_kv.checked_mul(cap).ok_or(AttnError::Overflow)?;
                alloc_i8_storage(plane, rows, n_layer)
            }
        };
        Ok(KvState {
            carrier,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            scale,
            occupied: vec![0usize; n_layer],
            storage,
            map: None,
            blocks: None,
        })
    }

    /// Allocate a cache whose planes are disjoint ranges of one writable `MAP_SHARED` cache file.
    ///
    /// The file holds the same words an owned cache would: `2 · n_layer` position-major planes of
    /// `n_head_kv · cap · head_dim` elements, the key plane of layer `0`, its value plane, then
    /// layer `1`, and so on. It is allocated to its full extent with `posix_fallocate` at creation,
    /// so a capacity the filesystem cannot hold is refused here with [`KvInitError::File`] instead
    /// of `SIGBUS` on a later store. The file is a cache: its bytes carry no header and are
    /// host-endian fixed-point or binary32 words.
    ///
    /// # Errors
    ///
    /// [`KvInitError::Geometry`] for a geometry [`KvState::with_carrier`] refuses, and
    /// [`KvInitError::File`] for a refused create, allocation or mapping. A zero-length file is
    /// never created.
    ///
    /// # Complexity
    ///
    /// `O(1)` file operations and page-table entries; no plane page is touched until a token is
    /// appended. The file's `2 · n_layer · plane · size_of::<T>()` bytes are reserved by the
    /// filesystem, which is what the error path checks.
    #[cfg(target_os = "linux")]
    pub fn with_file(
        path: &str,
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
    ) -> Result<KvState, KvInitError> {
        Self::with_file_carrier(
            path,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            carrier_from_env(),
        )
    }

    /// [`KvState::with_file`] with an explicit numerical carrier.
    ///
    /// # Errors
    ///
    /// As [`KvState::with_file`].
    ///
    /// # Complexity
    ///
    /// As [`KvState::with_file`].
    #[cfg(target_os = "linux")]
    pub(crate) fn with_file_carrier(
        path: &str,
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
        carrier: Carrier,
    ) -> Result<KvState, KvInitError> {
        let (plane, scale) = checked_geometry(n_layer, n_head, n_head_kv, head_dim, cap)?;
        let elem = match carrier {
            Carrier::Q32 => core::mem::size_of::<i64>(),
            Carrier::F32 => core::mem::size_of::<F32>(),
        };
        let plane_bytes = plane
            .checked_mul(elem)
            .ok_or(AttnError::Overflow)
            .map_err(KvInitError::Geometry)?;
        let bytes = plane_bytes
            .checked_mul(n_layer)
            .and_then(|x| x.checked_mul(2))
            .ok_or(AttnError::Overflow)
            .map_err(KvInitError::Geometry)?;
        let map = Arc::new(KvMap::create(path, bytes)?);
        let storage = match carrier {
            Carrier::Q32 => {
                let mut k = Vec::with_capacity(n_layer);
                let mut v = Vec::with_capacity(n_layer);
                let mut layer = 0usize;
                while layer < n_layer {
                    let (koff, voff) = plane_offsets(plane_bytes, layer)
                        .ok_or(AttnError::Overflow)
                        .map_err(KvInitError::Geometry)?;
                    k.push(KvPlane::<i64>::mapped(Arc::clone(&map), koff, plane)?);
                    v.push(KvPlane::<i64>::mapped(Arc::clone(&map), voff, plane)?);
                    layer += 1;
                }
                KvStorage::Q32 { k, v }
            }
            Carrier::F32 => {
                let mut k = Vec::with_capacity(n_layer);
                let mut v = Vec::with_capacity(n_layer);
                let mut layer = 0usize;
                while layer < n_layer {
                    let (koff, voff) = plane_offsets(plane_bytes, layer)
                        .ok_or(AttnError::Overflow)
                        .map_err(KvInitError::Geometry)?;
                    k.push(KvPlane::<F32>::mapped(Arc::clone(&map), koff, plane)?);
                    v.push(KvPlane::<F32>::mapped(Arc::clone(&map), voff, plane)?);
                    layer += 1;
                }
                KvStorage::F32 { k, v }
            }
        };
        Ok(KvState {
            carrier: match carrier {
                Carrier::Q32 => KvCarrier::Q32,
                Carrier::F32 => KvCarrier::F32,
            },
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            scale,
            occupied: vec![0usize; n_layer],
            storage,
            map: Some(map),
            blocks: None,
        })
    }

    /// A file-backed cache on a target without writable mappings refuses.
    ///
    /// # Errors
    ///
    /// Always [`KvInitError::File`] with [`KvFileError::Unsupported`].
    #[cfg(not(target_os = "linux"))]
    pub fn with_file(
        _path: &str,
        _n_layer: usize,
        _n_head: usize,
        _n_head_kv: usize,
        _head_dim: usize,
        _cap: usize,
    ) -> Result<KvState, KvInitError> {
        Err(KvInitError::File(KvFileError::Unsupported))
    }

    /// Write every dirty cache-file page back and wait for the writes; a no-op for an owned cache.
    ///
    /// # Errors
    ///
    /// The mapping's `msync` refusal.
    ///
    /// # Complexity
    ///
    /// `O(dirty pages)` I/O; one `msync` call.
    pub fn sync_file(&self) -> Result<(), KvFileError> {
        match &self.map {
            Some(map) => map.sync(),
            None => Ok(()),
        }
    }

    /// The cache file's byte length, or `None` for an owned cache.
    #[must_use]
    pub fn file_bytes(&self) -> Option<usize> {
        self.map.as_ref().map(|m| m.len())
    }

    /// The fewest occupied positions across the store's layers. A snapshot may be filed under a
    /// prompt of this length or shorter; a longer prompt would resume at a layer's short offset.
    pub(crate) fn min_occupied(&self) -> Result<usize, AttnError> {
        let mut m = usize::MAX;
        for l in 0..self.n_layer {
            m = m.min(self.occupied(l)?);
        }
        Ok(m)
    }

    /// The numerical carrier for this key/value cache.
    #[must_use]
    pub(crate) fn carrier(&self) -> KvCarrier {
        self.carrier
    }

    /// The resident bytes of one position across every layer and both K and V planes, in this
    /// state's carrier. `I8` adds its one `f32` scale per row per plane. The prefix cache charges
    /// `bytes_per_position() × cap()` per snapshot, and `serve` uses it to decide whether a pinned
    /// capacity fits a byte budget.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        let values = 2 * self.n_head_kv * self.head_dim * self.n_layer;
        match self.carrier() {
            KvCarrier::Q32 => values * 8,
            KvCarrier::F32 => values * 4,
            KvCarrier::I8 => values + 2 * self.n_head_kv * self.n_layer * 4,
        }
    }

    /// The configured layer count.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// The configured query-head count.
    #[must_use]
    pub fn n_head(&self) -> usize {
        self.n_head
    }

    /// The configured KV-head count.
    #[must_use]
    pub fn n_head_kv(&self) -> usize {
        self.n_head_kv
    }

    /// The per-head width.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The per-layer position capacity.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The load-time Q32.32 score scale `fixed_rsqrt(head_dim)`.
    #[must_use]
    pub fn scale(&self) -> i64 {
        self.scale
    }

    /// Positions appended for `layer`.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`] for a layer index at or past the layer count.
    pub fn occupied(&self, layer: usize) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        Ok(self.occupied[layer])
    }

    /// Append one token's KV rows at the layer's next position and return that position.
    ///
    /// `k_rows` and `v_rows` each hold the `n_head_kv` heads of one token's `head_dim`-wide rows
    /// (`n_head_kv · head_dim` values), KV head `h` first. This is the single-token step of the
    /// fold `kv_write`; the position advances only after the whole token's rows are written.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`] for a layer index at or past the layer count,
    /// [`AttnError::RowLength`] when a block is not `n_head_kv · head_dim` wide, and
    /// [`AttnError::CapacityExceeded`] when the layer already holds `cap` positions. No row is
    /// clamped or truncated.
    ///
    /// **The destination plane's contents are unspecified on `Err`.** The `F32` carrier writes
    /// the whole row before raising [`AttnError::Overflow`], so a rejecting element leaves a
    /// partially-written row past that element rather than an untouched tail; the earlier
    /// raise-at-first-failure guarantee is not preserved. `occupied` is not advanced and no
    /// caller reads the plane on `Err`.
    ///
    /// # Complexity
    ///
    /// `O(n_head_kv · head_dim)` time and no allocation: the two row blocks are copied into the
    /// layer planes.
    pub fn append(
        &mut self,
        layer: usize,
        k_rows: &[i64],
        v_rows: &[i64],
    ) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        let width = self
            .n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(AttnError::Overflow)?;
        if k_rows.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: k_rows.len(),
            });
        }
        if v_rows.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: v_rows.len(),
            });
        }
        let pos = self.occupied[layer];
        if pos >= self.cap {
            return Err(AttnError::CapacityExceeded { cap: self.cap });
        }
        // Checked forms throughout: a plain `*`/`+` on `usize` is lowered by the pinned Charon
        // preset to `checked.*` with an overflow assert, which the Aeneas fragment rejects as an
        // unimplemented binary operation. Each product is bounded by the checked `width` and the
        // admitted `pos < cap`, so no reachable input changes branch.
        let base = pos.checked_mul(width).ok_or(AttnError::Overflow)?;
        let end = base.checked_add(width).ok_or(AttnError::Overflow)?;
        match &mut self.storage {
            KvStorage::Q32 { k, v } => {
                k[layer][base..end].copy_from_slice(k_rows);
                v[layer][base..end].copy_from_slice(v_rows);
            }
            // Index `while` loops, not `iter_mut().zip`, and one `Err` after the loops. Both
            // removals are forced by the pinned Aeneas model, and neither is a change of shape
            // choice: the model has no field for `Iterator::zip` (prepass reports `Could not
            // find the information for item 'zip'`), and a `return` whose target is the function
            // boundary crossed from inside a loop nest is refused outright (`Returns inside of
            // nested loops are not supported yet`). Arithmetic is unchanged: the same
            // `q32_to_f32` on the same element in the same order, the same store to the same
            // slot, and the same `is_finite` test on the same value, over the index range
            // `0..width` that the two length checks above have already made equal to
            // `k_dst.len() == v_dst.len()`.
            KvStorage::F32 { k, v } => {
                let mut bad = 0u32;
                let k_dst = &mut k[layer][base..end];
                let mut i = 0usize;
                while i < width {
                    let f = q32_to_f32(k_rows[i]);
                    bad += u32::from(!f.is_finite());
                    k_dst[i] = f;
                    i += 1;
                }
                let v_dst = &mut v[layer][base..end];
                let mut i = 0usize;
                while i < width {
                    let f = q32_to_f32(v_rows[i]);
                    bad += u32::from(!f.is_finite());
                    v_dst[i] = f;
                    i += 1;
                }
                if bad != 0 {
                    return Err(AttnError::Overflow);
                }
            }
            // One quantisation per KV head and per K/V row: each row has its own scale, and the
            // codes are written KV-head-major so the group read sees one contiguous run. The
            // cursors advance by whole-head strides, so the loop body holds no checked index
            // product and no early return.
            KvStorage::I8 { k, v, ks, vs } => {
                let hd = self.head_dim;
                let cap = self.cap;
                let kv = self.n_head_kv;
                let head_stride = cap.checked_mul(hd).ok_or(AttnError::Overflow)?;
                let mut at = pos.checked_mul(hd).ok_or(AttnError::Overflow)?;
                let mut s_at = pos;
                let (k_plane, v_plane) = (&mut k[layer], &mut v[layer]);
                let (k_scales, v_scales) = (&mut ks[layer], &mut vs[layer]);
                let mut row = 0usize;
                let mut h = 0usize;
                let mut err: Option<AttnError> = None;
                while h < kv && err.is_none() {
                    let row_end = row + hd;
                    let at_end = at + hd;
                    match i8::quantise_row(&k_rows[row..row_end], &mut k_plane[at..at_end]) {
                        Ok(s) => k_scales[s_at] = s,
                        Err(e) => err = Some(e),
                    }
                    if err.is_none() {
                        match i8::quantise_row(&v_rows[row..row_end], &mut v_plane[at..at_end]) {
                            Ok(s) => v_scales[s_at] = s,
                            Err(e) => err = Some(e),
                        }
                    }
                    row += hd;
                    at += head_stride;
                    s_at += cap;
                    h += 1;
                }
                if let Some(e) = err {
                    return Err(e);
                }
            }
        }
        self.occupied[layer] = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        let len = self.occupied[layer];
        self.seal_completed_block(layer, len)?;
        Ok(pos)
    }

    /// Seal the block summary of `layer` when `len` positions complete one.
    ///
    /// Called by [`KvState::append`] after the position counter advances. A block is sealed once,
    /// when its last position is written; the summary is a pure function of the key plane's
    /// `0..len` rows, so a sealed block is never revisited unless a truncation drops it.
    ///
    /// # Complexity
    ///
    /// `O(1)` when no block completes. At a boundary, `O(n_head_kv · block · head_dim)` additions
    /// and comparisons: one mean pass and one deviation pass over the completed block for each KV
    /// head, so the amortised cost is `O(n_head_kv · head_dim)` per appended token.
    fn seal_completed_block(&mut self, layer: usize, len: usize) -> Result<(), AttnError> {
        let block = match &self.blocks {
            Some(idx) => idx.block(),
            None => return Ok(()),
        };
        if block == 0 || !len.is_multiple_of(block) {
            return Ok(());
        }
        let index = len / block;
        let kplane: &[i64] = match &self.storage {
            KvStorage::Q32 { k, .. } => &k[layer],
            // The block index is Q32-only (`enable_block_index` refuses the other carriers), so
            // neither arm is reachable with `blocks` set.
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Ok(()),
        };
        if let Some(idx) = self.blocks.as_mut() {
            idx.seal(
                layer,
                kplane,
                index.checked_sub(1).ok_or(AttnError::Overflow)?,
            );
        }
        Ok(())
    }

    /// Build the exact-on-certificate block index over the positions already appended, in blocks of
    /// `block` positions, and use it for every later [`KvState::attn_head`].
    ///
    /// The index stores, for each `(layer, kv_head, block)`, the rounded component mean `μ` of the
    /// block's keys and the largest `radius = Σ_i |k_{j,i} − μ_i|` over the block. For any query
    /// `q`, `|q·k_j − q·μ| ≤ max_i |q_i| · radius`, which bounds the block's exact score in both
    /// directions ([`KvState::attn_head_certified`]). The summaries are rebuilt for every completed
    /// block; a partial trailing block is not summarised and is always scanned.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadGeometry`] for `block == 0` or an `F32` carrier, and the geometry refusals
    /// the state already passed.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv · occupied · head_dim)` time to rebuild, and
    /// `n_layer · n_head_kv · ⌈occupied/block⌉ · head_dim · 8` bytes plus the radii: one `head_dim`
    /// mean per block. `block == 1` is the full index (never useful); `block >= occupied` is empty.
    pub fn enable_block_index(&mut self, block: usize) -> Result<(), AttnError> {
        if block == 0 || self.carrier != KvCarrier::Q32 {
            return Err(AttnError::BadGeometry);
        }
        let mut index = cert::KvBlockIndex::new(self.n_layer, self.n_head_kv, self.head_dim, block);
        match &self.storage {
            KvStorage::Q32 { k, .. } => {
                let mut layer = 0usize;
                while layer < self.n_layer {
                    index.rebuild_layer(layer, &k[layer], self.occupied[layer]);
                    layer += 1;
                }
            }
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        }
        self.blocks = Some(index);
        Ok(())
    }

    /// Drop the block index, restoring the full-scan read.
    pub fn disable_block_index(&mut self) {
        self.blocks = None;
    }

    /// Whether a block index is installed.
    #[must_use]
    pub fn has_block_index(&self) -> bool {
        self.blocks.is_some()
    }

    /// The running [`CertStats`] of every certified read through the installed index, or `None`
    /// when no index is installed.
    ///
    /// The counters accumulate over the state's lifetime; [`KvState::reset_cert_stats`] clears
    /// them (for example at the start of one served request).
    #[must_use]
    pub fn cert_stats(&self) -> Option<CertStats> {
        self.blocks.as_ref().map(cert::KvBlockIndex::totals)
    }

    /// Clear the running certificate counters, leaving the summaries in place.
    pub fn reset_cert_stats(&mut self) {
        if let Some(index) = self.blocks.as_mut() {
            index.reset_totals();
        }
    }

    /// Read one query head at position `pos` over the causal prefix `0..=pos`.
    ///
    /// The score is `round(round(Σ q·k_j / 2^32) · scale / 2^32)` — [`row_dot_fixed`] then one
    /// value-path epilogue with the load-time `scale`. The softmax subtracts the exact maximum and
    /// evaluates [`fixed_exp`] on `shifted = score - mx >= -16·ONE`; a `shifted < -16·ONE` saturates
    /// its numerator to `0`. The total is a checked `i128` sum, each weight is the round-to-nearest
    /// `p_j = round(e_j·2^32 / Σ e)`, and the output is `round(Σ p_j·v_j / 2^32)` under the same
    /// epilogue. Query head `q_head` reads KV head `q_head / (n_head / n_head_kv)` (GQA); positions
    /// past `pos` are never read (causal mask).
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`], [`AttnError::HeadOutOfRange`],
    /// [`AttnError::RowLength`] or [`AttnError::PositionOutOfRange`] for a bad index or width,
    /// [`AttnError::ZeroSum`] only for a genuinely empty total, and [`AttnError::Overflow`] when an
    /// `i128` accumulation or an `i64` narrowing leaves its range. Scores are exact; only the
    /// softmax numerator below `-16·ONE` saturates, so no attended input is clamped.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked integer multiply-adds and `fixed_exp` calls, with
    /// `O(pos + 1)` scratch space for the scores and exponentials; `out` is `head_dim`
    /// wide.
    #[allow(
        clippy::too_many_lines,
        reason = "the one linear score/softmax/weight/value composition of port-map row 5; the \
                  four bodies keep the fold's stage order and each carries the extraction \
                  fragment's explicit failure-flag loop shape"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
                  immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
                  extraction audit rejects"
    )]
    pub fn attn_head(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        if self.carrier == KvCarrier::F32 {
            return self.attn_head_f32(layer, q_head, q, pos, out);
        }
        if self.blocks.is_some() {
            // The certified read returns the same words as the full scan below (the skipped
            // blocks' `fixed_exp` words are exactly zero), so the caller sees one read.
            return self.attn_head_certified(layer, q_head, q, pos, out, &mut CertStats::default());
        }
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: q.len(),
            });
        }
        if out.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: out.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let group = self.n_head / self.n_head_kv;
        let kvh = q_head / group;
        if self.carrier == KvCarrier::I8 {
            // The per-head read is the group read with one slot: the same codes, the same
            // binary32 arithmetic and the same Q32.32 readout conversion.
            return self.attn_head_group_i8(layer, kvh, q, pos, out, 1, hd);
        }
        let (kplane, vplane) = match &self.storage {
            KvStorage::Q32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        };
        let kv = self.n_head_kv;
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        // The position-major row `(s·kv + kvh)·hd` is the hoisted `first + s·stride`; the guards
        // admit only `s <= pos < occupied <= cap`, so `base + hd <= plane.len()` and the product is
        // exact without a per-key checked chain. One bound check per key is the same refusal the
        // checked index chain produced.
        let stride = kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let first = kvh.checked_mul(hd).ok_or(AttnError::Overflow)?;

        // Scores: one Q32.32 dot per causal key, then one epilogue against the load-time scale.
        // Each loop carries a `bool` failure flag and stops on the first refusal, so no early
        // return appears inside a body; the refusals, arithmetic and order are unchanged.
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            let base = first + s * stride;
            if base > kplane.len().saturating_sub(hd) {
                err = Some(AttnError::Overflow);
                ok = false;
            } else {
                let end = base + hd;
                if let Ok(raw) = row_dot_fixed(&kplane[base..end], q) {
                    if let Some(prod) = i128::from(raw).checked_mul(i128::from(self.scale)) {
                        let rounded = round_shift_32(prod);
                        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                            err = Some(AttnError::Overflow);
                            ok = false;
                        } else {
                            let score = rounded as i64;
                            if score > mx {
                                mx = score;
                            }
                            scores.push(score);
                        }
                    } else {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    }
                } else {
                    err = Some(AttnError::Overflow);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }

        // Softmax: exact max shift, `fixed_exp` over `[-16·ONE, 0]`, underflow saturation below
        // `-16·ONE`, checked i128 total.
        let mut exps: Vec<i64> = Vec::with_capacity(slots);
        let mut total: i128 = 0;
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match exp_at(scores[s], mx) {
                Ok(e) => {
                    if let Some(t) = total.checked_add(i128::from(e)) {
                        total = t;
                        exps.push(e);
                    } else {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    }
                }
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        if total <= 0 {
            return Err(AttnError::ZeroSum);
        }

        // The weights fold into the output pass: each key's round-to-nearest division is resolved
        // immediately before that key's row accumulates, replacing the separate `weights` vector
        // and its pass. The reciprocal and its Option are resolved once per call; the output loop
        // only picks the fast or the literal i128 path. `exps` are non-negative with `total >= ONE`
        // here, so the weight never refuses where the old weights pass would have; a refusal at key
        // `s` is raised with the same `AttnError` at the same ascending key. A refused read still
        // leaves `out` untouched, because `out` is written only after every key folds.
        let recip = recip_u64(total);

        // Output: one key-outer pass. Each key's weight is resolved, then its `head_dim` row is read
        // contiguously once and its `weight·v[s, i]` products accumulate into the independent
        // per-component checked `i128` sums. For each `i` the terms are the same checked products in
        // the same ascending `s` order as the per-component oracle, so every `acc[i]` is the
        // identical sum and the single `round_shift_32` epilogue plus its `i64` range check are
        // unchanged; only the interleaving of independent sums changes (integer addition is
        // associative and commutative). One `bool` failure flag per loop, and no early return inside
        // a body, keep the pinned extraction fragment's shape.
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r),
                None => weight_at(exps[s], total),
            };
            match w {
                Ok(weight) => {
                    let base = first + s * stride;
                    if base > vplane.len().saturating_sub(hd) {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    } else {
                        let end = base + hd;
                        let row = &vplane[base..end];
                        let w = i128::from(weight);
                        let mut i = 0usize;
                        while i < hd && ok {
                            if let Some(term) = w.checked_mul(i128::from(row[i])) {
                                if let Some(next) = acc[i].checked_add(term) {
                                    acc[i] = next;
                                } else {
                                    err = Some(AttnError::Overflow);
                                    ok = false;
                                }
                            } else {
                                err = Some(AttnError::Overflow);
                                ok = false;
                            }
                            i += 1;
                        }
                    }
                }
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < hd && ok {
            let rounded = round_shift_32(acc[i]);
            if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                err = Some(AttnError::Overflow);
                ok = false;
            } else {
                out[i] = rounded as i64;
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    /// The exact-on-certificate read: [`KvState::attn_head`]'s words, with blocks whose score
    /// upper bound is provably below the survivor maximum removed from every pass.
    ///
    /// Let `ub_b` be [`cert::KvBlockIndex::bounds`]' upper bound for block `b`, `mx_lb` the largest
    /// lower bound over the summarised blocks, and `limit = mx_lb - 16·ONE`. A block with
    /// `ub_b < limit` has every score below `limit`, and the block attaining `mx_lb` is never
    /// skipped (`ub >= mx_lb > limit`), so the maximum over the scanned blocks is at least `mx_lb`
    /// and every skipped score is more than `16·ONE` below it. [`exp_at`] therefore returns exactly
    /// `0` for every skipped key, its weight is `0`, and every skipped term is exactly `0` in the
    /// `i128` sums: the output is bit-identical to the full scan.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain, [`AttnError::BadGeometry`] when no index is
    /// installed, and the same overflow refusals. `stats` is filled on `Ok` and unspecified on
    /// `Err`.
    ///
    /// # Complexity
    ///
    /// `O(n_blocks · head_dim)` to bound every block, then
    /// `O(scanned_positions · head_dim)` for the scores, exponentials and value fold. With
    /// `B`-position blocks the bound pass is `1/B` of the full score pass; a skipped block saves
    /// its keys' and values' reads.
    #[allow(
        clippy::too_many_lines,
        reason = "the three passes of the read, each with the fragment's flag loop; splitting them \
                  would copy the block cursor and bounds into every pass"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
                  immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
                  extraction audit rejects"
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "`q`/`k`/`v` and the block cursor `b` are the names of the full scan this read \
                  reproduces word for word"
    )]
    #[allow(
        clippy::manual_let_else,
        reason = "`src/engine/attn.rs` is extraction source: the pinned Charon frontend models the \
                  `match ... None => return` form, not `let ... else`"
    )]
    pub fn attn_head_certified(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        out: &mut [i64],
        stats: &mut CertStats,
    ) -> Result<(), AttnError> {
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: q.len(),
            });
        }
        if out.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: out.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let (kplane, vplane) = match &self.storage {
            KvStorage::Q32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        };
        let index = match &self.blocks {
            Some(i) => i,
            None => return Err(AttnError::BadGeometry),
        };
        let group = self.n_head / self.n_head_kv;
        let kvh = q_head / group;
        let kv = self.n_head_kv;
        let stride = kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let first = kvh.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let block = index.block();
        if block == 0 {
            return Err(AttnError::BadGeometry);
        }
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        let nb = slots.div_ceil(block);

        // The bound pass: one `head_dim` dot per block, not per position. `None` is a block with
        // no summary (the partial tail) and is always scanned.
        let mut bnds: Vec<Option<(i64, i64)>> = Vec::with_capacity(nb);
        let mut mx_lb = i64::MIN;
        let mut any = false;
        let mut b = 0usize;
        while b < nb {
            let bound = index.bounds(layer, kvh, b, len, q, self.scale);
            if let Some((lb, _)) = bound {
                any = true;
                if lb > mx_lb {
                    mx_lb = lb;
                }
            }
            bnds.push(bound);
            b += 1;
        }
        let mut limit = i64::MIN;
        if any {
            if let Some(v) = mx_lb.checked_sub(16 * ONE) {
                limit = v;
            }
        }
        let mut skip_blocks = 0u64;
        let mut scanned_blocks = 0u64;
        let mut skipped_positions = 0u64;
        let mut skip: Vec<bool> = Vec::with_capacity(nb);
        b = 0usize;
        while b < nb {
            let s0 = block.checked_mul(b).ok_or(AttnError::Overflow)?;
            let s1 = core::cmp::min(s0.checked_add(block).ok_or(AttnError::Overflow)?, slots);
            let positions = u64::try_from(s1 - s0).map_err(|_| AttnError::Overflow)?;
            let skippable = match bnds[b] {
                Some((_, ub)) => ub < limit,
                None => false,
            };
            skip.push(skippable);
            if skippable {
                skip_blocks = skip_blocks.checked_add(1).ok_or(AttnError::Overflow)?;
                skipped_positions = skipped_positions
                    .checked_add(positions)
                    .ok_or(AttnError::Overflow)?;
            } else {
                scanned_blocks = scanned_blocks.checked_add(1).ok_or(AttnError::Overflow)?;
            }
            b += 1;
        }

        // Scores over the scanned blocks only, in ascending position order.
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        b = 0usize;
        while b < nb && ok {
            if skip[b] {
                b += 1;
            } else {
                let s0 = block.checked_mul(b).ok_or(AttnError::Overflow)?;
                let s1 = core::cmp::min(s0.checked_add(block).ok_or(AttnError::Overflow)?, slots);
                let mut s = s0;
                while s < s1 && ok {
                    let base = first + s * stride;
                    if base > kplane.len().saturating_sub(hd) {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    } else {
                        let end = base + hd;
                        if let Ok(raw) = row_dot_fixed(&kplane[base..end], q) {
                            if let Some(prod) = i128::from(raw).checked_mul(i128::from(self.scale))
                            {
                                let rounded = round_shift_32(prod);
                                if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN)
                                {
                                    err = Some(AttnError::Overflow);
                                    ok = false;
                                } else {
                                    let score = rounded as i64;
                                    if score > mx {
                                        mx = score;
                                    }
                                    scores.push(score);
                                }
                            } else {
                                err = Some(AttnError::Overflow);
                                ok = false;
                            }
                        } else {
                            err = Some(AttnError::Overflow);
                            ok = false;
                        }
                    }
                    s += 1;
                }
                b += 1;
            }
        }
        if let Some(e) = err {
            return Err(e);
        }

        // Softmax over the scanned blocks. A skipped block's shifted score is below `-16·ONE`,
        // so its `exp_at` word is exactly zero and it adds nothing to the exact total.
        let mut exps: Vec<i64> = Vec::with_capacity(scores.len());
        let mut total: i128 = 0;
        let mut cursor = 0usize;
        b = 0usize;
        while b < nb && ok {
            if skip[b] {
                b += 1;
            } else {
                let s0 = block.checked_mul(b).ok_or(AttnError::Overflow)?;
                let s1 = core::cmp::min(s0.checked_add(block).ok_or(AttnError::Overflow)?, slots);
                let mut s = s0;
                while s < s1 && ok {
                    match exp_at(scores[cursor], mx) {
                        Ok(e) => {
                            if let Some(t) = total.checked_add(i128::from(e)) {
                                total = t;
                                exps.push(e);
                            } else {
                                err = Some(AttnError::Overflow);
                                ok = false;
                            }
                        }
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    s += 1;
                    cursor += 1;
                }
                b += 1;
            }
        }
        if let Some(e) = err {
            return Err(e);
        }
        if total <= 0 {
            return Err(AttnError::ZeroSum);
        }

        // Output over the scanned blocks: the same weight per surviving key and the same checked
        // `i128` per-component sums, in the same ascending key order.
        let recip = recip_u64(total);
        let mut acc: Vec<i128> = vec![0i128; hd];
        cursor = 0;
        b = 0usize;
        while b < nb && ok {
            if skip[b] {
                b += 1;
            } else {
                let s0 = block.checked_mul(b).ok_or(AttnError::Overflow)?;
                let s1 = core::cmp::min(s0.checked_add(block).ok_or(AttnError::Overflow)?, slots);
                let mut s = s0;
                while s < s1 && ok {
                    let w = match recip {
                        Some(r) => weight_at_recip(exps[cursor], total, r),
                        None => weight_at(exps[cursor], total),
                    };
                    match w {
                        Ok(weight) => {
                            let base = first + s * stride;
                            if base > vplane.len().saturating_sub(hd) {
                                err = Some(AttnError::Overflow);
                                ok = false;
                            } else {
                                let end = base + hd;
                                let row = &vplane[base..end];
                                let w = i128::from(weight);
                                let mut i = 0usize;
                                while i < hd && ok {
                                    if let Some(term) = w.checked_mul(i128::from(row[i])) {
                                        if let Some(next) = acc[i].checked_add(term) {
                                            acc[i] = next;
                                        } else {
                                            err = Some(AttnError::Overflow);
                                            ok = false;
                                        }
                                    } else {
                                        err = Some(AttnError::Overflow);
                                        ok = false;
                                    }
                                    i += 1;
                                }
                            }
                        }
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    s += 1;
                    cursor += 1;
                }
                b += 1;
            }
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut i = 0usize;
        while i < hd && ok {
            let rounded = round_shift_32(acc[i]);
            if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                err = Some(AttnError::Overflow);
                ok = false;
            } else {
                out[i] = rounded as i64;
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        stats.blocks = u64::try_from(nb).map_err(|_| AttnError::Overflow)?;
        stats.scanned = scanned_blocks;
        stats.skipped = skip_blocks;
        stats.skipped_positions = skipped_positions;
        index.add_read(stats);
        Ok(())
    }

    /// Read one query head at position `pos` over the causal prefix `0..=pos` in binary32 (f32).
    ///
    /// Reads slices directly from the native binary32 KV cache with zero per-call float conversion,
    /// executes the online-softmax attention core via [`f32_core::attn_core_f32_strided`], and
    /// converts output components back to Q32.32 via round-to-nearest. Refuses non-finite numbers
    /// and out-of-range outputs with [`AttnError::Overflow`].
    ///
    /// # Errors
    ///
    /// Same guard chain as [`KvState::attn_head`]: [`AttnError::LayerOutOfRange`],
    /// [`AttnError::HeadOutOfRange`], [`AttnError::RowLength`], [`AttnError::PositionOutOfRange`],
    /// [`AttnError::Overflow`] on non-finite floats or overflow, and [`AttnError::ZeroSum`] if
    /// the softmax sum is not positive.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time: dot products and value accumulations.
    /// Zero heap allocation on hot path for `head_dim <= 256`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::similar_names,
        reason = "Q32.32 to f32 conversions and stack scratch buffers"
    )]
    pub fn attn_head_f32(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: q.len(),
            });
        }
        if out.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: out.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let group = self.n_head / self.n_head_kv;
        let kvh = q_head / group;
        let kv = self.n_head_kv;

        let stride = kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let first = kvh.checked_mul(hd).ok_or(AttnError::Overflow)?;

        let (kplane, vplane) = match &self.storage {
            KvStorage::F32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::Q32 { .. } | KvStorage::I8 { .. } => {
                return Err(AttnError::BadGeometry);
            }
        };

        let last_base = first + pos * stride;
        if last_base > kplane.len().saturating_sub(hd)
            || last_base > vplane.len().saturating_sub(hd)
        {
            return Err(AttnError::Overflow);
        }

        // Each branch lends its scratch buffers to `attn_head_f32_rows`: a join of a stack and a heap
        // `&mut` is refused by the pinned Aeneas model. Stack for hd <= 256, heap otherwise.
        if hd <= 256 {
            let mut q_stack = [F32::ZERO; 256];
            let mut out_stack = [F32::ZERO; 256];
            Self::attn_head_f32_rows(
                &kplane[first..],
                &vplane[first..],
                q,
                hd,
                stride,
                pos,
                out,
                &mut q_stack[..hd],
                &mut out_stack[..hd],
            )
        } else {
            let mut q_heap = vec![F32::ZERO; hd];
            let mut out_heap = vec![F32::ZERO; hd];
            Self::attn_head_f32_rows(
                &kplane[first..],
                &vplane[first..],
                q,
                hd,
                stride,
                pos,
                out,
                &mut q_heap[..],
                &mut out_heap[..],
            )
        }
    }

    /// Q32.32 query to f32, the f32 core over `kx`/`vx`, and the f32 result back to Q32.32, with the
    /// two scratch buffers supplied by the caller (`q_f32`, `out_f32`, each of length `hd`).
    #[allow(
        clippy::too_many_arguments,
        reason = "one scratch buffer per caller branch keeps the Aeneas-visible borrows single-level"
    )]
    fn attn_head_f32_rows(
        kx: &[F32],
        vx: &[F32],
        q: &[i64],
        hd: usize,
        stride: usize,
        pos: usize,
        out: &mut [i64],
        q_f32: &mut [F32],
        out_f32: &mut [F32],
    ) -> Result<(), AttnError> {
        // Index `while` loop, and the one `Err` raised after it. The removal is forced by the
        // pinned Aeneas model, which refuses this body with `Breaks to outer loops are not
        // supported yet`: the `return Err(AttnError::Overflow)` sat inside the `for` body, and an
        // early exit from a loop body is the shape its translator cannot lower. No `zip` appears in
        // this body, so that refusal does not apply here. Arithmetic is unchanged: the same
        // `q32_to_f32` on the same element in the same ascending order, the same store to the same
        // slot, and the same `is_finite` test on the same value, over `0..hd`, which is exactly the
        // old `for` range. `q_f32` is a local scratch buffer with no reader after this loop, so
        // writing it whole before raising rather than up to the first non-finite element is not
        // observable.
        let mut bad = 0u32;
        let mut i = 0usize;
        while i < hd {
            let f = q32_to_f32(q[i]);
            bad += u32::from(!f.is_finite());
            q_f32[i] = f;
            i += 1;
        }
        if bad != 0 {
            return Err(AttnError::Overflow);
        }

        // `hd as i64` is exact: `hd` is a head width, far below 2^53.
        #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
        let scale = F32::ONE / F32::from_i64(hd as i64).sqrt();

        f32_core::attn_core_f32_strided(q_f32, kx, vx, hd, stride, pos, scale, out_f32)?;

        // Index `while` loop with a failure flag, and the one `Err` raised after it, for the same
        // asserted reason: the `?` inside the `for` body is an early exit from a loop body and is
        // refused with `Breaks to outer loops are not supported yet`. The flag is what holds the
        // early exit outside the loop without changing what `out` carries: the old `?` stored
        // `out[i]` at every index before the first failing conversion and returned at that index,
        // and `while i < hd && ok` stores the same prefix and stops at the same index, raising the
        // same `AttnError` after the loop.
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < hd && ok {
            match f32_core::f32_to_q32_checked(out_f32[i]) {
                Ok(v) => out[i] = v,
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }

        Ok(())
    }

    /// Attend every query head of one KV head over the causal prefix `0..=pos`.
    ///
    /// `queries` holds the `group = n_head / n_head_kv` query rows of KV head `kv_head`, each
    /// `head_dim` Q32.32 words, query head `kv_head · group + g` in slot `g`; `outs` receives the
    /// same layout. Under [`KvCarrier::Q32`] and [`KvCarrier::F32`] row `g` is read by
    /// [`KvState::attn_head`] when `tile` is zero and by [`KvState::attn_head_fused`] otherwise,
    /// with the arguments the per-query-head dispatch passed, in ascending `g`: the words are
    /// bit-identical to that dispatch. Under [`KvCarrier::I8`] one call reads each K and V row
    /// once and folds all `group` slots against it (`tile` is ignored: the int8 carrier has one
    /// read and no certificate).
    ///
    /// # Errors
    ///
    /// [`AttnError::HeadOutOfRange`] when `kv_head >= n_head_kv` (its `n_head` field carries the
    /// KV-head count at this level), [`AttnError::RowLength`] when either slice is not
    /// `group · head_dim` wide, and the read's own refusals.
    ///
    /// # Complexity
    ///
    /// `O(group · head_dim · (pos + 1))` words read: the `group` per-head calls it replaces, or
    /// one pass over the int8 rows with `O(group · head_dim)` scratch.
    pub fn attn_head_group(
        &self,
        layer: usize,
        kv_head: usize,
        queries: &[i64],
        pos: usize,
        tile: usize,
        outs: &mut [i64],
    ) -> Result<(), AttnError> {
        if kv_head >= self.n_head_kv {
            return Err(AttnError::HeadOutOfRange {
                head: kv_head,
                n_head: self.n_head_kv,
            });
        }
        let group = self.n_head / self.n_head_kv;
        let width = group
            .checked_mul(self.head_dim)
            .ok_or(AttnError::Overflow)?;
        if queries.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: queries.len(),
            });
        }
        if outs.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: outs.len(),
            });
        }
        if matches!(self.storage, KvStorage::I8 { .. }) {
            return self.attn_head_group_i8(layer, kv_head, queries, pos, outs, group, width);
        }
        let first_head = kv_head.checked_mul(group).ok_or(AttnError::Overflow)?;
        let mut g = 0usize;
        while g < group {
            let lo = g.checked_mul(self.head_dim).ok_or(AttnError::Overflow)?;
            let hi = lo.checked_add(self.head_dim).ok_or(AttnError::Overflow)?;
            let head = first_head.checked_add(g).ok_or(AttnError::Overflow)?;
            let query = &queries[lo..hi];
            let out = &mut outs[lo..hi];
            if tile == 0 {
                self.attn_head(layer, head, query, pos, out)?;
            } else {
                self.attn_head_fused(layer, head, query, pos, tile, out)?;
            }
            g += 1;
        }
        Ok(())
    }

    /// The [`KvCarrier::F32`] arm of [`KvState::attn_head_group`]: one K/V row read per tile of
    /// consecutive causal columns instead of one per column.
    ///
    /// `queries` holds `nq · group · head_dim` Q32.32 words, column `j`'s `group` query rows in
    /// slot `j`, and `outs` receives the same layout. Column `j`'s last position is `first + j`, so
    /// the columns are consecutive and the tile's reads of K and V are shared. The same query
    /// conversion, the same scale, [`f32_core::attn_core_f32_tile`]'s per-tile rescale and the
    /// same readout conversion produce words within the per-tile-rescale bound of the per-column
    /// [`KvState::attn_head_group`] call's; the tests assert that bound.
    ///
    /// # Errors
    ///
    /// [`AttnError::HeadOutOfRange`], [`AttnError::LayerOutOfRange`],
    /// [`AttnError::PositionOutOfRange`] when `first + nq - 1` is not occupied,
    /// [`AttnError::RowLength`] when the slices are not a whole number of `group · head_dim` rows,
    /// [`AttnError::BadGeometry`] under another carrier, and the tile's own read refusals.
    ///
    /// # Complexity
    ///
    /// `O((first + nq) · nq · group · head_dim)` arithmetic, against `nq` per-column calls' sum;
    /// the K and V rows are read once each (`O((first + nq) · head_dim)` words), not once per
    /// column. Scratch `O(nq · group · head_dim)` words plus the tile's score stage.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer, KV head, query tile, first position and output are the call"
    )]
    pub fn attn_head_group_tile_f32(
        &self,
        layer: usize,
        kv_head: usize,
        queries: &[i64],
        first: usize,
        outs: &mut [i64],
    ) -> Result<(), AttnError> {
        let (hd, group, last) = self.tile_group_shape(layer, kv_head, queries, first, outs)?;
        let (kplane, vplane) = match &self.storage {
            KvStorage::F32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::Q32 { .. } | KvStorage::I8 { .. } => return Err(AttnError::BadGeometry),
        };
        let stride = self.n_head_kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let base = kv_head.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let last_base = base
            .checked_add(last.checked_mul(stride).ok_or(AttnError::Overflow)?)
            .ok_or(AttnError::Overflow)?;
        if last_base > kplane.len().saturating_sub(hd)
            || last_base > vplane.len().saturating_sub(hd)
        {
            return Err(AttnError::Overflow);
        }
        let mut qf = vec![F32::ZERO; queries.len()];
        let mut bad = 0u32;
        let mut i = 0usize;
        while i < queries.len() {
            let f = q32_to_f32(queries[i]);
            bad += u32::from(!f.is_finite());
            qf[i] = f;
            i += 1;
        }
        if bad != 0 {
            return Err(AttnError::Overflow);
        }
        let positions: Vec<usize> = (first..=last).collect();
        let mut of = vec![F32::ZERO; outs.len()];
        // `hd as i64` is exact: `hd` is a head width, far below 2^53.
        #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
        let scale = F32::ONE / F32::from_i64(hd as i64).sqrt();
        f32_core::attn_core_f32_tile(
            &qf,
            &kplane[base..],
            &vplane[base..],
            hd,
            group,
            stride,
            &positions,
            scale,
            &mut of,
        )?;
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < outs.len() && ok {
            match f32_core::f32_to_q32_checked(of[i]) {
                Ok(v) => outs[i] = v,
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    /// The geometry and occupancy checks shared by [`KvState::attn_head_group_tile_f32`] and
    /// [`KvState::attn_head_group_tile_i8`]: returns `(head_dim, group, last)`.
    ///
    /// # Errors
    ///
    /// [`AttnError::LayerOutOfRange`], [`AttnError::HeadOutOfRange`],
    /// [`AttnError::RowLength`] for a query block that is not whole `group · head_dim` rows,
    /// [`AttnError::PositionOutOfRange`] when the block's last column is not occupied, and
    /// [`AttnError::Overflow`] for a block whose last column does not fit `usize`.
    fn tile_group_shape(
        &self,
        layer: usize,
        kv_head: usize,
        queries: &[i64],
        first: usize,
        outs: &[i64],
    ) -> Result<(usize, usize, usize), AttnError> {
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if kv_head >= self.n_head_kv {
            return Err(AttnError::HeadOutOfRange {
                head: kv_head,
                n_head: self.n_head_kv,
            });
        }
        let group = self.n_head / self.n_head_kv;
        let width = group.checked_mul(hd).ok_or(AttnError::Overflow)?;
        if width == 0 || queries.len() != outs.len() || !queries.len().is_multiple_of(width) {
            return Err(AttnError::RowLength {
                expected: width,
                got: queries.len(),
            });
        }
        let nq = queries.len() / width;
        if nq == 0 {
            return Err(AttnError::RowLength {
                expected: width,
                got: 0,
            });
        }
        let last = first
            .checked_add(nq)
            .and_then(|v| v.checked_sub(1))
            .ok_or(AttnError::Overflow)?;
        let len = self.occupied[layer];
        if last >= len {
            return Err(AttnError::PositionOutOfRange { pos: last, len });
        }
        Ok((hd, group, last))
    }

    /// The [`KvCarrier::I8`] arm of [`KvState::attn_head_group_tile_f32`]: one K/V tile read per
    /// tile of consecutive causal columns, over signed-8-bit code rows.
    ///
    /// The code planes are KV-head-major, so KV head `kv_head`'s rows start at
    /// `kv_head · cap · head_dim` and row `s` is at `s · head_dim`; the scale planes hold row `s`
    /// at `kv_head · cap + s`. The query conversion, the scale and the readout conversion match
    /// the `f32` arm's, and [`f32_core::attn_core_i8_tile`] applies each row's scales in the
    /// per-column int8 group core's order.
    ///
    /// # Errors
    ///
    /// As [`KvState::attn_head_group_tile_f32`], and [`AttnError::BadGeometry`] under another
    /// carrier.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller's guards and geometry are the arguments; the arm is one call"
    )]
    pub fn attn_head_group_tile_i8(
        &self,
        layer: usize,
        kv_head: usize,
        queries: &[i64],
        first: usize,
        outs: &mut [i64],
    ) -> Result<(), AttnError> {
        let (hd, group, last) = self.tile_group_shape(layer, kv_head, queries, first, outs)?;
        let (k, v, ks, vs) = match &self.storage {
            KvStorage::I8 { k, v, ks, vs } => (&k[layer], &v[layer], &ks[layer], &vs[layer]),
            KvStorage::Q32 { .. } | KvStorage::F32 { .. } => return Err(AttnError::BadGeometry),
        };
        let cap = self.cap;
        let head_stride = cap.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let code_first = kv_head
            .checked_mul(head_stride)
            .ok_or(AttnError::Overflow)?;
        let scale_first = kv_head.checked_mul(cap).ok_or(AttnError::Overflow)?;
        let rows = last.checked_add(1).ok_or(AttnError::Overflow)?;
        let count = rows.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let code_end = code_first.checked_add(count).ok_or(AttnError::Overflow)?;
        let scale_end = scale_first.checked_add(rows).ok_or(AttnError::Overflow)?;
        if code_end > k.len() || code_end > v.len() || scale_end > ks.len() || scale_end > vs.len()
        {
            return Err(AttnError::Overflow);
        }
        let mut qf = vec![F32::ZERO; queries.len()];
        let mut bad = 0u32;
        let mut i = 0usize;
        while i < queries.len() {
            let f = q32_to_f32(queries[i]);
            bad += u32::from(!f.is_finite());
            qf[i] = f;
            i += 1;
        }
        if bad != 0 {
            return Err(AttnError::Overflow);
        }
        let positions: Vec<usize> = (first..=last).collect();
        let mut of = vec![F32::ZERO; outs.len()];
        // `hd as i64` is exact: `hd` is a head width, far below 2^53.
        #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
        let scale = F32::ONE / F32::from_i64(hd as i64).sqrt();
        f32_core::attn_core_i8_tile(
            &qf,
            &k[code_first..code_end],
            &v[code_first..code_end],
            &ks[scale_first..scale_end],
            &vs[scale_first..scale_end],
            hd,
            group,
            hd,
            &positions,
            scale,
            &mut of,
        )?;
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < outs.len() && ok {
            match f32_core::f32_to_q32_checked(of[i]) {
                Ok(v) => outs[i] = v,
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    /// The [`KvCarrier::I8`] arm of [`KvState::attn_head_group`]: one pass over KV head
    /// `kv_head`'s code rows, all `group` slots folded against each row.
    ///
    /// The query words convert to binary32 once, the read runs as binary32 arithmetic on
    /// `code · scale`, and the result converts back to Q32.32 with the same round-to-nearest and
    /// the same `Overflow` refusal as [`KvState::attn_head_f32`].
    ///
    /// # Complexity
    ///
    /// `O(group · head_dim · (pos + 1))` time; `O(group · head_dim)` scratch.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller's guards and geometry are the arguments; the arm is one call"
    )]
    fn attn_head_group_i8(
        &self,
        layer: usize,
        kv_head: usize,
        queries: &[i64],
        pos: usize,
        outs: &mut [i64],
        group: usize,
        width: usize,
    ) -> Result<(), AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let hd = self.head_dim;
        let cap = self.cap;
        let KvStorage::I8 { k, v, ks, vs } = &self.storage else {
            return Err(AttnError::BadGeometry);
        };
        let head_stride = cap.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let code_first = kv_head
            .checked_mul(head_stride)
            .ok_or(AttnError::Overflow)?;
        let rows = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        let count = rows.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let code_end = code_first.checked_add(count).ok_or(AttnError::Overflow)?;
        let scale_first = kv_head.checked_mul(cap).ok_or(AttnError::Overflow)?;
        let scale_end = scale_first.checked_add(rows).ok_or(AttnError::Overflow)?;

        let mut qf = vec![F32::ZERO; width];
        let mut bad = 0u32;
        let mut i = 0usize;
        while i < width {
            let f = q32_to_f32(queries[i]);
            bad += u32::from(!f.is_finite());
            qf[i] = f;
            i += 1;
        }
        if bad != 0 {
            return Err(AttnError::Overflow);
        }
        let mut of = vec![F32::ZERO; width];
        // `hd as i64` is exact: `hd` is a head width, far below 2^53.
        #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
        let scale = F32::ONE / F32::from_i64(hd as i64).sqrt();
        i8::attn_core_i8_group(
            &qf,
            group,
            &k[layer][code_first..code_end],
            &v[layer][code_first..code_end],
            &ks[layer][scale_first..scale_end],
            &vs[layer][scale_first..scale_end],
            hd,
            rows,
            scale,
            &mut of,
        )?;
        let mut i = 0usize;
        let mut err: Option<AttnError> = None;
        while i < width && err.is_none() {
            match f32_core::f32_to_q32_checked(of[i]) {
                Ok(w) => outs[i] = w,
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    /// Rewind every layer to `m` positions. Causal attention makes the rows at `0..m` independent
    /// of later appends, so the rewound state equals one that stopped at `m`; rows at `m..` stay in
    /// the planes and the next [`KvState::append`] overwrites position `m`.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::PositionOutOfRange`] when a layer holds fewer than `m` positions; no
    /// layer is changed.
    ///
    /// # Complexity
    ///
    /// `O(n_layer)` time: one check pass and one store pass over the per-layer counters; no
    /// allocation.
    pub fn truncate(&mut self, m: usize) -> Result<(), AttnError> {
        let mut bad: Option<usize> = None;
        let mut i = 0usize;
        while i < self.occupied.len() && bad.is_none() {
            if self.occupied[i] < m {
                bad = Some(self.occupied[i]);
            }
            i += 1;
        }
        if let Some(len) = bad {
            return Err(AttnError::PositionOutOfRange { pos: m, len });
        }
        self.occupied.fill(m);
        if let Some(idx) = self.blocks.as_mut() {
            // A block summary covers only a completed block; rewinding into one drops it so the
            // next append re-seals the block from the rows actually stored.
            idx.truncate(m);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "test harnesses, carrier comparisons and multi-token sequence setups"
)]
mod tests {
    use super::*;

    /// A deterministic Q32.32 word in `[-4·ONE, 4·ONE)` from a 64-bit LCG state.
    fn word(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let wide = i64::try_from(*state >> 1).expect("the top 63 bits fit i64");
        wide % (8 * ONE) - 4 * ONE
    }

    /// The per-component oracle of the output stage: the score, softmax and weight stages of
    /// [`KvState::attn_head`] rebuilt from the live helpers, then one test-only [`value_component`]
    /// call per output component. This is the scan the key-outer production pass replaces.
    fn oracle_head(
        kv: &KvState,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let hd = kv.head_dim;
        let kv_n = kv.n_head_kv;
        let kvh = q_head / (kv.n_head / kv_n);
        let (kplane, vplane) = match &kv.storage {
            KvStorage::Q32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => {
                return Err(AttnError::BadGeometry);
            }
        };
        let slots = pos + 1;
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s <= pos {
            let sc = score_at(kplane, q, kv.scale, kv_n, kvh, hd, s)?;
            if sc > mx {
                mx = sc;
            }
            scores.push(sc);
            s += 1;
        }
        let mut exps: Vec<i64> = Vec::with_capacity(slots);
        let mut total: i128 = 0;
        let mut s = 0usize;
        while s <= pos {
            let e = exp_at(scores[s], mx)?;
            total = total
                .checked_add(i128::from(e))
                .ok_or(AttnError::Overflow)?;
            exps.push(e);
            s += 1;
        }
        if total <= 0 {
            return Err(AttnError::ZeroSum);
        }
        let mut weights: Vec<i64> = Vec::with_capacity(slots);
        let mut s = 0usize;
        while s <= pos {
            weights.push(weight_at(exps[s], total)?);
            s += 1;
        }
        let mut out: Vec<i64> = Vec::with_capacity(hd);
        let mut i = 0usize;
        while i < hd {
            out.push(value_component(vplane, &weights, kv_n, kvh, hd, pos, i)?);
            i += 1;
        }
        Ok(out)
    }

    #[test]
    fn attn_head_group_matches_the_per_head_read() {
        // n_head 4 over n_head_kv 2 is a group of 2; head_dim 8 exercises the vector width. The
        // group read must return the per-head words for both numeric carriers and for both the
        // plain and the fused tile read.
        for carrier in [KvCarrier::Q32, KvCarrier::F32] {
            let mut kv = KvState::with_carrier(1, 4, 2, 8, 6, carrier).expect("geometry");
            let mut seed = 0xDEAD_BEEF_1234_5678_u64;
            let mut s = 0usize;
            while s < 6 {
                let k: Vec<i64> = (0..16).map(|_| word(&mut seed)).collect();
                let v: Vec<i64> = (0..16).map(|_| word(&mut seed)).collect();
                assert_eq!(kv.append(0, &k, &v), Ok(s));
                s += 1;
            }
            let query: Vec<i64> = (0..32).map(|_| word(&mut seed)).collect();
            let mut kvh = 0usize;
            while kvh < 2 {
                let lo = kvh * 16;
                let mut got = [0i64; 16];
                kv.attn_head_group(0, kvh, &query[lo..lo + 16], 5, 0, &mut got)
                    .expect("group read");
                let mut g = 0usize;
                while g < 2 {
                    let (qlo, glo) = (lo + g * 8, g * 8);
                    let mut want = [0i64; 8];
                    kv.attn_head(0, kvh * 2 + g, &query[qlo..qlo + 8], 5, &mut want)
                        .expect("per-head read");
                    assert_eq!(
                        &got[glo..glo + 8],
                        &want[..],
                        "carrier {carrier:?}, KV head {kvh}, group slot {g}"
                    );
                    g += 1;
                }
                // The group passes its tile to every head: the fused group read is the per-head
                // fused read, which is the per-position path within its documented bound, not
                // bit-identical to it.
                if carrier == KvCarrier::Q32 {
                    let mut fused = [0i64; 16];
                    kv.attn_head_group(0, kvh, &query[lo..lo + 16], 5, 32, &mut fused)
                        .expect("fused group read");
                    let mut g = 0usize;
                    while g < 2 {
                        let (qlo, glo) = (lo + g * 8, g * 8);
                        let mut want = [0i64; 8];
                        kv.attn_head_fused(0, kvh * 2 + g, &query[qlo..qlo + 8], 5, 32, &mut want)
                            .expect("fused per-head read");
                        assert_eq!(
                            &fused[glo..glo + 8],
                            &want[..],
                            "fused carrier Q32, KV head {kvh}, group slot {g}"
                        );
                        g += 1;
                    }
                }
                kvh += 1;
            }
        }
    }

    #[test]
    fn attn_head_group_refuses_a_bad_group_slice() {
        let kv = KvState::with_carrier(1, 4, 2, 4, 2, KvCarrier::Q32).expect("geometry");
        let q = [0i64; 8];
        assert_eq!(
            kv.attn_head_group(0, 2, &q, 0, 0, &mut [0i64; 8]),
            Err(AttnError::HeadOutOfRange { head: 2, n_head: 2 })
        );
        assert_eq!(
            kv.attn_head_group(0, 0, &q[..4], 0, 0, &mut [0i64; 8]),
            Err(AttnError::RowLength {
                expected: 8,
                got: 4,
            })
        );
    }

    /// The int8 group read is the per-head read of each of its slots, bit for bit, and the int8
    /// carrier refuses the Q32-only certificate.
    #[test]
    fn i8_group_read_equals_the_per_head_reads() {
        let (n_head, n_kv, hd) = (4usize, 2usize, 8usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_kv, hd, 6, KvCarrier::I8).expect("geometry");
        let mut seed = 0xA5A5_5A5A_1234_5678u64;
        let mut s = 0usize;
        while s < 6 {
            let k: Vec<i64> = (0..n_kv * hd).map(|_| word(&mut seed)).collect();
            let v: Vec<i64> = (0..n_kv * hd).map(|_| word(&mut seed)).collect();
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let query: Vec<i64> = (0..n_head * hd).map(|_| word(&mut seed)).collect();
        assert!(matches!(kv.storage, KvStorage::I8 { .. }));
        let width = 2 * hd;
        let mut kvh = 0usize;
        while kvh < n_kv {
            let lo = kvh * width;
            let mut group = vec![0i64; width];
            kv.attn_head_group(0, kvh, &query[lo..lo + width], 5, 0, &mut group)
                .expect("group read");
            let mut g = 0usize;
            while g < 2 {
                let (qlo, glo) = (lo + g * hd, g * hd);
                let mut head = vec![0i64; hd];
                kv.attn_head(0, kvh * 2 + g, &query[qlo..qlo + hd], 5, &mut head)
                    .expect("per-head read");
                assert_eq!(&group[glo..glo + hd], &head[..], "kv head {kvh} slot {g}");
                g += 1;
            }
            kvh += 1;
        }
        assert_eq!(
            kv.enable_block_index(4),
            Err(AttnError::BadGeometry),
            "the block certificate is Q32-only"
        );
    }

    /// The int8 read is the causal softmax of the stored codes and scales, checked against an f64
    /// oracle over the same rows.
    #[test]
    fn i8_group_read_matches_a_float_oracle() {
        let (n_head, n_kv, hd, rows) = (4usize, 2usize, 8usize, 5usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_kv, hd, rows, KvCarrier::I8).expect("geometry");
        let mut seed = 0x0F1E_2D3C_4B5A_6978u64;
        let mut s = 0usize;
        while s < rows {
            let k: Vec<i64> = (0..n_kv * hd).map(|_| word(&mut seed)).collect();
            let v: Vec<i64> = (0..n_kv * hd).map(|_| word(&mut seed)).collect();
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let query: Vec<i64> = (0..n_head * hd).map(|_| word(&mut seed)).collect();
        let (kc, vc, ks, vs) = match &kv.storage {
            KvStorage::I8 { k, v, ks, vs } => (&k[0], &v[0], &ks[0], &vs[0]),
            _ => panic!("the state carries the int8 storage"),
        };
        let native = |x: &F32| f64::from(f32::from_bits(x.to_bits()));
        let qf: Vec<f64> = query
            .iter()
            .map(|w| f64::from(f32::from_bits(q32_to_f32(*w).to_bits())))
            .collect();
        let scale = 1.0f64 / f64::from(u32::try_from(hd).expect("head_dim fits u32")).sqrt();
        let group = n_head / n_kv;
        let width = group * hd;
        let cap = kv.cap;
        let mut kvh = 0usize;
        while kvh < n_kv {
            let lo = kvh * width;
            let mut got = vec![0i64; width];
            kv.attn_head_group(0, kvh, &query[lo..lo + width], rows - 1, 0, &mut got)
                .expect("group read");
            let mut g = 0usize;
            while g < group {
                let qrow = &qf[lo + g * hd..lo + (g + 1) * hd];
                let mut scores = vec![0.0f64; rows];
                let mut mx = f64::NEG_INFINITY;
                let mut s = 0usize;
                while s < rows {
                    let row = kvh * cap * hd + s * hd;
                    let mut dot = 0.0f64;
                    let mut i = 0usize;
                    while i < hd {
                        dot += qrow[i] * f64::from(kc[row + i]);
                        i += 1;
                    }
                    scores[s] = dot * native(&ks[kvh * cap + s]) * scale;
                    if scores[s] > mx {
                        mx = scores[s];
                    }
                    s += 1;
                }
                let mut den = 0.0f64;
                let mut acc = vec![0.0f64; hd];
                let mut s = 0usize;
                while s < rows {
                    let w = (scores[s] - mx).exp();
                    let v_scale = native(&vs[kvh * cap + s]);
                    den += w;
                    let row = kvh * cap * hd + s * hd;
                    let mut i = 0usize;
                    while i < hd {
                        acc[i] += w * f64::from(vc[row + i]) * v_scale;
                        i += 1;
                    }
                    s += 1;
                }
                let mut i = 0usize;
                while i < hd {
                    let want = acc[i] / den;
                    let have = f64::from(f32::from_bits(q32_to_f32(got[g * hd + i]).to_bits()));
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (have - want).abs() <= tol,
                        "kv head {kvh} slot {g} component {i}: have {have}, want {want}"
                    );
                    i += 1;
                }
                g += 1;
            }
            kvh += 1;
        }
    }

    #[test]
    fn attn_head_key_outer_output_matches_component_oracle() {
        // n_head 4 over n_head_kv 2 gives a GQA group of 2; head_dim 3 is neither 1 nor a power of
        // two, so several query heads read distinct KV rows over non-trivial causal prefixes.
        let mut kv = KvState::with_carrier(1, 4, 2, 3, 6, KvCarrier::Q32).expect("geometry");
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut s = 0usize;
        while s < 6 {
            let k = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let v = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let mut pos = 0usize;
        while pos < 6 {
            let mut head = 0usize;
            while head < 4 {
                let expected = oracle_head(&kv, 0, head, &q, pos).expect("per-component oracle");
                let mut got = [0i64; 3];
                kv.attn_head(0, head, &q, pos, &mut got)
                    .expect("key-outer read");
                assert_eq!(got.as_slice(), expected.as_slice(), "pos {pos} head {head}");
                head += 1;
            }
            pos += 1;
        }

        // A non-`head_dim`-wide output refuses before the output pass runs.
        assert_eq!(
            kv.attn_head(0, 0, &q, 0, &mut [0i64; 2]),
            Err(AttnError::RowLength {
                expected: 3,
                got: 2,
            })
        );
        // A position at or past the occupied count refuses before the output pass runs.
        let len = kv.occupied(0).expect("occupied");
        assert_eq!(
            kv.attn_head(0, 0, &q, len, &mut [0i64; 3]),
            Err(AttnError::PositionOutOfRange { pos: len, len })
        );
    }

    #[test]
    fn attn_head_f32_carrier_matches_oracle() {
        let mut kv_q32 = KvState::with_carrier(1, 4, 2, 3, 6, KvCarrier::Q32).expect("geometry");
        let mut kv_f32 = KvState::with_carrier(1, 4, 2, 3, 6, KvCarrier::F32).expect("geometry");
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut s = 0usize;
        while s < 6 {
            let k = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let v = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv_q32.append(0, &k, &v), Ok(s));
            assert_eq!(kv_f32.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let mut pos = 0usize;
        while pos < 6 {
            let mut head = 0usize;
            while head < 4 {
                let mut got_q32 = [0i64; 3];
                let mut got_f32 = [0i64; 3];
                kv_q32
                    .attn_head(0, head, &q, pos, &mut got_q32)
                    .expect("q32 read");
                kv_f32
                    .attn_head(0, head, &q, pos, &mut got_f32)
                    .expect("f32 read");
                for (&g, &e) in got_f32.iter().zip(got_q32.iter()) {
                    let diff = (g - e).abs();
                    assert!(
                        diff <= 2000,
                        "pos {pos} head {head}: got_f32 {g} got_q32 {e} diff {diff}"
                    );
                }
                head += 1;
            }
            pos += 1;
        }
    }

    /// The reciprocal weight path must equal the literal [`weight_at`] oracle (or refuse with the
    /// same `AttnError`) on the window boundary, the `e = 2^32` argmax fallback, xorshift draws,
    /// and the parity ties of each total.
    #[test]
    fn weight_at_recip_matches_weight_at_oracle() {
        let totals: [i128; 7] = [
            2,
            3,
            1 << 31,
            1 << 32,
            (1 << 32) + 1,
            1 << 40,
            i128::from(u64::MAX / 2),
        ];
        let mut x = 0x243F_6A88_85A3_08D3_u64;
        for &total in &totals {
            let recip = recip_u64(total).expect("total is in the reciprocal window");
            let total_i64 = i64::try_from(total).expect("total <= i64::MAX");
            let mut e_values: Vec<i64> = vec![
                0,
                1,
                total_i64 / 2,
                total_i64 - 1,
                total_i64,
                (1 << 32) - 1,
                1 << 32,
                -1,
                i64::MIN,
            ];
            let mut draw = 0usize;
            while draw < 128 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                e_values.push(i64::try_from(x >> 1).expect("the top 63 bits fit i64"));
                draw += 1;
            }
            // The parity ties: `e << 32` congruent to `total/2` mod total, when such an `e` exists
            // in the small search window. `total/2` is the oracle's own rounding bias.
            let half = (total / 2) % total;
            let mut e = 0i64;
            while e < 4096 {
                if (i128::from(e) << 32) % total == half {
                    e_values.push(e);
                }
                e += 1;
            }
            for &e in &e_values {
                let expected = weight_at(e, total);
                let got = weight_at_recip(e, total, recip);
                assert_eq!(got, expected, "total {total} e {e}");
            }
        }
        // `recip_u64` refuses outside its window, and the `None` dispatch is the literal oracle.
        assert!(recip_u64(0).is_none());
        assert!(recip_u64(1).is_none());
        assert!(recip_u64(i128::from(u64::MAX)).is_none());
        let total = i128::from(u64::MAX);
        let e = 7i64;
        let dispatched = match recip_u64(total) {
            Some(r) => weight_at_recip(e, total, r),
            None => weight_at(e, total),
        };
        assert_eq!(dispatched, weight_at(e, total));
    }

    /// The hoisted `first + s·stride` base must equal the checked index oracles [`row_bounds`] and
    /// [`plane_at`] on the rows the output test reads.
    #[test]
    fn hoisted_row_base_matches_plane_index_oracles() {
        let kv = 2usize;
        let hd = 3usize;
        let stride = kv.checked_mul(hd).expect("stride");
        let mut kvh = 0usize;
        while kvh < kv {
            let first = kvh.checked_mul(hd).expect("first");
            let mut s = 0usize;
            while s < 6 {
                let base = first + s * stride;
                let (rb_base, rb_end) = row_bounds(kv, kvh, hd, s).expect("row_bounds");
                assert_eq!(rb_base, base, "kvh {kvh} s {s}");
                assert_eq!(rb_end, base + hd, "kvh {kvh} s {s}");
                let mut i = 0usize;
                while i < hd {
                    assert_eq!(
                        plane_at(kv, kvh, hd, s, i).expect("plane_at"),
                        base + i,
                        "kvh {kvh} s {s} i {i}"
                    );
                    i += 1;
                }
                s += 1;
            }
            kvh += 1;
        }
    }

    /// The round epilogue shared by the two fold-shape helpers: one `round_shift_32` plus the same
    /// `i64` range check [`KvState::attn_head`] applies.
    ///
    /// # Complexity
    ///
    /// `O(hd)` time and space: one `head_dim`-wide output.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
                  immediately before it"
    )]
    fn round_acc(acc: &[i128]) -> Result<Vec<i64>, AttnError> {
        let mut out: Vec<i64> = Vec::with_capacity(acc.len());
        let mut i = 0usize;
        while i < acc.len() {
            let rounded = round_shift_32(acc[i]);
            if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                return Err(AttnError::Overflow);
            }
            out.push(rounded as i64);
            i += 1;
        }
        Ok(out)
    }

    /// The key-outer output fold in the pre-inline shape: the whole prefix's weights are resolved
    /// into a `Vec` first (the r26 dispatch), then each key's row is read and folded. Test-only
    /// oracle for the inline shape; the production inline path is [`KvState::attn_head`].
    ///
    /// # Complexity
    ///
    /// `O(keys · hd)` time, `O(keys + hd)` space (the weights vector and the accumulator).
    fn weights_vec_fold(
        vplane: &[i64],
        exps: &[i64],
        total: i128,
        kv: usize,
        kvh: usize,
        hd: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let recip = recip_u64(total);
        let mut weights: Vec<i64> = Vec::with_capacity(exps.len());
        let mut s = 0usize;
        while s < exps.len() {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r)?,
                None => weight_at(exps[s], total)?,
            };
            weights.push(w);
            s += 1;
        }
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        while s < exps.len() {
            let base = plane_at(kv, kvh, hd, s, 0)?;
            if base > vplane.len().saturating_sub(hd) {
                return Err(AttnError::Overflow);
            }
            let row = &vplane[base..base + hd];
            let w = i128::from(weights[s]);
            let mut i = 0usize;
            while i < hd {
                acc[i] = acc[i]
                    .checked_add(
                        w.checked_mul(i128::from(row[i]))
                            .ok_or(AttnError::Overflow)?,
                    )
                    .ok_or(AttnError::Overflow)?;
                i += 1;
            }
            s += 1;
        }
        round_acc(&acc)
    }

    /// The key-outer output fold in the inline shape: each key's weight is resolved immediately
    /// before its row is read, with no weights buffer. Mirrors the production loop of
    /// [`KvState::attn_head`] so the two shapes can be compared for the same `exps`/`total`.
    ///
    /// # Complexity
    ///
    /// `O(keys · hd)` time, `O(hd)` space (no weights vector).
    fn inline_fold(
        vplane: &[i64],
        exps: &[i64],
        total: i128,
        kv: usize,
        kvh: usize,
        hd: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let recip = recip_u64(total);
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        while s < exps.len() {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r)?,
                None => weight_at(exps[s], total)?,
            };
            let base = plane_at(kv, kvh, hd, s, 0)?;
            if base > vplane.len().saturating_sub(hd) {
                return Err(AttnError::Overflow);
            }
            let row = &vplane[base..base + hd];
            let w = i128::from(w);
            let mut i = 0usize;
            while i < hd {
                acc[i] = acc[i]
                    .checked_add(
                        w.checked_mul(i128::from(row[i]))
                            .ok_or(AttnError::Overflow)?,
                    )
                    .ok_or(AttnError::Overflow)?;
                i += 1;
            }
            s += 1;
        }
        round_acc(&acc)
    }

    /// The inline weight fold must equal the weights-`Vec` fold for the same scores/exps/total,
    /// including a total outside the reciprocal window (`recip_u64` returns `None`, so the literal
    /// [`weight_at`] dispatch runs). `attn_head`'s reachable totals are all `>= ONE` (the argmax key
    /// contributes `fixed_exp(0) = ONE`), so its dispatch never returns `None`; the `None` case is
    /// therefore exercised at the fold level, where the two shapes are the only variable.
    #[test]
    fn inline_weight_fold_matches_weights_vec_oracle() {
        // hd 3, kv 2, kvh 1: row `s` starts at `(2·s + 1)·3`, so the two shapes cannot share a
        // trivial base. Three keys fit the 18-word plane.
        let (kv, kvh, hd) = (2usize, 1usize, 3usize);
        let mut vplane = vec![0i64; 3 * kv * hd];
        let mut seed = 0xDEAD_BEEF_1234_5678_u64;
        let mut at = 0usize;
        while at < vplane.len() {
            vplane[at] = word(&mut seed);
            at += 1;
        }
        let cases: [(&[i64], i128); 4] = [
            (&[ONE, ONE, ONE / 2], 5 * i128::from(ONE) / 2),
            (&[ONE, 0, ONE / 4], 5 * i128::from(ONE) / 4),
            // `total = 1` is below `recip_u64`'s lower window edge, so the dispatch takes `None`.
            (&[0, 1], 1),
            // A total above `ONE` but well inside the window.
            (&[3, 5, 8], 16),
        ];
        for (exps, total) in cases {
            let expected = weights_vec_fold(&vplane, exps, total, kv, kvh, hd);
            let got = inline_fold(&vplane, exps, total, kv, kvh, hd);
            assert_eq!(got, expected, "exps {exps:?} total {total}");
            if total < 2 {
                assert!(
                    recip_u64(total).is_none(),
                    "total {total} leaves the window"
                );
            }
        }
    }

    /// The first refusing key and its `AttnError` must be the same in the inline and weights-`Vec`
    /// shapes: a weight refusal at key 1 with key 0 admitted, and a bound refusal at key 1 with a
    /// one-row-short plane. Both shapes scan keys ascending, so both stop at the same key.
    #[test]
    fn first_refusing_key_matches_weights_vec_shape() {
        let (kv, kvh, hd) = (2usize, 1usize, 3usize);
        let mut vplane = vec![0i64; 3 * kv * hd];
        let mut seed = 0x0F1E_2D3C_4B5A_6978_u64;
        let mut at = 0usize;
        while at < vplane.len() {
            vplane[at] = word(&mut seed);
            at += 1;
        }
        // Weight refusal at key 1: key 0's weight is admitted, key 1's `e·2^32` leaves `i64`.
        let exps = [1i64, i64::MAX];
        let total = 2i128;
        assert_eq!(weight_at(exps[0], total), Ok(1i64 << 31));
        assert_eq!(weight_at(exps[1], total), Err(AttnError::Overflow));
        assert_eq!(
            weights_vec_fold(&vplane, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
        assert_eq!(
            inline_fold(&vplane, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );

        // Bound refusal at key 1: `short` holds key 0's row but cuts key 1's.
        let short = &vplane[..kvh * hd + hd];
        let exps = [ONE, ONE];
        let total = 2 * i128::from(ONE);
        assert_eq!(
            weights_vec_fold(short, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
        assert_eq!(
            inline_fold(short, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
    }

    #[test]
    fn carrier_explicit_selection() {
        let kv_q32 = KvState::with_carrier(1, 2, 2, 8, 4, KvCarrier::Q32).expect("geometry q32");
        let kv_f32 = KvState::with_carrier(1, 2, 2, 8, 4, KvCarrier::F32).expect("geometry f32");
        assert_eq!(kv_q32.carrier(), KvCarrier::Q32);
        assert_eq!(kv_f32.carrier(), KvCarrier::F32);
        assert!(matches!(kv_q32.storage, KvStorage::Q32 { .. }));
        assert!(matches!(kv_f32.storage, KvStorage::F32 { .. }));

        let k = [0i64; 16];
        let v = [0i64; 16];
        let mut kv_q32 = kv_q32;
        let mut kv_f32 = kv_f32;
        kv_q32.append(0, &k, &v).expect("append q32");
        kv_f32.append(0, &k, &v).expect("append f32");

        let q = [0i64; 8];
        let mut out_q32 = [0i64; 8];
        let mut out_f32 = [0i64; 8];
        kv_q32
            .attn_head(0, 0, &q, 0, &mut out_q32)
            .expect("read q32");
        kv_f32
            .attn_head(0, 0, &q, 0, &mut out_f32)
            .expect("read f32");
        assert_eq!(out_q32, out_f32);
    }

    #[test]
    fn truncate_and_rollback_hold_for_both_carriers() {
        for &carrier in &[KvCarrier::Q32, KvCarrier::F32] {
            let mut kv = KvState::with_carrier(1, 2, 2, 4, 8, carrier).expect("geometry");
            let mut seed = 0x1234_5678_9ABC_DEF0_u64;

            let mut k_tokens = Vec::new();
            let mut v_tokens = Vec::new();
            for _ in 0..6 {
                let k = [
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                ];
                let v = [
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                ];
                k_tokens.push(k);
                v_tokens.push(v);
            }

            for s in 0..6 {
                assert_eq!(kv.append(0, &k_tokens[s], &v_tokens[s]), Ok(s));
            }
            assert_eq!(kv.occupied(0), Ok(6));

            // Truncate to 3 positions
            assert_eq!(kv.truncate(3), Ok(()));
            assert_eq!(kv.occupied(0), Ok(3));

            // Append 2 replacement tokens
            let new_k1 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_v1 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_k2 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_v2 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv.append(0, &new_k1, &new_v1), Ok(3));
            assert_eq!(kv.append(0, &new_k2, &new_v2), Ok(4));
            assert_eq!(kv.occupied(0), Ok(5));

            // Create a fresh kv state that directly appended the exact same sequence
            let mut fresh = KvState::with_carrier(1, 2, 2, 4, 8, carrier).expect("geometry");
            for s in 0..3 {
                assert_eq!(fresh.append(0, &k_tokens[s], &v_tokens[s]), Ok(s));
            }
            assert_eq!(fresh.append(0, &new_k1, &new_v1), Ok(3));
            assert_eq!(fresh.append(0, &new_k2, &new_v2), Ok(4));

            let q = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let mut out_rewound = [0i64; 4];
            let mut out_direct = [0i64; 4];

            kv.attn_head(0, 0, &q, 4, &mut out_rewound)
                .expect("rewound read");
            fresh
                .attn_head(0, 0, &q, 4, &mut out_direct)
                .expect("direct read");

            assert_eq!(
                out_rewound, out_direct,
                "carrier {carrier:?} rewound vs direct mismatch"
            );

            // Truncate past occupied refuses and changes nothing
            assert_eq!(
                kv.truncate(6),
                Err(AttnError::PositionOutOfRange { pos: 6, len: 5 })
            );
            assert_eq!(kv.occupied(0), Ok(5));
        }
    }

    /// A unique path under the temp dir for a cache-file test.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "apeiron-kvstate-{tag}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn certified_read_equals_full_scan_and_skips_distant_blocks() {
        let (n_head, n_head_kv, hd, cap, block) = (2usize, 2usize, 4usize, 64usize, 8usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::Q32).expect("geometry");
        let mut seed = 0x05EE_D0FA_11CE_u64;
        let mut s = 0usize;
        while s < cap {
            // The first block is the maximum; the rest are 20 units below it, so every later
            // block clears the 16·ONE floor by four units.
            let base = if s < block { 2 * ONE } else { -3 * ONE };
            let k = vec![base; n_head_kv * hd];
            let mut v = vec![0i64; n_head_kv * hd];
            let mut i = 0usize;
            while i < v.len() {
                v[i] = word(&mut seed);
                i += 1;
            }
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        kv.enable_block_index(block).expect("index");
        let mut full = kv.clone();
        full.disable_block_index();
        assert!(kv.has_block_index() && !full.has_block_index());

        let q = [2 * ONE; 4];
        let mut pos = 0usize;
        while pos < cap {
            let mut head = 0usize;
            while head < n_head {
                let mut got = [0i64; 4];
                let mut want = [0i64; 4];
                kv.attn_head(0, head, &q, pos, &mut got)
                    .expect("certified read");
                full.attn_head(0, head, &q, pos, &mut want)
                    .expect("full scan");
                assert_eq!(got, want, "pos {pos} head {head}");
                head += 1;
            }
            pos += 1;
        }

        let mut out = [0i64; 4];
        let mut stats = CertStats::default();
        kv.attn_head_certified(0, 0, &q, cap - 1, &mut out, &mut stats)
            .expect("certified read");
        assert_eq!(stats.blocks, 8);
        assert_eq!(stats.scanned + stats.skipped, stats.blocks);
        assert_eq!(stats.skipped, 7, "every block but the maximum is certified");
        assert_eq!(stats.skipped_positions, 56);
    }

    #[test]
    #[allow(
        clippy::many_single_char_names,
        reason = "the test lays out `k`/`v`/`q` exactly as the plane it checks"
    )]
    fn certified_read_after_truncate_and_reappend_equals_full_scan() {
        let (n_head, n_head_kv, hd, cap, block) = (2usize, 2usize, 3usize, 48usize, 4usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::Q32).expect("geometry");
        kv.enable_block_index(block).expect("index");
        let mut seed = 0xC0FF_EE12_3456_789A_u64;
        let mut s = 0usize;
        while s < 24 {
            let mut k = vec![0i64; n_head_kv * hd];
            let mut v = vec![0i64; n_head_kv * hd];
            let mut i = 0usize;
            while i < k.len() {
                k[i] = word(&mut seed);
                v[i] = word(&mut seed);
                i += 1;
            }
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        assert_eq!(kv.truncate(18), Ok(()));
        // Re-append different rows over positions 18..23; blocks 4 and 5 must re-seal.
        let mut s = 18usize;
        while s < 24 {
            let mut k = vec![0i64; n_head_kv * hd];
            let mut v = vec![0i64; n_head_kv * hd];
            let mut i = 0usize;
            while i < k.len() {
                k[i] = word(&mut seed);
                v[i] = word(&mut seed);
                i += 1;
            }
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let mut full = kv.clone();
        full.disable_block_index();
        let q = vec![word(&mut seed); hd];
        let mut pos = 0usize;
        while pos < 24 {
            let mut head = 0usize;
            while head < n_head {
                let mut got = [0i64; 3];
                let mut want = [0i64; 3];
                kv.attn_head(0, head, &q, pos, &mut got)
                    .expect("certified read");
                full.attn_head(0, head, &q, pos, &mut want)
                    .expect("full scan");
                assert_eq!(got, want, "pos {pos} head {head}");
                head += 1;
            }
            pos += 1;
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[allow(
        clippy::many_single_char_names,
        reason = "the test lays out `k`/`v`/`q` exactly as the plane it checks"
    )]
    fn file_backed_state_matches_owned_state() {
        let path = temp_path("q32");
        let path_s = path.to_str().expect("utf-8 temp path");
        let (n_layer, n_head, n_head_kv, hd, cap) = (1usize, 4usize, 2usize, 3usize, 6usize);
        let mut owned = KvState::with_carrier(n_layer, n_head, n_head_kv, hd, cap, KvCarrier::Q32)
            .expect("owned");
        let mut file =
            KvState::with_file_carrier(path_s, n_layer, n_head, n_head_kv, hd, cap, Carrier::Q32)
                .expect("file");
        assert_eq!(
            file.file_bytes(),
            Some(n_layer * 2 * n_head_kv * cap * hd * 8)
        );
        assert_eq!(owned.file_bytes(), None);

        let mut seed = 0x0BAD_C0DE_DEAD_BEEF_u64;
        let mut s = 0usize;
        while s < cap {
            let mut k = vec![0i64; n_head_kv * hd];
            let mut v = vec![0i64; n_head_kv * hd];
            let mut i = 0usize;
            while i < k.len() {
                k[i] = word(&mut seed);
                v[i] = word(&mut seed);
                i += 1;
            }
            assert_eq!(owned.append(0, &k, &v), Ok(s));
            assert_eq!(file.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let mut pos = 0usize;
        while pos < cap {
            let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
            let mut head = 0usize;
            while head < n_head {
                let (mut a, mut b) = ([0i64; 3], [0i64; 3]);
                owned
                    .attn_head(0, head, &q, pos, &mut a)
                    .expect("owned read");
                file.attn_head(0, head, &q, pos, &mut b).expect("file read");
                assert_eq!(a, b, "attn_head pos {pos} head {head}");
                let (mut fa, mut fb) = ([0i64; 3], [0i64; 3]);
                owned
                    .attn_head_fused(0, head, &q, pos, 2, &mut fa)
                    .expect("owned fused read");
                file.attn_head_fused(0, head, &q, pos, 2, &mut fb)
                    .expect("file fused read");
                assert_eq!(fa, fb, "attn_head_fused pos {pos} head {head}");
                assert_eq!(
                    owned.attn_max(0, head, &q, pos),
                    file.attn_max(0, head, &q, pos)
                );
                assert_eq!(
                    owned.attn_fused_state(0, head, &q, pos, fa[0], 2),
                    file.attn_fused_state(0, head, &q, pos, fa[0], 2)
                );
                head += 1;
            }
            pos += 1;
        }
        file.sync_file().expect("msync");

        // A mapped state rewinds and re-appends exactly like an owned one.
        assert_eq!(file.truncate(3), owned.truncate(3));
        let mut k = vec![0i64; n_head_kv * hd];
        let mut v = vec![0i64; n_head_kv * hd];
        let mut i = 0usize;
        while i < k.len() {
            k[i] = word(&mut seed);
            v[i] = word(&mut seed);
            i += 1;
        }
        assert_eq!(owned.append(0, &k, &v), Ok(3));
        assert_eq!(file.append(0, &k, &v), Ok(3));
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let (mut a, mut b) = ([0i64; 3], [0i64; 3]);
        owned.attn_head(0, 0, &q, 3, &mut a).expect("owned read");
        file.attn_head(0, 0, &q, 3, &mut b).expect("file read");
        assert_eq!(a, b);

        drop(file);
        std::fs::remove_file(&path).expect("remove q32 file");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn file_backed_f32_state_matches_owned_f32_state() {
        let path = temp_path("f32");
        let path_s = path.to_str().expect("utf-8 temp path");
        let (n_layer, n_head, n_head_kv, hd, cap) = (1usize, 4usize, 2usize, 3usize, 6usize);
        let mut owned = KvState::with_carrier(n_layer, n_head, n_head_kv, hd, cap, KvCarrier::F32)
            .expect("owned");
        let mut file =
            KvState::with_file_carrier(path_s, n_layer, n_head, n_head_kv, hd, cap, Carrier::F32)
                .expect("file");
        assert_eq!(
            file.file_bytes(),
            Some(n_layer * 2 * n_head_kv * cap * hd * 4)
        );
        let mut seed = 0x1357_9BDF_2468_ACE0_u64;
        let mut s = 0usize;
        while s < cap {
            let mut k = vec![0i64; n_head_kv * hd];
            let mut v = vec![0i64; n_head_kv * hd];
            let mut i = 0usize;
            while i < k.len() {
                k[i] = word(&mut seed);
                v[i] = word(&mut seed);
                i += 1;
            }
            assert_eq!(owned.append(0, &k, &v), Ok(s));
            assert_eq!(file.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let mut pos = 0usize;
        while pos < cap {
            let mut head = 0usize;
            while head < n_head {
                let (mut a, mut b) = ([0i64; 3], [0i64; 3]);
                owned.attn_head(0, head, &q, pos, &mut a).expect("owned");
                file.attn_head(0, head, &q, pos, &mut b).expect("file");
                assert_eq!(a, b, "pos {pos} head {head}");
                head += 1;
            }
            pos += 1;
        }
        drop(file);
        std::fs::remove_file(&path).expect("remove f32 file");
    }

    /// The resident and anonymous kilobytes of every `/proc/self/smaps` region backed by `path`.
    ///
    /// Kernels differ in the field names: older ones report `RssAnon:`/`RssFile:`, this one
    /// reports `Anonymous:` and only `Rss:`. The file count is the reported `RssFile` when present,
    /// else the region's resident pages less its anonymous ones.
    #[cfg(target_os = "linux")]
    fn smaps_rss(path: &str) -> (u64, u64) {
        fn kb(v: &str) -> u64 {
            v.split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0)
        }
        fn flush(anon: &mut u64, file: &mut u64, cur: (u64, u64, u64)) {
            *anon += cur.1;
            *file += if cur.2 != 0 {
                cur.2
            } else {
                cur.0.saturating_sub(cur.1)
            };
        }
        let text = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
        let mut anon = 0u64;
        let mut file = 0u64;
        let mut ours = false;
        let mut cur = (0u64, 0u64, 0u64);
        for line in text.lines() {
            let header =
                line.as_bytes().first().is_some_and(u8::is_ascii_hexdigit) && line.contains(" rw");
            if header {
                flush(&mut anon, &mut file, cur);
                cur = (0, 0, 0);
                ours = line.trim_end().ends_with(path);
                continue;
            }
            if !ours {
                continue;
            }
            if let Some(v) = line.strip_prefix("Rss:") {
                cur.0 += kb(v);
            }
            if let Some(v) = line.strip_prefix("RssAnon:") {
                cur.1 += kb(v);
            }
            if let Some(v) = line.strip_prefix("Anonymous:") {
                cur.1 += kb(v);
            }
            if let Some(v) = line.strip_prefix("RssFile:") {
                cur.2 += kb(v);
            }
        }
        flush(&mut anon, &mut file, cur);
        (anon, file)
    }

    /// The tile read of `group` query heads agrees with the per-column read head for head, and
    /// each word within the per-tile-rescale bound, across tile boundaries and the last partial
    /// tile.
    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        reason = "fixture words are built from moduli below 1001, far inside i64"
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "the test's geometry indices are the call's own (kvh, first, nq, i, m)"
    )]
    fn the_tile_group_read_agrees_with_the_per_column_read() {
        let (n_head, n_head_kv, hd, cap) = (4usize, 2usize, 8usize, 300usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::F32).expect("geometry");
        let kv_out = n_head_kv * hd;
        let mut s = 0usize;
        while s < cap {
            let mut k = vec![0i64; kv_out];
            let mut v = vec![0i64; kv_out];
            let mut i = 0usize;
            while i < kv_out {
                // Small Q32.32 words: |value| < 2^31, so every sum and product stays in range.
                k[i] = ((s * 131 + i * 17) % 257) as i64 - 128;
                v[i] = ((s * 53 + i * 29) % 193) as i64 - 96;
                k[i] <<= 23;
                v[i] <<= 23;
                i += 1;
            }
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let group = n_head / n_head_kv;
        let width = group * hd;
        let mut kvh = 0usize;
        while kvh < n_head_kv {
            // `first` 250 with `nq` 50 crosses the 64-position key-tile boundary at 256.
            for first in [0usize, 1, 7, 8, 9, 31, 250, 255] {
                for nq in [1usize, 2, 3, 8, 9, 50] {
                    if first + nq > cap {
                        continue;
                    }
                    let q: Vec<i64> = (0..nq * width)
                        .map(|i| (((i * 37 + kvh * 11) % 1001) as i64 - 500) << 24)
                        .collect();
                    let mut tiled = vec![0i64; nq * width];
                    kv.attn_head_group_tile_f32(0, kvh, &q, first, &mut tiled)
                        .expect("tile");
                    for j in 0..nq {
                        let mut one = vec![0i64; width];
                        kv.attn_head_group(
                            0,
                            kvh,
                            &q[j * width..(j + 1) * width],
                            first + j,
                            0,
                            &mut one,
                        )
                        .expect("per column");
                        for i in 0..width {
                            let (x, y) = (tiled[j * width + i], one[i]);
                            let m = i128::from(x.abs().max(y.abs()));
                            let prefix = (first + j + 1) as i128;
                            let tol = 64
                                + i64::try_from((m * 512 * (prefix + 64)) >> 32)
                                    .expect("the tile bound fits i64");
                            assert!(
                                (x - y).abs() <= tol,
                                "kvh {kvh} first {first} nq {nq} column {j} word {i}: {x} vs {y} (tol {tol})"
                            );
                        }
                    }
                }
            }
            kvh += 1;
        }
    }

    /// The int8 tile read agrees with the per-column int8 group read over the same quantised
    /// rows, within the per-tile rescale bound plus the `head_dim` score-rounding difference; the
    /// tile's `f32` and AVX2 cores are word-equal by the `f32_core` test.
    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        reason = "fixture words are built from moduli below 1001, far inside i64"
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "the test's geometry indices are the call's own (kvh, first, nq, i, m)"
    )]
    fn the_i8_tile_group_read_agrees_with_the_per_column_read() {
        let (n_head, n_head_kv, hd, cap) = (4usize, 2usize, 8usize, 300usize);
        let mut kv =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::I8).expect("geometry");
        let kv_out = n_head_kv * hd;
        let mut s = 0usize;
        while s < cap {
            let mut k = vec![0i64; kv_out];
            let mut v = vec![0i64; kv_out];
            let mut i = 0usize;
            while i < kv_out {
                k[i] = ((s * 137 + i * 19) % 263) as i64 - 131;
                v[i] = ((s * 59 + i * 31) % 197) as i64 - 98;
                k[i] <<= 23;
                v[i] <<= 23;
                i += 1;
            }
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let group = n_head / n_head_kv;
        let width = group * hd;
        let mut kvh = 0usize;
        while kvh < n_head_kv {
            for first in [0usize, 1, 7, 8, 9, 31, 250, 255] {
                for nq in [1usize, 2, 3, 8, 9, 50] {
                    if first + nq > cap {
                        continue;
                    }
                    let q: Vec<i64> = (0..nq * width)
                        .map(|i| (((i * 37 + kvh * 11) % 1001) as i64 - 500) << 24)
                        .collect();
                    let mut tiled = vec![0i64; nq * width];
                    kv.attn_head_group_tile_i8(0, kvh, &q, first, &mut tiled)
                        .expect("tile");
                    for j in 0..nq {
                        let mut one = vec![0i64; width];
                        kv.attn_head_group(
                            0,
                            kvh,
                            &q[j * width..(j + 1) * width],
                            first + j,
                            0,
                            &mut one,
                        )
                        .expect("per column");
                        for i in 0..width {
                            let (x, y) = (tiled[j * width + i], one[i]);
                            let m = i128::from(x.abs().max(y.abs()));
                            let prefix = (first + j + 1) as i128 + hd as i128;
                            // The per-tile rescale adds `(prefix + hd)` roundings; the per-column
                            // core's eight-lane dot adds a 2e-3 relative term the sequential GEMM
                            // does not share (see the f32_core i8 test).
                            let tol = 64
                                + i64::try_from((m * 512 * (prefix + 64)) >> 32)
                                    .expect("the tile bound fits i64")
                                + i64::try_from(m / 500).expect("the tile bound fits i64");
                            assert!(
                                (x - y).abs() <= tol,
                                "i8 kvh {kvh} first {first} nq {nq} column {j} word {i}: {x} vs {y} (tol {tol})"
                            );
                        }
                    }
                }
            }
            kvh += 1;
        }
    }

    /// The KV pages of a file-backed state are file-backed: they appear as `RssFile`, never
    /// `RssAnon`, and the kernel may evict them without swapping.
    #[cfg(target_os = "linux")]
    #[test]
    fn mapped_cache_pages_are_file_backed() {
        let path = temp_path("smaps");
        let path_s = path.to_str().expect("utf-8 temp path");
        let (n_layer, n_head, n_head_kv, hd, cap) = (1usize, 1usize, 1usize, 128usize, 64usize);
        let mut kv =
            KvState::with_file_carrier(path_s, n_layer, n_head, n_head_kv, hd, cap, Carrier::Q32)
                .expect("file");
        let k = vec![ONE; hd];
        let v = vec![ONE; hd];
        let mut s = 0usize;
        while s < cap {
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let (anon_kb, file_kb) = smaps_rss(path_s);
        drop(kv);
        std::fs::remove_file(&path).expect("remove");
        assert_eq!(anon_kb, 0, "mapped KV must not be anonymous");
        assert!(
            file_kb >= 64,
            "the touched 128 KiB cache file shows {file_kb} KiB of file-backed pages"
        );
    }
}
