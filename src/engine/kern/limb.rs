//! The one shared Q32.32 product primitive on the `Q8_0` kernel: [`gemm_q32`], [`syrk_q32`],
//! [`gemv_q32`] and [`rowdot_q32`].
//!
//! **`Q8_0` raw layout.** A block is 2 little-endian bytes of `f16` scale followed by 32 signed code
//! bytes ([`Q8_0_BLOCK_BYTES`] = 34); a row of `n` inputs is `⌈n/32⌉` blocks
//! ([`Q8_0_QK`] = 32). Code `q` of a block decodes to the Q32.32 word `d·q/2^32` with
//! `d = f16_to_fixed(scale) = m·2^k`, `|m| ≤ 2047`, `k ∈ [8, 37]` ([`super::pow2::f16_split`]).
//!
//! **Exactness on the kernel.** [`gemm_q8_0`] computes
//! `round_shift_32(Σ_b d_b·S_b·D_b)` with `D_b` the integer code dot of block `b` and `S_b` the
//! activation block scale. Two facts make every term a shift and the call exact: `d = m·2^k` and
//! `S = 2^e` a power of two ([`crate::engine::chart::quant_act`] only ever returns a power of two),
//! so a block contributes `(m·D_b)·2^{k+e}`. Here every weight scale is the exact power of two
//! `2^{dw}` (`m = 1024`, `k = dw − 10`) and every activation scale the exact power of two `2^σ`, so
//! the block term is `1024·D·2^{dw+σ−10}` with `|1024·D| < 2^30` for the digits below; nothing is
//! rounded until the call's single `round_shift_32`.
//!
//! **The limb expansion.** Each 32-element block of each operand row is written
//! `x = 2^e·Σ_{s<limbs} d_s·2^{−7(s+1)} + 2^e·ρ` with balanced signed-byte digits
//! `|d_s| ≤ 64` and `|ρ| ≤ 2^{−7·limbs}/2`, `e` the block exponent shared by the block's rows
//! (`2^{e−1} > max|x|`, raised per block only when the pair placement needs it). A pair of
//! operands then expands into the kept limb pairs `(s, t)` with `s + t ≤ limbs − 1`
//! (`limbs(limbs+1)/2` of them: 6 of 9 at `limbs = 3`), each pair contributing the block term
//! `α_s·β_t·2^{ea+eb−7(s+t+2)−32}` in Q32.32 words. A pair's two scales are placed as
//! `d = 2^{dw}`, `S = 2^σ` with `dw = clamp(T, 8, 47)` and `σ = T − dw`, `T = ea + eb − 7(s+t+2)`
//! the pair's Q32.32 exponent: `dw ∈ [8, 47]` is exactly the range of a finite `f16` power of two
//! (`2^{−24} … 2^{15}` scaled by `2^32`), and `σ ∈ [0, 44]` keeps `S·32·128·127 ≤ i64::MAX`, the
//! fast-row bound [`crate::engine::chart::q8act_fast_bound_ok`] tests. `T < 8` cannot be placed
//! (the activation scale would be fractional); such a block pair's whole contribution is
//! `≤ 2^{T−15} < 2^{−8}` words and is dropped, and the bound below counts it. `T > 91` exceeds
//! every placement and is refused as [`LimbError::Range`].
//!
//! **Refusals.** [`LimbError::Shape`]: a zero or mismatched dimension, a slice whose length is not
//! `rows·n`, `limbs == 0`, or `n > MAX_INNER`; [`LimbError::Range`]:
//! a block exponent pair whose product the scale placement cannot carry, or a group partial sum
//! that leaves `i64`. [`LimbError::NoKernel`] is never returned: a host without AVX-VNNI, or a
//! width past [`gemm_q8_0`]'s 512 blocks, makes that kernel return `None` and the same planes go to
//! [`matmat_q8act`], which is the `Q8_0` contract itself; only a `MatvecError` refusal survives, and
//! that is `Range`.
//!
//! **Trust boundary.** The AVX2/AVX-VNNI cores of `gemm_q8_0` and `matmat_q8act` are not
//! extracted; their equality with the scalar contract is `tested` in `gemm.rs` and `matmat.rs`.
//! The error bound below is `tested` against an `i128` oracle, not yet `proved`: its Lean theorem
//! is open (the design notes), so this file carries a `tested` grade only.
#![allow(
    clippy::many_single_char_names,
    reason = "`a`, `b`, `m`, `p`, `n`, `s`, `t` are the matrix-product and limb indices the \
              API, the kernel's own signatures and the derivation all use; renaming them would \
              make the bound's arithmetic unreadable"
)]
#![allow(
    dead_code,
    reason = "the four entry points land in this wave; the KERN port lanes wire their call sites \
              in the next wave, and the tests call every one of them under cfg(test)"
)]

use super::gemm::gemm_q8_0;
use super::matmat::matmat_q8act;
use crate::engine::chart::pool::{chunks_mut_global, width_global};
use crate::engine::chart::task::ChunkTask;
use crate::engine::chart::QuantAct;
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES, Q8_0_QK};

/// Element count below which a phase runs on the calling thread.
const POOL_MIN: usize = 1 << 15;

/// Rows (or bytes, columns) per pooled task for `len` units: two tasks per pool worker, so a
/// worker that finishes early claims another piece and the tail does not idle the pool.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
fn pool_chunk(len: usize) -> usize {
    let workers = width_global().max(1);
    len.div_ceil(workers * 2).max(1)
}

/// Lanes of one limb block, the kernel's activation block width.
const ACT_QK: usize = Q8_0_QK;

/// The widest contraction the limb products take, in blocks of 32. Above `gemm_q8_0`'s own
/// `MAX_BLOCKS` (512) the product falls back to [`matmat_q8act`] on the same planes — the same
/// `Q8_0` contract, slower — so the accepted width is not the kernel's stack-array bound. A 7B
/// `ffn_down` contracts over `n_ff = 18944 > 512·32` (Qwen2.5-7B), which is why it is 1024.
pub(crate) const MAX_INNER: usize = 1024 * Q8_0_QK;

/// The most limbs one block may be split into; the digit shift `e + 7(limbs − 1)` must stay
/// inside the `i128` digit accumulator.
const MAX_LIMBS: u32 = 8;

/// `2^−7`: one limb step.
const LIMB_BITS: i32 = 7;

/// The smallest pair exponent the scale placement carries: `dw ≥ 8` (the smallest finite `f16`
/// power of two is `2^−24`, i.e. `2^8` in Q32.32) with `σ ≥ 0`.
const MIN_PAIR_EXP: i32 = 8;

/// The largest pair exponent: `dw ≤ 47` (the largest `f16` power of two, `2^15`, is `2^47` in
/// Q32.32) with `σ ≤ 44` (`2^44·32·128·127 ≤ i64::MAX`).
const MAX_PAIR_EXP: i32 = 91;

/// The largest Q32.32 exponent of a finite `f16` power of two: `2^15` is `2^47` in Q32.32.
const MAX_WEIGHT_EXP: i32 = 47;

/// Why a limb product refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimbError {
    /// A dimension, a slice length, or a contraction width the kernel call cannot carry.
    Shape,
    /// A word whose block exponents the `f16`/`i64` scale placement cannot carry, or an output
    /// that leaves `i64`.
    Range,
    /// No kernel: never returned. A host without AVX-VNNI falls back to `matmat_q8act` on the same
    /// planes, and the variant exists only so a caller can name the condition it cannot see.
    NoKernel,
}

/// `C[i·p + j] = Σ_l a[i·n + l]·b[j·n + l]` in Q32.32, both operands row-major, the contraction
/// contiguous, each 32-element block split into `limbs` signed-byte limbs and every limb product
/// carried on [`gemm_q8_0`].
///
/// # Error bound
///
/// Write `Ea = ⌊log₂ max|a|⌋ + 2`, `Eb = ⌊log₂ max|b|⌋ + 2` (so every block exponent `ea ≤ Ea`,
/// `eb ≤ Eb`), `L = limbs`, `P = L(L+1)/2` kept pairs, `B = ⌈n/32⌉` blocks. Then every entry obeys
/// `|C_limb − C_exact| ≤ G(n, L, max|a|, max|b|)` words, with
///
/// ```text
/// G = ⌈P/2⌉ + 1 + B·(1 + ⌈C_L·2^{Ea+Eb−32}⌉),
/// C_L = (33 + 16(L−1))·2^{−7L},
/// ```
///
/// which is [`bound_ulps`] rounded out to the integers. The three terms are the call's own
/// rounding (`⌈P/2⌉`: at most one `round_shift_32` per kept pair, each `≤ 1/2` ulp), the per-block
/// residual and dropped-pair mass `C_L·2^{Ea+Eb−32}` (both operands truncated to `L` digits, and
/// the pairs with `s+t ≥ L` dropped), and the per-block `+1` covering the `T < 8` blocks whose
/// whole contribution is `≤ 2^{−8}` words. Truncation dominates:
/// `|ρ_a|·|b| + |ρ_b|·|a| + |ρ_aρ_b| ≤ 2^{−7L}(1 + 2^{−7L}/4)` per element, and
/// `Σ_{s+t≥L}|α_s||β_t|2^{−7(s+t+2)} ≤ 64²·D_L ≤ 64²(L−1)·2^{−7L−13}` per element; both times 32
/// elements and `2^{ea+eb−32}` words give the `C_L` coefficient. At `L = 3` that is
/// `C_3 = 65·2^{−21}`, i.e. a relative per-block truncation of about `2^{−15}` of `2^{Ea+Eb}`.
///
/// # Errors
///
/// [`LimbError::Shape`] on a zero or mismatched dimension, `limbs == 0` or `n > MAX_INNER`;
/// [`LimbError::Range`] on a block pair the scale placement cannot carry, a group partial sum
/// that leaves `i64`, or an entry that leaves `i64`. A group partial is refused as `Range` even
/// when the exact total would cancel back into range: the refusal is conservative, never a wrong
/// word.
///
/// # Complexity
///
/// Temporal `O(n·(m + p)·limbs)` to build the limb planes plus `O(P·m·p·n)` byte multiply-adds —
/// the kernel's `vpdpbusd` counts 32 per instruction, so the executed instruction count is
/// `P·m·p·n/32` — plus the same `P·m·p·n` activation-code byte reads; each kernel row
/// task reads the `P·n` weight bytes of its own rows once per call. Spatial
/// `O((m + p)·n·limbs)` limb bytes, `O(P·m·n)` weight bytes for one pair group and `O(m·p)` `i128`
/// accumulators.
pub(crate) fn gemm_q32(
    a: &[i64],
    m: usize,
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    let need_a = m.checked_mul(n).ok_or(LimbError::Shape)?;
    let need_b = p.checked_mul(n).ok_or(LimbError::Shape)?;
    if a.len() != need_a || b.len() != need_b {
        return Err(LimbError::Shape);
    }
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    product(&pa, &pb)
}

/// The upper triangle of `A·Aᵀ`, mirrored: `C[i·m + j] = Σ_l a[i·n + l]·a[j·n + l]`, `C` is `m×m`.
///
/// Only the columns `j` and the rows `i ≤ j` the upper triangle needs are computed (a column chunk
/// `[j0, j1)` runs the kernel over rows `0..j1`; the entries `i > j` are discarded), and the lower
/// triangle is the transpose of the upper, so the kernel work is `≈ m²n/2` rather than `m²n`. Each
/// column is computed by the same planes and the same pair grouping as [`gemm_q32`], so the two
/// agree bit for bit and the mirror is exact.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|a|, max|a|)`.
///
/// # Errors
///
/// As [`gemm_q32`].
///
/// # Complexity
///
/// Temporal `O(m·n·limbs + m²·n·limbs/2)` byte multiply-adds — half of [`gemm_q32`]'s full
/// product — and spatial `O(m·n·limbs)` for the one limb plane plus `O(m·COLS)` accumulators.
pub(crate) fn syrk_q32(a: &[i64], m: usize, n: usize, limbs: u32) -> Result<Vec<i64>, LimbError> {
    prepare(a, m, n, limbs)?.syrk()
}

/// [`gemm_q32`]'s operands as a device kernel reads them: the two limb planes, the kept pairs, the
/// pair-group width and the product exponent of every `(pair, block)`.
///
/// [`gemm_q32`]'s value per pair group is `round_shift_32(Σ_{(pair, block) live} D·2^T)`, `D` the
/// 32-lane digit dot of the pair's two limbs and `T = dw + sigma` the placement exponent; a dead
/// block pair contributes 0. `T` depends on the block and on `s + t` only, never on the row.
#[cfg(feature = "gpu")]
pub(crate) struct DeviceOperands<'a> {
    /// `[row][block][limb][lane]` digits of the first operand.
    pub(crate) a_dig: &'a [i8],
    /// `[row][block][limb][lane]` digits of the second operand.
    pub(crate) b_dig: &'a [i8],
    /// Rows of the first operand.
    pub(crate) m: usize,
    /// Rows of the second operand.
    pub(crate) p: usize,
    /// Blocks per row, `⌈n/32⌉`.
    pub(crate) blocks: usize,
    /// Limbs per block.
    pub(crate) limbs: u32,
    /// The kept pairs `(s, t)` in [`kept_pairs`] order.
    pub(crate) pairs: Vec<(u32, u32)>,
    /// Pairs per kernel call; the calls are consecutive runs of `pairs`.
    pub(crate) group: usize,
    /// `[pair][block]`: `Some(T)` for a live block pair, `None` below the placement floor.
    pub(crate) exp: Vec<Option<i32>>,
}

/// The placement [`Plane::product_rows`] computes for the whole of `pa · pbᵀ`, with its refusals.
///
/// # Errors
///
/// `Shape` on mismatched planes, `Range` on a block pair past the placement.
///
/// # Complexity
///
/// `O(P·⌈n/32⌉)` placements; the planes are borrowed.
#[cfg(feature = "gpu")]
fn device_operands<'a>(pa: &'a Plane, pb: &'a Plane) -> Result<DeviceOperands<'a>, LimbError> {
    if pa.n != pb.n || pa.limbs != pb.limbs {
        return Err(LimbError::Shape);
    }
    let pairs = kept_pairs(pa.limbs);
    let place = Layout::placement(pa, pb, &pairs)?;
    let group = (MAX_INNER / pa.padded).max(1).min(pairs.len());
    let exp = place
        .iter()
        .map(|pl| pl.live.then_some(pl.dw + pl.sigma))
        .collect();
    Ok(DeviceOperands {
        a_dig: &pa.dig,
        b_dig: &pb.dig,
        m: pa.rows,
        p: pb.rows,
        blocks: pa.blocks,
        limbs: pa.limbs,
        pairs,
        group,
        exp,
    })
}

/// The full product `pa · pbᵀ`: on the device ([`crate::gpu::limb::product`]) when it takes the
/// call, else the CPU kernel. Both are bit-identical to [`Plane::product_rows`].
///
/// # Errors
///
/// As [`gemm_q32`].
///
/// # Complexity
///
/// As [`Plane::product_rows`] over all rows.
fn product(pa: &Plane, pb: &Plane) -> Result<Vec<i64>, LimbError> {
    #[cfg(feature = "gpu")]
    if crate::gpu::limb::takes(pa.rows, pb.rows, pa.n) {
        if let Some(c) = crate::gpu::limb::product(&device_operands(pa, pb)?)? {
            return Ok(c);
        }
    }
    pa.product_rows(pb, 0, pa.rows, 0, pb.rows, pa.n)
}

/// [`gemm_q32`] on the CPU kernel only, and its operands as the device reads them: the reference
/// and the input of the device test.
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn with_device_operands<R>(
    a: &[i64],
    m: usize,
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
    f: impl FnOnce(&DeviceOperands<'_>) -> R,
) -> Result<(Vec<i64>, R), LimbError> {
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    let cpu = pa.product_rows(&pb, 0, m, 0, p, n)?;
    let r = f(&device_operands(&pa, &pb)?);
    Ok((cpu, r))
}

/// The placement of `a · bᵀ` as [`device_operands`] makes it, without running a product: both
/// planes' block exponents, the kept pairs, the pair-group width, and the `[pair][block]` exponent
/// table or [`Layout::placement`]'s `Range`. The reference of the device placement.
#[cfg(all(test, feature = "gpu"))]
pub(crate) type PlacementParts = (
    Vec<u32>,
    Vec<u32>,
    Vec<(u32, u32)>,
    usize,
    Result<Vec<Option<i32>>, LimbError>,
);

/// See [`PlacementParts`].
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn placement_parts(
    a: &[i64],
    m: usize,
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<PlacementParts, LimbError> {
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    let pairs = kept_pairs(limbs);
    let group = (MAX_INNER / pa.padded).max(1).min(pairs.len());
    let exp = device_operands(&pa, &pb).map(|ops| ops.exp);
    Ok((pa.e, pb.e, pairs, group, exp))
}

/// The block exponents and digits of [`Plane::build`]: the reference of the device plane build.
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn plane_parts(
    data: &[i64],
    rows: usize,
    n: usize,
    limbs: u32,
) -> Result<(Vec<u32>, Vec<i8>), LimbError> {
    let p = Plane::build(data, rows, n, limbs)?;
    Ok((p.e, p.dig))
}

/// One pair group's word from its accumulator `v = Σ D·2^{T − tmin}`: `round_shift_32(v·2^tmin)`
/// as [`gemm_q32`]'s kernel call rounds it, `None` when the word leaves `i64` (its `Range`).
///
/// # Complexity
///
/// `O(1)`.
#[cfg(feature = "gpu")]
pub(crate) fn device_group_word(v: i64, tmin: u32) -> Option<i64> {
    // tmin ≤ MAX_PAIR_EXP = 91: below 32 the shift is < 2^94 in i128; at or above it the product is
    // a multiple of 2^32, the rounding is exact and the shift by tmin − 32 ≤ 59 is < 2^122.
    let r = if tmin >= 32 {
        i128::from(v) << (tmin - 32)
    } else {
        crate::engine::num::value::round_shift_32(i128::from(v) << tmin)
    };
    i64::try_from(r).ok()
}

/// The kept pairs and the pair-group width of a product of inner length `n` at `limbs`, as
/// [`device_operands`] sets them; they depend on the shape only, never on the operands.
///
/// # Complexity
///
/// `O(limbs²)`.
#[cfg(feature = "gpu")]
pub(crate) fn device_pairs(n: usize, limbs: u32) -> (Vec<Pair>, usize) {
    let pairs = kept_pairs(limbs);
    let group = (MAX_INNER / (n.div_ceil(ACT_QK) * ACT_QK))
        .max(1)
        .min(pairs.len());
    (pairs, group)
}

/// A built limb plane whose rows are reused across several products.
///
/// [`gemm_q32`] builds the first operand's plane on every call. A call site that runs several
/// products against the same first operand (the resq's gate, up, qg and qu against one slab, and
/// its `pp` and `pc` against one transposed `pq`) builds it once here and pays only the second
/// operand's plane per call.
pub(crate) struct Prepared {
    plane: Plane,
}

/// Build `a`'s limb plane once for [`Prepared::gemm`] and [`Prepared::syrk`].
///
/// # Errors
///
/// [`LimbError::Shape`] on a mismatched length or a zero dimension; [`LimbError::Range`] as
/// [`Plane::build`].
///
/// # Complexity
///
/// Temporal and spatial `O(rows·n·limbs)`, the first operand's plane.
pub(crate) fn prepare(a: &[i64], rows: usize, n: usize, limbs: u32) -> Result<Prepared, LimbError> {
    if a.len() != rows.checked_mul(n).ok_or(LimbError::Shape)? {
        return Err(LimbError::Shape);
    }
    Ok(Prepared {
        plane: Plane::build(a, rows, n, limbs)?,
    })
}

impl Prepared {
    /// `a·bᵀ` with `a` the prepared `rows × n` plane and `b` a `p × n` operand.
    ///
    /// # Errors
    ///
    /// [`LimbError::Shape`] on a zero or mismatched dimension; [`LimbError::Range`] as
    /// [`gemm_q32`].
    ///
    /// # Complexity
    ///
    /// As [`gemm_q32`] without the first operand's plane build.
    pub(crate) fn gemm(&self, b: &[i64], p: usize) -> Result<Vec<i64>, LimbError> {
        let n = self.plane.n;
        if b.len() != p.checked_mul(n).ok_or(LimbError::Shape)? {
            return Err(LimbError::Shape);
        }
        let pb = Plane::build(b, p, n, self.plane.limbs)?;
        product(&self.plane, &pb)
    }

    /// `a·aᵀ` on the prepared plane: [`syrk_q32`]'s body after its plane build.
    ///
    /// # Errors
    ///
    /// As [`syrk_q32`].
    ///
    /// # Complexity
    ///
    /// As [`syrk_q32`] without the plane build.
    pub(crate) fn syrk(&self) -> Result<Vec<i64>, LimbError> {
        let plane = &self.plane;
        let m = plane.rows;
        let n = plane.n;
        // The device computes the full square: `gemm_q32(a, a)` equals the mirrored triangle.
        #[cfg(feature = "gpu")]
        if crate::gpu::limb::takes(m, m, n) {
            if let Some(c) = crate::gpu::limb::product(&device_operands(plane, plane)?)? {
                return Ok(c);
            }
        }
        let mut c = vec![0i64; m.checked_mul(m).ok_or(LimbError::Shape)?];
        let mut j0 = 0usize;
        while j0 < m {
            let j1 = (j0 + SYRK_COLS).min(m);
            let width = j1 - j0;
            let block = plane.product_rows(plane, 0, j1, j0, width, n)?;
            // Row `i` keeps the contiguous run `j ∈ [max(i, j0), j1)` of the tile.
            let mut i = 0usize;
            while i < j1 {
                let lo = j0.max(i);
                if lo < j1 {
                    let from = i * width + (lo - j0);
                    c[i * m + lo..i * m + j1].copy_from_slice(&block[from..from + (j1 - lo)]);
                }
                i += 1;
            }
            j0 = j1;
        }
        // The lower triangle is the upper one's transpose.
        let mut i = 0usize;
        while i < m {
            let mut j = i + 1;
            while j < m {
                c[j * m + i] = c[i * m + j];
                j += 1;
            }
            i += 1;
        }
        Ok(c)
    }
}

/// One row against many: `y[j] = Σ_l x[l]·b[j·n + l]`.
///
/// Identical to [`gemm_q32`] with `m = 1` on the same planes, so the words are bit-equal.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|x|, max|b|)`.
///
/// # Errors
///
/// As [`gemm_q32`].
///
/// # Complexity
///
/// Temporal `O(n·(1 + p)·limbs + P·p·n)` byte multiply-adds; spatial `O((p + 1)·n·limbs)` limb
/// bytes and `O(p)` accumulators.
pub(crate) fn gemv_q32(
    x: &[i64],
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    if x.len() != n || b.len() != p.checked_mul(n).ok_or(LimbError::Shape)? {
        return Err(LimbError::Shape);
    }
    let px = Plane::build(x, 1, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    px.product_rows(&pb, 0, 1, 0, p, n)
}

/// Batched row-pair dots: `y[i] = Σ_l a[i·n + l]·b[i·n + l]`.
///
/// The diagonal of the same planes and the same pair grouping as [`gemm_q32`], taken from
/// [`ROWDOT_TILE`]-square diagonal tiles: each tile is one [`gemm_q8_0`] call over its own row
/// block against its own column block, so the words are the per-entry words of [`gemm_q32`] and
/// bit-equal to them. A kernel call's every entry depends only on its weight row, its activation
/// column and the contraction planes, never on the other columns of the call, which is what lets
/// one tile replace `tile` separate one-row calls.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|a|, max|b|)`, entrywise.
///
/// # Errors
///
/// As [`gemm_q32`], at the first refusing tile.
///
/// # Complexity
///
/// Temporal `O((m + m)·n·limbs)` plane packing plus `⌈m/ROWDOT_TILE⌉` calls of
/// `O(P·ROWDOT_TILE²·n)` byte multiply-adds each — `m·ROWDOT_TILE·P·n` in total, a factor
/// `ROWDOT_TILE/P` over the per-row calls' `m·P·n`, paid for `⌈m/ROWDOT_TILE⌉` dispatches instead
/// of `m`. Spatial `O((m + m)·n·limbs)` limb bytes and one tile's `i128` accumulators.
pub(crate) fn rowdot_q32(
    a: &[i64],
    b: &[i64],
    m: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    let need = m.checked_mul(n).ok_or(LimbError::Shape)?;
    if a.len() != need || b.len() != need {
        return Err(LimbError::Shape);
    }
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, m, n, limbs)?;
    let mut y = vec![0i64; m];
    let mut i0 = 0usize;
    while i0 < m {
        let i1 = (i0 + ROWDOT_TILE).min(m);
        let w = i1 - i0;
        let tile = pa.product_rows(&pb, i0, w, i0, w, n)?;
        let mut i = 0usize;
        while i < w {
            y[i0 + i] = tile[i * w + i];
            i += 1;
        }
        i0 = i1;
    }
    Ok(y)
}

/// Columns per [`syrk_q32`] kernel call: the row band the upper triangle needs at that column,
/// small enough that the discarded lower-triangle rows stay a sixth of the work.
const SYRK_COLS: usize = 16;

/// Rows and columns per [`rowdot_q32`] diagonal tile: the work multiplier is `ROWDOT_TILE/P`
/// (8/6 at the default 3 limbs) and the number of kernel calls is `⌈m/ROWDOT_TILE⌉`.
const ROWDOT_TILE: usize = 8;

/// The per-block limb expansion of one operand matrix.
struct Plane {
    /// Rows of the operand.
    rows: usize,
    /// Inputs per row (`n`), unpadded.
    n: usize,
    /// Blocks per row, `⌈n/32⌉`.
    blocks: usize,
    /// `blocks·32`: the padded row length the kernel reads.
    padded: usize,
    /// Limbs per block.
    limbs: u32,
    /// Per block, the exponent shared by every row: `max_i(bitlen(max|x|) + 1)`, `≥ 1`.
    e: Vec<u32>,
    /// `[row][block][limb][lane]`, the balanced digits of `2^{−e}·x`.
    dig: Vec<i8>,
}

/// One kept limb pair: weight digit `s` against activation digit `t`.
type Pair = (u32, u32);

/// The kept pairs `(s, t)`, `s + t ≤ limbs − 1`, ordered by `s + t` then `s`.
///
/// The order matters twice: every prefix of it is closed under `(s, t) ↦ (t, s)` — which is what
/// makes [`syrk_q32`]'s mirror equal to [`gemm_q32`]'s transpose bit for bit — and the pair groups
/// it induces are the ones the two share.
///
/// # Complexity
///
/// `O(limbs²)` time and space; the double loop runs `limbs(limbs+1)/2` accepting steps.
fn kept_pairs(limbs: u32) -> Vec<Pair> {
    let mut out = Vec::with_capacity((limbs * (limbs + 1) / 2) as usize);
    let mut sum = 0u32;
    while sum < limbs {
        let mut s = 0u32;
        while s <= sum {
            let t = sum - s;
            if t < limbs {
                out.push((s, t));
            }
            s += 1;
        }
        sum += 1;
    }
    out
}

/// `⌊log₂ x⌋ + 1`, `0` for `x = 0`.
#[must_use]
fn bitlen(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

/// The exact `f16` word of the Q32.32 power of two `2^{dw}`, `dw ∈ [8, 47]`.
///
/// `2^{dw}` is `2^{dw−32}` as a real: a normal `f16` `(1024 + 0)·2^{e−25}` with
/// `e = dw − 17` when `dw ≥ 18`, and the subnormal `f = 2^{dw−8}` for `dw ∈ [8, 17]`. Reversing
/// through [`super::pow2::f16_split`] gives `(1024, dw − 10)` and `(2^{dw−8}, 8)`, both exactly
/// `2^{dw}`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "dw is clamped to [8, 47]; the normals land in [1, 30] and the subnormal mantissa in \
              [1, 9], so both shifts are exact and the u16 is a 5- or 10-bit field"
)]
fn f16_pow2(dw: i32) -> u16 {
    let p = dw - 32;
    if p >= -14 {
        ((p + 15) as u16) << 10
    } else {
        1u16 << (p + 24)
    }
}

/// The largest `|word|` of one 32-element block.
///
/// # Complexity
///
/// `O(32)`.
fn block_absmax(words: &[i64]) -> u64 {
    let mut m = 0u64;
    for &w in words {
        m = m.max(w.unsigned_abs());
    }
    m
}

/// The block exponent of an absolute maximum: `bitlen(m) + 1` (`1` for an all-zero block), so that
/// `|x| < 2^{e−1}` and the first digit `round(x/2^{e−7})` obeys `|d₀| ≤ 64`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
fn block_exp(absmax: u64) -> u32 {
    if absmax == 0 {
        1
    } else {
        bitlen(absmax) + 1
    }
}

/// The `limbs` balanced digits of `x` at block exponent `e`, written to `out`.
///
/// `x = 2^e·Σ_s d_s·2^{−7(s+1)} + ρ` with `|d_s| ≤ 64`, `|ρ| ≤ 2^{e−7·limbs}/2`. Digit `s` is
/// `⌊(u_s·2^7 + 2^{e−1})/2^e⌋` on the running residual `u_0 = x`, `u_{s+1} = u_s − d_s·2^{e−7}`;
/// that closed form is the same value the earlier residual recurrence
/// `⌊(rem_s + 2^{sh−1})/2^{sh}⌋` computed in units of `2^{e−7·limbs}`, with the `2^{7(limbs−1−s)}`
/// common factor cancelled, so the digits are unchanged. It states the step in the operand's own
/// scale: an `i128` for the first digit only (where `x + 2^{e−8}` can pass `i64::MAX`), and `i64`
/// shifts with the block exponent for the rest, each bounded by `|u_s| ≤ 2^{e−1−7s}`.
///
/// # Complexity
///
/// `O(limbs)` integer shifts, no allocation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "|d_s| ≤ 64 by the block-exponent choice, so the i64-to-i8 cast is exact; the shift \
              amounts are bounded by e ≤ 65"
)]
fn split_digits(x: i64, e: u32, limbs: u32, out: &mut [i8]) {
    // u_s carries the residual at the operand's own scale; the first step needs i128 because
    // `x + 2^{e−8}` cannot leave [−2^63, 2^63) for the largest block exponent.
    let mut u = x;
    let mut s = 0u32;
    while s < limbs {
        let k = e as i32 - LIMB_BITS * (s as i32 + 1);
        let d = if s == 0 {
            let num = (i128::from(u) << LIMB_BITS) + (1i128 << (e - 1));
            (num >> e) as i64
        } else if k >= 1 {
            u.wrapping_add(1i64 << (k - 1)) >> k
        } else {
            // The residual is an exact multiple of the (fractional) step: the remaining digits
            // are the whole residual and the residual is zero.
            u << (-k) as u32
        };
        out[s as usize] = d as i8;
        if k >= 1 {
            u = u.wrapping_sub(d << k);
        } else {
            u = 0;
        }
        s += 1;
    }
}

/// The largest block exponent the wide digit split takes: `e ≤ 62` keeps `|x| + 2^{e−8}` (the
/// rounding offset of the first digit) inside `i64`, so the vector core's `i64` lanes cannot wrap
/// where the scalar `i128` step does not. A block above it takes [`split_digits`] itself.
const SIMD_MAX_E: u32 = 62;

/// The digits of one full block: `ACT_QK` words at the block exponent `e`, written `[limb][lane]`.
///
/// The `AVX2` core runs four `i64` lanes per step when the host has it and the block exponent is
/// inside [`SIMD_MAX_E`]; every other case is the scalar [`split_digits`] the pinned outputs were
/// produced with. Both compute the identical digits (`split_digits`'s own closed form, with the
/// first step's `i128` numerator divided through by `2^7`):
/// `d_s = ⌊(u_s + 2^{k−1})/2^k⌋`, `u_{s+1} = u_s − d_s·2^k`, `k = e − 7(s+1)`, and `d_s = u_s·2^{−k}`
/// (the whole residual) when `k ≤ 0`.
///
/// # Complexity
///
/// `O(limbs·ACT_QK)` digit steps, four lanes per vector step on the `AVX2` core.
fn split_block(x: &[i64], e: u32, limbs: u32, out: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if e <= SIMD_MAX_E && std::is_x86_feature_detected!("avx2") {
            // SAFETY: the caller passes one full block (`ACT_QK` words, `limbs·ACT_QK` bytes) and
            // the feature was just detected.
            unsafe { simd::split_block_avx2(x, e, limbs, out) };
            return;
        }
    }
    let mut tmp = [0i8; MAX_LIMBS as usize];
    let mut lane = 0usize;
    while lane < ACT_QK {
        split_digits(x[lane], e, limbs, &mut tmp);
        let mut s = 0usize;
        while s < limbs as usize {
            out[s * ACT_QK + lane] = tmp[s];
            s += 1;
        }
        lane += 1;
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    #[allow(
        clippy::wildcard_imports,
        reason = "the `core::arch` lane types and the three intrinsics named in the body are the \
                  AVX2 core's whole surface"
    )]
    use std::arch::x86_64::*;

    use super::{ACT_QK, LIMB_BITS};

    /// `v >> k` arithmetic, `0 ≤ k ≤ 63`, on four `i64` lanes.
    ///
    /// `vp sravq` is an AVX-512VL instruction, not AVX2 (AVX2 has only the logical `vpsrlvq` for
    /// qwords), so the arithmetic shift is the logical one plus the sign bits `vpsllvq` puts back.
    ///
    /// # Safety
    ///
    /// AVX2 must be present and `0 ≤ k ≤ 63`.
    #[target_feature(enable = "avx2")]
    unsafe fn sra_epi64(v: __m256i, k: i32) -> __m256i {
        let sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
        let logical = _mm256_srlv_epi64(v, _mm256_set1_epi64x(i64::from(k)));
        let fill = _mm256_sll_epi64(sign, _mm_cvtsi32_si128(64 - k));
        _mm256_or_si256(logical, fill)
    }

    /// The digits of one full block, four `i64` lanes at a time.
    ///
    /// # Safety
    ///
    /// AVX2 must be present; `x` holds `ACT_QK` words, `out` `limbs·ACT_QK` bytes, and
    /// `e ≤ SIMD_MAX_E` so the rounding offset stays inside `i64`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_ptr_alignment,
        clippy::cast_sign_loss,
        reason = "the shift amounts are `e − 7(s+1)` with `e ≤ 62` and `s < 8`, so both `i32` \
                  casts are exact; the load and the store are unaligned `i64`/`i8` reads and \
                  writes by construction"
    )]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn split_block_avx2(x: &[i64], e: u32, limbs: u32, out: &mut [i8]) {
        // Per 128-bit lane, byte 0 of each 64-bit element into positions 0 and 1.
        let pick = _mm256_setr_epi8(
            0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let mut lane0 = 0usize;
        while lane0 < ACT_QK {
            // SAFETY: `lane0 + 4 ≤ ACT_QK`, and `x` holds `ACT_QK` words.
            let mut u = unsafe { _mm256_loadu_si256(x.as_ptr().add(lane0).cast::<__m256i>()) };
            let mut s = 0u32;
            while s < limbs {
                let k = e.cast_signed() - LIMB_BITS * (s.cast_signed() + 1);
                let d = if k >= 1 {
                    let sh = _mm256_set1_epi64x(i64::from(k));
                    let half = _mm256_set1_epi64x(1i64 << (k - 1));
                    let d = sra_epi64(_mm256_add_epi64(u, half), k);
                    u = _mm256_sub_epi64(u, _mm256_sllv_epi64(d, sh));
                    d
                } else {
                    let d = _mm256_sllv_epi64(u, _mm256_set1_epi64x(i64::from(-k)));
                    u = _mm256_setzero_si256();
                    d
                };
                let bytes = _mm256_shuffle_epi8(d, pick);
                let packed = _mm_unpacklo_epi16(
                    _mm256_castsi256_si128(bytes),
                    _mm256_extracti128_si256(bytes, 1),
                );
                let word = _mm_cvtsi128_si32(packed);
                // SAFETY: `s·ACT_QK + lane0 + 4 ≤ limbs·ACT_QK`, the size of `out`.
                unsafe {
                    out.as_mut_ptr()
                        .add(s as usize * ACT_QK + lane0)
                        .cast::<i32>()
                        .write_unaligned(word);
                }
                s += 1;
            }
            lane0 += 4;
        }
    }
}

/// Pass one of [`Plane::build`]: the per-row block exponents, written as `[row][block]` so the
/// caller can take the per-block maximum.
struct RowExpTask<'a> {
    data: &'a [i64],
    n: usize,
    blocks: usize,
}

impl RowExpTask<'_> {
    /// The body of [`ChunkTask::run`] with the borrowed row slice as a plain parameter: the
    /// pinned extraction refuses a loop that reads a reference field of a struct, so `run` passes
    /// `data` in and this function owns every loop. The arithmetic is the task's own.
    fn body(data: &[i64], n: usize, blocks: usize, start: usize, piece: &mut [u32]) {
        let row0 = start / blocks;
        let mut k = 0usize;
        while k < piece.len() {
            let r = row0 + k / blocks;
            let b = k % blocks;
            let base = r * n;
            let lo = base + b * ACT_QK;
            let hi = (lo + ACT_QK).min(base + n);
            piece[k] = block_exp(block_absmax(&data[lo..hi]));
            k += 1;
        }
    }
}

impl ChunkTask<u32> for RowExpTask<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [u32]) {
        Self::body(self.data, self.n, self.blocks, start, piece);
    }
}

/// Pass two of [`Plane::build`]: the digits of the rows in the piece.
struct DigitsTask<'a> {
    data: &'a [i64],
    n: usize,
    blocks: usize,
    lane_words: usize,
    limbs: u32,
    e: &'a [u32],
}

impl DigitsTask<'_> {
    /// The body of [`ChunkTask::run`] with the borrowed row and exponent slices as plain
    /// parameters; see [`RowExpTask::body`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the pool task's fields, passed explicitly because the extraction refuses a loop \
                  that reads a reference field of a struct"
    )]
    fn body(
        data: &[i64],
        n: usize,
        blocks: usize,
        lane_words: usize,
        limbs: u32,
        e: &[u32],
        start: usize,
        piece: &mut [i8],
    ) {
        let per_row = blocks * lane_words;
        let row0 = start / per_row;
        let row_count = piece.len() / per_row;
        let mut k = 0usize;
        while k < row_count {
            let r = row0 + k;
            let base = r * n;
            let out = &mut piece[k * per_row..(k + 1) * per_row];
            let mut b = 0usize;
            while b < blocks {
                let lo = base + b * ACT_QK;
                let hi = (lo + ACT_QK).min(base + n);
                let at = b * lane_words;
                if lo + ACT_QK <= hi {
                    split_block(
                        &data[lo..lo + ACT_QK],
                        e[b],
                        limbs,
                        &mut out[at..at + lane_words],
                    );
                } else {
                    // The tail block zero-padded to a full block: a zero word has zero digits,
                    // so the digits are those of the per-lane split, at one vector call.
                    let mut pad = [0i64; ACT_QK];
                    let mut lane = 0usize;
                    while lane < hi - lo {
                        pad[lane] = data[lo + lane];
                        lane += 1;
                    }
                    split_block(&pad, e[b], limbs, &mut out[at..at + lane_words]);
                }
                b += 1;
            }
            k += 1;
        }
    }
}

impl ChunkTask<i8> for DigitsTask<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i8]) {
        Self::body(
            self.data,
            self.n,
            self.blocks,
            self.lane_words,
            self.limbs,
            self.e,
            start,
            piece,
        );
    }
}

impl Plane {
    /// The limb expansion of `data` (`rows × n`, row-major), or the shape refusal.
    ///
    /// Both passes are pooled over rows when the operand holds at least [`POOL_MIN`] elements;
    /// each task writes its own rows of the per-row exponents or of the digit plane, and the
    /// per-block maximum over rows is the serial fold of the first pass's output.
    ///
    /// # Complexity
    ///
    /// Temporal `O(rows·n·limbs)` digit steps in two passes (block maxima, then digits), split
    /// into `⌈rows·n / 32·CHUNK⌉` pool tasks; spatial `O(rows·⌈n/32⌉·limbs·32)` bytes for the
    /// digits plus `O(rows·⌈n/32⌉)` per-row exponents and `O(⌈n/32⌉)` block exponents.
    fn build(data: &[i64], rows: usize, n: usize, limbs: u32) -> Result<Self, LimbError> {
        if rows == 0 || n == 0 || limbs == 0 || limbs > MAX_LIMBS || n > MAX_INNER {
            return Err(LimbError::Shape);
        }
        if data.len() != rows.checked_mul(n).ok_or(LimbError::Shape)? {
            return Err(LimbError::Shape);
        }
        let blocks = n.div_ceil(ACT_QK);
        let padded = blocks * ACT_QK;
        let mut row_e = vec![0u32; rows * blocks];
        let max = RowExpTask { data, n, blocks };
        if data.len() >= POOL_MIN {
            let chunk = pool_chunk(rows) * blocks;
            chunks_mut_global(&mut row_e, chunk, &max);
        } else {
            max.run(0, 0, &mut row_e);
        }
        let mut e = vec![1u32; blocks];
        let mut r = 0usize;
        while r < rows {
            let mut b = 0usize;
            while b < blocks {
                let cand = row_e[r * blocks + b];
                if cand > e[b] {
                    e[b] = cand;
                }
                b += 1;
            }
            r += 1;
        }
        let lane_words = limbs as usize * ACT_QK;
        let per_row = blocks * lane_words;
        let mut dig = vec![0i8; rows * per_row];
        let digits = DigitsTask {
            data,
            n,
            blocks,
            lane_words,
            limbs,
            e: &e,
        };
        if data.len() >= POOL_MIN {
            let chunk = pool_chunk(rows) * per_row;
            chunks_mut_global(&mut dig, chunk, &digits);
        } else {
            digits.run(0, 0, &mut dig);
        }
        Ok(Self {
            rows,
            n,
            blocks,
            padded,
            limbs,
            e,
            dig,
        })
    }

    /// The 32 limb bytes of `(row, block, limb)`, the block's `lane` order contiguous.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    fn digit_block(&self, row: usize, block: usize, limb: u32) -> &[i8] {
        let lane_words = self.limbs as usize * ACT_QK;
        let at = (row * self.blocks + block) * lane_words + limb as usize * ACT_QK;
        &self.dig[at..at + ACT_QK]
    }

    /// The same bytes read as the `Q8_0` code bytes: `i8` and `u8` share layout and every `i8`
    /// bit pattern is a `u8`.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    fn digit_block_u8(&self, row: usize, block: usize, limb: u32) -> &[u8] {
        // SAFETY: `digit_block` returns `ACT_QK` initialised `i8` inside `dig`; `i8` and `u8`
        // have the same size, alignment and validity, so the reinterpretation is sound.
        unsafe {
            std::slice::from_raw_parts(self.digit_block(row, block, limb).as_ptr().cast(), ACT_QK)
        }
    }
}

/// Where one `(pair, block)` lands on the kernel: the weight scale `2^dw`, the activation scale
/// `2^sigma`, and whether the block pair is carried at all.
#[derive(Clone, Copy)]
struct Place {
    /// Q32.32 exponent of the `f16` weight scale, `[8, 47]`.
    dw: i32,
    /// Exponent of the activation block scale, `[0, 44]`.
    sigma: i32,
    /// `false` when `T < 8`: the block pair's contribution is dropped and the bound counts it.
    live: bool,
}

/// One pair group's kernel inputs: the placement table, the kept pairs and the two `Plane`s that
/// supply the digits.
struct Layout<'a> {
    /// The weight side (its rows are the output rows).
    a: &'a Plane,
    /// The activation side (its rows are the output columns).
    b: &'a Plane,
    /// The kept pairs, in [`kept_pairs`] order.
    pairs: Vec<Pair>,
    /// `[pair][block]` placement.
    place: Vec<Place>,
}

impl Layout<'_> {
    /// The placement of every `(pair, block)`, or the range refusal when a block pair's product
    /// exponent is past the widest `f16` weight scale times the widest `i64` activation scale.
    ///
    /// # Complexity
    ///
    /// `O(P·blocks)` time and space.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "block exponents are ≤ 65 and pair exponents ≤ 2·MAX_LIMBS, far below i32::MAX"
    )]
    fn placement(a: &Plane, b: &Plane, pairs: &[Pair]) -> Result<Vec<Place>, LimbError> {
        let blocks = a.blocks;
        let mut place = vec![
            Place {
                dw: MIN_PAIR_EXP,
                sigma: 0,
                live: false,
            };
            pairs.len() * blocks
        ];
        let mut gi = 0usize;
        let mut err: Option<LimbError> = None;
        while gi < pairs.len() && err.is_none() {
            let (s, t) = pairs[gi];
            let weight = LIMB_BITS * ((s + t) as i32 + 2);
            let mut blk = 0usize;
            while blk < blocks && err.is_none() {
                let t_exp = a.e[blk] as i32 + b.e[blk] as i32 - weight;
                if t_exp > MAX_PAIR_EXP {
                    err = Some(LimbError::Range);
                } else {
                    let slot = gi * blocks + blk;
                    if t_exp >= MIN_PAIR_EXP {
                        let dw = t_exp.clamp(MIN_PAIR_EXP, MAX_WEIGHT_EXP);
                        place[slot] = Place {
                            dw,
                            sigma: t_exp - dw,
                            live: true,
                        };
                    }
                    // Below `MIN_PAIR_EXP` the pair cannot be placed (the activation scale would
                    // be fractional); the block pair's whole contribution is below 2^-8 words and
                    // the bound counts it, so the slot keeps `live: false`.
                }
                blk += 1;
            }
            gi += 1;
        }
        match err {
            Some(e) => Err(e),
            None => Ok(place),
        }
    }

    /// The `Q8_0` payload of rows `a0..a0+ra`: `ng` pair blocks per 32 inputs, each a `f16` power
    /// of two followed by the weight-side digits.
    ///
    /// # Complexity
    ///
    /// `O(ra·ng·blocks·32)` bytes written, one 32-byte copy per live limb block, split into
    /// `⌈ra·row_bytes / POOL_BYTES⌉` pool tasks; one allocation of `ra·ng·blocks·34`.
    fn weight_payload(&self, a0: usize, ra: usize, g0: usize, g1: usize) -> Vec<u8> {
        let blocks = self.a.blocks;
        let row_bytes = (g1 - g0) * blocks * Q8_0_BLOCK_BYTES;
        let mut raw = vec![0u8; ra * row_bytes];
        let task = PayloadTask {
            layout: self,
            a0,
            g0,
            g1,
            blocks,
            row_bytes,
        };
        if raw.len() >= POOL_MIN {
            let chunk = pool_chunk(ra) * row_bytes;
            chunks_mut_global(&mut raw, chunk, &task);
        } else {
            task.run(0, 0, &mut raw);
        }
        raw
    }

    /// The activation columns of rows `b0..b0+rb`: `ng` pair blocks per 32 inputs, each a power of
    /// two and the activation-side digits.
    ///
    /// # Complexity
    ///
    /// `O(rb·ng·blocks·32)` bytes written, one 32-byte copy per live limb block, split into
    /// `⌈rb·col_bytes / POOL_BYTES⌉` pool tasks; `rb` allocations of `ng·32·blocks` codes.
    fn columns(&self, b0: usize, rb: usize, g0: usize, g1: usize) -> Vec<QuantAct> {
        let blocks = self.b.blocks;
        let padded = self.b.padded;
        ColumnsTask::dispatch(self, b0, rb, g0, g1, blocks, padded)
    }
}

/// Fill one piece of the weight payload: rows `[a0 + start/row_bytes, …)`.
struct PayloadTask<'a> {
    layout: &'a Layout<'a>,
    a0: usize,
    g0: usize,
    g1: usize,
    blocks: usize,
    row_bytes: usize,
}

impl PayloadTask<'_> {
    /// The body of [`ChunkTask::run`] with the three borrowed pieces of the layout as plain
    /// parameters; see [`RowExpTask::body`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the pool task's fields, passed explicitly because the extraction refuses a loop \
                  that reads a reference field of a struct"
    )]
    fn body(
        a: &Plane,
        pairs: &[Pair],
        place: &[Place],
        a0: usize,
        g0: usize,
        g1: usize,
        blocks: usize,
        row_bytes: usize,
        start: usize,
        piece: &mut [u8],
    ) {
        let row0 = start / row_bytes;
        let row_count = piece.len() / row_bytes;
        let mut k = 0usize;
        while k < row_count {
            let row = a0 + row0 + k;
            let out = &mut piece[k * row_bytes..(k + 1) * row_bytes];
            let mut g = g0;
            while g < g1 {
                let (s, _) = pairs[g];
                let mut blk = 0usize;
                while blk < blocks {
                    let p = place[g * blocks + blk];
                    if p.live {
                        let at = (g - g0) * blocks * Q8_0_BLOCK_BYTES + blk * Q8_0_BLOCK_BYTES;
                        out[at..at + 2].copy_from_slice(&f16_pow2(p.dw).to_le_bytes());
                        out[at + 2..at + Q8_0_BLOCK_BYTES]
                            .copy_from_slice(a.digit_block_u8(row, blk, s));
                    }
                    blk += 1;
                }
                g += 1;
            }
            k += 1;
        }
    }
}

impl ChunkTask<u8> for PayloadTask<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [u8]) {
        let layout = self.layout;
        Self::body(
            layout.a,
            &layout.pairs,
            &layout.place,
            self.a0,
            self.g0,
            self.g1,
            self.blocks,
            self.row_bytes,
            start,
            piece,
        );
    }
}

/// Fill one piece of the activation columns: columns `[start, start + piece.len())`.
struct ColumnsTask<'a> {
    layout: &'a Layout<'a>,
    b0: usize,
    g0: usize,
    g1: usize,
    blocks: usize,
    padded: usize,
}

impl ColumnsTask<'_> {
    /// The `rb` empty activation columns the pool task fills.
    ///
    /// The fill is its own function because the pinned extraction refuses a loop in a function
    /// whose parameters include a reference to a struct that itself holds references; `dispatch`
    /// carries the one `&Layout` parameter and owns no loop.
    ///
    /// # Complexity
    ///
    /// Temporal and spatial `O(rb)`: `rb` pushes, one allocation of `rb` headers.
    fn empty_columns(rb: usize) -> Vec<QuantAct> {
        let mut parts: Vec<QuantAct> = Vec::with_capacity(rb);
        let mut z = 0usize;
        while z < rb {
            parts.push(QuantAct::from_parts(Vec::new(), Vec::new()));
            z += 1;
        }
        parts
    }

    /// Build the `rb` empty activation columns and fill them: pooled above [`POOL_MIN`] output
    /// bytes, serially on the calling thread below it.
    ///
    /// This is `Layout::columns`'s body over parameters only: the two `blocks`/`padded` reads stay
    /// in `Layout::columns`, and the one loop stays in [`ColumnsTask::empty_columns`].
    ///
    /// # Complexity
    ///
    /// As [`Layout::columns`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller's scoped variables, passed explicitly because the extraction refuses \
                  a loop beside a reference-field read"
    )]
    fn dispatch(
        layout: &Layout<'_>,
        b0: usize,
        rb: usize,
        g0: usize,
        g1: usize,
        blocks: usize,
        padded: usize,
    ) -> Vec<QuantAct> {
        let ng = g1 - g0;
        // Empty columns; the pool task allocates and zeroes each column's blocks, so the fill and
        // the zeroing both run off the calling thread.
        let mut parts = Self::empty_columns(rb);
        let task = ColumnsTask {
            layout,
            b0,
            g0,
            g1,
            blocks,
            padded,
        };
        let col_bytes = ng * padded;
        if rb * col_bytes >= POOL_MIN {
            let chunk = pool_chunk(rb);
            chunks_mut_global(&mut parts, chunk, &task);
        } else {
            task.run(0, 0, &mut parts);
        }
        parts
    }

    /// The body of [`ChunkTask::run`] with the three borrowed pieces of the layout as plain
    /// parameters; see [`RowExpTask::body`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the pool task's fields, passed explicitly because the extraction refuses a loop \
                  that reads a reference field of a struct"
    )]
    fn body(
        b: &Plane,
        pairs: &[Pair],
        place: &[Place],
        b0: usize,
        g0: usize,
        g1: usize,
        blocks: usize,
        padded: usize,
        start: usize,
        piece: &mut [QuantAct],
    ) {
        let mut k = 0usize;
        while k < piece.len() {
            let row = b0 + start + k;
            let mut scales = vec![0i64; (g1 - g0) * blocks];
            let mut codes = vec![0i8; (g1 - g0) * padded];
            let mut g = g0;
            while g < g1 {
                let (_, t) = pairs[g];
                let mut blk = 0usize;
                while blk < blocks {
                    let p = place[g * blocks + blk];
                    scales[(g - g0) * blocks + blk] = 1i64 << p.sigma;
                    if p.live {
                        let at = (g - g0) * padded + blk * ACT_QK;
                        codes[at..at + ACT_QK].copy_from_slice(b.digit_block(row, blk, t));
                    }
                    blk += 1;
                }
                g += 1;
            }
            piece[k] = QuantAct::from_parts(scales, codes);
            k += 1;
        }
    }
}

impl ChunkTask<QuantAct> for ColumnsTask<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [QuantAct]) {
        let layout = self.layout;
        Self::body(
            layout.b,
            &layout.pairs,
            &layout.place,
            self.b0,
            self.g0,
            self.g1,
            self.blocks,
            self.padded,
            start,
            piece,
        );
    }
}

/// One kernel projection on the packed planes, or the `Q8_0` contract on the same planes.
///
/// # Complexity
///
/// `gemm_q8_0`'s `O(ra·rb·n_in)` byte multiply-adds.
fn projection(
    raw: &[u8],
    ra: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Result<Vec<Vec<i64>>, LimbError> {
    match gemm_q8_0(GGML_TYPE_Q8_0, raw, ra, n_in, qs) {
        Some(cols) => Ok(cols),
        // No AVX-VNNI, an unsupported width or a non-finite scale: the same planes go to the
        // `Q8_0` contract itself, never to a scalar Q32.32 loop.
        None => matmat_q8act(GGML_TYPE_Q8_0, raw, ra, n_in, qs).map_err(|_| LimbError::Range),
    }
}

/// Row-major `out[i·rb + j] = cols[j][i]` for `i < ra`, tiled [`TRANSPOSE_TILE`] square so both
/// sides stay in cache.
///
/// # Complexity
///
/// `O(ra·rb)` word moves; pooled over rows into `⌈ra·rb / POOL_WORDS⌉` tasks when the output is
/// at least [`POOL_MIN`] words.
#[allow(
    clippy::cast_possible_truncation,
    reason = "POOL_MIN/8 is a constant word count; the chunk is rounded to a whole row count"
)]
fn transpose_rows(out: &mut [i64], cols: &[Vec<i64>], ra: usize, rb: usize) {
    const TRANSPOSE_TILE: usize = 32;
    struct TransposeTask<'a> {
        cols: &'a [Vec<i64>],
        rb: usize,
    }
    impl ChunkTask<i64> for TransposeTask<'_> {
        fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
            Self::body(self.cols, self.rb, start, piece);
        }
    }

    impl TransposeTask<'_> {
        /// The body of [`ChunkTask::run`] with the borrowed columns as a plain parameter; see
        /// [`RowExpTask::body`].
        fn body(cols: &[Vec<i64>], rb: usize, start: usize, piece: &mut [i64]) {
            let row0 = start / rb;
            let row_count = piece.len() / rb;
            let mut j0 = 0usize;
            while j0 < rb {
                let j1 = (j0 + TRANSPOSE_TILE).min(rb);
                let mut i = 0usize;
                while i < row_count {
                    let i1 = (i + TRANSPOSE_TILE).min(row_count);
                    let mut j = j0;
                    while j < j1 {
                        let col = &cols[j];
                        let mut r = i;
                        while r < i1 {
                            piece[r * rb + j] = col[row0 + r];
                            r += 1;
                        }
                        j += 1;
                    }
                    i = i1;
                }
                j0 = j1;
            }
        }
    }
    if ra * rb < POOL_MIN {
        let mut j0 = 0usize;
        while j0 < rb {
            let j1 = (j0 + TRANSPOSE_TILE).min(rb);
            let mut i0 = 0usize;
            while i0 < ra {
                let i1 = (i0 + TRANSPOSE_TILE).min(ra);
                let mut j = j0;
                while j < j1 {
                    let col = &cols[j];
                    let mut i = i0;
                    while i < i1 {
                        out[i * rb + j] = col[i];
                        i += 1;
                    }
                    j += 1;
                }
                i0 = i1;
            }
            j0 = j1;
        }
        return;
    }
    // A band of at least one tile: every column's band slice then shares cache lines across the
    // tile's rows, so each column is read once over the whole transpose.
    let rows_per_task = pool_chunk(ra).max(TRANSPOSE_TILE);
    chunks_mut_global(out, rows_per_task * rb, &TransposeTask { cols, rb });
}

impl Plane {
    /// The kernel product of rows `a0..a0+ra` against rows `b0..b0+rb` (both over the same `n`),
    /// row-major `ra × rb`.
    ///
    /// The one-group case (every kept pair in a single kernel call, which the default 3 limbs and
    /// the production widths always take) transposes the kernel's columns straight into the output
    /// with no accumulator; a multi-group call keeps a column-major `i128` accumulator, one
    /// contiguous add per kernel output word, and converts once.
    ///
    /// # Complexity
    ///
    /// Temporal `O(ra·n·limbs)` plane-to-payload packing plus `O(P·ra·rb·n)` byte multiply-adds
    /// over `⌈P/g⌉` kernel calls, `g = ⌊MAX_INNER/padded⌋` pairs per call; spatial `O(ra·g·n)` payload
    /// bytes plus `O(ra·rb)` output words (`i128` only when the call spans several groups).
    fn product_rows(
        &self,
        b: &Plane,
        a0: usize,
        ra: usize,
        b0: usize,
        rb: usize,
        n: usize,
    ) -> Result<Vec<i64>, LimbError> {
        if n != self.n
            || n != b.n
            || ra == 0
            || rb == 0
            || a0 + ra > self.rows
            || b0 + rb > b.rows
            || self.limbs != b.limbs
        {
            return Err(LimbError::Shape);
        }
        let pairs = kept_pairs(self.limbs);
        let place = Layout::placement(self, b, &pairs)?;
        let layout = Layout {
            a: self,
            b,
            pairs,
            place,
        };
        let group = (MAX_INNER / self.padded).max(1).min(layout.pairs.len());
        let mut out = vec![0i64; ra * rb];
        if layout.pairs.len() <= group {
            let n_in = layout.pairs.len() * self.padded;
            let raw = layout.weight_payload(a0, ra, 0, layout.pairs.len());
            let qs = layout.columns(b0, rb, 0, layout.pairs.len());
            let cols = projection(&raw, ra, n_in, &qs)?;
            transpose_rows(&mut out, &cols, ra, rb);
            return Ok(out);
        }
        // Column-major `[j·ra + i]`: the kernel output and its accumulator are both contiguous in
        // `i`, and the transpose happens once on the finished words.
        let mut acc = vec![0i128; ra * rb];
        let mut g0 = 0usize;
        let mut err: Option<LimbError> = None;
        while g0 < layout.pairs.len() && err.is_none() {
            let g1 = (g0 + group).min(layout.pairs.len());
            let n_in = (g1 - g0) * self.padded;
            let raw = layout.weight_payload(a0, ra, g0, g1);
            let qs = layout.columns(b0, rb, g0, g1);
            match projection(&raw, ra, n_in, &qs) {
                Ok(cols) => {
                    let mut j = 0usize;
                    while j < rb {
                        let col = &cols[j];
                        let base = j * ra;
                        let mut i = 0usize;
                        while i < ra {
                            acc[base + i] += i128::from(col[i]);
                            i += 1;
                        }
                        j += 1;
                    }
                }
                Err(e) => err = Some(e),
            }
            g0 = g1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut i = 0usize;
        let mut err2: Option<LimbError> = None;
        while i < ra && err2.is_none() {
            let mut j = 0usize;
            while j < rb && err2.is_none() {
                if let Ok(v) = i64::try_from(acc[j * ra + i]) {
                    out[i * rb + j] = v;
                } else {
                    err2 = Some(LimbError::Range);
                }
                j += 1;
            }
            i += 1;
        }
        match err2 {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}

/// The per-entry word bound of [`gemm_q32`] for `n`, `limbs` and the two operands' largest
/// magnitudes: `⌈P/2⌉ + 1 + ⌈n/32⌉·(1 + ⌈C_L·2^{Ea+Eb−32}⌉)` with `C_L = (33 + 16(L−1))·2^{−7L}`.
///
/// Every term is rounded up and every vanished fraction is carried by the trailing `+1`, so the
/// value is a strict upper bound for every entry the construction can produce; it is the assertion
/// the tests run against.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the exponents are bounded by 64 + 64; the shifts stay inside i128's 127-bit range \
              and the `as u32` amounts are at most 66"
)]
pub(crate) fn bound_ulps(n: usize, limbs: u32, max_a: u64, max_b: u64) -> i128 {
    let l = i128::from(limbs);
    let pairs = l * (l + 1) / 2;
    let blocks = i128::from((n.div_ceil(ACT_QK)) as u64);
    let ea = i128::from(block_exp(max_a));
    let eb = i128::from(block_exp(max_b));
    let exp = ea + eb - 32 - i128::from(LIMB_BITS) * l;
    let c = 33 + 16 * (l - 1);
    let trunc: i128 = if exp >= 0 {
        let sh = exp as u32;
        if sh >= 120 {
            i128::MAX / 4
        } else {
            (c << sh) + 1
        }
    } else {
        let sh = (-exp) as u32;
        if sh >= 120 {
            1
        } else {
            let d = 1i128 << sh;
            (c + d - 1) / d + 1
        }
    };
    (pairs / 2 + 1) + blocks * (1 + trunc)
}

/// One `F16` weight matrix in the limb representation, packed once for repeated projections.
///
/// The weights are decoded to Q32.32 words, split into `limbs` balanced signed-byte digits by
/// [`Plane::build`], and repacked as one [`Q8_0_BLOCK_BYTES`] block per `(pair, block)` with the
/// `f16` scale `2^{clamp(e[b] − 7(s+1), 8, 47)}` ([`f16_pow2`]). The activation side is the chart's
/// one-limb [`QuantAct`], so one projection is `limbs` kernel passes over the same codes rather
/// than the `limbs(limbs+1)/2` pair passes of [`gemm_q32`]. The scale exponent is independent of the
/// activation, which is what lets the tensor be packed before any column is known.
///
/// The activation scale of `(column c, pair s, block b)` is
/// `sigma = e[b] + q(c, b) − 7(s+1) − dw(s, b)`, with `q` the exponent of the column's own
/// `QuantAct` block scale; `T = e[b] + q − 7(s+1)` is the block pair's Q32.32 exponent, and a
/// pair is dropped (scale `0`) exactly when `T < 8`, the same rule and the same `+1`-per-block
/// accounting as [`gemm_q32`]'s bound.
pub(crate) struct F16Tile {
    /// Output rows (weights per output).
    rows: usize,
    /// Inputs per row, unpadded.
    n: usize,
    /// Blocks per row, `⌈n/32⌉`.
    blocks: usize,
    /// `blocks·32`, the padded row length the kernel reads.
    padded: usize,
    /// Limbs per block (the weight-side pair count).
    limbs: u32,
    /// Per block, the exponent shared by the block's weight words.
    e: Vec<u32>,
    /// Per `(pair, block)`, the packed `f16` weight scale exponent.
    dw: Vec<i32>,
    /// One packed payload per kept-pair group.
    groups: Vec<F16Group>,
}

impl std::fmt::Debug for F16Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F16Tile")
            .field("rows", &self.rows)
            .field("n", &self.n)
            .field("limbs", &self.limbs)
            .finish_non_exhaustive()
    }
}

/// One `Q8_0` group of a packed [`F16Tile`]: pairs `[g0, g1)`.
struct F16Group {
    /// First pair of the group.
    g0: usize,
    /// One past the last pair.
    g1: usize,
    /// `rows × (g1−g0) × blocks` `Q8_0` blocks, row-major.
    raw: Vec<u8>,
}

/// Decode an `F16` payload and pack it as a [`F16Tile`], or a refusal.
///
/// # Errors
///
/// [`LimbError::Shape`] on a zero dimension, `limbs == 0`, `limbs > MAX_LIMBS`, `n > MAX_INNER`, or
/// a payload whose length is not `2·rows·n`; [`LimbError::Range`] on an `Inf`/`NaN` half word or a
/// block exponent the scale placement cannot carry.
///
/// # Complexity
///
/// Temporal `O(rows·n·limbs)` for the decode, the [`Plane::build`] passes and the repack; spatial
/// `O(rows·n)` `i64` words transiently, `O(rows·⌈n/32⌉·limbs)` digit bytes, and the returned
/// `O(rows·limbs·⌈n/32⌉·34)` payload bytes.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::manual_clamp,
    reason = "`e ≤ 65`, `s < MAX_LIMBS ≤ 8` and `g < limbs`, so every index and exponent cast is \
              exact; the clamp is written as two comparisons so the extraction sees plain `if`s"
)]
pub(crate) fn f16_tile(
    raw: &[u8],
    rows: usize,
    n: usize,
    limbs: u32,
) -> Result<F16Tile, LimbError> {
    let cells = rows.checked_mul(n).ok_or(LimbError::Shape)?;
    let bytes = cells.checked_mul(2).ok_or(LimbError::Shape)?;
    if rows == 0 || n == 0 || limbs == 0 || limbs > MAX_LIMBS || n > MAX_INNER || raw.len() != bytes
    {
        return Err(LimbError::Shape);
    }
    let mut words: Vec<i64> = Vec::with_capacity(cells);
    let mut ok = true;
    let mut i = 0usize;
    while i < cells && ok {
        let h = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        match super::pow2::f16_split(h) {
            Some((m, k)) => words.push(i64::from(m) << k),
            None => ok = false,
        }
        i += 1;
    }
    if !ok {
        return Err(LimbError::Range);
    }
    let plane = Plane::build(&words, rows, n, limbs)?;
    let blocks = plane.blocks;
    let padded = plane.padded;
    // The packed weight scale of `(pair, block)`: the pair's own exponent clamped into the finite
    // `f16` power-of-two range. It is activation-independent by construction.
    let mut dw = vec![0i32; limbs as usize * blocks];
    let mut s = 0u32;
    while s < limbs {
        let mut b = 0usize;
        while b < blocks {
            let t = plane.e[b] as i32 - LIMB_BITS * (s as i32 + 1);
            let d = if t < MIN_PAIR_EXP {
                MIN_PAIR_EXP
            } else if t > MAX_WEIGHT_EXP {
                MAX_WEIGHT_EXP
            } else {
                t
            };
            dw[s as usize * blocks + b] = d;
            b += 1;
        }
        s += 1;
    }
    let group = (MAX_INNER / padded).max(1).min(limbs as usize);
    let mut groups: Vec<F16Group> = Vec::new();
    let mut g0 = 0usize;
    while g0 < limbs as usize {
        let g1 = (g0 + group).min(limbs as usize);
        let row_bytes = (g1 - g0) * blocks * Q8_0_BLOCK_BYTES;
        let mut packed = vec![0u8; rows * row_bytes];
        let mut r = 0usize;
        while r < rows {
            let out = &mut packed[r * row_bytes..(r + 1) * row_bytes];
            let mut g = g0;
            while g < g1 {
                let mut b = 0usize;
                while b < blocks {
                    let at = (g - g0) * blocks * Q8_0_BLOCK_BYTES + b * Q8_0_BLOCK_BYTES;
                    out[at..at + 2].copy_from_slice(&f16_pow2(dw[g * blocks + b]).to_le_bytes());
                    out[at + 2..at + Q8_0_BLOCK_BYTES]
                        .copy_from_slice(plane.digit_block_u8(r, b, g as u32));
                    b += 1;
                }
                g += 1;
            }
            r += 1;
        }
        groups.push(F16Group {
            g0,
            g1,
            raw: packed,
        });
        g0 = g1;
    }
    Ok(F16Tile {
        rows,
        n,
        blocks,
        padded,
        limbs,
        e: plane.e,
        dw,
        groups,
    })
}

/// The one-limb activation columns of pairs `[g0, g1)` for every `QuantAct` in `qs`, or a refusal
/// when a block pair's scales cannot be placed.
///
/// # Errors
///
/// [`LimbError::Shape`] on a column whose length is not the tile's `n`; [`LimbError::Range`] on a
/// pair exponent past [`MAX_PAIR_EXP`], an activation scale that is not a non-negative power of two,
/// or a `sigma` past the kernel's `44`.
///
/// # Complexity
///
/// Temporal and spatial `O(k·(g1−g0)·blocks + k·(g1−g0)·padded)` bytes for the columns.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::manual_range_contains,
    reason = "`q` is a `trailing_zeros` of a positive `i64` (≤ 63) and `s` is a pair index below \
              `MAX_LIMBS`; both casts are exact, and the sigma range is written as two comparisons \
              so the extraction sees plain `if`s"
)]
fn f16_columns(
    tile: &F16Tile,
    qs: &[QuantAct],
    g0: usize,
    g1: usize,
) -> Result<Vec<QuantAct>, LimbError> {
    let blocks = tile.blocks;
    let padded = tile.padded;
    let ng = g1 - g0;
    let mut out: Vec<QuantAct> = Vec::with_capacity(qs.len());
    let mut c = 0usize;
    let mut err: Option<LimbError> = None;
    while c < qs.len() && err.is_none() {
        if qs[c].len() == tile.n {
            let codes_c = qs[c].codes();
            let scales_c = qs[c].scales();
            let mut scales = vec![0i64; ng * blocks];
            let mut codes = vec![0i8; ng * padded];
            let mut g = g0;
            while g < g1 && err.is_none() {
                let s = (g - g0) as u32;
                let mut b = 0usize;
                while b < blocks && err.is_none() {
                    let scale = scales_c[b];
                    if scale != 0 && (scale < 0 || (scale & (scale - 1)) != 0) {
                        err = Some(LimbError::Range);
                    } else {
                        let q = if scale == 0 {
                            0
                        } else {
                            scale.trailing_zeros() as i32
                        };
                        let t = tile.e[b] as i32 + q - LIMB_BITS * (s as i32 + 1);
                        if t > MAX_PAIR_EXP {
                            err = Some(LimbError::Range);
                        } else if t >= MIN_PAIR_EXP {
                            let sigma = t - tile.dw[g * blocks + b];
                            if sigma < 0 || sigma > 44 {
                                err = Some(LimbError::Range);
                            } else {
                                scales[(g - g0) * blocks + b] = 1i64 << sigma;
                                let lo = b * ACT_QK;
                                let take = ACT_QK.min(tile.n - lo);
                                let at = (g - g0) * padded + lo;
                                codes[at..at + take].copy_from_slice(&codes_c[lo..lo + take]);
                            }
                        }
                        // `t < 8`: the block pair's whole contribution is below `2^-8` words and is
                        // dropped, exactly as `Layout::placement` drops it; the scale stays `0`.
                    }
                    b += 1;
                }
                g += 1;
            }
            out.push(QuantAct::from_parts(scales, codes));
        }
        c += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// `C[j][i] = Σ_l w[i·n + l]·x_j[l]` in Q32.32 for the packed F16 weights of `tile` and the
/// activation columns `qs`, or a refusal the caller answers with the scalar oracle.
///
/// One projection is `tile.limbs` kernel passes ([`projection`]) over the same activation codes;
/// the passes are summed in `i128` and narrowed once, so the call-level rounding is one
/// `round_shift_32` per pass. The result is within [`gemm_q32`]'s `G(n, limbs, max|w|, max|x|)`
/// of the scalar [`matvec_q8act`] oracle on the same words: the activation is exact in one limb
/// (its codes are the digits), the weight keeps `limbs` digits, and the `+1` per block covers the
/// dropped `T < 8` pairs.
///
/// # Errors
///
/// [`LimbError::Shape`] on an empty `qs` or a column of the wrong length; [`LimbError::Range`] on
/// the placements [`f16_columns`] refuses, a kernel refusal on the packed planes, or an output
/// that leaves `i64`.
///
/// # Complexity
///
/// Temporal `O(k·limbs·rows·n)` byte multiply-adds on the kernel plus the `O(k·limbs·n)` column
/// build; spatial `O(k·limbs·n)` activation bytes and `O(k·rows)` output words.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `i128`-to-`i64` narrowing follows an explicit `i64::MIN`/`i64::MAX` check; \
              `TryFrom` emits an unmodeled external the pinned fail-closed extraction audit rejects"
)]
pub(crate) fn f16_project(tile: &F16Tile, qs: &[QuantAct]) -> Result<Vec<Vec<i64>>, LimbError> {
    let k = qs.len();
    if k == 0 || tile.rows == 0 {
        return Err(LimbError::Shape);
    }
    let mut result: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut acc = vec![0i128; k * tile.rows];
    let mut err: Option<LimbError> = None;
    let mut gi = 0usize;
    while gi < tile.groups.len() && err.is_none() {
        let g0 = tile.groups[gi].g0;
        let g1 = tile.groups[gi].g1;
        let n_in = (g1 - g0) * tile.padded;
        match f16_columns(tile, qs, g0, g1) {
            Err(e) => err = Some(e),
            Ok(columns) => match projection(&tile.groups[gi].raw, tile.rows, n_in, &columns) {
                Err(e) => err = Some(e),
                Ok(cols) => {
                    let mut j = 0usize;
                    while j < k {
                        let col = &cols[j];
                        let base = j * tile.rows;
                        let mut i = 0usize;
                        while i < tile.rows {
                            acc[base + i] += i128::from(col[i]);
                            i += 1;
                        }
                        j += 1;
                    }
                }
            },
        }
        gi += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut j = 0usize;
    while j < k && err.is_none() {
        let mut col: Vec<i64> = Vec::with_capacity(tile.rows);
        let base = j * tile.rows;
        let mut i = 0usize;
        while i < tile.rows && err.is_none() {
            let v = acc[base + i];
            if v < i128::from(i64::MIN) || v > i128::from(i64::MAX) {
                err = Some(LimbError::Range);
            } else {
                col.push(v as i64);
            }
            i += 1;
        }
        result.push(col);
        j += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::chart::matvec_q8act;
    use crate::engine::num::value::round_shift_32;

    /// xorshift64*: the seeded generator the tests use (no `rand` dependency).
    fn step(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// `rows × n` Q32.32 words with `|x| ∈ [2^{k+32}, 2^{k+33})`, every 37th element zero and a
    /// random sign on the rest; `k = 0` means a magnitude around `1.0`.
    fn operand(state: &mut u64, rows: usize, n: usize, k: i32) -> Vec<i64> {
        let mut out = Vec::with_capacity(rows * n);
        let base = (k + 32).cast_unsigned();
        let mask = (1i64 << base) - 1;
        let mut i = 0usize;
        while i < rows * n {
            let r = step(state);
            if i % 37 == 3 {
                out.push(0);
            } else {
                let mag = (1i64 << base) | (r.cast_signed() & mask);
                out.push(if r >> 63 == 0 { mag } else { -mag });
            }
            i += 1;
        }
        out
    }

    /// The exact Q32.32 product: `round_shift_32(Σ_l a·b)`, the contract the kernel reproduces.
    fn reference(a: &[i64], m: usize, b: &[i64], p: usize, n: usize) -> Vec<i64> {
        let mut c = vec![0i64; m * p];
        let mut i = 0usize;
        while i < m {
            let mut j = 0usize;
            while j < p {
                let mut acc: i128 = 0;
                let mut l = 0usize;
                while l < n {
                    acc += i128::from(a[i * n + l]) * i128::from(b[j * n + l]);
                    l += 1;
                }
                c[i * p + j] = i64::try_from(round_shift_32(acc)).expect("reference fits i64");
                j += 1;
            }
            i += 1;
        }
        c
    }

    fn max_abs(v: &[i64]) -> u64 {
        v.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0)
    }

    /// The largest entry error of `got` against `want`.
    fn max_err(got: &[i64], want: &[i64]) -> i128 {
        let mut e: i128 = 0;
        for (g, w) in got.iter().zip(want) {
            let d = (i128::from(*g) - i128::from(*w)).abs();
            if d > e {
                e = d;
            }
        }
        e
    }

    /// (a) Every entry is inside [`bound_ulps`] at 3 and 4 limbs, over the shape grid and a
    /// magnitude grid that keeps every product inside `i64`.
    #[test]
    fn entries_are_inside_the_bound() {
        let shapes = [
            (1usize, 1usize),
            (1, 7),
            (7, 1),
            (7, 7),
            (64, 7),
            (7, 64),
            (64, 64),
        ];
        let ns = [1usize, 31, 32, 33, 896];
        let mags = [(-20, 20), (0, 0), (10, 7), (-20, -20), (20, -20), (-7, 10)];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut worst = 0i128;
        let mut checked = 0usize;
        for limbs in [3u32, 4] {
            for (t, &(m, p)) in shapes.iter().enumerate() {
                for (u, &n) in ns.iter().enumerate() {
                    let (ka, kb) = mags[(t + u) % mags.len()];
                    let a = operand(&mut seed, m, n, ka);
                    let b = operand(&mut seed, p, n, kb);
                    let got = gemm_q32(&a, m, &b, p, n, limbs).expect("carried");
                    let want = reference(&a, m, &b, p, n);
                    let bound = bound_ulps(n, limbs, max_abs(&a), max_abs(&b));
                    let err = max_err(&got, &want);
                    assert!(
                        err <= bound,
                        "limbs {limbs} m {m} p {p} n {n} ka {ka} kb {kb}: err {err} > bound {bound}"
                    );
                    worst = worst.max(if bound == 0 { 0 } else { err * 1000 / bound });
                    checked += 1;
                }
            }
        }
        println!("(a) {checked} products, worst error = {worst}⁄1000 of the bound");
    }

    /// (b) `syrk_q32` is `gemm_q32(a, a)` bit for bit and symmetric.
    #[test]
    fn syrk_matches_gemm_bit_for_bit() {
        let mut seed = 0xdead_beef_1234_5678u64;
        for limbs in [3u32, 4] {
            for (m, n) in [(1usize, 1usize), (1, 33), (7, 31), (64, 32), (7, 896)] {
                let a = operand(&mut seed, m, n, 3);
                let syrk = syrk_q32(&a, m, n, limbs).expect("carried");
                let gemm = gemm_q32(&a, m, &a, m, n, limbs).expect("carried");
                assert_eq!(syrk, gemm, "limbs {limbs} m {m} n {n}");
                for i in 0..m {
                    for j in 0..m {
                        assert_eq!(syrk[i * m + j], syrk[j * m + i], "asymmetric ({i},{j})");
                    }
                }
            }
        }
    }

    /// (c) `gemv_q32` is the `m = 1` `gemm_q32` bit for bit and inside the bound; `rowdot_q32` is
    /// the `gemm_q32` diagonal bit for bit.
    #[test]
    fn gemv_and_rowdot_agree_with_gemm() {
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for limbs in [3u32, 4] {
            for (p, n) in [(1usize, 1usize), (7, 31), (64, 32), (33, 33), (7, 896)] {
                let x = operand(&mut seed, 1, n, 4);
                let b = operand(&mut seed, p, n, -3);
                let y = gemv_q32(&x, &b, p, n, limbs).expect("carried");
                let g = gemm_q32(&x, 1, &b, p, n, limbs).expect("carried");
                assert_eq!(y, g, "gemv p {p} n {n}");
                let want = reference(&x, 1, &b, p, n);
                let bound = bound_ulps(n, limbs, max_abs(&x), max_abs(&b));
                assert!(max_err(&y, &want) <= bound, "gemv bound p {p} n {n}");
            }
            for (m, n) in [(1usize, 1usize), (7, 32), (7, 33), (64, 896)] {
                let a = operand(&mut seed, m, n, 3);
                let b = operand(&mut seed, m, n, -2);
                let dots = rowdot_q32(&a, &b, m, n, limbs).expect("carried");
                let full = gemm_q32(&a, m, &b, m, n, limbs).expect("carried");
                for i in 0..m {
                    assert_eq!(dots[i], full[i * m + i], "rowdot m {m} n {n} i {i}");
                }
            }
        }
    }

    /// A contraction past the old 512-block bound runs: Qwen2.5-7B's `ffn_down` contracts over
    /// `n_ff = 18944 > 512·32`, and the 7B fit refused at `Plane::build` before this bound moved.
    #[test]
    fn inner_past_512_blocks_products() {
        let n = 18_944usize;
        let a = vec![1i64 << 32; 2 * n];
        let b = vec![1i64 << 32; 3 * n];
        let want = i64::try_from(n).expect("n fits i64") << 32;
        let g = gemm_q32(&a, 2, &b, 3, n, 3).expect("wide contraction");
        assert!(g.iter().all(|&x| x == want), "gemm {g:?}");
        let s = syrk_q32(&a, 2, n, 3).expect("wide half Gram");
        assert!(s.iter().all(|&x| x == want), "syrk {s:?}");
        let d = rowdot_q32(&a, &a, 2, n, 3).expect("wide row dot");
        assert!(d.iter().all(|&x| x == want), "rowdot {d:?}");
    }

    /// (d) Shape refusal: a mismatched slice, a zero dimension, no limbs, an over-wide
    /// contraction.
    #[test]
    fn shape_refusals() {
        let a = vec![1i64 << 32; 4];
        let b = vec![1i64 << 32; 4];
        assert_eq!(gemm_q32(&a[..3], 2, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 3, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b[..3], 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 0, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b, 0, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b, 2, 2, 0), Err(LimbError::Shape));
        let wide = vec![1i64; MAX_INNER + 32];
        assert_eq!(
            gemm_q32(&wide, 1, &wide, 1, MAX_INNER + 32, 3),
            Err(LimbError::Shape)
        );
        assert_eq!(syrk_q32(&a, 3, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemv_q32(&a, &b, 2, 3, 3), Err(LimbError::Shape));
        assert_eq!(rowdot_q32(&a, &b, 3, 2, 3), Err(LimbError::Shape));
    }

    /// (e) Range refusal: words whose product the scale placement cannot carry.
    #[test]
    fn range_refusal_on_unplaceable_words() {
        let big = vec![i64::MAX; 32];
        assert_eq!(gemm_q32(&big, 1, &big, 1, 32, 3), Err(LimbError::Range));
        let mid = vec![1i64 << 62; 32];
        assert_eq!(gemm_q32(&mid, 1, &mid, 1, 32, 3), Err(LimbError::Range));
        assert_eq!(gemv_q32(&big, &big, 1, 32, 3), Err(LimbError::Range));
        assert_eq!(rowdot_q32(&big, &big, 1, 32, 3), Err(LimbError::Range));
        // A huge operand against a zero column is exactly zero: `Range` is about the product, not
        // the operand.
        let zero = vec![0i64; 32];
        assert_eq!(gemm_q32(&big, 1, &zero, 1, 32, 3), Ok(vec![0i64]));
    }

    /// (f) Integer words (multiples of `2^32`, `|v| ≤ 127`) are bit exact: every digit beyond the
    /// second is zero, so no kept pair is dropped and no residual remains.
    #[test]
    fn integer_operands_are_bit_exact() {
        let mut seed = 0xfeed_face_cafe_beefu64;
        for limbs in [3u32, 4] {
            for (m, p, n) in [
                (1usize, 1usize, 1usize),
                (7, 7, 32),
                (7, 7, 33),
                (64, 7, 896),
            ] {
                let mut a = Vec::with_capacity(m * n);
                let mut b = Vec::with_capacity(p * n);
                for _ in 0..m * n {
                    let v = (step(&mut seed) % 255).cast_signed() - 127;
                    a.push(v << 32);
                }
                for _ in 0..p * n {
                    let v = (step(&mut seed) % 255).cast_signed() - 127;
                    b.push(v << 32);
                }
                let got = gemm_q32(&a, m, &b, p, n, limbs).expect("carried");
                let want = reference(&a, m, &b, p, n);
                assert_eq!(got, want, "limbs {limbs} m {m} p {p} n {n}");
            }
        }
    }

    /// FNV-1a over the little-endian bytes of a word vector.
    fn digest(words: &[i64]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for w in words {
            for b in w.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// (g) The four entry points' outputs at three seeded shapes are pinned bit for bit: the
    /// digests below were taken from the body before the packing and accumulator were pooled, so a
    /// change to any limb, scale, pairing, grouping or rounding step shows up here.
    #[test]
    fn outputs_are_pinned_bit_identical() {
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let shapes = [(7usize, 5usize, 33usize), (64, 17, 896), (3, 3, 31)];
        // Pinned from the unoptimised body (commit b2d7587d) before the packing, accumulation and
        // `rowdot` batching were pooled; the values are the FNV-1a digests of the four outputs.
        let pinned: [[u64; 4]; 3] = [
            [
                0xbbb9_5d9a_a115_3673,
                0x14ee_6180_bd2a_afb6,
                0x3ff7_f88f_1cc7_a542,
                0xd65a_476e_1f9e_3e56,
            ],
            [
                0xd24f_fc48_7b6a_421e,
                0xcc66_5fd0_e2ac_4d68,
                0x47c7_120f_c529_242c,
                0xa412_c316_63ff_5d51,
            ],
            [
                0x8e0b_b844_9ee0_9b55,
                0xd9bc_4d2e_c2ab_3487,
                0xe98f_eb33_6c9f_c184,
                0xc942_6dd7_def7_050a,
            ],
        ];
        let mut all: Vec<[u64; 4]> = Vec::new();
        for &(m, p, n) in &shapes {
            let a = operand(&mut seed, m, n, 3);
            let b = operand(&mut seed, p, n, -2);
            let c = operand(&mut seed, m, n, 1);
            let out_prod = gemm_q32(&a, m, &b, p, n, 3).expect("carried");
            let out_syrk = syrk_q32(&c, m, n, 3).expect("carried");
            let out_vec = gemv_q32(&a[..n], &b, p, n, 3).expect("carried");
            let out_dots = rowdot_q32(&a, &c, m, n, 3).expect("carried");
            let got = [
                digest(&out_prod),
                digest(&out_syrk),
                digest(&out_vec),
                digest(&out_dots),
            ];
            println!(
                "shape {m} {p} {n}: gemm {:#018x} syrk {:#018x} gemv {:#018x} rowdot {:#018x}",
                got[0], got[1], got[2], got[3]
            );
            all.push(got);
        }
        for (t, got) in all.iter().enumerate() {
            assert_eq!(*got, pinned[t], "shape {:?} drifted", shapes[t]);
        }
    }

    /// (R8) Phase table: the cost of building the two limb planes, the weight payload, the
    /// activation columns, the kernel call and the `i128` accumulation, at the production shapes.
    #[test]
    #[ignore = "measurement: the phase table at the production shapes"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "m·p·n ≤ 2^32, exactly representable in f64, and only used to print ns/MAC"
    )]
    fn limb_phase_profile() {
        let shapes = [
            (512usize, 896usize, 896usize, 3u32),
            (512, 4864, 896, 3),
            (128, 128, 16, 8),
            (128, 128, 16, 3),
            (128, 16, 128, 8),
            (16, 16, 128, 8),
        ];
        let mut seed = 0xabcd_ef01_2345_6789u64;
        for (m, p, n, limbs) in shapes {
            let a = operand(&mut seed, m, n, 3);
            let b = operand(&mut seed, p, n, 3);
            // Warm the pool, the allocator and the page tables before the phase table.
            let warm = gemm_q32(&a, m, &b, p, n, limbs).expect("carried");
            drop(warm);
            let t = std::time::Instant::now();
            let pa = Plane::build(&a, m, n, limbs).expect("carried");
            let build_a = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let pb = Plane::build(&b, p, n, limbs).expect("carried");
            let build_b = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let pairs = kept_pairs(limbs);
            let place = Layout::placement(&pa, &pb, &pairs).expect("placed");
            let layout = Layout {
                a: &pa,
                b: &pb,
                pairs,
                place,
            };
            let place_s = t.elapsed().as_secs_f64();
            let group = (MAX_INNER / pa.padded).max(1).min(layout.pairs.len());
            // The same branch `product_rows` takes: one kernel call means the transpose is the
            // whole output stage, several calls mean the `i128` accumulator and its conversion.
            let single = layout.pairs.len() <= group;
            let mut out = vec![0i64; m * p];
            let mut acc = vec![0i128; m * p];
            let (mut payload, mut columns, mut kernel, mut accumulate) = (0.0, 0.0, 0.0, 0.0);
            let mut g0 = 0usize;
            while g0 < layout.pairs.len() {
                let g1 = (g0 + group).min(layout.pairs.len());
                let n_in = (g1 - g0) * pa.padded;
                let t = std::time::Instant::now();
                let raw = layout.weight_payload(0, m, g0, g1);
                payload += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let qs = layout.columns(0, p, g0, g1);
                columns += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let cols = gemm_q8_0(GGML_TYPE_Q8_0, &raw, m, n_in, &qs).expect("kernel");
                kernel += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                if single {
                    transpose_rows(&mut out, &cols, m, p);
                } else {
                    let mut j = 0usize;
                    while j < p {
                        let col = &cols[j];
                        let base = j * m;
                        let mut i = 0usize;
                        while i < m {
                            acc[base + i] += i128::from(col[i]);
                            i += 1;
                        }
                        j += 1;
                    }
                }
                accumulate += t.elapsed().as_secs_f64();
                g0 = g1;
            }
            if !single {
                let t = std::time::Instant::now();
                let mut i = 0usize;
                while i < m {
                    let mut j = 0usize;
                    while j < p {
                        out[i * p + j] = i64::try_from(acc[j * m + i]).expect("carried");
                        j += 1;
                    }
                    i += 1;
                }
                accumulate += t.elapsed().as_secs_f64();
            }
            let total = build_a + build_b + place_s + payload + columns + kernel + accumulate;
            let macs = (m * p * n) as f64;
            for (name, v) in [
                ("build_a", build_a),
                ("build_b", build_b),
                ("placement", place_s),
                ("payload", payload),
                ("columns", columns),
                ("kernel", kernel),
                ("accumulate", accumulate),
                ("total", total),
            ] {
                println!(
                    "m {m} p {p} n {n} limbs {limbs} {name:>10}: {:8.3} ms {:5.1}%  {:.4} ns/MAC",
                    v * 1e3,
                    100.0 * v / total,
                    v * 1e9 / macs
                );
            }
        }
    }

    /// (a) The production-shape receipt: [`gemm_q32`] at the K3 default 3 limbs against the exact
    /// `i128` body it replaces ([`crate::learn::solve::gemm::gemm_nt`], the fold's own product),
    /// five alternating paired runs at `m 512, p 896, n 896`, in ns per word-MAC.
    ///
    /// The exact body's inner kernel is a `4 × 4` `i128` tile that the compiler does not vectorise
    /// (`src/learn/solve/gemm.rs`), so both arms are measured on the same shape, the same words and
    /// the same host, and the per-pair ratio is the reported quantity.
    #[test]
    #[ignore = "measurement: 5 alternating runs against the exact solve::gemm body"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "m·p·n ≤ 2^32, exactly representable in f64, and only used to print ns/MAC"
    )]
    fn limb_vs_exact_body() {
        use crate::learn::solve::gemm::gemm_nt;
        const M: usize = 512;
        const P: usize = 896;
        const N: usize = 896;
        const LIMBS: u32 = 3;
        const RUNS: usize = 5;

        let mut seed = 0xabcd_ef01_2345_6789u64;
        let a = operand(&mut seed, M, N, 3);
        let b = operand(&mut seed, P, N, 3);
        let macs = (M * P * N) as f64;
        let want = gemm_nt(&a, &b, M, N, P).expect("exact body carried the operands");
        let got = gemm_q32(&a, M, &b, P, N, LIMBS).expect("limb product carried");
        let bound = bound_ulps(N, LIMBS, max_abs(&a), max_abs(&b));
        let err = max_err(&got, &want);
        assert!(err <= bound, "err {err} > bound {bound}");

        let (mut tl, mut te, mut pair) = (Vec::new(), Vec::new(), Vec::new());
        for run in 0..RUNS {
            let t = std::time::Instant::now();
            let k = gemm_q32(&a, M, &b, P, N, LIMBS).expect("limb product carried");
            let dl = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let e = gemm_nt(&a, &b, M, N, P).expect("exact body carried the operands");
            let de = t.elapsed().as_secs_f64();
            assert_eq!(k, got, "run {run}: limb output not reproducible");
            assert_eq!(e, want, "run {run}: exact output not reproducible");
            tl.push(dl);
            te.push(de);
            pair.push(dl / de);
        }
        let stat = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (m, lo, hi)
        };
        let (ml, lo_l, hi_l) = stat(&tl);
        let (me, lo_e, hi_e) = stat(&te);
        let (mp, lo_p, hi_p) = stat(&pair);
        println!("m {M} p {P} n {N} limbs {LIMBS}: {macs:.0} word-MACs, {RUNS} paired runs");
        println!(
            "limb {LIMBS}: mean {:.3} ms  spread [{:.3}, {:.3}] ms  {:.4} ns/word-MAC",
            ml * 1e3,
            lo_l * 1e3,
            hi_l * 1e3,
            ml * 1e9 / macs
        );
        println!(
            "exact i128: mean {:.3} ms  spread [{:.3}, {:.3}] ms  {:.4} ns/word-MAC",
            me * 1e3,
            lo_e * 1e3,
            hi_e * 1e3,
            me * 1e9 / macs
        );
        println!(
            "paired limb/exact {mp:.3}x (mean over {RUNS} pairs, spread [{lo_p:.3}, {hi_p:.3}])"
        );
        println!("error vs exact: {err} words, bound {bound}");
        assert!(
            mp < 1.0,
            "K5: the limb product is slower than the exact body: {mp:.3}x"
        );
    }

    /// (R8) Paired microbench at the Qwen2.5-0.5B shapes: 5 alternating runs of the limb product
    /// and of the scalar `i128` loop, mean and spread in ns per MAC.
    #[test]
    #[ignore = "measurement: 5 alternating runs at the production shapes"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "m·p·n ≤ 2^32, exactly representable in f64, and only used to print ns/MAC"
    )]
    fn limb_bench() {
        let shapes = [
            (512usize, 896usize, 896usize),
            (512, 4864, 896),
            (512, 896, 4864),
        ];
        let mut seed = 0xabcd_ef01_2345_6789u64;
        for (m, p, n) in shapes {
            let a = operand(&mut seed, m, n, 3);
            let b = operand(&mut seed, p, n, 3);
            // Warm both paths and validate the result against the oracle and the bound.
            let got = gemm_q32(&a, m, &b, p, n, 3).expect("carried");
            let want = reference(&a, m, &b, p, n);
            let bound = bound_ulps(n, 3, max_abs(&a), max_abs(&b));
            let err = max_err(&got, &want);
            assert!(err <= bound, "err {err} > bound {bound}");
            let macs = (m * p * n) as f64;
            let mut kern = Vec::new();
            let mut scal = Vec::new();
            for run in 0..5 {
                let t0 = std::time::Instant::now();
                let k = gemm_q32(&a, m, &b, p, n, 3).expect("carried");
                let dk = t0.elapsed();
                let t1 = std::time::Instant::now();
                let s = reference(&a, m, &b, p, n);
                let ds = t1.elapsed();
                assert_eq!(k, got, "run {run} kernel not reproducible");
                assert_eq!(s, want, "run {run} reference not reproducible");
                kern.push(dk.as_secs_f64());
                scal.push(ds.as_secs_f64());
                println!(
                    "{run}: limb {:>8.3} ms  i128 {:>8.3} ms",
                    dk.as_secs_f64() * 1e3,
                    ds.as_secs_f64() * 1e3
                );
            }
            for (name, v) in [("limb", &kern), ("i128", &scal)] {
                let mean = v.iter().sum::<f64>() / 5.0;
                let lo = v.iter().copied().fold(f64::MAX, f64::min);
                let hi = v.iter().copied().fold(0.0, f64::max);
                println!(
                    "m {m} p {p} n {n} {name}: mean {:.3} ms  spread [{:.3}, {:.3}] ms  \
                     {:.3} ns/MAC  bound {bound} observed {err}",
                    mean * 1e3,
                    lo * 1e3,
                    hi * 1e3,
                    mean * 1e9 / macs
                );
            }
        }
    }

    /// The `F16` projection on the packed tile is within [`bound_ulps`] of the scalar
    /// `matvec_q8act` oracle, and one flipped limb byte of that tile leaves the bound: the
    /// mutation control for the tile's digits.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::manual_is_multiple_of,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "one measured fixture with its oracle and mutation control; every cast is on a \
                  bounded test index or a masked half-word field"
    )]
    fn f16_tile_is_within_the_limb_bound_and_a_flipped_digit_breaks_it() {
        let (rows, n, k) = (64usize, 256usize, 5usize);
        // Half words with two exponent bands twelve powers apart, so a 21-bit block expansion
        // truncates the small band and the deviation is not identically zero.
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut raw: Vec<u8> = Vec::with_capacity(rows * n * 2);
        let mut r = 0usize;
        while r < rows * n {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let e: u16 = if st % 7 == 0 { 2 } else { 14 };
            let h = ((st >> 32) as u16 & 0x83ff) | (e << 10);
            raw.extend_from_slice(&h.to_le_bytes());
            r += 1;
        }
        let mut qs: Vec<QuantAct> = Vec::with_capacity(k);
        let mut c = 0usize;
        while c < k {
            let mut x: Vec<i64> = Vec::with_capacity(n);
            let mut j = 0usize;
            while j < n {
                let v = (((j * 37 + c * 11) % 257) as i64 - 128) << 20;
                x.push(v);
                j += 1;
            }
            qs.push(crate::engine::chart::quant_act(&x).expect("quant"));
            c += 1;
        }
        let tile = f16_tile(&raw, rows, n, 3).expect("tile");
        let got = f16_project(&tile, &qs).expect("projection");
        let mut max_dev: i128 = 0;
        let mut max_a: u64 = 0;
        let mut max_b: u64 = 0;
        let mut c = 0usize;
        while c < k {
            let want = matvec_q8act(crate::format::quant::GGML_TYPE_F16, &raw, rows, n, &qs[c])
                .expect("oracle");
            let mut i = 0usize;
            while i < rows {
                let d = i128::from(got[c][i]) - i128::from(want[i]);
                let d = d.abs();
                if d > max_dev {
                    max_dev = d;
                }
                i += 1;
            }
            let mut b = 0usize;
            while b < qs[c].scales().len() {
                let s = qs[c].scales()[b].unsigned_abs();
                let w = s.saturating_mul(127);
                if w > max_b {
                    max_b = w;
                }
                b += 1;
            }
            c += 1;
        }
        let mut r = 0usize;
        while r < rows * n {
            let h = u16::from_le_bytes([raw[2 * r], raw[2 * r + 1]]);
            let (m, kk) = super::super::pow2::f16_split(h).expect("finite");
            let w = (i64::from(m).unsigned_abs()) << kk;
            if w > max_a {
                max_a = w;
            }
            r += 1;
        }
        let bound = bound_ulps(n, 3, max_a, max_b);
        assert!(
            max_dev <= bound,
            "max |C_limb - C_scalar| {max_dev} > bound {bound}"
        );
        println!("f16 tile: max deviation {max_dev} words, bound {bound}");

        // Mutation control: replace the block's largest weight digit, in the cached payload,
        // with the opposite extreme. A check that cannot fail has no weight.
        let mut tile = tile;
        let block = &mut tile.groups[0].raw[2..Q8_0_BLOCK_BYTES];
        let mut at = 0usize;
        let mut best = 0i16;
        let mut i = 0usize;
        while i < ACT_QK {
            let d = i16::from(block[i].cast_signed()).abs();
            if d > best {
                best = d;
                at = i;
            }
            i += 1;
        }
        block[at] = if block[at].cast_signed() > 0 {
            0x80
        } else {
            0x7f
        };
        let mutated = f16_project(&tile, &qs).expect("mutated projection");
        let mut dev: i128 = 0;
        let mut c = 0usize;
        while c < k {
            let want = matvec_q8act(crate::format::quant::GGML_TYPE_F16, &raw, rows, n, &qs[c])
                .expect("oracle");
            let mut i = 0usize;
            while i < rows {
                let d = (i128::from(mutated[c][i]) - i128::from(want[i])).abs();
                if d > dev {
                    dev = d;
                }
                i += 1;
            }
            c += 1;
        }
        assert!(
            dev > bound,
            "a flipped limb byte stayed inside the bound: {dev} <= {bound}"
        );
        println!("f16 tile mutation: deviation {dev} > bound {bound}");
    }

    /// Measured report: the per-entry deviation and [`bound_ulps`] on qwen05b layer 0's seven
    /// projections with representative activation magnitudes. Run with
    /// `cargo test --release --lib -- --ignored --nocapture f16_qwen05b_layer0`.
    #[test]
    #[ignore = "measured report: reads models/gguf/qwen05b-f16.gguf"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "the printed relative error is a diagnostic ratio of two bounded word counts; the \
                  activation index is a bounded `usize` below 2049"
    )]
    fn f16_qwen05b_layer0_deviation_and_bound() {
        use crate::engine::loader::Engine;
        let path = crate::engine::testdata::model_path("models/gguf/qwen05b-f16.gguf");
        let Some(path) = path else {
            println!("qwen05b-f16.gguf absent; skipped");
            return;
        };
        let engine = Engine::open(&path).expect("open");
        let parts = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let k = 8usize;
        let entries = engine.table().entries();
        for name in parts {
            let Some(t) = entries.iter().find(|e| e.name == name) else {
                println!("{name}: absent");
                continue;
            };
            assert_eq!(t.ggml_type, crate::format::quant::GGML_TYPE_F16, "{name}");
            assert_eq!(t.dims.len(), 2, "{name}");
            let n_in = usize::try_from(t.dims[0]).expect("dim");
            let n_out = usize::try_from(t.dims[1]).expect("dim");
            let raw = engine
                .payload(t.byte_range.start, t.n_bytes.expect("bytes"))
                .expect("payload");
            let mut qs: Vec<QuantAct> = Vec::with_capacity(k);
            let mut c = 0usize;
            while c < k {
                // Representative hidden-state magnitude: |x| near 2^{34} in Q32.32, deterministic.
                let mut x: Vec<i64> = Vec::with_capacity(n_in);
                let mut j = 0usize;
                while j < n_in {
                    let v = (((j * 2_654_435_761usize + c * 40503) % 2049) as i64 - 1024) << 24;
                    x.push(v);
                    j += 1;
                }
                qs.push(crate::engine::chart::quant_act(&x).expect("quant"));
                c += 1;
            }
            let tile = f16_tile(raw, n_out, n_in, 3).expect("tile");
            let got = f16_project(&tile, &qs).expect("projection");
            let mut max_dev: i128 = 0;
            let mut max_a: u64 = 0;
            let mut max_b: u64 = 0;
            let mut c = 0usize;
            while c < k {
                let want = matvec_q8act(
                    crate::format::quant::GGML_TYPE_F16,
                    raw,
                    n_out,
                    n_in,
                    &qs[c],
                )
                .expect("oracle");
                let mut i = 0usize;
                while i < n_out {
                    let d = (i128::from(got[c][i]) - i128::from(want[i])).abs();
                    if d > max_dev {
                        max_dev = d;
                    }
                    i += 1;
                }
                let mut b = 0usize;
                while b < qs[c].scales().len() {
                    let w = qs[c].scales()[b].unsigned_abs().saturating_mul(127);
                    if w > max_b {
                        max_b = w;
                    }
                    b += 1;
                }
                c += 1;
            }
            let mut i = 0usize;
            while i < n_out * n_in {
                let h = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                let (m, kk) = super::super::pow2::f16_split(h).expect("finite");
                let w = (i64::from(m).unsigned_abs()) << kk;
                if w > max_a {
                    max_a = w;
                }
                i += 1;
            }
            let bound = bound_ulps(n_in, 3, max_a, max_b);
            println!(
                "{name}: n_out {n_out} n_in {n_in} max|C_limb-C_scalar| {max_dev} words, \
                 bound {bound}, relative {:.3e}",
                max_dev as f64 / bound as f64
            );
            assert!(max_dev <= bound, "{name}: {max_dev} > {bound}");
        }
    }
}
