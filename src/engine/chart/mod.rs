//! Integer chart dot: the int8-activation projection `matvec_q8act`, defined as
//! `matvec_fixed(W, quant(x))` bit for bit.
//!
//! `quant_act` is the llama.cpp `q8_0` activation shape with a power-of-two scale: each 32-wide
//! block `b` of a Q32.32 activation vector gets the smallest scale `S_b = 2^e_b` with
//! `127·S_b ≥ absmax_b` and codes `xq_j` in `[−127, 127]`, so the dequantized activation
//! `x̂_j = S_b·xq_j` is exact in Q32.32 and `|x_j − x̂_j| ≤ S_b/2`. A power of two turns every
//! block's activation scale into a shift in the kernels (`kern::gemm`); it is at most twice
//! `ceil(absmax_b / 127)`, one bit of activation precision.
//!
//! For a weight group `w_j = W_g·q_j − M_g` the row dot factors exactly:
//! `Σ_j w_j·x̂_j = Σ_b S_b·Σ_{g⊆b} (W_g·Σ_j q_j·xq_j − M_g·Σ_j xq_j)`. The inner sums are exact
//! `i32`; the group and block products are checked `i128`; the kernel ends with the single
//! `round_shift_32` epilogue and the same `i64` range check as `matvec_fixed`. Wherever a bound
//! cannot rule out an overflow the whole call falls back to `matvec_fixed` on the dequantized
//! activation, so no refusal can differ.
//!
//! The inner group dot dispatches at runtime to [`x86`] (AVX2), ported from
//! `fold/bedrock/src/serve/model/rowdot.rs`; the portable lane loop here is the fallback and the
//! test oracle. The AVX2 intrinsics are not extracted; their equality with the portable twin is
//! `tested`.
use crate::format::quant::{
    self, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, Q5_0_BLOCK_BYTES, Q5_0_QK, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, Q8_0_QK, QK_K,
};

use super::EngineError;
use crate::engine::num::value::round_shift_32;
use crate::engine::ops::fused::{
    abs_i128, finish_row, fold_group, fused_type, group_take, max_abs_code, q2_k_fields,
    q3_k_fields, q4_k_fields, q5_0_fields, q6_k_fields, q8_0_fields,
};
use crate::engine::ops::matvec::{matvec_fixed, MatvecError};

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "x86_64")]
use x86::{q4_k_scale_min_at, scale_word};

mod portable;

#[cfg(target_arch = "x86_64")]
mod kcols;

#[cfg(target_arch = "x86_64")]
pub(crate) use kcols::{
    lane_spread_limit, pack_cols, q4_k_row_lanes_avx2, q6_k_row_lanes_avx2, PackedCols, Q4KRow,
    Q6KRow,
};
#[cfg(all(test, target_arch = "x86_64"))]
use kcols::{q4_k_row_kcols_avx2, q6_k_row_kcols_avx2};

use portable::{
    block_xsums, f16_to_fixed_fast, group_dot, group_xsum, portable_q4_k_window_dots,
    portable_q5_0_dot, portable_u8_i8_dot,
};
#[cfg(test)]
use portable::{
    portable_i8_dot, q4_k_row_portable, q5_0_row_portable, q6_k_row_portable, q8_0_row_portable,
};

pub(crate) mod ffn;
pub(crate) mod pool;
pub(crate) mod task;

/// Lanes of one activation quantization block.
const ACT_QK: usize = 32;

/// The integer quantization of a Q32.32 activation vector: one scale per 32-wide block and one
/// int8 code per element.
///
/// Produced by [`quant_act`] and consumed by [`matvec_q8act`]. The fields are private: the type is
/// only meaningful with the invariant `codes.len() == n_in`, `scales.len() == n_in.div_ceil(32)`
/// and `|codes[j]| ≤ 127`, which [`quant_act`] establishes.
/// The `Default` value is empty and holds the invariant only for `n_in = 0`; a scratch holder sizes it
/// before use.
#[derive(Default)]
pub(crate) struct QuantAct {
    /// Per activation block `b`, the scale `S_b`.
    scales: Vec<i64>,
    /// Per activation element `j`, the code `xq_j`.
    codes: Vec<i8>,
}

/// The outcome of [`QuantAct::magnitude`]: the exact Q32.32 magnitude, a product that leaves `i64`
/// (so [`QuantAct::dequant`] is `None`), or an `i128` total that leaves range (`dequant` still
/// succeeds, so the caller falls back to the reference).
pub(crate) enum ActMagnitude {
    /// `Σ_j |S_b·xq_j|`, every product in `i64` and the total in `i128`.
    Exact(i128),
    /// Some `S_b·xq_j` leaves `i64`; the same condition as `dequant() == None`.
    NoQ32,
    /// The `i128` total of the exact products leaves range.
    TotalOverflow,
}

impl QuantAct {
    /// Per activation block `b`, the scale `S_b` (read access for the `kern` kernels).
    pub(crate) fn scales(&self) -> &[i64] {
        &self.scales
    }

    /// Per activation element `j`, the code `xq_j` (read access for the `kern` kernels).
    pub(crate) fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// The activation length `n_in` the quantizer was built for (`codes.len()`).
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    pub(crate) fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether [`QuantAct::dequant`] would return a value: every `S_b·xq_j` fits `i64`, checked
    /// without materializing the dequantized vector.
    ///
    /// A block is decided by the one product `S_b·127`: every code obeys `|xq_j| ≤ 127`, so
    /// `|S_b·xq_j| ≤ |S_b·127|`, and a fitting `S_b·127` (never `i64::MIN`, since `127 ∤ 2^63`)
    /// leaves every product of the block in `i64`. Only a block whose `S_b·127` leaves `i64` runs
    /// the per-element multiply, which decides exactly as the previous per-element loop.
    ///
    /// # Complexity
    ///
    /// `O(n)` time in the activation length `n` worst case (every block's bound leaves `i64`) and
    /// `O(1)` extra space; one checked `i64` multiply per block `b ∈ 0..codes.len().div_ceil(32)`
    /// plus one per element of a block whose bound left `i64`, no allocation.
    pub(crate) fn dequantizes(&self) -> bool {
        let n_blocks = self.codes.len().div_ceil(ACT_QK);
        let mut ok = true;
        let mut block = 0usize;
        while block < n_blocks && ok {
            if block >= self.scales.len() {
                ok = false;
            } else {
                let scale = self.scales[block];
                if scale.checked_mul(127).is_none() {
                    let base = block * ACT_QK;
                    let end = (base + ACT_QK).min(self.codes.len());
                    let mut lane = base;
                    while lane < end && ok {
                        ok = scale.checked_mul(i64::from(self.codes[lane])).is_some();
                        lane += 1;
                    }
                }
            }
            block += 1;
        }
        ok
    }

    /// `Σ_j |S_b·xq_j|` in `i128` without materializing `x̂`; the same checked products
    /// [`QuantAct::dequant`] uses, so the `NoQ32` case is exactly `dequant() == None`.
    ///
    /// # Complexity
    ///
    /// `O(n)` time and `O(1)` extra space in the activation length `n`; one checked `i64` multiply
    /// and one checked `i128` add per element, no allocation.
    pub(crate) fn magnitude(&self) -> ActMagnitude {
        let mut acc: i128 = 0;
        let mut idx = 0usize;
        while idx < self.codes.len() {
            let Some(&scale) = self.scales.get(idx / ACT_QK) else {
                return ActMagnitude::NoQ32;
            };
            let Some(value) = scale.checked_mul(i64::from(self.codes[idx])) else {
                return ActMagnitude::NoQ32;
            };
            let mag = i128::from(value).abs();
            match acc.checked_add(mag) {
                Some(total) => acc = total,
                None => return ActMagnitude::TotalOverflow,
            }
            idx += 1;
        }
        ActMagnitude::Exact(acc)
    }

    /// The dequantized activation `x̂_j = S_b·xq_j` in Q32.32, or `None` when any product leaves
    /// `i64`.
    ///
    /// # Complexity
    ///
    /// `O(n)` time and `O(n)` space in the activation length `n`; one checked `i64` multiply per
    /// element.
    pub(crate) fn dequant(&self) -> Option<Vec<i64>> {
        let mut out: Vec<i64> = Vec::with_capacity(self.codes.len());
        let mut idx = 0usize;
        let mut ok = true;
        while idx < self.codes.len() && ok {
            let value = if idx / ACT_QK < self.scales.len() {
                self.scales[idx / ACT_QK].checked_mul(i64::from(self.codes[idx]))
            } else {
                None
            };
            match value {
                Some(value) => out.push(value),
                None => ok = false,
            }
            idx += 1;
        }
        if ok {
            Some(out)
        } else {
            None
        }
    }

    /// Rebuild a `QuantAct` from the concatenated `scales` and `codes` of a 32-aligned partition's
    /// [`quant_act`] results, for a caller that quantized the slices locally.
    pub(crate) fn from_parts(scales: Vec<i64>, codes: Vec<i8>) -> Self {
        Self { scales, codes }
    }
}

/// The quotient `q` and remainder `r` of `m / s` for the code rounding. For a power-of-two `s`
/// (`pow2`, with `shift = s.trailing_zeros()` and `mask = s.wrapping_sub(1)`) these are the
/// shift and the mask instead of a hardware `divq`/`remq`; otherwise the checked `u64` division,
/// whose `s == 0` failure is `0` as in the previous body. Exact for `s = 2^e`, `e ≥ 0`:
/// `m >> e = m / s` and `m & (2^e − 1) = m % s` in Euclidean division, and `m ≤ 2^63` so the
/// shift cannot saturate.
///
/// # Complexity
///
/// `O(1)`.
#[inline]
fn act_quot_rem(m: u64, s: u64, pow2: bool, shift: u32, mask: u64) -> (u64, u64) {
    if pow2 {
        (m >> shift, m & mask)
    } else {
        (m.checked_div(s).unwrap_or(0), m.checked_rem(s).unwrap_or(0))
    }
}

/// Quantize a Q32.32 activation vector to one scale per 32-wide block and int8 codes.
///
/// Per block `b` (the last block may be ragged): `absmax_b = max_j |x_j|`,
/// `S_b = 2^⌈log₂ ⌈absmax_b / 127⌉⌉`, the smallest power of two with `127·S_b ≥ absmax_b`
/// (`0` when `absmax_b = 0`), and
/// `xq_j = 0` when `S_b = 0`, else `sign(x_j)·((2|x_j| + S_b) / (2·S_b))` with integer division
/// (round to nearest, ties away from zero). The result satisfies `xq_j ∈ [−127, 127]`.
///
/// Returns `None` exactly when [`QuantAct::dequant`] would return `None`; the final
/// [`QuantAct::dequantizes`] check is that contract, not a second rounding.
///
/// # Complexity
///
/// `O(n)` time and `O(n)` space in the activation length `n`; two passes over each block.
///
/// The loops are single `while` loops carrying their counters and the current block's `take`, so
/// the body can be extracted later without an early `return` inside a loop.
#[allow(
    clippy::cast_possible_truncation,
    reason = "`S_b ≤ i64::MAX` is checked before the cast and `|xq_j| ≤ 127`, \
              each checked in `i128` immediately before the cast; `TryFrom` emits an unmodeled \
              external the pinned fail-closed extraction audit rejects"
)]
pub(crate) fn quant_act(x: &[i64]) -> Option<QuantAct> {
    let blocks = x.len().div_ceil(ACT_QK);
    let mut scales: Vec<i64> = Vec::with_capacity(blocks);
    let mut codes: Vec<i8> = Vec::with_capacity(x.len());
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        let base = block * ACT_QK;
        let take = ACT_QK.min(x.len() - base);
        let mut absmax: u64 = 0;
        let mut lane = 0usize;
        while lane < take {
            let mag = x[base + lane].unsigned_abs();
            if mag > absmax {
                absmax = mag;
            }
            lane += 1;
        }
        // `absmax ≤ 2^63` (a `|i64|`), so `absmax + 126` fits `u64` and the `u64` divisions equal
        // the `i128` ones; the `≤ i64::MAX` guard below keeps the scale in `i64`.
        // The smallest power of two at least `ceil(absmax / 127)`; `0` stays `0` (the extracted
        // `kern::pow2::act_scale`).
        let scale_wide: u64 = super::kern::pow2::act_scale(absmax).unwrap_or(u64::MAX);
        if scale_wide > i64::MAX.unsigned_abs() {
            ok = false;
        } else {
            let scale = scale_wide.cast_signed();
            scales.push(scale);
            let s = scale.unsigned_abs();
            // `act_scale` returns only `0` or a power of two, so for `s != 0` `m / s` and `m % s`
            // are a shift and a mask, hoisted per block; the guard at `scale_wide` above already
            // rejected the `2^63` non-power value `act_scale` refuses with. `pow2` is false only
            // for `s == 0`, whose branch below still emits code `0`.
            let pow2 = s.is_power_of_two();
            let shift = s.trailing_zeros();
            let mask = s.wrapping_sub(1);
            let mut lane = 0usize;
            while lane < take && ok {
                let value = x[base + lane];
                let code: i8 = if s == 0 {
                    0
                } else {
                    // `floor((2|v| + S) / (2S))` by `u64` quotient/remainder: with
                    // `|v| = q·S + r`, `0 ≤ r < S`, the result is `q + [2r ≥ S]`; `2r < 2S ≤
                    // 2·2^57` and `|v| ≤ 2^63`, so no `u64` step leaves range and the `i128`
                    // division is not needed. For the power-of-two `S` `act_scale` returns, the
                    // quotient and remainder are exact shifts of the same integer values.
                    let m = value.unsigned_abs();
                    let (quotient, remainder) = act_quot_rem(m, s, pow2, shift, mask);
                    let magnitude = quotient + u64::from(2 * remainder >= s);
                    let signed = if value < 0 {
                        -i128::from(magnitude)
                    } else {
                        i128::from(magnitude)
                    };
                    if (-127..=127).contains(&signed) {
                        signed as i8
                    } else {
                        ok = false;
                        0
                    }
                };
                codes.push(code);
                lane += 1;
            }
        }
        block += 1;
    }
    if !ok {
        return None;
    }
    let quant = QuantAct { scales, codes };
    if !quant.dequantizes() {
        return None;
    }
    Some(quant)
}

/// The largest finite `f16` magnitude in Q32.32: `65504 · 2^32`.
const F16_MAX_FIXED: i128 = 65504i128 << 32;

/// Whether a fast per-row kernel can run with no per-group checked `i128` bound.
///
/// Every finite weight scale obeys `|d| ≤ 65504` (the `f16` magnitude bound) and every decoded
/// weight obeys `|w| ≤ (65504·2^32)·top`; with the call's largest activation block scale
/// `S_max = max_b |S_b|` and activation codes `|xq| ≤ 127`, the reference's largest partial sum —
/// and every running partial sum of it — obeys
/// `|Σ_i w_i·x_i| ≤ n_in · (65504·2^32) · top · 127 · S_max`.
/// `top` is the family's largest `|w| / (65504·2^32)`: `Q8_0` `128`, `Q5_0` `16`, `Q4_K`
/// `63·15 + 63 = 1008`, `Q6_K` `128·32 = 4096`. When
/// `n_in · (65504·2^32) · top · 127 · S_max` is at most `i128::MAX` the reference's checked `i128`
/// accumulator cannot overflow, so the fused row reproduces its exact value with plain
/// non-wrapping `i128`; the per-element `i64` product `|w_i| ≤ (65504·2^32)·top` is checked
/// separately per family and holds for all four.
///
/// With `narrow` (the pure `Q8_0`/`Q5_0` rows) the extra condition
/// `S_max · (32·top·127) ≤ i64::MAX` proves the per-block activation-side product `S_b·I_b` fits
/// `i64`, so the row folds each block with one `i64` product and one `i64`×`i64`→`i128` widening
/// product instead of two generic `i128` products. When any condition fails the caller keeps the
/// checked per-group kernel and its fallback.
///
/// # Complexity
///
/// `O(n_in/32)` time, `O(1)` extra space.
pub(crate) fn q8act_fast_bound_ok(n_in: usize, top: i128, q: &QuantAct, narrow: bool) -> bool {
    let mut s_max: i128 = 0;
    let mut b = 0usize;
    while b < q.scales.len() {
        let a = i128::from(q.scales[b]).abs();
        if a > s_max {
            s_max = a;
        }
        b += 1;
    }
    if narrow && s_max.saturating_mul(32 * top * 127) > i128::from(i64::MAX) {
        return false;
    }
    let bound = (n_in as i128)
        .saturating_mul(F16_MAX_FIXED)
        .saturating_mul(top)
        .saturating_mul(127)
        .saturating_mul(s_max);
    // Strict: a saturated product equals `i128::MAX`, which cannot be distinguished from an exact
    // bound at the maximum, so that case takes the checked fallback.
    bound < i128::MAX
}

/// AVX2 `Q8_0` fast row: the whole block loop is one `#[target_feature(enable = "avx2")]` body, so
/// the code dot inlines and no per-block bounds check, call or `[i8; 32]` copy remains.
///
/// # Safety
///
/// AVX2 must be present; `row.len() == blocks·Q8_0_BLOCK_BYTES`, `codes.len() == blocks·ACT_QK`
/// and `scales.len() == blocks`, as established by the caller's length checks.
///
/// # Complexity
///
/// `O(n_in)` time, `O(1)` extra space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) unsafe fn q8_0_row_avx2(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    scales: &[i64],
) -> Option<i128> {
    unsafe { q8act_accumulate_row_avx2::<Q8_0Decoder>(row, blocks * ACT_QK, codes, scales, &[]) }
}

/// [`q8_0_row_avx2`] for `acc.len()` activation columns at once: each four-block weight group's
/// scales are decoded once, and its 128 code bytes stay in L1 while every column's dots are formed
/// by the same `dot32x4_ptr`/`dot32_ptr` cores. Column `c` accumulates exactly the `i128` sum
/// [`q8_0_row_avx2`] forms for `qs[c]`, so each column equals the single-column row bit for bit.
///
/// Returns `None` for a non-finite weight scale; the caller then routes the call to its exact
/// fallback.
///
/// # Safety
///
/// AVX2 must be present; `row.len() == blocks·Q8_0_BLOCK_BYTES`; every `qs[c]` holds
/// `blocks·ACT_QK` codes and `blocks` scales and has passed [`q8act_fast_bound_ok`] (narrow), so
/// no product or partial sum leaves `i64`/`i128`; `acc.len() ≤ qs.len()`.
///
/// # Complexity
///
/// `O(n_in·k)` time, `O(1)` extra space; the row's bytes are read from memory once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q8_0_row_kcols_avx2(
    row: &[u8],
    blocks: usize,
    qs: &[QuantAct],
    acc: &mut [i128],
) -> Option<()> {
    let k = acc.len();
    // One feature decision per row: the column loop below must not re-read the atomic.
    let vnni = x86::avxvnni();
    let mut blk = 0usize;
    while blk + 4 <= blocks {
        let off = blk * Q8_0_BLOCK_BYTES;
        // SAFETY: `(blk + 3)·Q8_0_BLOCK_BYTES + 2 <= row.len()` by the caller's geometry.
        let ds = unsafe {
            [
                f16_to_fixed_fast(scale_word(row.as_ptr(), off))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + Q8_0_BLOCK_BYTES))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + 2 * Q8_0_BLOCK_BYTES))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + 3 * Q8_0_BLOCK_BYTES))?,
            ]
        };
        // The four code vectors are decoded once and reused by every column.
        let codes = unsafe { x86::q8_0_codes4(row.as_ptr().add(off + 2), Q8_0_BLOCK_BYTES) };
        let mut c = 0usize;
        while c < k {
            let q = &qs[c];
            // SAFETY: four blocks of lanes are in range by the caller's geometry.
            let dots =
                unsafe { x86::dot32x4_codes(&codes, q.codes.as_ptr().add(blk * ACT_QK), vnni) };
            let mut i = 0usize;
            while i < 4 {
                // SAFETY: `q.scales.len() == blocks >= blk + 4`.
                let scaled_dot = unsafe { *q.scales.get_unchecked(blk + i) } * i64::from(dots[i]);
                acc[c] += i128::from(ds[i]) * i128::from(scaled_dot);
                i += 1;
            }
            c += 1;
        }
        blk += 4;
    }
    while blk < blocks {
        let off = blk * Q8_0_BLOCK_BYTES;
        // SAFETY: `off + 2 <= row.len()`.
        let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
        let mut c = 0usize;
        while c < k {
            let q = &qs[c];
            // SAFETY: the block's 32 code bytes and 32 lanes are in range.
            let dot = unsafe {
                x86::dot32_ptr_vnni(
                    row.as_ptr().add(off + 2),
                    q.codes.as_ptr().add(blk * ACT_QK),
                    vnni,
                )
            };
            // SAFETY: `q.scales.len() == blocks > blk`.
            let scaled_dot = unsafe { *q.scales.get_unchecked(blk) } * i64::from(dot);
            acc[c] += i128::from(d) * i128::from(scaled_dot);
            c += 1;
        }
        blk += 1;
    }
    Some(())
}

/// The `Q5_0` twin of [`q8_0_row_kcols_avx2`]: one weight row's 5-bit blocks decoded once and
/// dotted against all `k` activation columns, with the row's `f16` block scale and each column's
/// activation scale applied in `i128`.
///
/// # Safety
///
/// The caller established AVX2, `row.len() >= blocks·Q5_0_BLOCK_BYTES`, and every column's
/// geometry and bound exactly as for [`q8_0_row_kcols_avx2`].
///
/// # Complexity
///
/// `O(blocks·k)` exact 32-lane dots, `O(k)` accumulator words.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn q5_0_row_kcols_avx2(
    row: &[u8],
    blocks: usize,
    qs: &[QuantAct],
    acc: &mut [i128],
) -> Option<()> {
    let k = acc.len();
    // One feature decision per row: the column loop below must not re-read the atomic.
    let vnni = x86::avxvnni();
    let mut blk = 0usize;
    while blk + 4 <= blocks {
        let off = blk * Q5_0_BLOCK_BYTES;
        // SAFETY: four block scales are in range by the caller's geometry.
        let ds = unsafe {
            [
                f16_to_fixed_fast(scale_word(row.as_ptr(), off))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + Q5_0_BLOCK_BYTES))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + 2 * Q5_0_BLOCK_BYTES))?,
                f16_to_fixed_fast(scale_word(row.as_ptr(), off + 3 * Q5_0_BLOCK_BYTES))?,
            ]
        };
        // The four 5-bit blocks are decoded once and reused by every column.
        let codes = unsafe {
            x86::q5_0_codes4(
                row.as_ptr().add(off + 2),
                row.as_ptr().add(off + 6),
                Q5_0_BLOCK_BYTES,
            )
        };
        let mut c = 0usize;
        while c < k {
            let q = &qs[c];
            // SAFETY: four blocks of lanes are in range by the caller's geometry.
            let dots =
                unsafe { x86::dot32x4_codes(&codes, q.codes.as_ptr().add(blk * ACT_QK), vnni) };
            let mut i = 0usize;
            while i < 4 {
                // SAFETY: `q.scales.len() == blocks >= blk + 4`.
                let scaled = unsafe { *q.scales.get_unchecked(blk + i) } * i64::from(dots[i]);
                acc[c] += i128::from(ds[i]) * i128::from(scaled);
                i += 1;
            }
            c += 1;
        }
        blk += 4;
    }
    while blk < blocks {
        let off = blk * Q5_0_BLOCK_BYTES;
        // SAFETY: `off + 6 <= row.len()`.
        let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
        // SAFETY: the block's `qh`/`qs` bytes are in range.
        let codes =
            unsafe { x86::q5_0_codes(row.as_ptr().add(off + 2), row.as_ptr().add(off + 6)) };
        let mut c = 0usize;
        while c < k {
            let q = &qs[c];
            // SAFETY: the block's 32 lanes are in range.
            let dot = unsafe { x86::dot32_codes(codes, q.codes.as_ptr().add(blk * ACT_QK), vnni) };
            // SAFETY: `q.scales.len() == blocks > blk`.
            let scaled = unsafe { *q.scales.get_unchecked(blk) } * i64::from(dot);
            acc[c] += i128::from(d) * i128::from(scaled);
            c += 1;
        }
        blk += 1;
    }
    Some(())
}

/// Receiver of the 32-lane groups a [`FastBlockDecoder`] step decodes: `emit(offset, w_scale, w_min,
/// dot, xsum)` once per group. A named trait, not a closure, so the extraction models the call.
pub(crate) trait GroupSink {
    fn emit(&mut self, i: usize, w_scale: i64, w_min: i64, dot: i64, xsum: i32);
}

/// The row accumulator: `acc += act_scales[base + i] * (w_scale * dot - w_min * xsum)` per group.
struct RowAcc<'a> {
    act_scales: &'a [i64],
    base: usize,
    acc: i128,
}

impl GroupSink for RowAcc<'_> {
    fn emit(&mut self, i: usize, w_scale: i64, w_min: i64, dot: i64, xsum: i32) {
        let act_scale = self.act_scales[self.base + i];
        let inner = i128::from(w_scale) * i128::from(dot) - i128::from(w_min) * i128::from(xsum);
        self.acc += i128::from(act_scale) * inner;
    }
}

/// Per-family block decoder for the unified fast row accumulator [`q8act_accumulate_row`].
pub(crate) trait FastBlockDecoder {
    /// Decode the next step of 32-lane groups using AVX2.
    ///
    /// Invokes `emit(offset, w_scale, w_min, dot, xsum)` for each group in the step, where:
    /// - `offset`: the relative 32-lane group index within this step (`0..groups_decoded`)
    /// - `w_scale`: weight scale (e.g. `d` or `d * scale_code`)
    /// - `w_min`: weight minimum offset (0 for offset-free families, `dmin * min_code` for affine `Q4_K`)
    /// - `dot`: inner product `dot32` between weight codes and activation codes
    /// - `xsum`: activation sum `Σ xq` for the group (0 for offset-free families, `xsums[g]` for `Q4_K`)
    ///
    /// # Safety
    ///
    /// The caller guarantees AVX2 is available, `g < total_groups`, and slice bounds hold.
    #[cfg(target_arch = "x86_64")]
    unsafe fn decode_step_avx2<S: GroupSink>(
        row: &[u8],
        g: usize,
        total_groups: usize,
        codes: &[i8],
        xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize>;

    /// Decode the next step of 32-lane groups using portable scalar logic.
    ///
    /// Invokes `emit(offset, w_scale, w_min, dot, xsum)` for each group in the step.
    fn decode_step_portable<S: GroupSink>(
        row: &[u8],
        g: usize,
        total_groups: usize,
        codes: &[i8],
        xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize>;
}

/// Zero-sized block decoder for `Q8_0`: offset-free, one group per block.
pub(crate) struct Q8_0Decoder;

impl FastBlockDecoder for Q8_0Decoder {
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    unsafe fn decode_step_avx2<S: GroupSink>(
        row: &[u8],
        g: usize,
        total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let off = g * Q8_0_BLOCK_BYTES;
        if g + 4 <= total_groups {
            // SAFETY: `(g + 3)·Q8_0_BLOCK_BYTES + 2 <= row.len()` by the caller's length check.
            let d0 = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            let d1 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + Q8_0_BLOCK_BYTES) })?;
            let d2 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 2 * Q8_0_BLOCK_BYTES) })?;
            let d3 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 3 * Q8_0_BLOCK_BYTES) })?;
            // SAFETY: four blocks are in range; `dot32x4_ptr` reads 128 code bytes and 128 lanes.
            let dots = unsafe {
                x86::dot32x4_ptr(
                    row.as_ptr().add(off + 2),
                    Q8_0_BLOCK_BYTES,
                    codes.as_ptr().add(g * ACT_QK),
                )
            };
            emit.emit(0, d0, 0, i64::from(dots[0]), 0);
            emit.emit(1, d1, 0, i64::from(dots[1]), 0);
            emit.emit(2, d2, 0, i64::from(dots[2]), 0);
            emit.emit(3, d3, 0, i64::from(dots[3]), 0);
            Some(4)
        } else {
            // SAFETY: `off + 2 <= row.len()` by the caller's length check.
            let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            // SAFETY: `off + 2 + 32 <= row.len()` and `g·32 + 32 <= codes.len()`.
            let dot = unsafe {
                x86::dot32_ptr(row.as_ptr().add(off + 2), codes.as_ptr().add(g * ACT_QK))
            };
            emit.emit(0, d, 0, i64::from(dot), 0);
            Some(1)
        }
    }

    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    fn decode_step_portable<S: GroupSink>(
        row: &[u8],
        g: usize,
        _total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let off = g * Q8_0_BLOCK_BYTES;
        let bits = u16::from_le_bytes([row[off], row[off + 1]]);
        let d = f16_to_fixed_fast(bits)?;
        let code_window = &row[off + 2..off + 2 + Q8_0_QK];
        let xbase = g * ACT_QK;
        let dot = portable_u8_i8_dot(code_window, &codes[xbase..xbase + ACT_QK]);
        emit.emit(0, d, 0, i64::from(dot), 0);
        Some(1)
    }
}

/// Zero-sized block decoder for `Q5_0`: offset-free in vector domain, one group per block.
pub(crate) struct Q5_0Decoder;

impl FastBlockDecoder for Q5_0Decoder {
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    unsafe fn decode_step_avx2<S: GroupSink>(
        row: &[u8],
        g: usize,
        total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let off = g * Q5_0_BLOCK_BYTES;
        if g + 4 <= total_groups {
            // SAFETY: `(g + 3)·Q5_0_BLOCK_BYTES + 2 <= row.len()` by the caller's length check.
            let d0 = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            let d1 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + Q5_0_BLOCK_BYTES) })?;
            let d2 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 2 * Q5_0_BLOCK_BYTES) })?;
            let d3 =
                f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 3 * Q5_0_BLOCK_BYTES) })?;
            // SAFETY: four blocks are in range; `q5_0_dot32x4_ptr` reads 4 `qh` bytes and 16 `qs`
            // bytes per block and 32 activation lanes per block.
            let dots = unsafe {
                x86::q5_0_dot32x4_ptr(
                    row.as_ptr().add(off + 2),
                    row.as_ptr().add(off + 6),
                    Q5_0_BLOCK_BYTES,
                    codes.as_ptr().add(g * ACT_QK),
                )
            };
            emit.emit(0, d0, 0, i64::from(dots[0]), 0);
            emit.emit(1, d1, 0, i64::from(dots[1]), 0);
            emit.emit(2, d2, 0, i64::from(dots[2]), 0);
            emit.emit(3, d3, 0, i64::from(dots[3]), 0);
            Some(4)
        } else {
            // SAFETY: `off + 2 <= row.len()` by the caller's length check.
            let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            // SAFETY: `off + 6 + 16 <= row.len()` and `g·32 + 32 <= codes.len()` by caller checks.
            let dot = unsafe {
                x86::q5_0_dot32_ptr(
                    row.as_ptr().add(off + 2),
                    row.as_ptr().add(off + 6),
                    codes.as_ptr().add(g * ACT_QK),
                )
            };
            emit.emit(0, d, 0, i64::from(dot), 0);
            Some(1)
        }
    }

    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    fn decode_step_portable<S: GroupSink>(
        row: &[u8],
        g: usize,
        _total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let off = g * Q5_0_BLOCK_BYTES;
        let bits = u16::from_le_bytes([row[off], row[off + 1]]);
        let d = f16_to_fixed_fast(bits)?;
        let qh = &row[off + 2..off + 6];
        let qs = &row[off + 6..off + 6 + Q5_0_QK / 2];
        let xbase = g * ACT_QK;
        let dot = portable_q5_0_dot(qh, qs, &codes[xbase..xbase + ACT_QK]);
        emit.emit(0, d, 0, i64::from(dot), 0);
        Some(1)
    }
}

/// Zero-sized block decoder for `Q4_K`: affine groups `w = d·sc·q − dmin·m`.
#[allow(
    non_camel_case_types,
    reason = "GGML quantization family name convention"
)]
pub(crate) struct Q4_KDecoder;

impl FastBlockDecoder for Q4_KDecoder {
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    unsafe fn decode_step_avx2<S: GroupSink>(
        row: &[u8],
        g: usize,
        total_groups: usize,
        codes: &[i8],
        xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let pp = g / 2;
        let b = pp / 4;
        let p = pp % 4;
        let off = b * Q4_K_BLOCK_BYTES;
        if p == 0 && g + 8 <= total_groups {
            // SAFETY: `off + 4 <= row.len()` by the caller's length check.
            let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            let dmin = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 2) })?;
            let mut sc = [0u8; Q4_K_SCALE_SIZE];
            // SAFETY: `off + 16 <= row.len()` by the caller's length check.
            sc.copy_from_slice(unsafe { row.get_unchecked(off + 4..off + 16) });
            // SAFETY: a full block is present; eight activation groups start at `g·32`.
            let dots = unsafe {
                x86::q4_k_block_dots(row.as_ptr().add(off + 16), codes.as_ptr().add(g * ACT_QK))
            };
            let mut c = 0usize;
            while c < 4 {
                let (sl, ml) = q4_k_scale_min_at(&sc, 2 * c);
                let (sh, mh) = q4_k_scale_min_at(&sc, 2 * c + 1);
                let slope_lo = d * i64::from(sl);
                let intercept_lo = dmin * i64::from(ml);
                let slope_hi = d * i64::from(sh);
                let intercept_hi = dmin * i64::from(mh);
                let x_lo = *xsums.get_unchecked(g + 2 * c);
                let x_hi = *xsums.get_unchecked(g + 2 * c + 1);
                emit.emit(2 * c, slope_lo, intercept_lo, i64::from(dots[2 * c]), x_lo);
                emit.emit(
                    2 * c + 1,
                    slope_hi,
                    intercept_hi,
                    i64::from(dots[2 * c + 1]),
                    x_hi,
                );
                c += 1;
            }
            Some(8)
        } else {
            // SAFETY: `off + 4 <= row.len()`.
            let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off) })?;
            let dmin = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + 2) })?;
            let mut sc = [0u8; Q4_K_SCALE_SIZE];
            // SAFETY: `off + 16 <= row.len()`.
            sc.copy_from_slice(unsafe { row.get_unchecked(off + 4..off + 16) });
            let (sl, ml) = q4_k_scale_min_at(&sc, 2 * p);
            let (sh, mh) = q4_k_scale_min_at(&sc, 2 * p + 1);
            let slope_lo = d * i64::from(sl);
            let intercept_lo = dmin * i64::from(ml);
            let slope_hi = d * i64::from(sh);
            let intercept_hi = dmin * i64::from(mh);
            // SAFETY: `off + 16 + 32p + 32 <= row.len()` and `(g + 1)·32 + 32 <= codes.len()`.
            let (dot_lo, dot_hi) = unsafe {
                x86::q4_k_window_dot_ptr(
                    row.as_ptr().add(off + 16 + 32 * p),
                    codes.as_ptr().add(g * ACT_QK),
                    codes.as_ptr().add((g + 1) * ACT_QK),
                )
            };
            let x_lo = *xsums.get_unchecked(g);
            let x_hi = *xsums.get_unchecked(g + 1);
            emit.emit(0, slope_lo, intercept_lo, i64::from(dot_lo), x_lo);
            emit.emit(1, slope_hi, intercept_hi, i64::from(dot_hi), x_hi);
            Some(2)
        }
    }

    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    fn decode_step_portable<S: GroupSink>(
        row: &[u8],
        g: usize,
        _total_groups: usize,
        codes: &[i8],
        xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let pp = g / 2;
        let b = pp / 4;
        let p = pp % 4;
        let off = b * Q4_K_BLOCK_BYTES;
        let d = f16_to_fixed_fast(u16::from_le_bytes([row[off], row[off + 1]]))?;
        let dmin = f16_to_fixed_fast(u16::from_le_bytes([row[off + 2], row[off + 3]]))?;
        let mut sc = [0u8; Q4_K_SCALE_SIZE];
        sc.copy_from_slice(&row[off + 4..off + 16]);
        let (sl, ml) = quant::q4_k_scale_min(&sc, 2 * p);
        let (sh, mh) = quant::q4_k_scale_min(&sc, 2 * p + 1);
        let slope_lo = d * i64::from(sl);
        let intercept_lo = dmin * i64::from(ml);
        let slope_hi = d * i64::from(sh);
        let intercept_hi = dmin * i64::from(mh);
        let qs = &row[off + 16 + 32 * p..off + 16 + 32 * p + 32];
        let (dot_lo, dot_hi) = portable_q4_k_window_dots(
            qs,
            &codes[g * ACT_QK..(g + 1) * ACT_QK],
            &codes[(g + 1) * ACT_QK..(g + 2) * ACT_QK],
        );
        emit.emit(0, slope_lo, intercept_lo, i64::from(dot_lo), xsums[g]);
        emit.emit(1, slope_hi, intercept_hi, i64::from(dot_hi), xsums[g + 1]);
        Some(2)
    }
}

/// Zero-sized block decoder for `Q6_K`: offset-free `w = d·sc·q`, eight 32-lane groups per block.
#[allow(
    non_camel_case_types,
    reason = "GGML quantization family name convention"
)]
pub(crate) struct Q6_KDecoder;

impl FastBlockDecoder for Q6_KDecoder {
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    unsafe fn decode_step_avx2<S: GroupSink>(
        row: &[u8],
        g: usize,
        _total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let b = g / 8;
        let off = b * Q6_K_BLOCK_BYTES;
        // SAFETY: `off + Q6_K_BLOCK_BYTES <= row.len()` by the caller's length check.
        let d = f16_to_fixed_fast(unsafe { scale_word(row.as_ptr(), off + Q6_K_BLOCK_BYTES - 2) })?;
        let base = row.as_ptr().add(off);
        let mut n = 0usize;
        while n < 2 {
            // SAFETY: `base + 64n + 64 <= row end`, `base + 128 + 32n + 32 <= row end` and
            // `codes + 256b + 128n + 128 <= codes end` by the caller's length checks.
            let dots = unsafe {
                x86::q6_k_half_dots(
                    base.add(64 * n),
                    base.add(128 + 32 * n),
                    codes.as_ptr().add(b * QK_K + 128 * n),
                )
            };
            let mut c = 0usize;
            while c < 4 {
                let sclo = i64::from(unsafe { *base.add(192 + 8 * n + 2 * c) }.cast_signed());
                let schi = i64::from(unsafe { *base.add(192 + 8 * n + 2 * c + 1) }.cast_signed());
                let group_dot = sclo * i64::from(dots[2 * c]) + schi * i64::from(dots[2 * c + 1]);
                emit.emit(4 * n + c, d, 0, group_dot, 0);
                c += 1;
            }
            n += 1;
        }
        Some(8)
    }

    #[inline]
    #[allow(
        clippy::many_single_char_names,
        reason = "quantization math uses standard GGML/llama.cpp single-letter names"
    )]
    fn decode_step_portable<S: GroupSink>(
        row: &[u8],
        g: usize,
        _total_groups: usize,
        codes: &[i8],
        _xsums: &[i32],
        emit: &mut S,
    ) -> Option<usize> {
        let b = g / 8;
        let off = b * Q6_K_BLOCK_BYTES;
        let d = f16_to_fixed_fast(u16::from_le_bytes([
            row[off + Q6_K_BLOCK_BYTES - 2],
            row[off + Q6_K_BLOCK_BYTES - 1],
        ]))?;
        let blk = &row[off..off + Q6_K_BLOCK_BYTES];
        let block_codes = quant::q6_k_codes(blk).ok()?;
        let mut c = 0usize;
        while c < 8 {
            let group0 = 2 * c;
            let group1 = 2 * c + 1;
            let sclo = i64::from(blk[192 + group0].cast_signed());
            let schi = i64::from(blk[192 + group1].cast_signed());
            let mut dot0 = 0i64;
            let mut l = 0usize;
            while l < 16 {
                dot0 += i64::from(block_codes[16 * group0 + l])
                    * i64::from(codes[(b * 16 + group0) * 16 + l]);
                l += 1;
            }
            let mut dot1 = 0i64;
            let mut l = 0usize;
            while l < 16 {
                dot1 += i64::from(block_codes[16 * group1 + l])
                    * i64::from(codes[(b * 16 + group1) * 16 + l]);
                l += 1;
            }
            let group_dot = sclo * dot0 + schi * dot1;
            emit.emit(c, d, 0, group_dot, 0);
            c += 1;
        }
        Some(8)
    }
}

/// One accumulator kernel that walks 32-lane groups in the existing per-group order and computes
/// each group's contribution as:
///
///   `acc += act_scale * (w_scale * dot - w_min * xsum)`
///
/// where `act_scale` comes from `q.scales`, and `w_scale`/`w_min`/`dot`/`xsum` come from a
/// per-family [`FastBlockDecoder`].
pub(crate) fn q8act_accumulate_row<D: FastBlockDecoder>(
    row: &[u8],
    cols: usize,
    q: &QuantAct,
    xsums: &[i32],
    avx: bool,
) -> Option<i128> {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present, caller established slice extents.
        return unsafe { q8act_accumulate_row_avx2::<D>(row, cols, &q.codes, &q.scales, xsums) };
    }
    let _ = avx;
    q8act_accumulate_row_portable::<D>(row, cols, &q.codes, &q.scales, xsums)
}

/// AVX2 accumulator loop over 32-lane groups in per-group order.
///
/// # Safety
///
/// AVX2 must be present; slice extents must match `cols` geometry.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q8act_accumulate_row_avx2<D: FastBlockDecoder>(
    row: &[u8],
    cols: usize,
    codes: &[i8],
    act_scales: &[i64],
    xsums: &[i32],
) -> Option<i128> {
    let total_groups = cols / ACT_QK;
    let mut sink = RowAcc {
        act_scales,
        base: 0,
        acc: 0,
    };
    let mut g = 0usize;
    while g < total_groups {
        sink.base = g;
        let step = unsafe { D::decode_step_avx2(row, g, total_groups, codes, xsums, &mut sink)? };
        g += step;
    }
    Some(sink.acc)
}

/// Portable accumulator loop over 32-lane groups in per-group order.
pub(crate) fn q8act_accumulate_row_portable<D: FastBlockDecoder>(
    row: &[u8],
    cols: usize,
    codes: &[i8],
    act_scales: &[i64],
    xsums: &[i32],
) -> Option<i128> {
    let total_groups = cols / ACT_QK;
    let mut sink = RowAcc {
        act_scales,
        base: 0,
        acc: 0,
    };
    let mut g = 0usize;
    while g < total_groups {
        sink.base = g;
        let step = D::decode_step_portable(row, g, total_groups, codes, xsums, &mut sink)?;
        g += step;
    }
    Some(sink.acc)
}

/// AVX2 `Q5_0` fast row.
///
/// # Safety
///
/// AVX2 must be present; caller established slice extents.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) unsafe fn q5_0_row_avx2(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    scales: &[i64],
) -> Option<i128> {
    unsafe { q8act_accumulate_row_avx2::<Q5_0Decoder>(row, blocks * ACT_QK, codes, scales, &[]) }
}

/// AVX2 `Q4_K` fast row.
///
/// # Safety
///
/// AVX2 must be present; caller established slice extents.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) unsafe fn q4_k_row_avx2(
    row: &[u8],
    pairs: usize,
    codes: &[i8],
    act_scales: &[i64],
    xsums: &[i32],
) -> Option<i128> {
    unsafe { q8act_accumulate_row_avx2::<Q4_KDecoder>(row, pairs * 64, codes, act_scales, xsums) }
}

/// AVX2 `Q6_K` fast row.
///
/// # Safety
///
/// AVX2 must be present; caller established slice extents.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) unsafe fn q6_k_row_avx2(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    act_scales: &[i64],
) -> Option<i128> {
    unsafe { q8act_accumulate_row_avx2::<Q6_KDecoder>(row, blocks * QK_K, codes, act_scales, &[]) }
}

/// `Q8_0` fast row dispatching to the unified accumulator.
fn q8act_q8_0_row_fast(row: &[u8], blocks: usize, q: &QuantAct, avx: bool) -> Option<i128> {
    q8act_accumulate_row::<Q8_0Decoder>(row, blocks * ACT_QK, q, &[], avx)
}

/// `Q5_0` fast row dispatching to the unified accumulator.
fn q8act_q5_0_row_fast(row: &[u8], blocks: usize, q: &QuantAct, avx: bool) -> Option<i128> {
    q8act_accumulate_row::<Q5_0Decoder>(row, blocks * ACT_QK, q, &[], avx)
}

/// `Q4_K` fast row dispatching to the unified accumulator.
fn q8act_q4_k_row_fast(
    row: &[u8],
    pairs: usize,
    q: &QuantAct,
    xsums: &[i32],
    avx: bool,
) -> Option<i128> {
    q8act_accumulate_row::<Q4_KDecoder>(row, pairs * 64, q, xsums, avx)
}

/// `Q6_K` fast row dispatching to the unified accumulator.
fn q8act_q6_k_row_fast(row: &[u8], blocks: usize, q: &QuantAct, avx: bool) -> Option<i128> {
    q8act_accumulate_row::<Q6_KDecoder>(row, blocks * QK_K, q, &[], avx)
}

/// One weight row's exact pre-rounding `i128` accumulator over `cols` columns on the family's
/// fully-fast path, with the slice-local quantization `q` (`q.len() == cols`). The row begins at a
/// family block boundary and `cols` is a multiple of the fast alignment (32 for `Q8_0`/`Q5_0`, 64
/// for `Q4_K`, 256 for `Q6_K`); `None` for another family or a non-finite weight scale, which
/// routes the caller to the whole-row kernel. `O(cols)` time, `O(1)` extra space.
pub(crate) fn q8act_row_i128_fast(
    ggml_type: u32,
    row: &[u8],
    cols: usize,
    q: &QuantAct,
    plan: &Q8ActFast,
) -> Option<i128> {
    match ggml_type {
        quant::GGML_TYPE_Q8_0 => q8act_q8_0_row_fast(row, cols / ACT_QK, q, plan.fast_avx),
        quant::GGML_TYPE_Q5_0 => q8act_q5_0_row_fast(row, cols / ACT_QK, q, plan.fast_avx),
        quant::GGML_TYPE_Q4_K => q8act_q4_k_row_fast(row, cols / 64, q, &plan.xsums, plan.fast_avx),
        quant::GGML_TYPE_Q6_K => q8act_q6_k_row_fast(row, cols / QK_K, q, plan.fast_avx),
        _ => None,
    }
}

/// Whether the executing host exposes AVX2.
#[inline]
fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        x86::avx2()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// One pure group `w = slope·q`: the contribution `S·slope·Σ q·xq` and the weight magnitude bound
/// `|slope|·max|q|`, or `None` on a checked overflow.
///
/// # Complexity
///
/// `O(take + count)` time, `O(1)` extra space.
#[allow(
    clippy::too_many_arguments,
    reason = "the block scale, the slope, the code window and the matching activation window are \
              the group's full input; bundling them would hide which window is read"
)]
fn q8act_pure(
    s: i64,
    slope: i64,
    codes: &[i8],
    cbase: usize,
    xq: &[i8],
    xbase: usize,
    take: usize,
    count: usize,
) -> Option<(i128, i128)> {
    let dot = group_dot(codes, cbase, xq, xbase, take);
    let factor = abs_i128(slope).checked_mul(i128::from(max_abs_code(codes, cbase, count)))?;
    let inner = i128::from(slope).checked_mul(i128::from(dot))?;
    let contribution = i128::from(s).checked_mul(inner)?;
    Some((contribution, factor))
}

/// One affine group `w = slope·q − intercept`: the contribution `S·(slope·Σ q·xq − intercept·Σ xq)`
/// and the weight magnitude bound `|slope|·max|q| + |intercept|`, or `None` on a checked overflow.
///
/// # Complexity
///
/// `O(take + count)` time, `O(1)` extra space.
#[allow(
    clippy::too_many_arguments,
    reason = "the block scale, the slope/intercept, the code window and the activation window are \
              the group's full input; bundling them would hide which window is read"
)]
fn q8act_affine(
    s: i64,
    slope: i64,
    intercept: i64,
    codes: &[i8],
    cbase: usize,
    xq: &[i8],
    xbase: usize,
    take: usize,
    count: usize,
) -> Option<(i128, i128)> {
    let dot = group_dot(codes, cbase, xq, xbase, take);
    q8act_affine_with_dot(
        s, slope, intercept, dot, codes, cbase, count, xq, xbase, take,
    )
}

/// [`q8act_affine`] with the group dot already computed (the `Q4_K` AVX2 window returns both
/// planes' dots at once).
///
/// # Complexity
///
/// `O(take + count)` time, `O(1)` extra space.
#[allow(
    clippy::too_many_arguments,
    reason = "the block scale, the slope/intercept, the precomputed dot and the two windows are \
              the group's full input; bundling them would hide which window is read"
)]
fn q8act_affine_with_dot(
    s: i64,
    slope: i64,
    intercept: i64,
    dot: i32,
    codes: &[i8],
    cbase: usize,
    count: usize,
    xq: &[i8],
    xbase: usize,
    take: usize,
) -> Option<(i128, i128)> {
    let xsum = group_xsum(xq, xbase, take);
    let factor = abs_i128(slope)
        .checked_mul(i128::from(max_abs_code(codes, cbase, count)))?
        .checked_add(abs_i128(intercept))?;
    let a = i128::from(slope).checked_mul(i128::from(dot))?;
    let b = i128::from(intercept).checked_mul(i128::from(xsum))?;
    let inner = a.checked_sub(b)?;
    let contribution = i128::from(s).checked_mul(inner)?;
    Some((contribution, factor))
}

/// `Q8_0` chart row: `w = d·q`, one 32-lane pure group per block.
fn q8act_q8_0_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(Q8_0_QK);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q8_0_fields(row, block) {
            Some((scale, codes)) => {
                let xbase = block * Q8_0_QK;
                let take = Q8_0_QK.min(n_in - xbase);
                match q.scales.get(xbase / ACT_QK) {
                    Some(&s) => {
                        if take > 0 {
                            fold_group(
                                &mut total,
                                &mut wmax,
                                &mut ok,
                                q8act_pure(s, scale, &codes, 0, &q.codes, xbase, take, Q8_0_QK),
                            );
                        }
                    }
                    None => ok = false,
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// `Q5_0` chart row: `w = d·q`, one 32-lane pure group per block.
fn q8act_q5_0_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(Q5_0_QK);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q5_0_fields(row, block) {
            Some((scale, codes)) => {
                let xbase = block * Q5_0_QK;
                let take = Q5_0_QK.min(n_in - xbase);
                match q.scales.get(xbase / ACT_QK) {
                    Some(&s) => {
                        if take > 0 {
                            fold_group(
                                &mut total,
                                &mut wmax,
                                &mut ok,
                                q8act_pure(s, scale, &codes, 0, &q.codes, xbase, take, Q5_0_QK),
                            );
                        }
                    }
                    None => ok = false,
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// `Q2_K` chart row: `w = d·sc·q − dmin·m`, sixteen 16-lane affine groups per block.
fn q8act_q2_k_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q2_k_fields(row, block) {
            Some((d, dmin, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let (scale_code, min_code) = quant::q2_k_scale_min(scales[group]);
                    let slope = if let Some(v) = d.checked_mul(i64::from(scale_code)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    let intercept = if let Some(v) = dmin.checked_mul(i64::from(min_code)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        let take_g = group_take(take, gbase, 16);
                        if take_g > 0 {
                            match q.scales.get((xblock + gbase) / ACT_QK) {
                                Some(&s) => fold_group(
                                    &mut total,
                                    &mut wmax,
                                    &mut ok,
                                    q8act_affine(
                                        s,
                                        slope,
                                        intercept,
                                        &codes,
                                        gbase,
                                        &q.codes,
                                        xblock + gbase,
                                        take_g,
                                        16,
                                    ),
                                ),
                                None => ok = false,
                            }
                        }
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// `Q3_K` chart row: `w = d·(sc − 32)·q`, sixteen 16-lane pure groups per block.
fn q8act_q3_k_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q3_k_fields(row, block) {
            Some((d, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = if let Some(v) = d.checked_mul(i64::from(scales[group]) - 32) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        let take_g = group_take(take, gbase, 16);
                        if take_g > 0 {
                            match q.scales.get((xblock + gbase) / ACT_QK) {
                                Some(&s) => fold_group(
                                    &mut total,
                                    &mut wmax,
                                    &mut ok,
                                    q8act_pure(
                                        s,
                                        slope,
                                        &codes,
                                        gbase,
                                        &q.codes,
                                        xblock + gbase,
                                        take_g,
                                        16,
                                    ),
                                ),
                                None => ok = false,
                            }
                        }
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// The exact `Σ code·xq` of a `Q4_K` `qs` window's low and high 32-lane groups: AVX2 on the packed
/// nibble planes when both groups are full, the portable lane loop otherwise.
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
#[allow(
    clippy::too_many_arguments,
    reason = "the row, the block byte base, the pair index, the decoded codes and the two \
              activation windows are the pair's full input; bundling them would hide which \
              window is read"
)]
fn q4_k_pair_dots(
    row: &[u8],
    blk_off: usize,
    pair: usize,
    codes: &[i8],
    xq: &[i8],
    xblock: usize,
    take_lo: usize,
    take_hi: usize,
) -> (i32, i32) {
    let lo_base = 64 * pair;
    let hi_base = lo_base + 32;
    #[cfg(target_arch = "x86_64")]
    {
        if take_lo == 32 && take_hi == 32 && avx2() {
            let qs_off = blk_off + 4 + Q4_K_SCALE_SIZE + 32 * pair;
            if let Some(qs) = row.get(qs_off..qs_off + 32) {
                // SAFETY: AVX2 is present, `qs` holds the 32 packed bytes, and the two activation
                // windows hold the 32 lanes the core loads (`take_lo == take_hi == 32`).
                return unsafe {
                    x86::q4_k_window_dot(
                        qs,
                        &xq[xblock + lo_base..][..32],
                        &xq[xblock + hi_base..][..32],
                    )
                };
            }
        }
    }
    (
        group_dot(codes, lo_base, xq, xblock + lo_base, take_lo),
        group_dot(codes, hi_base, xq, xblock + hi_base, take_hi),
    )
}

/// `Q4_K` chart row: `w = d·sc·q − dmin·m`, four 64-lane pairs of two 32-lane affine groups.
#[allow(
    clippy::too_many_lines,
    reason = "the pair loop mirrors `fused_q4_k_row`'s group structure and carries the two plane \
              scales per pair; splitting it would duplicate the checked slope/intercept derivation"
)]
fn q8act_q4_k_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q4_k_fields(row, block) {
            Some((scale, min_scale, sc, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let blk_off = block * Q4_K_BLOCK_BYTES;
                let mut pair = 0usize;
                while pair < 4 && ok {
                    let (scale_lo, min_lo) = quant::q4_k_scale_min(&sc, 2 * pair);
                    let (scale_hi, min_hi) = quant::q4_k_scale_min(&sc, 2 * pair + 1);
                    let slope_lo = if let Some(v) = scale.checked_mul(i64::from(scale_lo)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    let intercept_lo = if let Some(v) = min_scale.checked_mul(i64::from(min_lo)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    let slope_hi = if let Some(v) = scale.checked_mul(i64::from(scale_hi)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    let intercept_hi = if let Some(v) = min_scale.checked_mul(i64::from(min_hi)) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let lo_base = 64 * pair;
                        let hi_base = lo_base + 32;
                        let take_lo = group_take(take, lo_base, 32);
                        let take_hi = group_take(take, hi_base, 32);
                        // The low and high 32-lane groups sit in two consecutive activation blocks.
                        let s_lo = if take_lo > 0 {
                            q.scales.get((xblock + lo_base) / ACT_QK).copied()
                        } else {
                            Some(0)
                        };
                        let s_hi = if take_hi > 0 {
                            q.scales.get((xblock + hi_base) / ACT_QK).copied()
                        } else {
                            Some(0)
                        };
                        match (s_lo, s_hi) {
                            (Some(s_lo), Some(s_hi)) => {
                                let (dot_lo, dot_hi) = q4_k_pair_dots(
                                    row, blk_off, pair, &codes, &q.codes, xblock, take_lo, take_hi,
                                );
                                if take_lo > 0 {
                                    fold_group(
                                        &mut total,
                                        &mut wmax,
                                        &mut ok,
                                        q8act_affine_with_dot(
                                            s_lo,
                                            slope_lo,
                                            intercept_lo,
                                            dot_lo,
                                            &codes,
                                            lo_base,
                                            32,
                                            &q.codes,
                                            xblock + lo_base,
                                            take_lo,
                                        ),
                                    );
                                }
                                if take_hi > 0 {
                                    fold_group(
                                        &mut total,
                                        &mut wmax,
                                        &mut ok,
                                        q8act_affine_with_dot(
                                            s_hi,
                                            slope_hi,
                                            intercept_hi,
                                            dot_hi,
                                            &codes,
                                            hi_base,
                                            32,
                                            &q.codes,
                                            xblock + hi_base,
                                            take_hi,
                                        ),
                                    );
                                }
                            }
                            _ => ok = false,
                        }
                    }
                    pair += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// `Q6_K` chart row: `w = d·sc·q`, sixteen 16-lane pure groups per block.
fn q8act_q6_k_row(row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut ok = true;
    let mut block = 0usize;
    while block < blocks && ok {
        match q6_k_fields(row, block) {
            Some((d, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = if let Some(v) = d.checked_mul(i64::from(scales[group])) {
                        v
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        let take_g = group_take(take, gbase, 16);
                        if take_g > 0 {
                            match q.scales.get((xblock + gbase) / ACT_QK) {
                                Some(&s) => fold_group(
                                    &mut total,
                                    &mut wmax,
                                    &mut ok,
                                    q8act_pure(
                                        s,
                                        slope,
                                        &codes,
                                        gbase,
                                        &q.codes,
                                        xblock + gbase,
                                        take_g,
                                        16,
                                    ),
                                ),
                                None => ok = false,
                            }
                        }
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, xabs)
}

/// The chart row for one weight row of `ggml_type`, or `None` when the fast path cannot prove
/// equality with [`matvec_fixed`] and the caller must fall back.
fn q8act_row(ggml_type: u32, row: &[u8], n_in: usize, q: &QuantAct, xabs: i128) -> Option<i128> {
    match ggml_type {
        quant::GGML_TYPE_Q8_0 => q8act_q8_0_row(row, n_in, q, xabs),
        quant::GGML_TYPE_Q5_0 => q8act_q5_0_row(row, n_in, q, xabs),
        quant::GGML_TYPE_Q2_K => q8act_q2_k_row(row, n_in, q, xabs),
        quant::GGML_TYPE_Q3_K => q8act_q3_k_row(row, n_in, q, xabs),
        quant::GGML_TYPE_Q4_K => q8act_q4_k_row(row, n_in, q, xabs),
        quant::GGML_TYPE_Q6_K => q8act_q6_k_row(row, n_in, q, xabs),
        _ => None,
    }
}

/// The row-independent part of an exact [`matvec_q8act`] call: the family's fast-row decision and
/// the activation sums those rows read. Every row chunk of one call shares one value, so a chunked
/// caller computes it once instead of once per task.
pub(crate) struct Q8ActFast {
    /// Whether the family's in-place fast row kernel is admitted for this width and bound.
    pub(crate) fast: bool,
    /// Whether AVX2 is present, resolved once per call.
    fast_avx: bool,
    /// `Σ_j |x̂_j|`, read only by the checked per-group kernel (`!fast`).
    xabs: i128,
    /// Per-block activation data read only by the fast rows, computed once per call: raw `Σ xq`
    /// per 32 lanes for the affine `Q4_K` row, one `[c(2j), c(2j+1), c(2j), c(2j+1)]` vector of
    /// `c = 16·Σ xq` per block pair for the `Q5_0` row's offset-decode correction, empty for the
    /// families that read neither.
    xsums: Vec<i32>,
}

/// The row-independent part of a [`matvec_q8act`] call: the exact row kernels with their shared
/// plan, or the whole call as the reference `matvec_fixed` on the dequantized activation.
pub(crate) enum Q8ActPlan {
    /// The exact row kernels apply; the plan is shared by every row chunk.
    Fused(Q8ActFast),
    /// A non-fused family or an `i128` magnitude overflow: the call is `matvec_fixed` on `x̂`.
    Reference,
}

/// Compute the row-independent part of [`matvec_q8act`] once for a call.
///
/// `NoQ32` and a non-fused family both take the reference, which materializes `x̂`; the magnitude
/// and the call-level fast bound are otherwise computed from `scales`/`codes` with no allocation.
///
/// # Complexity
///
/// `O(n_in)` time for the magnitude (and, for `Q4_K`, the `Σ xq` sums), `O(1)` for the bound.
pub(crate) fn q8act_plan(ggml_type: u32, n_in: usize, q: &QuantAct) -> Q8ActPlan {
    let xabs = match q.magnitude() {
        ActMagnitude::Exact(v) if fused_type(ggml_type) => v,
        // `NoQ32` is `dequant() == None`, and a non-fused family is exactly `matvec_fixed` on `x̂`;
        // the unreachable `i128` total overflow takes the same reference path.
        _ => return Q8ActPlan::Reference,
    };
    let fast_avx = avx2();
    let fast = match ggml_type {
        quant::GGML_TYPE_Q8_0 if n_in.is_multiple_of(ACT_QK) => {
            q8act_fast_bound_ok(n_in, 128, q, true)
        }
        quant::GGML_TYPE_Q5_0 if n_in.is_multiple_of(ACT_QK) => {
            q8act_fast_bound_ok(n_in, 16, q, true)
        }
        quant::GGML_TYPE_Q4_K if n_in.is_multiple_of(64) => {
            q8act_fast_bound_ok(n_in, 1008, q, false)
        }
        quant::GGML_TYPE_Q6_K if n_in.is_multiple_of(QK_K) => {
            q8act_fast_bound_ok(n_in, 4096, q, false)
        }
        _ => false,
    };
    // `Q4_K` is affine, so every row needs the per-32-lane `Σ xq`; the `Q5_0` fast rows decode in
    // the offset representation `c + 16` and subtract `16·Σ xq` per block as their correction. Both
    // are call-global, so compute the one the family reads once for the call.
    let xsums: Vec<i32> = if fast && ggml_type == quant::GGML_TYPE_Q4_K {
        block_xsums(&q.codes)
    } else if fast && ggml_type == quant::GGML_TYPE_Q5_0 {
        // The `Q5_0` rows correct in the vector domain, one `__m128i` per block pair
        // `[c(2j), c(2j+1), c(2j), c(2j+1)]` with `c = 16·Σxq`, matching the `reduce4` lane order
        // `[dot(row0,2j), dot(row0,2j+1), dot(row1,2j), dot(row1,2j+1)]`.
        let sums = block_xsums(&q.codes);
        let mut out: Vec<i32> = Vec::with_capacity(2 * sums.len());
        let mut j = 0usize;
        while j < sums.len() {
            let c0 = sums[j] * 16;
            let c1 = sums.get(j + 1).copied().unwrap_or(0) * 16;
            out.extend_from_slice(&[c0, c1, c0, c1]);
            j += 2;
        }
        out
    } else {
        Vec::new()
    };
    Q8ActPlan::Fused(Q8ActFast {
        fast,
        fast_avx,
        xabs,
        xsums,
    })
}

/// Two tiled rows' exact pre-rounding accumulators, or the one refusal the single-row path also
/// reports: a non-finite weight scale in one of the two rows.
#[cfg(target_arch = "x86_64")]
enum Tiled2 {
    /// Row `r`'s per-block fold, before the caller's single `round_shift_32`.
    Accs([i128; 2]),
    /// A non-finite weight scale: the caller falls the whole payload back to the reference.
    NonFinite,
}

/// `round_shift_32(acc)` when it fits `i64`, `None` otherwise.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check, as in `matvec_q8act`"
)]
fn round_to_i64(acc: i128) -> Option<i64> {
    let rounded = round_shift_32(acc);
    if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
        None
    } else {
        Some(rounded as i64)
    }
}

/// The exact row kernels over `n_out` whole rows of `raw` with a plan shared by the call's chunks.
///
/// `row_bytes` is the row stride and `need` the `n_out·row_bytes` extent the caller validated. The
/// fast rows read the block bytes in place; a row that cannot prove the reference stays in range
/// falls the whole payload's rows back to `matvec_fixed` on the dequantized activation. The `Q8_0`
/// / `Q5_0` / `Q4_K` / `Q6_K` selection and the single `round_shift_32` epilogue are
/// [`matvec_q8act`]'s; this is the row loop alone.
///
/// # Errors
///
/// The same [`MatvecError`] values [`matvec_q8act`] returns: a short payload, an output that leaves
/// `i64`, or the fallback's own refusal.
///
/// # Complexity
///
/// `O(n_out·n_in)` time in exact integer multiply-adds; `O(n_out)` output.
#[allow(
    clippy::too_many_arguments,
    reason = "the call's geometry (`ggml_type`, `raw`, `n_out`, `n_in`, `row_bytes`, `need`) and the \
              shared plan are the kernel's full input; bundling them would hide which extent is \
              read"
)]
pub(crate) fn q8act_rows(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    row_bytes: usize,
    need: usize,
    q: &QuantAct,
    plan: &Q8ActFast,
) -> Result<Vec<i64>, MatvecError> {
    let mut out: Vec<i64> = Vec::with_capacity(n_out);
    let mut o = 0usize;
    let mut ok = true;
    let mut fallback = false;
    while o < n_out && ok && !fallback {
        // Cross-row tiling: two rows share each weight block's activation code vector and its
        // per-block `S_b`. Every row keeps its own fold order and its own single rounding, so the
        // pushed values are the single-row kernels' bit for bit.
        #[cfg(target_arch = "x86_64")]
        if plan.fast && plan.fast_avx && o + 2 <= n_out && o * row_bytes <= raw.len() {
            let base = o * row_bytes;
            if raw.len() - base >= 2 * row_bytes {
                if let Some(tiled) = x86::tiled2(
                    ggml_type,
                    &raw[base..],
                    row_bytes,
                    n_in,
                    &q.codes,
                    &q.scales,
                    &plan.xsums,
                ) {
                    match tiled {
                        Tiled2::Accs(accs) => {
                            match round_to_i64(accs[0]) {
                                Some(v) => out.push(v),
                                None => ok = false,
                            }
                            match round_to_i64(accs[1]) {
                                Some(v) => out.push(v),
                                None => ok = false,
                            }
                            o += 2;
                        }
                        Tiled2::NonFinite => fallback = true,
                    }
                    continue;
                }
            }
        }
        let start = o * row_bytes;
        let row = raw
            .get(start..start + row_bytes)
            .ok_or(MatvecError::ShortPayload {
                need,
                have: raw.len(),
            })?;
        let row_acc = if plan.fast {
            q8act_row_i128_fast(ggml_type, row, n_in, q, plan)
        } else {
            q8act_row(ggml_type, row, n_in, q, plan.xabs)
        };
        match row_acc {
            Some(acc) => match round_to_i64(acc) {
                Some(v) => out.push(v),
                None => ok = false,
            },
            None => fallback = true,
        }
        o += 1;
    }
    if fallback {
        // The fast row could not prove the reference stays in range: the dequantized activation is
        // materialized here, on the fallback only.
        let Some(xhat) = q.dequant() else {
            return Err(MatvecError::Dequant(EngineError::FixedOverflow));
        };
        return matvec_fixed(ggml_type, raw, n_out, n_in, &xhat);
    }
    if !ok {
        return Err(MatvecError::OutputOverflow);
    }
    Ok(out)
}

/// The int8-activation projection: `matvec_fixed(W, x̂)` bit for bit, with `x̂ = quant(x)`.
///
/// For `Q8_0`, `Q5_0`, `Q2_K`, `Q3_K`, `Q4_K` and `Q6_K` the row is reduced per activation block
/// and weight group as described in the module docs; when any bound cannot prove the reference's
/// arithmetic stays in range, the whole call falls back to [`matvec_fixed`] on the dequantized
/// activation and returns that result. Any other type is exactly [`matvec_fixed`] on `x̂`.
///
/// # Errors
///
/// The same [`MatvecError`] values, in the same order, as
/// `matvec_fixed(ggml_type, raw, n_out, n_in, &q.dequant()` — including the `Err` value. When
/// `q.dequant()` is `None` (only reachable for a `QuantAct` not produced by [`quant_act`], whose
/// contract makes that impossible) the activation has no Q32.32 value and the call refuses with
/// [`MatvecError::Dequant`]`(`[`EngineError::FixedOverflow`]`)`.
///
/// # Complexity
///
/// `O(n_out·n_in)` time in exact integer multiply-adds — one `i32` dot plus one checked `i128`
/// product per weight group instead of one `i128` product per weight — with `O(n_in)` for the
/// dequantized activation and its magnitude sum, and `O(n_out)` output.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check; `TryFrom` emits an unmodeled external the pinned \
              fail-closed extraction audit rejects"
)]
pub(crate) fn matvec_q8act(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    q: &QuantAct,
) -> Result<Vec<i64>, MatvecError> {
    match q8act_plan(ggml_type, n_in, q) {
        // A non-fused family or an activation with no Q32.32 value is the reference itself; the
        // dequantized activation is built here and nowhere else on that path.
        Q8ActPlan::Reference => {
            let Some(xhat) = q.dequant() else {
                return Err(MatvecError::Dequant(EngineError::FixedOverflow));
            };
            matvec_fixed(ggml_type, raw, n_out, n_in, &xhat)
        }
        Q8ActPlan::Fused(plan) => {
            let (qk, bb) =
                quant::block_geom(ggml_type).ok_or(MatvecError::UnsupportedQuant { ggml_type })?;
            if q.len() != n_in {
                return Err(MatvecError::ShapeMismatch { x: q.len(), n_in });
            }
            let row_bytes = n_in
                .div_ceil(qk)
                .checked_mul(bb)
                .ok_or(MatvecError::LengthOverflow)?;
            let need = n_out
                .checked_mul(row_bytes)
                .ok_or(MatvecError::LengthOverflow)?;
            if raw.len() < need {
                return Err(MatvecError::ShortPayload {
                    need,
                    have: raw.len(),
                });
            }
            q8act_rows(ggml_type, raw, n_out, n_in, row_bytes, need, q, &plan)
        }
    }
}

#[cfg(test)]
mod tests;
