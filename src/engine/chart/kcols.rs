//! Multi-column `Q4_K` and `Q6_K` row kernels: one weight row's bytes are decoded once and folded
//! against a group of activation columns. Column `c` of either kernel is exactly the `i128`
//! [`super::q8act_q4_k_row_fast`] / [`super::q8act_q6_k_row_fast`] form for `qs[c]` with `avx =
//! true`, and the AVX2 cores are the only unsafe code here; their equality with the single-column
//! row is `tested`, not extracted (trust boundary).
#![allow(
    dead_code,
    reason = "`matmat` is the caller for the packed lanes; the single-column folds below are called \
              by the equality tests under cfg(test)"
)]

use crate::format::quant::{Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, Q6_K_BLOCK_BYTES};

use super::portable::f16_to_fixed_fast;
use super::round_to_i64;
use super::x86;
use super::x86::{q4_k_scale_min_at, q4_k_window_dot_ptr, q6_k_half_dots, scale_word};
use super::{QuantAct, ACT_QK};

/// Activation columns per pass of the block loop.
///
/// The row's bytes are re-fetched once per column group and are `n_in` bytes wide, while a column's
/// activation codes for one `Q4_K` block are `QK_K` bytes (`Q4_K` 256, `Q6_K` 256): 32 columns keep a
/// group's code working set at 8 KiB, well inside a 48 KiB L1, so every block pass reads its codes
/// from L1. A wider group (128 columns = 32 KiB) also fits but leaves no room for the row and the
/// streamed next group; a narrower one re-fetches the row more often with no locality left to gain.
const COL_GROUP: usize = 32;

use crate::engine::kern::pow2::{f16_split, finish};
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mul_epi32, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setzero_si256,
    _mm256_slli_epi16, _mm256_sllv_epi64, _mm256_srli_epi16, _mm256_srli_epi64,
    _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_epi64, _mm256_sub_epi8, _mm_add_epi32,
    _mm_storeu_si128,
};

/// VEX-encoded `vpdpbusd` accumulating four `u8·i8` products into each `i32` lane: the same
/// instruction, and the same asm-inlining reason, as the private `x86::dpbusd_vex` (written as `asm!`
/// so no AVX2→AVX-VNNI inlining barrier or `incompatible_msrv` warning is introduced).
///
/// `vpdpbusd(0, q, x)` equals `vpmaddubsw` + `vpmaddwd` lane for lane: `q ≤ 15` and `|x| ≤ 127`, so
/// `2·15·127 = 3810 < 2^15` and the `i16` widening add cannot saturate.
///
/// # Safety
///
/// The caller must have AVX-VNNI (checked at runtime before the call).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dpbusd_zero(q: __m256i, x: __m256i) -> __m256i {
    let mut acc = _mm256_setzero_si256();
    // SAFETY: the caller guarantees AVX-VNNI.
    unsafe {
        core::arch::asm!(
            "{{vex}} vpdpbusd {acc}, {q}, {x}",
            acc = inout(ymm_reg) acc,
            q = in(ymm_reg) q,
            x = in(ymm_reg) x,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// `Σ_{pairs} q·x` per `i32` lane for one nibble plane and 32 activation codes, by `vpdpbusd` or the
/// `vpmaddubsw`/`vpmaddwd` pair; both are exact and lane-identical. `O(1)`.
///
/// # Safety
///
/// Register arithmetic only; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn plane_lanes(q: __m256i, x: __m256i, vnni: bool) -> __m256i {
    if vnni {
        // SAFETY: the caller resolved AVX-VNNI present once per kernel call.
        unsafe { dpbusd_zero(q, x) }
    } else {
        _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), _mm256_set1_epi16(1))
    }
}

/// The four unpacked nibble planes `[low(a), low(b), high(a), high(b)]` of the 32-byte windows at
/// `qs` and `qs + 32`, so a column group pays the unpack once. `O(1)`.
///
/// # Safety
///
/// `qs` must point at 64 readable bytes and the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4_k_planes2(qs: *const u8) -> [__m256i; 4] {
    // SAFETY: the caller guarantees the two loads' 64 bytes and AVX2.
    unsafe {
        let nib = _mm256_set1_epi8(0x0F);
        let a = _mm256_loadu_si256(qs.cast());
        let b = _mm256_loadu_si256(qs.add(32).cast());
        [
            _mm256_and_si256(a, nib),
            _mm256_and_si256(b, nib),
            _mm256_and_si256(_mm256_srli_epi16(a, 4), nib),
            _mm256_and_si256(_mm256_srli_epi16(b, 4), nib),
        ]
    }
}

/// The four exact `i32` sums of the lanes of `p0..p3`: the three-`vphaddd` tree of the private
/// `x86::reduce4`, restated because that one is not visible outside `chart::x86`. `O(1)`.
///
/// # Safety
///
/// Register arithmetic only; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn reduce4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    let h01 = _mm256_hadd_epi32(p0, p1);
    let h23 = _mm256_hadd_epi32(p2, p3);
    let h = _mm256_hadd_epi32(h01, h23);
    _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1))
}

/// The four exact 16-lane dots of a `Q4_K` window pair against one column, in the order
/// `[lo(a), hi(a), lo(b), hi(b)]` of [`super::x86::q4_k_block_dots`]. `O(1)`.
///
/// # Safety
///
/// `xq` must point at 128 readable lanes and the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4_k_plane_dots4(planes: &[__m256i; 4], xq: *const i8, vnni: bool) -> [i32; 4] {
    // SAFETY: the caller guarantees the four activation loads' 128 lanes and AVX2.
    unsafe {
        let r = reduce4(
            plane_lanes(planes[0], _mm256_loadu_si256(xq.cast()), vnni),
            plane_lanes(planes[2], _mm256_loadu_si256(xq.add(32).cast()), vnni),
            plane_lanes(planes[1], _mm256_loadu_si256(xq.add(64).cast()), vnni),
            plane_lanes(planes[3], _mm256_loadu_si256(xq.add(96).cast()), vnni),
        );
        let mut o = [0i32; 4];
        // SAFETY: `o` is four `i32` = 16 bytes, exactly one 128-bit store.
        _mm_storeu_si128(o.as_mut_ptr().cast(), r);
        o
    }
}

/// One group's contribution `act_scale·(w_scale·dot − w_min·xsum)` folded into `a`: the exact
/// expression [`super::RowAcc`]'s `emit` forms, in the same order of operations.
///
/// `#[inline(always)]` so the caller's group folds stay independent `i128` multiply chains in
/// registers; a call would serialize them.
#[inline]
fn fold_affine(a: &mut i128, w_scale: i64, w_min: i64, act_scale: i64, dot: i32, xsum: i32) {
    let inner = i128::from(w_scale) * i128::from(dot) - i128::from(w_min) * i128::from(xsum);
    *a += i128::from(act_scale) * inner;
}

/// One `Q4_K` row's exact pre-rounding `i128` sums for `k = acc.len()` columns: `acc[c]` is the value
/// [`super::q8act_q4_k_row_fast`] returns for `row`, `pairs` and `qs[c]` with `avx = true`, or `None`
/// where that call refuses (`Inf`/`NaN` block scale, which is row-scoped). `xsums` runs
/// `[column][32-lane group]`, as [`super::block_xsums`] lays it out. `O(pairs·k)` multiply-adds.
///
/// # Safety
///
/// AVX2 must be present; `row.len() == (pairs/4)·Q4_K_BLOCK_BYTES`, every `qs[c]` holds `2·pairs·32`
/// codes and `2·pairs` scales, `xsums.len() >= k·2·pairs`, and `acc.len() <= qs.len()`.
#[target_feature(enable = "avx2")]
#[allow(
    clippy::many_single_char_names,
    reason = "quantization math uses standard GGML/llama.cpp single-letter names"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the block-scale decode, the full-block column loop and the partial-block tail are one  row walk; splitting them would put the shared d/dmin/scales decode behind a call"
)]
pub(crate) unsafe fn q4_k_row_kcols_avx2(
    row: &[u8],
    pairs: usize,
    qs: &[QuantAct],
    xsums: &[i32],
    acc: &mut [i128],
) -> Option<()> {
    let k = acc.len();
    let total_groups = pairs * 2;
    // Resolved once: the plane cores branch on it per column group, not per plane call.
    let vnni = x86::avxvnni();
    let mut c0 = 0usize;
    while c0 < k {
        let c1 = if c0 + COL_GROUP < k {
            c0 + COL_GROUP
        } else {
            k
        };
        let mut g = 0usize;
        while g < total_groups {
            let pp = g / 2;
            let b = pp / 4;
            let p = pp % 4;
            let off = b * Q4_K_BLOCK_BYTES;
            // SAFETY: `off + 4 <= row.len()` by the caller's geometry.
            let d = unsafe { f16_to_fixed_fast(scale_word(row.as_ptr(), off)) }?;
            let dmin = unsafe { f16_to_fixed_fast(scale_word(row.as_ptr(), off + 2)) }?;
            let mut sc = [0u8; Q4_K_SCALE_SIZE];
            // SAFETY: `off + 16 <= row.len()` by the caller's geometry.
            sc.copy_from_slice(unsafe { row.get_unchecked(off + 4..off + 16) });
            if p == 0 && g + 8 <= total_groups {
                let (r0, n0) = q4_k_scale_min_at(&sc, 0);
                let (r1, n1) = q4_k_scale_min_at(&sc, 1);
                let (r2, n2) = q4_k_scale_min_at(&sc, 2);
                let (r3, n3) = q4_k_scale_min_at(&sc, 3);
                let (r4, n4) = q4_k_scale_min_at(&sc, 4);
                let (r5, n5) = q4_k_scale_min_at(&sc, 5);
                let (r6, n6) = q4_k_scale_min_at(&sc, 6);
                let (r7, n7) = q4_k_scale_min_at(&sc, 7);
                let slope = [
                    d * i64::from(r0),
                    d * i64::from(r1),
                    d * i64::from(r2),
                    d * i64::from(r3),
                    d * i64::from(r4),
                    d * i64::from(r5),
                    d * i64::from(r6),
                    d * i64::from(r7),
                ];
                let intercept = [
                    dmin * i64::from(n0),
                    dmin * i64::from(n1),
                    dmin * i64::from(n2),
                    dmin * i64::from(n3),
                    dmin * i64::from(n4),
                    dmin * i64::from(n5),
                    dmin * i64::from(n6),
                    dmin * i64::from(n7),
                ];
                // Two window-pair passes of four groups each: four unpacked planes stay in
                // registers for a whole column group, and the eight group sums per column use the
                // block-wide slopes decoded once above.
                let mut w = 0usize;
                while w < 2 {
                    // SAFETY: windows `2w` and `2w + 1` of a full block are inside its 128 bytes.
                    let planes = unsafe { q4_k_planes2(row.as_ptr().add(off + 16 + 64 * w)) };
                    let gw = g + 4 * w;
                    let q4 = 4 * w;
                    let sl0 = slope[q4];
                    let sl1 = slope[q4 + 1];
                    let sl2 = slope[q4 + 2];
                    let sl3 = slope[q4 + 3];
                    let it0 = intercept[q4];
                    let it1 = intercept[q4 + 1];
                    let it2 = intercept[q4 + 2];
                    let it3 = intercept[q4 + 3];
                    let mut c = c0;
                    while c < c1 {
                        let q = &qs[c];
                        let codes = q.codes();
                        let scales = q.scales();
                        // SAFETY: `codes.len() == total_groups·32` and `gw + 4 <= total_groups`.
                        let dots = unsafe {
                            q4_k_plane_dots4(&planes, codes.as_ptr().add(gw * ACT_QK), vnni)
                        };
                        let base = c * total_groups + gw;
                        // SAFETY: `scales.len() == total_groups > gw + 3` and
                        // `xsums.len() >= k·total_groups > base + 3`.
                        let (x0, x1, x2, x3) = unsafe {
                            (
                                *xsums.get_unchecked(base),
                                *xsums.get_unchecked(base + 1),
                                *xsums.get_unchecked(base + 2),
                                *xsums.get_unchecked(base + 3),
                            )
                        };
                        let (s0, s1, s2, s3) = unsafe {
                            (
                                *scales.get_unchecked(gw),
                                *scales.get_unchecked(gw + 1),
                                *scales.get_unchecked(gw + 2),
                                *scales.get_unchecked(gw + 3),
                            )
                        };
                        // Two accumulator chains: consecutive `i128` multiply chains are
                        // independent, so one group's multiply latency overlaps the next.
                        let mut a0 = acc[c];
                        let mut a1 = 0i128;
                        fold_affine(&mut a0, sl0, it0, s0, dots[0], x0);
                        fold_affine(&mut a1, sl1, it1, s1, dots[1], x1);
                        fold_affine(&mut a0, sl2, it2, s2, dots[2], x2);
                        fold_affine(&mut a1, sl3, it3, s3, dots[3], x3);
                        acc[c] = a0 + a1;
                        c += 1;
                    }
                    w += 1;
                }
                g += 8;
            } else {
                let (rl, nl) = q4_k_scale_min_at(&sc, 2 * p);
                let (rh, nh) = q4_k_scale_min_at(&sc, 2 * p + 1);
                let slope_lo = d * i64::from(rl);
                let intercept_lo = dmin * i64::from(nl);
                let slope_hi = d * i64::from(rh);
                let intercept_hi = dmin * i64::from(nh);
                let mut c = c0;
                while c < c1 {
                    let q = &qs[c];
                    let codes = q.codes();
                    let scales = q.scales();
                    // SAFETY: `off + 16 + 32p + 32 <= row.len()`, and both of the column's groups
                    // `g`, `g + 1` are below `total_groups`, so their 32-lane windows are in range.
                    let (dot_lo, dot_hi) = unsafe {
                        q4_k_window_dot_ptr(
                            row.as_ptr().add(off + 16 + 32 * p),
                            codes.as_ptr().add(g * ACT_QK),
                            codes.as_ptr().add((g + 1) * ACT_QK),
                        )
                    };
                    let base = c * total_groups + g;
                    // SAFETY: `scales.len() == total_groups > g + 1` and `xsums.len() >= base + 2`.
                    let s_lo = unsafe { *scales.get_unchecked(g) };
                    let s_hi = unsafe { *scales.get_unchecked(g + 1) };
                    let x_lo = unsafe { *xsums.get_unchecked(base) };
                    let x_hi = unsafe { *xsums.get_unchecked(base + 1) };
                    let inner_lo = i128::from(slope_lo) * i128::from(dot_lo)
                        - i128::from(intercept_lo) * i128::from(x_lo);
                    let inner_hi = i128::from(slope_hi) * i128::from(dot_hi)
                        - i128::from(intercept_hi) * i128::from(x_hi);
                    acc[c] += i128::from(s_lo) * inner_lo + i128::from(s_hi) * inner_hi;
                    c += 1;
                }
                g += 2;
            }
        }
        c0 = c1;
    }
    Some(())
}

/// One `Q6_K` row's exact pre-rounding `i128` sums for `k = acc.len()` columns: `acc[c]` is the value
/// [`super::q8act_q6_k_row_fast`] returns for `row`, `blocks` and `qs[c]` with `avx = true`, or `None`
/// where that call refuses (an `Inf`/`NaN` block `d`, row-scoped). `O(blocks·k)` exact integer
/// multiply-adds, `O(1)` space.
///
/// # Safety
///
/// AVX2 must be present; `row.len() == blocks·Q6_K_BLOCK_BYTES`, every `qs[c]` holds `blocks·256` codes
/// and `blocks·8` scales, and `acc.len() <= qs.len()`.
#[target_feature(enable = "avx2")]
#[allow(
    clippy::many_single_char_names,
    reason = "quantization math uses standard GGML/llama.cpp single-letter names"
)]
pub(crate) unsafe fn q6_k_row_kcols_avx2(
    row: &[u8],
    blocks: usize,
    qs: &[QuantAct],
    acc: &mut [i128],
) -> Option<()> {
    let k = acc.len();
    let total_groups = blocks * 8;
    let mut c0 = 0usize;
    while c0 < k {
        let c1 = if c0 + COL_GROUP < k {
            c0 + COL_GROUP
        } else {
            k
        };
        let mut g = 0usize;
        while g < total_groups {
            let b = g / 8;
            let off = b * Q6_K_BLOCK_BYTES;
            // SAFETY: `off + Q6_K_BLOCK_BYTES <= row.len()` by the caller's geometry.
            let d =
                unsafe { f16_to_fixed_fast(scale_word(row.as_ptr(), off + Q6_K_BLOCK_BYTES - 2)) }?;
            // SAFETY: the block's 210 bytes are in range.
            let base = unsafe { row.as_ptr().add(off) };
            let mut c = c0;
            while c < c1 {
                let q = &qs[c];
                let codes = q.codes();
                let scales = q.scales();
                let mut a0 = acc[c];
                let mut a1 = 0i128;
                let mut n = 0usize;
                while n < 2 {
                    // SAFETY: `base + 64n + 64 <= block end`, `base + 128 + 32n + 32 <= block end`,
                    // and the column's 128 lanes at `g·32 + 128n` are below its `total_groups·32`
                    // codes.
                    let dots = unsafe {
                        q6_k_half_dots(
                            base.add(64 * n),
                            base.add(128 + 32 * n),
                            codes.as_ptr().add(g * ACT_QK + 128 * n),
                        )
                    };
                    let mut i = 0usize;
                    while i < 4 {
                        // SAFETY: the block's 16 scale bytes sit at `base + 192`, and
                        // `8n + 2i + 1 < 16`.
                        let sclo =
                            i64::from(unsafe { *base.add(192 + 8 * n + 2 * i) }.cast_signed());
                        let schi =
                            i64::from(unsafe { *base.add(192 + 8 * n + 2 * i + 1) }.cast_signed());
                        let group_dot =
                            sclo * i64::from(dots[2 * i]) + schi * i64::from(dots[2 * i + 1]);
                        // SAFETY: `scales.len() == total_groups > g + 7`.
                        let s = unsafe { *scales.get_unchecked(g + 4 * n + i) };
                        // Two accumulators keep consecutive groups' `i128` multiply chains
                        // independent; the value is the single-column row's, exactly.
                        if i.is_multiple_of(2) {
                            a0 += i128::from(s) * (i128::from(d) * i128::from(group_dot));
                        } else {
                            a1 += i128::from(s) * (i128::from(d) * i128::from(group_dot));
                        }
                        i += 1;
                    }
                    n += 1;
                }
                acc[c] = a0 + a1;
                c += 1;
            }
            g += 8;
        }
        c0 = c1;
    }
    Some(())
}

// ── the packed-lane kernel ───────────────────────────────────────────────────────────────────────
//
// The shared `matmat` carrier's `Q4_K` and `Q6_K` lanes share this kernel: the same activation
// pack (8 columns per group as `x + 128` bytes, four consecutive inputs of column `c` in 32-bit
// lane `c`, one `vpdpbusd` per weight word against a broadcast), the same per-(row, column-group)
// shift grid in `i64` lanes, and the same `i128` fallback past the [`lane_spread_limit`] bound.
// Every sub-block term is an integer coefficient times `2^{k + e}` (`d = m_d·2^k_d`,
// `dmin = m_m·2^k_m` from `f16_split`, `S_s = 2^{e_s}` from `act_scale`), so the lane holds the
// exact `i128` sum of
// [`super::q8act_q4_k_row_fast`] / [`super::q8act_q6_k_row_fast`] divided by `2^{k_min + e_min(c)}`,
// and [`finish`] applies that power once: [`super::round_to_i64`] of the same sum.
//
// **The bound.** A `Q4_K` 32-input term is at most `2^11·2^6·2^16 = 2^33` (`m_d`, `sc`, `D`) plus
// `2^29` for its `X` partner, so `G = n_in/32` groups are below `G·2^34`; a `Q6_K` 16-input term is
// at most `2^11·2^7·2^16 = 2^34`, so `G = n_in/16` groups are below `G·2^34`. With every shift at
// most `spread` (the range of `k + e` over the group grid) the lane stays below
// `2^(34 + ⌈log₂ G⌉ + spread)`. Past the bound the (row, column group) takes the exact `i128` fold of
// [`q4_k_row_kcols_avx2`] / [`q6_k_row_kcols_avx2`]. The AVX2/AVX-VNNI core is not extracted; its
// equality with the `i128` kernel and with `matvec_q8act` is `tested`.

/// Columns per packed activation group: one 32-bit lane each in a 256-bit register.
const LANES: usize = 8;

/// `34 + ⌈log₂ groups⌉ + spread ≤ 62`: the largest shift spread an `i64` lane admits. `O(1)`.
pub(crate) fn lane_spread_limit(groups: usize) -> u32 {
    let log = usize::BITS - groups.saturating_sub(1).leading_zeros();
    62u32.saturating_sub(34 + log)
}

/// Test-only count of (row, column-group) pairs that took the exact `i128` fold; [`reset_i128_groups`]
/// zeroes it, so a fixture can assert which path it exercised.
#[cfg(test)]
pub(crate) static I128_GROUPS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Zeroes [`I128_GROUPS`].
#[cfg(test)]
pub(crate) fn reset_i128_groups() {
    I128_GROUPS.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// The call's activation columns, packed per group of [`LANES`] columns: `words` the `x + 128` code
/// quads `[group][block][quad][lane]`, `ke_e`/`ke_o` the per-column `e_b − e_min(c)` in the order
/// `_mm256_mul_epi32` leaves the columns in, `xs`/`xs_col` the `Q4_K` block sums (packed and
/// `[column][block]`), `emin` per column and `spread` per group.
pub(crate) struct PackedCols {
    /// Column groups, `⌈k / LANES⌉`.
    groups: usize,
    /// `[group][block][quad][lane]`: four `x + 128` bytes of column `8·group + lane`.
    words: Vec<u32>,
    /// `[group][block][lane]`: `e_b − e_min(c)` as `i64`, in the order `_mm256_mul_epi32` leaves
    /// the columns in — `ke_e` the even lanes `8·group + {0,2,4,6}`, `ke_o` the odd ones.
    ke_e: Vec<i64>,
    ke_o: Vec<i64>,
    /// `[group][block][lane]`: `X_b = Σ xq` of the 32-lane block, `Q4_K` only (empty otherwise),
    /// and `xs_col` the same sums in `[column][block]` order for the `i128` fold.
    xs: Vec<i32>,
    xs_col: Vec<i32>,
    /// `e_min(c)` per column and the largest `e_b − e_min` in each group.
    emin: Vec<u32>,
    spread: Vec<u32>,
}

/// Packs the call's columns, or `None` when a scale is not a power of two or the geometry is ragged.
/// `O(k·blocks·8)` `u32` writes plus `O(k·blocks·LANES)` shift words, serial, once per call.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    reason = "c, b, g index the lane-by-block grid; the codes are signed bytes widened to a lane word;  the pack is one grid walk"
)]
pub(crate) fn pack_cols(qs: &[QuantAct], blocks: usize, need_xs: bool) -> Option<PackedCols> {
    let k = qs.len();
    let groups = k.div_ceil(LANES);
    let mut words = vec![u32::MAX; groups * blocks * 8 * LANES];
    let mut ke_e = vec![0i64; groups * blocks * 4];
    let mut ke_o = vec![0i64; groups * blocks * 4];
    let mut xs = vec![0i32; if need_xs { groups * blocks * LANES } else { 0 }];
    let mut xs_col = vec![0i32; if need_xs { k * blocks } else { 0 }];
    let mut emin = vec![0u32; k];
    let mut spread = vec![0u32; groups];
    let mut g = 0usize;
    while g < groups {
        let lo_c = g * LANES;
        let hi_c = (lo_c + LANES).min(k);
        let mut group_spread = 0u32;
        let mut lane = 0usize;
        while lane < LANES {
            let c = lo_c + lane;
            if c < hi_c {
                let q = &qs[c];
                let scales = q.scales();
                if scales.len() != blocks {
                    return None;
                }
                let mut lo = u32::MAX;
                let mut b = 0usize;
                while b < blocks {
                    let s = scales[b];
                    if s < 0 || (s != 0 && s.count_ones() != 1) {
                        return None;
                    }
                    if s != 0 {
                        lo = lo.min(s.trailing_zeros());
                    }
                    b += 1;
                }
                let lo = if lo == u32::MAX { 0 } else { lo };
                emin[c] = lo;
                let (ei, oi) = if lane.is_multiple_of(2) {
                    (Some((c - lo_c) / 2), None)
                } else {
                    (None, Some((c - lo_c) / 2))
                };
                let codes = q.codes();
                let mut sum_base = 0usize;
                if need_xs {
                    sum_base = (g * blocks) * LANES + lane;
                }
                let bx: Vec<i32> = if need_xs {
                    super::portable::block_xsums(codes)
                } else {
                    Vec::new()
                };
                let mut b = 0usize;
                while b < blocks {
                    let s = scales[b];
                    let sh = if s == 0 { 0 } else { s.trailing_zeros() - lo };
                    group_spread = group_spread.max(sh);
                    if let Some(i) = ei {
                        ke_e[(g * blocks + b) * 4 + i] = i64::from(sh);
                    }
                    if let Some(i) = oi {
                        ke_o[(g * blocks + b) * 4 + i] = i64::from(sh);
                    }
                    let mut quad = 0usize;
                    while quad < 8 {
                        let at = b * ACT_QK + 4 * quad;
                        // SAFETY: `codes` holds `blocks·ACT_QK` entries and `at + 4` is inside it;
                        // the word's four little-endian bytes are the quad's four codes.
                        let raw = unsafe { codes.as_ptr().add(at).cast::<u32>().read_unaligned() };
                        words[((g * blocks + b) * 8 + quad) * LANES + lane] = raw ^ 0x8080_8080;
                        quad += 1;
                    }
                    if need_xs {
                        if let Some(v) = bx.get(b) {
                            xs[sum_base + b * LANES] = *v;
                            xs_col[c * blocks + b] = *v;
                        }
                    }
                    b += 1;
                }
            } else {
                // A padding lane: no column, zero shift, a code word whose `x + 128` is `0`.
                let mut b = 0usize;
                while b < blocks {
                    let mut quad = 0usize;
                    while quad < 8 {
                        words[((g * blocks + b) * 8 + quad) * LANES + lane] = 0x8080_8080;
                        quad += 1;
                    }
                    b += 1;
                }
            }
            lane += 1;
        }
        spread[g] = group_spread;
        g += 1;
    }
    Some(PackedCols {
        groups,
        words,
        ke_e,
        ke_o,
        xs,
        xs_col,
        emin,
        spread,
    })
}

/// `vpdpbusd` accumulating into `acc`; AVX-VNNI must be present.
///
/// # Safety
///
/// AVX-VNNI must be present.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dpbusd_acc(acc: __m256i, q: __m256i, x: __m256i) -> __m256i {
    let mut out = acc;
    // SAFETY: the caller guarantees AVX-VNNI.
    unsafe {
        core::arch::asm!(
            "{{vex}} vpdpbusd {acc}, {q}, {x}",
            acc = inout(ymm_reg) out,
            q = in(ymm_reg) q,
            x = in(ymm_reg) x,
            options(pure, nomem, nostack)
        );
    }
    out
}

/// `acc + q·x` per 32-bit lane for four activation bytes and four weight bytes: one `vpdpbusd`
/// where the host has AVX-VNNI, the `vpmaddubsw`/`vpmaddwd` pair otherwise. Both are exact and
/// lane-identical (`|2·255·32| < 2^15`, so neither widening step saturates).
///
/// # Safety
///
/// AVX2, and AVX-VNNI when `vnni`.
///
/// # Complexity
///
/// `O(1)`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lane_acc(acc: __m256i, q: __m256i, x: __m256i, vnni: bool) -> __m256i {
    if vnni {
        // SAFETY: the caller resolved AVX-VNNI present before the kernel call.
        unsafe { dpbusd_acc(acc, q, x) }
    } else {
        _mm256_add_epi32(
            acc,
            _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), _mm256_set1_epi16(1)),
        )
    }
}

/// The `i32` sums of the 32 bytes of `v` and of its two 16-byte halves. `O(1)`; AVX2.
///
/// # Safety
///
/// AVX2.
#[target_feature(enable = "avx2")]
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    reason = "a sum of 32 bytes is below 2^14; the store into the four-lane stack buffer is the  unaligned store `_mm256_storeu_si256` names"
)]
unsafe fn sum_bytes(v: __m256i) -> [i32; 3] {
    let s = _mm256_sad_epu8(v, _mm256_setzero_si256());
    let mut q = [0u64; 4];
    // SAFETY: `q` is four `u64` = 32 bytes, exactly one store.
    unsafe { _mm256_storeu_si256(q.as_mut_ptr().cast::<__m256i>(), s) };
    [
        (q[0] + q[1]) as i32,
        (q[2] + q[3]) as i32,
        (q[0] + q[1] + q[2] + q[3]) as i32,
    ]
}

/// One row's decoded `Q4_K` block data, refilled per row and reused across the column groups: the
/// 32 weight bytes of every 32-input group, its `128·Σ q` bias, `m_d·sc_g`, `m_m·m_g`, and the row's
/// `k_d`/`k_m` per block.
pub(crate) struct Q4KRow {
    /// `[block][group][lane]`: the 32 weight bytes of each 32-input group.
    bytes: Vec<u8>,
    /// `[block][group]`: `128·Σ q`, the bias the packed `x + 128` codes carry.
    bias: Vec<i32>,
    /// `[block][group]`: `m_d·sc_g`.
    coef: Vec<i32>,
    /// `[block][group]`: `m_m·m_g`.
    cmin: Vec<i32>,
    /// `[block]`: the row's `k_d` and `k_m`.
    kd: Vec<u32>,
    km: Vec<u32>,
}

impl Q4KRow {
    /// The scratch for `blocks` `Q4_K` super-blocks.
    ///
    /// # Complexity
    ///
    /// `O(blocks)` time and six allocations; `blocks·256` bytes and `blocks·26` words.
    pub(crate) fn new(blocks: usize) -> Self {
        Q4KRow {
            bytes: vec![0u8; blocks * 256],
            bias: vec![0i32; blocks * 8],
            coef: vec![0i32; blocks * 8],
            cmin: vec![0i32; blocks * 8],
            kd: vec![0u32; blocks],
            km: vec![0u32; blocks],
        }
    }

    /// Decodes `row` into the scratch, returning the row's `(kmin, kmax)` over `d` and `dmin`, or
    /// `None` for a non-finite block scale — the single-column row's own, row-scoped refusal.
    ///
    /// # Safety
    ///
    /// AVX2; `row.len() == blocks·Q4_K_BLOCK_BYTES`.
    ///
    /// # Complexity
    ///
    /// `O(blocks)`: four 32-byte windows, eight biases, eight scale pairs and two `f16` splits per
    /// block, each a constant number of vector steps.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::many_single_char_names,
        reason = "d, dmin, sc, m, k are the standard GGML/llama.cpp names of these fields"
    )]
    pub(crate) unsafe fn fill(&mut self, row: &[u8], blocks: usize) -> Option<(u32, u32)> {
        let mut kmin = u32::MAX;
        let mut kmax = 0u32;
        let nib = _mm256_set1_epi8(0x0F);
        let mut b = 0usize;
        while b < blocks {
            let off = b * Q4_K_BLOCK_BYTES;
            // SAFETY: `off + 4 <= row.len()` by the caller's geometry.
            let (md, kd) = f16_split(unsafe { scale_word(row.as_ptr(), off) })?;
            let (mm, km) = f16_split(unsafe { scale_word(row.as_ptr(), off + 2) })?;
            let mut sc = [0u8; Q4_K_SCALE_SIZE];
            // SAFETY: `off + 16 <= row.len()`.
            sc.copy_from_slice(unsafe { row.get_unchecked(off + 4..off + 16) });
            self.kd[b] = kd;
            self.km[b] = km;
            kmin = kmin.min(kd).min(km);
            kmax = kmax.max(kd).max(km);
            let mut w = 0usize;
            while w < 4 {
                // SAFETY: window `w` is the 32 bytes at `off + 16 + 32w` of the block's 128.
                let v = unsafe { _mm256_loadu_si256(row.as_ptr().add(off + 16 + 32 * w).cast()) };
                let lo = _mm256_and_si256(v, nib);
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nib);
                let mut half = 0usize;
                while half < 2 {
                    let gi = b * 8 + 2 * w + half;
                    let vec = if half == 0 { lo } else { hi };
                    // SAFETY: `gi < blocks·8` group slots of 32 bytes each.
                    unsafe {
                        _mm256_storeu_si256(self.bytes.as_mut_ptr().add(gi * 32).cast(), vec);
                    }
                    // SAFETY: register arithmetic over `vec`, which lives above.
                    let sums = unsafe { sum_bytes(vec) };
                    self.bias[gi] = 128 * sums[2];
                    let (s, m) = q4_k_scale_min_at(&sc, 2 * w + half);
                    self.coef[gi] = md * i32::from(s);
                    self.cmin[gi] = mm * i32::from(m);
                    half += 1;
                }
                w += 1;
            }
            b += 1;
        }
        Some((kmin, kmax))
    }
}

/// One `Q4_K` row's packed-lane fold: `out[c]` is column `c`'s rounded value, or `false` when a lane
/// leaves `i64`, a scale is non-finite or [`finish`] refuses. A column group takes the `i64` lane
/// while `kspread + spread[g] <= limit`; one past it takes [`q4_k_row_kcols_avx2`] on its [`LANES`]
/// columns. Temporal `O(qs.len()·blocks)`, spatial `O(1)`.
///
/// # Safety
///
/// AVX2; `row` holds `blocks` `Q4_K` super-blocks; `p` was packed from `qs` at `8·blocks` 32-input
/// groups; `out.len() == qs.len()`.
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment,
    reason = "the row's decoded scales and the pack travel as one argument each, so the group loop  keeps them in registers; g, b, s, d, x, kd, km name the kernel's grid; every load is  the unaligned load the packed buffers require"
)]
pub(crate) unsafe fn q4_k_row_lanes_avx2(
    row: &[u8],
    blocks: usize,
    qs: &[QuantAct],
    p: &PackedCols,
    sc: &Q4KRow,
    kmin: u32,
    kspread: u32,
    limit: u32,
    out: &mut [i64],
) -> bool {
    let k = qs.len();
    let vnni = x86::avxvnni();
    let n32 = blocks * 8;
    let mut g = 0usize;
    while g < p.groups {
        let lo_c = g * LANES;
        let hi_c = (lo_c + LANES).min(k);
        if kspread + p.spread[g] > limit {
            #[cfg(test)]
            I128_GROUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut acc8 = [0i128; LANES];
            // SAFETY: `row` holds `blocks` super-blocks (`4·blocks` pairs), the columns hold `n32`
            // 32-lane groups of codes and scales, and `xs_col` is `[column][n32]`.
            let got = unsafe {
                q4_k_row_kcols_avx2(
                    row,
                    4 * blocks,
                    &qs[lo_c..hi_c],
                    &p.xs_col[lo_c * n32..hi_c * n32],
                    &mut acc8[..hi_c - lo_c],
                )
            };
            if got.is_none() {
                return false;
            }
            let mut c = lo_c;
            while c < hi_c {
                match round_to_i64(acc8[c - lo_c]) {
                    Some(v) => out[c] = v,
                    None => return false,
                }
                c += 1;
            }
        } else {
            let mut acc_e = _mm256_setzero_si256();
            let mut acc_o = _mm256_setzero_si256();
            let mut b = 0usize;
            while b < blocks {
                let kd = i64::from(sc.kd[b] - kmin);
                let km = i64::from(sc.km[b] - kmin);
                let mut s = 0usize;
                while s < 8 {
                    let gi = b * 8 + s;
                    let at = g * n32 + gi;
                    let w = unsafe { p.words.as_ptr().add(at * 8 * LANES) };
                    let a = unsafe { sc.bytes.as_ptr().add(gi * 32) };
                    let mut d = _mm256_setzero_si256();
                    let mut quad = 0usize;
                    while quad < 8 {
                        // SAFETY: `w` has eight quads of `LANES` words for this group and `a` its
                        // 32 weight bytes.
                        unsafe {
                            let act = _mm256_loadu_si256(w.add(quad * LANES).cast::<__m256i>());
                            let wb =
                                _mm256_set1_epi32(a.add(4 * quad).cast::<i32>().read_unaligned());
                            d = lane_acc(d, act, wb, vnni);
                        }
                        quad += 1;
                    }
                    let d = _mm256_sub_epi32(d, _mm256_set1_epi32(sc.bias[gi]));
                    let cd = _mm256_set1_epi32(sc.coef[gi]);
                    let cm = _mm256_set1_epi32(sc.cmin[gi]);
                    // SAFETY: `ke_e`/`ke_o` hold four `i64` per group and `xs` eight `i32`.
                    unsafe {
                        let ke = _mm256_loadu_si256(p.ke_e.as_ptr().add(at * 4).cast::<__m256i>());
                        let ko = _mm256_loadu_si256(p.ke_o.as_ptr().add(at * 4).cast::<__m256i>());
                        let x = _mm256_loadu_si256(p.xs.as_ptr().add(at * LANES).cast::<__m256i>());
                        let sh_e = _mm256_add_epi64(ke, _mm256_set1_epi64x(kd));
                        let sh_o = _mm256_add_epi64(ko, _mm256_set1_epi64x(kd));
                        acc_e = _mm256_add_epi64(
                            acc_e,
                            _mm256_sllv_epi64(_mm256_mul_epi32(d, cd), sh_e),
                        );
                        acc_o = _mm256_add_epi64(
                            acc_o,
                            _mm256_sllv_epi64(_mm256_mul_epi32(_mm256_srli_epi64(d, 32), cd), sh_o),
                        );
                        let mh_e = _mm256_add_epi64(ke, _mm256_set1_epi64x(km));
                        let mh_o = _mm256_add_epi64(ko, _mm256_set1_epi64x(km));
                        acc_e = _mm256_sub_epi64(
                            acc_e,
                            _mm256_sllv_epi64(_mm256_mul_epi32(x, cm), mh_e),
                        );
                        acc_o = _mm256_sub_epi64(
                            acc_o,
                            _mm256_sllv_epi64(_mm256_mul_epi32(_mm256_srli_epi64(x, 32), cm), mh_o),
                        );
                    }
                    s += 1;
                }
                b += 1;
            }
            let mut ve = [0i64; LANES / 2];
            let mut vo = [0i64; LANES / 2];
            // SAFETY: two four-lane stores into the stack buffers above.
            unsafe {
                _mm256_storeu_si256(ve.as_mut_ptr().cast::<__m256i>(), acc_e);
                _mm256_storeu_si256(vo.as_mut_ptr().cast::<__m256i>(), acc_o);
            }
            let mut i = 0usize;
            while i < LANES / 2 {
                let c0 = lo_c + 2 * i;
                let c1 = c0 + 1;
                if c0 < hi_c {
                    match finish(ve[i], kmin + p.emin[c0]) {
                        Some(v) => out[c0] = v,
                        None => return false,
                    }
                }
                if c1 < hi_c {
                    match finish(vo[i], kmin + p.emin[c1]) {
                        Some(v) => out[c1] = v,
                        None => return false,
                    }
                }
                i += 1;
            }
        }
        g += 1;
    }
    true
}

/// One 32-input `Q6_K` group's unsigned six-bit weight bytes, `(ql nibble | qh bits << 4) ∈ 0..63`,
/// for window `j ∈ 0..4` of block half `n`, in the output order
/// [`super::quant::q6_k_codes`](crate::format::quant::q6_k_codes) gives. The `−32` the codes carry is
/// applied at the use site, where its byte sum feeds the `x + 128` bias (the `sad` sum is unsigned).
/// `O(1)`.
///
/// # Safety
///
/// AVX2; `ql` and `qh` were loaded from 32 readable bytes each.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q6_group_raw(ql: __m256i, qh: __m256i, j: usize) -> __m256i {
    let nib = _mm256_set1_epi8(0x0F);
    let two = _mm256_set1_epi8(0x03);
    let lo = if j < 2 {
        _mm256_and_si256(ql, nib)
    } else {
        _mm256_and_si256(_mm256_srli_epi16(ql, 4), nib)
    };
    let hi = match j {
        0 => _mm256_and_si256(qh, two),
        1 => _mm256_and_si256(_mm256_srli_epi16(qh, 2), two),
        2 => _mm256_and_si256(_mm256_srli_epi16(qh, 4), two),
        _ => _mm256_and_si256(_mm256_srli_epi16(qh, 6), two),
    };
    _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4))
}

/// One row's decoded `Q6_K` block data, refilled per row: the 32 signed weight bytes of every 32-input
/// group, its `128·Σ w` bias and `m_d·sc` per 16-lane half, and the row's `k_d` per block.
pub(crate) struct Q6KRow {
    /// `[block][group][lane]`: the 32 signed weight bytes of each 32-input group.
    bytes: Vec<u8>,
    /// `[block][group]`: `128·Σ w` over the group's first and second 16 lanes.
    bias_lo: Vec<i32>,
    bias_hi: Vec<i32>,
    /// `[block][group]`: `m_d·sc` of the group's first and second 16-lane 16-group.
    coef_lo: Vec<i32>,
    coef_hi: Vec<i32>,
    /// `[block]`: the row's `k_d`.
    kd: Vec<u32>,
}

impl Q6KRow {
    /// The scratch for `blocks` `Q6_K` super-blocks.
    ///
    /// # Complexity
    ///
    /// `O(blocks)` time and six allocations; `blocks·256` bytes and `blocks·33` words.
    pub(crate) fn new(blocks: usize) -> Self {
        Q6KRow {
            bytes: vec![0u8; blocks * 256],
            bias_lo: vec![0i32; blocks * 8],
            bias_hi: vec![0i32; blocks * 8],
            coef_lo: vec![0i32; blocks * 8],
            coef_hi: vec![0i32; blocks * 8],
            kd: vec![0u32; blocks],
        }
    }

    /// Decodes `row` into the scratch, returning the row's `(kmin, kmax)` over `d`, or `None` for a
    /// non-finite block scale — the single-column row's own, row-scoped refusal.
    ///
    /// # Safety
    ///
    /// AVX2; `row.len() == blocks·Q6_K_BLOCK_BYTES`.
    ///
    /// # Complexity
    ///
    /// `O(blocks)`: two halves of four groups each per block, with one `f16` split and eight `i8`
    /// scale bytes.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::many_single_char_names,
        reason = "n, j, k are the standard GGML block-half and window names"
    )]
    pub(crate) unsafe fn fill(&mut self, row: &[u8], blocks: usize) -> Option<(u32, u32)> {
        let mut kmin = u32::MAX;
        let mut kmax = 0u32;
        let mut b = 0usize;
        while b < blocks {
            let off = b * Q6_K_BLOCK_BYTES;
            // SAFETY: the block's last two bytes are its `f16` scale.
            let (md, kd) =
                f16_split(unsafe { scale_word(row.as_ptr(), off + Q6_K_BLOCK_BYTES - 2) })?;
            self.kd[b] = kd;
            kmin = kmin.min(kd);
            kmax = kmax.max(kd);
            // SAFETY: the block's 210 bytes are in range.
            let base = unsafe { row.as_ptr().add(off) };
            let mut n = 0usize;
            while n < 2 {
                // SAFETY: `ql` windows sit at `64n` and `64n + 32`, `qh` at `128 + 32n`, and the
                // block's 16 `i8` scales at `192`.
                unsafe {
                    let qh = _mm256_loadu_si256(base.add(128 + 32 * n).cast());
                    let mut j = 0usize;
                    while j < 4 {
                        let ql = _mm256_loadu_si256(base.add(64 * n + 32 * (j % 2)).cast());
                        let raw = q6_group_raw(ql, qh, j);
                        let v = _mm256_sub_epi8(raw, _mm256_set1_epi8(32));
                        let gi = b * 8 + 4 * n + j;
                        _mm256_storeu_si256(self.bytes.as_mut_ptr().add(gi * 32).cast(), v);
                        // The codes are `raw - 32`; the bias's byte sum is the raw sum less `32·16`
                        // per half, and the `sad` sum is unsigned while `raw` is `0..63`.
                        let sums = sum_bytes(raw);
                        self.bias_lo[gi] = 128 * (sums[0] - 512);
                        self.bias_hi[gi] = 128 * (sums[1] - 512);
                        self.coef_lo[gi] =
                            md * i32::from(base.add(192 + 8 * n + 2 * j).read().cast_signed());
                        self.coef_hi[gi] =
                            md * i32::from(base.add(192 + 8 * n + 2 * j + 1).read().cast_signed());
                        j += 1;
                    }
                }
                n += 1;
            }
            b += 1;
        }
        Some((kmin, kmax))
    }
}

/// One `Q6_K` row's packed-lane fold: `out[c]` is column `c`'s rounded value, or `false` when a lane
/// leaves `i64`, a scale is non-finite or [`finish`] refuses. A column group takes the `i64` lane
/// while `kspread + spread[g] <= limit`; one past it takes [`q6_k_row_kcols_avx2`]. Temporal
/// `O(qs.len()·blocks)`, spatial `O(1)`.
///
/// # Safety
///
/// AVX2; `row` holds `blocks` `Q6_K` super-blocks; `p` was packed from `qs` at `8·blocks` 32-input
/// groups; `out.len() == qs.len()`.
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment,
    reason = "the row's decoded scales and the pack travel as one argument each, so the group loop  keeps them in registers; g, b, s, d_lo, d_hi, kd name the kernel's grid; every load  is the unaligned load the packed buffers require"
)]
pub(crate) unsafe fn q6_k_row_lanes_avx2(
    row: &[u8],
    blocks: usize,
    qs: &[QuantAct],
    p: &PackedCols,
    sc: &Q6KRow,
    kmin: u32,
    kspread: u32,
    limit: u32,
    out: &mut [i64],
) -> bool {
    let k = qs.len();
    let vnni = x86::avxvnni();
    let n32 = blocks * 8;
    let mut g = 0usize;
    while g < p.groups {
        let lo_c = g * LANES;
        let hi_c = (lo_c + LANES).min(k);
        if kspread + p.spread[g] > limit {
            #[cfg(test)]
            I128_GROUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut acc8 = [0i128; LANES];
            // SAFETY: `row` holds `blocks` blocks, the columns hold `8·blocks` 32-lane groups of
            // codes and scales, and `acc8` covers the group's real columns.
            let got = unsafe {
                q6_k_row_kcols_avx2(row, blocks, &qs[lo_c..hi_c], &mut acc8[..hi_c - lo_c])
            };
            if got.is_none() {
                return false;
            }
            let mut c = lo_c;
            while c < hi_c {
                match round_to_i64(acc8[c - lo_c]) {
                    Some(v) => out[c] = v,
                    None => return false,
                }
                c += 1;
            }
        } else {
            let mut acc_e = _mm256_setzero_si256();
            let mut acc_o = _mm256_setzero_si256();
            let mut b = 0usize;
            while b < blocks {
                let kd = i64::from(sc.kd[b] - kmin);
                let mut s = 0usize;
                while s < 8 {
                    let gi = b * 8 + s;
                    let at = g * n32 + gi;
                    let w = unsafe { p.words.as_ptr().add(at * 8 * LANES) };
                    let a = unsafe { sc.bytes.as_ptr().add(gi * 32) };
                    let mut d_lo = _mm256_setzero_si256();
                    let mut d_hi = _mm256_setzero_si256();
                    let mut quad = 0usize;
                    while quad < 8 {
                        // SAFETY: `w` has eight quads of `LANES` words for this group and `a` its
                        // 32 weight bytes.
                        unsafe {
                            let act = _mm256_loadu_si256(w.add(quad * LANES).cast::<__m256i>());
                            let wb =
                                _mm256_set1_epi32(a.add(4 * quad).cast::<i32>().read_unaligned());
                            if quad < 4 {
                                d_lo = lane_acc(d_lo, act, wb, vnni);
                            } else {
                                d_hi = lane_acc(d_hi, act, wb, vnni);
                            }
                        }
                        quad += 1;
                    }
                    let d_lo = _mm256_sub_epi32(d_lo, _mm256_set1_epi32(sc.bias_lo[gi]));
                    let d_hi = _mm256_sub_epi32(d_hi, _mm256_set1_epi32(sc.bias_hi[gi]));
                    let cl = _mm256_set1_epi32(sc.coef_lo[gi]);
                    let ch = _mm256_set1_epi32(sc.coef_hi[gi]);
                    // SAFETY: `ke_e`/`ke_o` hold four `i64` per group.
                    unsafe {
                        let ke = _mm256_loadu_si256(p.ke_e.as_ptr().add(at * 4).cast::<__m256i>());
                        let ko = _mm256_loadu_si256(p.ke_o.as_ptr().add(at * 4).cast::<__m256i>());
                        let sh_e = _mm256_add_epi64(ke, _mm256_set1_epi64x(kd));
                        let sh_o = _mm256_add_epi64(ko, _mm256_set1_epi64x(kd));
                        let pe = _mm256_add_epi64(
                            _mm256_mul_epi32(d_lo, cl),
                            _mm256_mul_epi32(d_hi, ch),
                        );
                        let po = _mm256_add_epi64(
                            _mm256_mul_epi32(_mm256_srli_epi64(d_lo, 32), cl),
                            _mm256_mul_epi32(_mm256_srli_epi64(d_hi, 32), ch),
                        );
                        acc_e = _mm256_add_epi64(acc_e, _mm256_sllv_epi64(pe, sh_e));
                        acc_o = _mm256_add_epi64(acc_o, _mm256_sllv_epi64(po, sh_o));
                    }
                    s += 1;
                }
                b += 1;
            }
            let mut ve = [0i64; LANES / 2];
            let mut vo = [0i64; LANES / 2];
            // SAFETY: two four-lane stores into the stack buffers above.
            unsafe {
                _mm256_storeu_si256(ve.as_mut_ptr().cast::<__m256i>(), acc_e);
                _mm256_storeu_si256(vo.as_mut_ptr().cast::<__m256i>(), acc_o);
            }
            let mut i = 0usize;
            while i < LANES / 2 {
                let c0 = lo_c + 2 * i;
                let c1 = c0 + 1;
                if c0 < hi_c {
                    match finish(ve[i], kmin + p.emin[c0]) {
                        Some(v) => out[c0] = v,
                        None => return false,
                    }
                }
                if c1 < hi_c {
                    match finish(vo[i], kmin + p.emin[c1]) {
                        Some(v) => out[c1] = v,
                        None => return false,
                    }
                }
                i += 1;
            }
        }
        g += 1;
    }
    true
}
