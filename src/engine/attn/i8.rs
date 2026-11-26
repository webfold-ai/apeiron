//! The signed-8-bit key/value carrier: per-row codes plus one binary32 scale.
//!
//! # Contract
//!
//! A `head_dim`-wide Q32.32 row `x` becomes
//!
//! ```text
//! s     = max_i |x_i| / 127            (s = 1 for an all-zero row)
//! code_i = clamp(round_ties_even(x_i / s), -127, 127)
//! ```
//!
//! and the read reconstructs `x_i ≈ code_i · s`. The scale is per `(kv_head, position, K|V)`:
//! one value per row, not one per element, so a row costs `head_dim` bytes plus four scale bytes
//! instead of `4 · head_dim`. The codes are signed; `-128` is never stored, so `-code_i` is exact
//! and the magnitude is at most 127.
//!
//! # The group read
//!
//! One call serves all `group = n_head / n_head_kv` query heads of one KV head. The position loop
//! is outer and the group slots are inner, so each K and V row is loaded once and used `group`
//! times: the cache issues `L · n_head_kv · (2 · head_dim + 8)` bytes per position instead of the
//! `n_head / n_head_kv`-times larger per-query-head read.
//!
//! The online-softmax recurrence is the one in
//! [`f32_core::attn_core_f32_strided_scalar`](super::f32_core::attn_core_f32_strided_scalar):
//! running maximum, `den` and accumulator per slot, `alpha = exp(m_prev − m_next)` and
//! `w = exp(score − m_next)` per position. The arithmetic is binary32 `mul`/`add`, not fused, so
//! the scalar body and the AVX2 body perform the same two roundings per term and are
//! bit-identical; `the_vector_source_equals_the_scalar_source` checks that over the boundary
//! lengths.
//!
//! # Trust boundary
//!
//! The scalar body is extracted. The AVX2 body is not: Aeneas does not model `core::arch` or an
//! inline-assembly lane reduce, so it is opaque to the extraction
//! (`proof/boundary-axioms.txt`, SIMD twins) and `tested` equal to the scalar body. rustc's code
//! generation of the scalar body, MXCSR state and the hardware are outside every claim.

use super::AttnError;
use crate::engine::mixer::f32_carrier::{exp_f32_avx2, exp_f32_scalar, q32_to_f32};
use crate::engine::num::fp::F32;

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::wildcard_imports,
    reason = "x86_64 SIMD intrinsics imported en masse for AVX2+FMA"
)]
use std::arch::x86_64::*;

/// The largest code magnitude. `-128` is never stored so that negation is exact.
pub(super) const CODE_MAX: i32 = 127;

/// Quantise one Q32.32 row to signed 8-bit codes, returning the binary32 scale the read
/// multiplies each code by.
///
/// # Errors
///
/// [`AttnError::BadGeometry`] for a zero width or a `codes` length other than `row.len()`, and
/// [`AttnError::Overflow`] when a row word is not finite as binary32. No code is written on
/// `Err`.
///
/// # Complexity
///
/// `O(hd)` time in two passes over the row (one maximum, one quantisation) and no allocation.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the clamp bounds `code` to ±127 before the cast, so the low byte is the value"
)]
pub(super) fn quantise_row(row: &[i64], codes: &mut [i8]) -> Result<F32, AttnError> {
    let hd = row.len();
    if hd == 0 || codes.len() != hd {
        return Err(AttnError::BadGeometry);
    }
    let mut max = F32::ZERO;
    let mut bad = 0u32;
    let mut i = 0usize;
    while i < hd {
        let x = q32_to_f32(row[i]);
        bad += u32::from(!x.is_finite());
        let mag = x.abs();
        if mag.gt(max) {
            max = mag;
        }
        i += 1;
    }
    if bad != 0 {
        return Err(AttnError::Overflow);
    }
    // An all-zero row carries no direction; a unit scale keeps every code zero without a division
    // by zero, and the reconstructed row is still zero.
    let unit = F32::from_i64(i64::from(CODE_MAX));
    let mut scale = if max.le(F32::ZERO) {
        F32::ONE
    } else {
        max / unit
    };
    if scale.le(F32::ZERO) || !scale.is_finite() {
        scale = F32::ONE;
    }
    let mut i = 0usize;
    while i < hd {
        let q = (q32_to_f32(row[i]) / scale).round_ties_even().to_i32_sat();
        codes[i] = q.clamp(-CODE_MAX, CODE_MAX) as i8;
        i += 1;
    }
    Ok(scale)
}

/// `Σ_i q_i · code_i` in binary32, eight independent lanes summed in ascending lane order.
///
/// The lane order is the AVX2 body's: lanes `0..8` accumulate `hd / 8` products each, then the
/// eight lanes fold left to right. A width that is not a multiple of eight takes one serial
/// chain, which only the scalar body can reach.
///
/// # Complexity
///
/// `O(hd)` multiplies and adds; no allocation.
fn dot_codes(q: &[F32], codes: &[i8], hd: usize) -> F32 {
    if hd.is_multiple_of(8) {
        let mut lanes = [F32::ZERO; 8];
        let chunks = hd / 8;
        let mut c = 0usize;
        while c < chunks {
            let mut l = 0usize;
            while l < 8 {
                let code = F32::from_i64(i64::from(codes[c * 8 + l]));
                lanes[l] += q[c * 8 + l] * code;
                l += 1;
            }
            c += 1;
        }
        let mut raw = lanes[0];
        let mut l = 1usize;
        while l < 8 {
            raw += lanes[l];
            l += 1;
        }
        raw
    } else {
        let mut raw = F32::ZERO;
        let mut i = 0usize;
        while i < hd {
            raw += q[i] * F32::from_i64(i64::from(codes[i]));
            i += 1;
        }
        raw
    }
}

/// `acc_i = acc_i · alpha + wv · code_i` for every component.
///
/// # Complexity
///
/// `O(hd)` time; no allocation.
fn axpy_codes(acc: &mut [F32], codes: &[i8], hd: usize, alpha: F32, wv: F32) {
    let mut i = 0usize;
    while i < hd {
        acc[i] = acc[i] * alpha + wv * F32::from_i64(i64::from(codes[i]));
        i += 1;
    }
}

/// Causal attention of `group` query heads over one KV head's int8 rows, scalar body.
///
/// `q` is `group · head_dim` Q32.32-derived binary32 query words, slot `g` at `g · head_dim`;
/// `kc`/`vc` are `count · head_dim` code rows with row `s` at `s · head_dim`; `ks`/`vs` are the
/// `count` per-row scales; `scale` is `1 / sqrt(head_dim)`; `out` receives `group · head_dim`
/// binary32 words. Position `s` contributes `exp(score_s − m)` to slot `g` alone: each slot runs
/// its own maximum, denominator and accumulator.
///
/// # Errors
///
/// [`AttnError::BadGeometry`] for a zero width, a slice that is not the stated length, or an
/// `out`/`q` length other than `group · head_dim`; [`AttnError::Overflow`] for a non-finite score
/// or readout; [`AttnError::ZeroSum`] for a denominator that is not positive and finite.
///
/// # Complexity
///
/// `O(group · head_dim · count)` multiplies and adds, `O(count)` exponentials per slot, and
/// `O(group · head_dim)` space for the accumulators.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::if_not_else,
    reason = "the carrier's geometry and its four planes are the call; the recurrence names follow \
              the f32 core it mirrors, whose refusal arm reads first"
)]
pub(super) fn attn_core_i8_group_scalar(
    q: &[F32],
    group: usize,
    kc: &[i8],
    vc: &[i8],
    ks: &[F32],
    vs: &[F32],
    head_dim: usize,
    count: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    let width = group.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    let rows = count.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if head_dim == 0 || group == 0 {
        return Err(AttnError::BadGeometry);
    }
    if q.len() != width || out.len() != width || kc.len() < rows || vc.len() < rows {
        return Err(AttnError::BadGeometry);
    }
    if ks.len() < count || vs.len() < count {
        return Err(AttnError::BadGeometry);
    }
    let mut acc = vec![F32::ZERO; width];
    let mut m = vec![F32::ZERO; group];
    let mut den = vec![F32::ZERO; group];

    // Position 0 seeds every slot: `m = score`, `den = 1`, `acc = v_row · v_scale`.
    let k_scale0 = ks[0] * scale;
    let mut g = 0usize;
    while g < group {
        let lo = g * head_dim;
        let score = dot_codes(&q[lo..lo + head_dim], &kc[..head_dim], head_dim) * k_scale0;
        if !score.is_finite() {
            return Err(AttnError::Overflow);
        }
        m[g] = score;
        den[g] = F32::ONE;
        let vs0 = vs[0];
        let mut i = 0usize;
        while i < head_dim {
            acc[lo + i] = F32::from_i64(i64::from(vc[i])) * vs0;
            i += 1;
        }
        g += 1;
    }

    let mut ok = true;
    let mut s = 1usize;
    while s < count && ok {
        let row = s * head_dim;
        let k_scale = ks[s] * scale;
        let v_scale = vs[s];
        let mut g = 0usize;
        while g < group && ok {
            let lo = g * head_dim;
            let score =
                dot_codes(&q[lo..lo + head_dim], &kc[row..row + head_dim], head_dim) * k_scale;
            if !score.is_finite() {
                ok = false;
            } else {
                let m_prev = m[g];
                let m_next = if score.gt(m_prev) { score } else { m_prev };
                let alpha = exp_f32_scalar(m_prev - m_next);
                let w = exp_f32_scalar(score - m_next);
                den[g] = den[g] * alpha + w;
                let wv = w * v_scale;
                axpy_codes(
                    &mut acc[lo..lo + head_dim],
                    &vc[row..row + head_dim],
                    head_dim,
                    alpha,
                    wv,
                );
                m[g] = m_next;
            }
            g += 1;
        }
        s += 1;
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    let mut g = 0usize;
    while g < group {
        let d = den[g];
        if d.le(F32::ZERO) || !d.is_finite() {
            return Err(AttnError::ZeroSum);
        }
        let lo = g * head_dim;
        let inv = F32::ONE / d;
        let mut i = 0usize;
        while i < head_dim {
            let res = acc[lo + i] * inv;
            if !res.is_finite() {
                return Err(AttnError::Overflow);
            }
            out[lo + i] = res;
            i += 1;
        }
        g += 1;
    }
    Ok(())
}

/// AVX2 body of [`attn_core_i8_group_scalar`], bit-identical to it.
///
/// Full eight-position tiles score with eight independent dot chains (the latency of one chain is
/// hidden by the others) and take both exponentials as one eight-lane vector; the tail runs the
/// per-position body so no padding value enters the recurrence. The den and accumulator updates
/// keep the scalar body's order and its separate multiply and add, so the two bodies round
/// identically.
///
/// # Safety
///
/// Requires AVX2; every slice must satisfy the scalar body's length contract and `head_dim` must
/// be a multiple of eight.
///
/// # Complexity
///
/// `O(group · head_dim · count / 8)` vector multiply-adds, `O(count / 8)` vector exponentials per
/// slot and `O(group · head_dim)` scratch; no allocation beyond the accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::if_not_else,
    reason = "the carrier's geometry and its four planes are the call; lane names follow the \
              scalar body it matches, including its refusal arm"
)]
unsafe fn attn_core_i8_group_avx2_inner(
    q: &[F32],
    group: usize,
    kc: &[i8],
    vc: &[i8],
    ks: &[F32],
    vs: &[F32],
    head_dim: usize,
    count: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    const TILE: usize = 8;
    let width = group.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    let rows = count.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if head_dim == 0 || group == 0 || !head_dim.is_multiple_of(8) {
        return Err(AttnError::BadGeometry);
    }
    if q.len() != width || out.len() != width || kc.len() < rows || vc.len() < rows {
        return Err(AttnError::BadGeometry);
    }
    if ks.len() < count || vs.len() < count {
        return Err(AttnError::BadGeometry);
    }
    let chunks = head_dim / 8;
    let mut acc = vec![F32::ZERO; width];
    let mut m = vec![F32::ZERO; group];
    let mut den = vec![F32::ZERO; group];

    // SAFETY: the guards above bound every load: `q` and `out` are `width` wide, the code planes
    // are at least `count · head_dim`, and `head_dim` is a positive multiple of eight.
    unsafe {
        // Position 0 seeds every slot: `m = score`, `den = 1`, `acc = v_row · v_scale`.
        let k_scale0 = nat(ks[0] * scale);
        let mut g = 0usize;
        while g < group {
            let lo = g * head_dim;
            let qp = q.as_ptr().add(lo).cast::<f32>();
            let mut dots = _mm256_setzero_ps();
            let mut c = 0usize;
            while c < chunks {
                let vq = _mm256_loadu_ps(qp.add(c * 8));
                let vk = codes_f32(kc.as_ptr().add(c * 8));
                dots = _mm256_add_ps(dots, _mm256_mul_ps(vq, vk));
                c += 1;
            }
            let score = hsum256(dots) * k_scale0;
            if !score.is_finite() {
                return Err(AttnError::Overflow);
            }
            m[g] = fp(score);
            den[g] = F32::ONE;
            let vs0 = nat(vs[0]);
            let vw = _mm256_set1_ps(vs0);
            let mut c = 0usize;
            while c < chunks {
                let vv = codes_f32(vc.as_ptr().add(c * 8));
                let dst = acc.as_mut_ptr().add(lo + c * 8).cast::<f32>();
                _mm256_storeu_ps(dst, _mm256_mul_ps(vw, vv));
                c += 1;
            }
            g += 1;
        }

        let mut s = 1usize;
        while count - s >= TILE {
            let mut g = 0usize;
            while g < group {
                let lo = g * head_dim;
                let qp = q.as_ptr().add(lo).cast::<f32>();
                let mut dots = [_mm256_setzero_ps(); TILE];
                let mut c = 0usize;
                while c < chunks {
                    let vq = _mm256_loadu_ps(qp.add(c * 8));
                    let mut j = 0usize;
                    while j < TILE {
                        let vk = codes_f32(kc.as_ptr().add((s + j) * head_dim + c * 8));
                        dots[j] = _mm256_add_ps(dots[j], _mm256_mul_ps(vq, vk));
                        j += 1;
                    }
                    c += 1;
                }
                // The running maximum is the scalar recurrence over the tile's scores, so each
                // `prev`/`next` pair is the pair the per-position body would have used.
                let mut score = [0.0f32; TILE];
                let mut prev = [0.0f32; TILE];
                let mut next = [0.0f32; TILE];
                let mut run = nat(m[g]);
                let mut j = 0usize;
                while j < TILE {
                    let k_scale = nat(ks[s + j] * scale);
                    score[j] = hsum256(dots[j]) * k_scale;
                    if !score[j].is_finite() {
                        return Err(AttnError::Overflow);
                    }
                    prev[j] = run;
                    if score[j] > run {
                        run = score[j];
                    }
                    next[j] = run;
                    j += 1;
                }
                let valpha = exp_f32_avx2(_mm256_sub_ps(
                    _mm256_loadu_ps(prev.as_ptr()),
                    _mm256_loadu_ps(next.as_ptr()),
                ));
                let vw = exp_f32_avx2(_mm256_sub_ps(
                    _mm256_loadu_ps(score.as_ptr()),
                    _mm256_loadu_ps(next.as_ptr()),
                ));
                let mut alpha = [0.0f32; TILE];
                let mut w = [0.0f32; TILE];
                _mm256_storeu_ps(alpha.as_mut_ptr(), valpha);
                _mm256_storeu_ps(w.as_mut_ptr(), vw);
                let mut d = nat(den[g]);
                let mut j = 0usize;
                while j < TILE {
                    // Separate multiply and add: the scalar body's two roundings, no FMA.
                    d = d * alpha[j] + w[j];
                    j += 1;
                }
                den[g] = fp(d);
                let mut wv = [0.0f32; TILE];
                let mut j = 0usize;
                while j < TILE {
                    wv[j] = w[j] * nat(vs[s + j]);
                    j += 1;
                }
                let mut c = 0usize;
                while c < chunks {
                    let dst = acc.as_mut_ptr().add(lo + c * 8).cast::<f32>();
                    let mut a = _mm256_loadu_ps(dst);
                    let mut j = 0usize;
                    while j < TILE {
                        let vv = codes_f32(vc.as_ptr().add((s + j) * head_dim + c * 8));
                        a = _mm256_add_ps(
                            _mm256_mul_ps(a, _mm256_set1_ps(alpha[j])),
                            _mm256_mul_ps(_mm256_set1_ps(wv[j]), vv),
                        );
                        j += 1;
                    }
                    _mm256_storeu_ps(dst, a);
                    c += 1;
                }
                m[g] = fp(run);
                g += 1;
            }
            s += TILE;
        }

        // Tail: the per-position body, so no padded position enters the recurrence.
        let mut ok = true;
        while s < count && ok {
            let row = s * head_dim;
            let k_scale = nat(ks[s] * scale);
            let v_scale = nat(vs[s]);
            let mut g = 0usize;
            while g < group && ok {
                let lo = g * head_dim;
                let qp = q.as_ptr().add(lo).cast::<f32>();
                let mut dots = _mm256_setzero_ps();
                let mut c = 0usize;
                while c < chunks {
                    let vq = _mm256_loadu_ps(qp.add(c * 8));
                    let vk = codes_f32(kc.as_ptr().add(row + c * 8));
                    dots = _mm256_add_ps(dots, _mm256_mul_ps(vq, vk));
                    c += 1;
                }
                let score = hsum256(dots) * k_scale;
                if !score.is_finite() {
                    ok = false;
                } else {
                    let m_prev = nat(m[g]);
                    let m_next = if score > m_prev { score } else { m_prev };
                    let alpha = nat(exp_f32_scalar(fp(m_prev - m_next)));
                    let w = nat(exp_f32_scalar(fp(score - m_next)));
                    den[g] = fp(nat(den[g]) * alpha + w);
                    let wv = w * v_scale;
                    let va = _mm256_set1_ps(alpha);
                    let vwv = _mm256_set1_ps(wv);
                    let mut c = 0usize;
                    while c < chunks {
                        let vv = codes_f32(vc.as_ptr().add(row + c * 8));
                        let dst = acc.as_mut_ptr().add(lo + c * 8).cast::<f32>();
                        let prev = _mm256_loadu_ps(dst);
                        _mm256_storeu_ps(
                            dst,
                            _mm256_add_ps(_mm256_mul_ps(prev, va), _mm256_mul_ps(vwv, vv)),
                        );
                        c += 1;
                    }
                    m[g] = fp(m_next);
                }
                g += 1;
            }
            s += 1;
        }
        if !ok {
            return Err(AttnError::Overflow);
        }

        let mut g = 0usize;
        while g < group {
            let d = nat(den[g]);
            if d <= 0.0 || !d.is_finite() {
                return Err(AttnError::ZeroSum);
            }
            let inv = 1.0f32 / d;
            let vinv = _mm256_set1_ps(inv);
            let lo = g * head_dim;
            let op = out.as_mut_ptr().add(lo).cast::<f32>();
            let mut bad = 0u32;
            let mut c = 0usize;
            while c < chunks {
                let src = acc.as_ptr().add(lo + c * 8).cast::<f32>();
                let res = _mm256_mul_ps(_mm256_loadu_ps(src), vinv);
                _mm256_storeu_ps(op.add(c * 8), res);
                let mut lanes = [0.0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), res);
                let mut l = 0usize;
                while l < 8 {
                    bad += u32::from(!lanes[l].is_finite());
                    l += 1;
                }
                c += 1;
            }
            if bad != 0 {
                return Err(AttnError::Overflow);
            }
            g += 1;
        }
    }
    Ok(())
}

/// Eight signed code bytes at `src` widened to binary32 lanes.
///
/// # Safety
///
/// `src` must point at eight readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "`_mm_loadl_epi64` reads the low eight bytes; the pointer need not be 16-byte aligned"
)]
unsafe fn codes_f32(src: *const i8) -> __m256 {
    // SAFETY: the caller guarantees eight readable bytes.
    unsafe {
        let bytes = _mm_loadl_epi64(src.cast::<__m128i>());
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes))
    }
}

/// Left-to-right sum of the eight lanes, the order [`dot_codes`] folds in.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` holds eight floats, the width of the store.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
    lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
}

/// Native binary32 view of an [`F32`] for the AVX2 body.
#[cfg(target_arch = "x86_64")]
#[inline]
fn nat(x: F32) -> f32 {
    f32::from_bits(x.to_bits())
}

/// [`F32`] view of a native binary32 value computed inside the AVX2 body.
#[cfg(target_arch = "x86_64")]
#[inline]
fn fp(x: f32) -> F32 {
    F32::from_bits(x.to_bits())
}

/// Dispatch of the group read: AVX2 when the width is a multiple of eight and the processor has
/// it, the scalar body otherwise.
///
/// # Complexity
///
/// As [`attn_core_i8_group_scalar`].
#[allow(
    clippy::too_many_arguments,
    reason = "the carrier's four planes and geometry"
)]
pub(super) fn attn_core_i8_group(
    q: &[F32],
    group: usize,
    kc: &[i8],
    vc: &[i8],
    ks: &[F32],
    vs: &[F32],
    head_dim: usize,
    count: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    #[cfg(target_arch = "x86_64")]
    if head_dim.is_multiple_of(8) && crate::engine::num::fp::prim::cpu_avx2_fma() {
        // SAFETY: `cpu_avx2_fma` checked AVX2 on this processor; the scalar body's guards run
        // again inside the AVX2 body before any load.
        return unsafe {
            attn_core_i8_group_avx2_inner(q, group, kc, vc, ks, vs, head_dim, count, scale, out)
        };
    }
    attn_core_i8_group_scalar(q, group, kc, vc, ks, vs, head_dim, count, scale, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let wide = i64::try_from(*state >> 1).expect("the top 63 bits fit i64");
        wide % (8 << 32) - 4 * (1i64 << 32)
    }

    fn bits(v: &[F32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The scalar and AVX2 bodies are bit-identical over the widths the models use and over the
    /// odd widths the scalar body admits.
    #[test]
    fn the_vector_source_equals_the_scalar_source() {
        if !crate::engine::num::fp::prim::cpu_avx2_fma() {
            return;
        }
        let widths = [8usize, 16, 128, 40, 6];
        let groups = [1usize, 2, 4];
        // Two full eight-position tiles and a four-position tail: both paths run.
        let count = 21usize;
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        for hd in widths {
            for group in groups {
                let width = group * hd;
                let q: Vec<F32> = (0..width)
                    .map(|_| F32::from_i64(lcg(&mut seed) >> 8))
                    .collect();
                let kc: Vec<i8> = (0..count * hd)
                    .map(|i| {
                        let v = i64::try_from(i % 255).expect("small") - 127;
                        i8::try_from(v).expect("one code in range")
                    })
                    .collect();
                let vc: Vec<i8> = (0..count * hd)
                    .map(|i| {
                        let v = i64::try_from((i * 7) % 255).expect("small") - 127;
                        i8::try_from(v).expect("one code in range")
                    })
                    .collect();
                let ks: Vec<F32> = (0..count)
                    .map(|_| F32::from_i64((lcg(&mut seed) >> 40) + 1))
                    .collect();
                let vs: Vec<F32> = (0..count)
                    .map(|_| F32::from_i64((lcg(&mut seed) >> 40) + 1))
                    .collect();
                let scale = F32::from_i64(2);
                let mut scalar = vec![F32::ZERO; width];
                let mut vector = vec![F32::ZERO; width];
                attn_core_i8_group_scalar(
                    &q,
                    group,
                    &kc,
                    &vc,
                    &ks,
                    &vs,
                    hd,
                    count,
                    scale,
                    &mut scalar,
                )
                .expect("scalar body");
                if hd.is_multiple_of(8) {
                    attn_core_i8_group(
                        &q,
                        group,
                        &kc,
                        &vc,
                        &ks,
                        &vs,
                        hd,
                        count,
                        scale,
                        &mut vector,
                    )
                    .expect("vector body");
                    assert_eq!(bits(&scalar), bits(&vector), "width {hd}, group {group}");
                }
            }
        }
    }

    /// Quantisation reconstructs the row within one scale step and never emits `-128`.
    #[test]
    fn the_quantiser_bounds_the_row_error() {
        let mut seed = 0x0BAD_F00D_DEAD_10CCu64;
        let row: Vec<i64> = (0..128).map(|_| lcg(&mut seed)).collect();
        let mut codes = vec![0i8; row.len()];
        let scale = f32::from_bits(quantise_row(&row, &mut codes).expect("quantise").to_bits());
        let mut worst = 0.0f32;
        let mut i = 0usize;
        while i < row.len() {
            let original = f32::from_bits(q32_to_f32(row[i]).to_bits());
            let reconstructed = f32::from(codes[i]) * scale;
            let err = (original - reconstructed).abs();
            assert!(codes[i] >= -127, "the code is never -128");
            if err > worst {
                worst = err;
            }
            i += 1;
        }
        assert!(worst <= scale * 0.5 + 1e-6, "worst {worst} step {scale}");
    }

    /// An all-zero row takes the unit scale and all-zero codes.
    #[test]
    fn the_zero_row_is_all_zero_codes() {
        let row = vec![0i64; 32];
        let mut codes = vec![9i8; 32];
        let scale = quantise_row(&row, &mut codes).expect("quantise");
        assert_eq!(scale.to_bits(), F32::ONE.to_bits());
        assert!(codes.iter().all(|c| *c == 0));
    }

    /// The extreme Q32.32 words take the full code range and a finite scale.
    #[test]
    fn the_extreme_words_quantise_to_finite_codes() {
        // A Q32.32 word is at most 2^31 in binary32 (`from_i64` then `/ 2^32`), so no word is
        // non-finite; the largest magnitudes must still take the full code range and a finite
        // scale.
        let mut row = vec![0i64; 8];
        row[3] = i64::MAX;
        row[5] = i64::MIN;
        let mut codes = vec![0i8; 8];
        let scale = quantise_row(&row, &mut codes).expect("quantise");
        assert!(scale.is_finite());
        assert!(scale.gt(F32::ZERO));
        assert!(codes.iter().all(|c| *c >= -127));
        assert!(codes[3] > 100, "positive extreme took {}", codes[3]);
        assert!(codes[5] < -100, "negative extreme took {}", codes[5]);
    }
}
