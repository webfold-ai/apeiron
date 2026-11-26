//! IEEE binary32 (f32) carrier for the softmax attention core (Milakov & Gimelshein 2018).
//!
//! # Contract
//!
//! Computes causal scaled dot-product attention for one query head over positions `0..=pos`:
//! 1. Scores `score_s = (q · k_s) · scale` for each causal key `s ∈ 0..=pos`.
//! 2. Online running-max softmax: maintains running maximum `m`, denominator `den = Σ exp(score_s - m)`,
//!    and numerator accumulator `acc = Σ exp(score_s - m) · v_s`.
//! 3. Normalized readout: `out = acc / den`.
//!
//! # Conversions and Refusals
//!
//! - Input conversion: Q32.32 fixed-point words `x` map to binary32 via `(x as f32) / 2^32`.
//! - Readout conversion: binary32 output values map to Q32.32 via round-to-nearest (`round() · 2^32`).
//! - Refusals:
//!   - Non-finite floating-point inputs (NaN, ±Inf) or outputs exceeding the Q32.32 range refuse with
//!     [`AttnError::Overflow`].
//!   - Non-positive or non-finite softmax denominator sums refuse with [`AttnError::ZeroSum`].
//!   - Geometry violations (empty head dimension, row length mismatches) refuse with [`AttnError::BadGeometry`].
//!
//! # Trust Boundary
//!
//! The binary32 kernels (scalar twin and AVX2+FMA) are not Aeneas-extracted. The prepass now
//! admits both bodies — the loop-shape refusals are cleared — but the pinned Aeneas then stops in
//! its module analysis with an internal assert before emitting Lean, so extraction is not
//! established. Correctness relies on
//! IEEE 754-2019 standard conformance, exact bit-for-bit identity between the AVX2 kernel and its
//! scalar twin specification, test-measured bounded relative error on the polynomial exponential,
//! and bounded deviation against the Q32.32 fixed-point oracle.

#![allow(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "bit-cast views are one instruction; the scalar and strided cores take the kernel operands"
)]

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::wildcard_imports,
    reason = "x86_64 SIMD intrinsics imported en masse for AVX2+FMA"
)]
use std::arch::x86_64::*;

use crate::engine::attn::AttnError;
#[cfg(target_arch = "x86_64")]
use crate::engine::mixer::f32_carrier::exp_f32_avx2;
use crate::engine::mixer::f32_carrier::exp_f32_scalar;
use crate::engine::num::fp::F32;

/// Native binary32 view of an [`F32`] for the AVX2 inners, which keep `f32` intrinsic bodies.
#[inline(always)]
fn nat(x: F32) -> f32 {
    f32::from_bits(x.to_bits())
}

/// [`F32`] view of a native binary32 value computed inside an AVX2 inner.
#[inline(always)]
fn fp(x: f32) -> F32 {
    F32::from_bits(x.to_bits())
}

/// Reinterprets a `&[F32]` as `&[f32]`. `F32` is `repr(transparent)` over `u32` with the layout of
/// `f32`, so the pointer and length carry over unchanged; every bit pattern is a valid `f32`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn native(s: &[F32]) -> &[f32] {
    // SAFETY: `F32` is `repr(transparent)` over `u32`, which has the size and alignment of `f32`;
    // the slice length counts the same elements.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<f32>(), s.len()) }
}

/// Mutable form of [`native`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn native_mut(s: &mut [F32]) -> &mut [f32] {
    // SAFETY: as for `native`; the exclusive borrow is passed through unchanged.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f32>(), s.len()) }
}

/// Convert a binary32 float back to Q32.32 fixed-point `i64` using round-to-nearest.
///
/// Refuses non-finite floats (NaN, ±Inf) and values outside the representable `i64` Q32.32 range
/// `[-2^31, 2^31)` with [`AttnError::Overflow`].
///
/// # Complexity
///
/// O(1) temporal and spatial: 1 float test, 1 multiplication, 1 rounding, 1 range check.
#[inline]
pub fn f32_to_q32_checked(v: F32) -> Result<i64, AttnError> {
    // `None` covers non-finite input and a result outside `[-2^63, 2^63)`: both were `Overflow`.
    v.to_q32().ok_or(AttnError::Overflow)
}

/// Compute scaled dot product `(q · k) · scale` in scalar arithmetic with fixed 8-lane reduction.
#[inline]
#[allow(
    clippy::needless_range_loop,
    reason = "fixed lane loop order pins execution order"
)]
fn dot_f32_scalar(q: &[F32], k: &[F32], scale: F32) -> F32 {
    let hd = q.len();
    if hd.is_multiple_of(8) {
        let mut lanes = [F32::ZERO; 8];
        let chunks = hd / 8;
        let mut c = 0usize;
        while c < chunks {
            let mut lane = 0usize;
            while lane < 8 {
                lanes[lane] = q[c * 8 + lane].fma(k[c * 8 + lane], lanes[lane]);
                lane += 1;
            }
            c += 1;
        }
        let raw =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        raw * scale
    } else {
        let mut raw = F32::ZERO;
        let mut i = 0usize;
        while i < hd {
            raw = q[i].fma(k[i], raw);
            i += 1;
        }
        raw * scale
    }
}

/// Compute scaled dot product `(q · k) · scale` using AVX2+FMA.
/// Matches [`dot_f32_scalar`] bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_avx2(q: &[f32], k: &[f32], scale: f32) -> f32 {
    let hd = q.len();
    let chunks = hd / 8;
    let mut v_dot = _mm256_setzero_ps();
    for c in 0..chunks {
        let v_q = _mm256_loadu_ps(q.as_ptr().add(c * 8));
        let v_k = _mm256_loadu_ps(k.as_ptr().add(c * 8));
        v_dot = _mm256_fmadd_ps(v_q, v_k, v_dot);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v_dot);
    let raw = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    raw * scale
}

/// Scalar twin implementation of the binary32 online-softmax attention core with strided K/V layout.
///
/// Evaluates causal attention over positions `0..=pos` using the Milakov & Gimelshein 2018
/// online normalizer recurrence. Serves as the specification for [`attn_core_f32_strided_avx2_inner`].
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time in floating-point multiply-adds (FMAs) and `O(pos + 1)`
/// exponential evaluations. Zero heap allocation for `head_dim <= 256` (stack buffer).
#[allow(
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, fixed lane loop and explicit strided geometry"
)]
pub fn attn_core_f32_strided_scalar(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || out.len() != head_dim || q.len() != head_dim || stride < head_dim {
        return Err(AttnError::BadGeometry);
    }
    let count = pos.checked_add(1).ok_or(AttnError::Overflow)?;
    let required_len = pos
        .checked_mul(stride)
        .and_then(|p| p.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required_len || v.len() < required_len {
        return Err(AttnError::BadGeometry);
    }

    let score0 = dot_f32_scalar(q, &k[..head_dim], scale);
    if !score0.is_finite() {
        return Err(AttnError::Overflow);
    }

    // Each branch lends one buffer to the recurrence: a join of a stack and a heap `&mut` is refused
    // by the pinned Aeneas model, so the buffer is a parameter of `attn_core_f32_scalar_rows`.
    if head_dim <= 256 {
        let mut acc_stack = [F32::ZERO; 256];
        attn_core_f32_scalar_rows(
            q,
            k,
            v,
            head_dim,
            stride,
            count,
            score0,
            scale,
            &mut acc_stack[..head_dim],
            out,
        )
    } else {
        let mut acc_heap = vec![F32::ZERO; head_dim];
        attn_core_f32_scalar_rows(
            q,
            k,
            v,
            head_dim,
            stride,
            count,
            score0,
            scale,
            &mut acc_heap[..],
            out,
        )
    }
}

/// Recurrence of [`attn_core_f32_strided_scalar`] after the first score, over a caller-owned
/// accumulator `acc` of length `head_dim`. Same operations, order and early exits as the body it was
/// split from.
#[allow(
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, fixed lane loop and explicit strided geometry"
)]
fn attn_core_f32_scalar_rows(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    count: usize,
    score0: F32,
    scale: F32,
    acc: &mut [F32],
    out: &mut [F32],
) -> Result<(), AttnError> {
    let mut m = score0;
    let mut den = F32::ONE;

    // One `ok` flag carries every in-loop `Overflow` refusal out of the three loops below, and the
    // two `return Err` sites that consume it stand after the position loop and after the readout
    // loop. This is forced by the pinned Aeneas model, which refuses this body with `Returns inside
    // of nested loops are not supported yet`: each `return Err(AttnError::Overflow)` sat inside a
    // loop body, and an early exit from a loop is the shape its translator cannot lower. No `zip`
    // appears in this body, so that refusal does not apply here. The flag keeps `Overflow` ahead of
    // the `ZeroSum` test, because the position loop's `if !ok` return precedes that test.
    //
    // `while i < head_dim && ok` writes the same prefix and stops at the same index as the `return`
    // it replaces, so `acc` and `out` carry exactly what the early exits left behind: an element is
    // stored only when the old code would have stored it before returning. Arithmetic is unchanged:
    // the same `v[i]`, `v_row[i]`, `acc[i].mul_add(alpha, w * v_row[i])` and `acc[i] * inv_den` on
    // the same elements in the same ascending order, over `0..head_dim` and `1..count`, which are
    // exactly the old `for` ranges. `m = m_next` stays guarded by `ok` because the old code reached
    // it only after a whole finite row, and `den` keeps its single `mul_add` per position. The
    // residual cost is bounded: at most one extra `head_dim` scan per failing position, since `ok`
    // stops the inner loop at the same index the `return` did.
    let mut ok = true;

    let mut i = 0usize;
    while i < head_dim && ok {
        let val = v[i];
        if val.is_finite() {
            acc[i] = val;
        } else {
            ok = false;
        }
        i += 1;
    }

    let mut s = 1usize;
    while s < count && ok {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_row = &v[base..base + head_dim];

        let score_s = dot_f32_scalar(q, k_row, scale);
        if score_s.is_finite() {
            let m_prev = m;
            let m_next = if score_s.gt(m_prev) { score_s } else { m_prev };
            let alpha = exp_f32_scalar(m_prev - m_next);
            let w = exp_f32_scalar(score_s - m_next);

            den = den.fma(alpha, w);

            let mut i = 0usize;
            while i < head_dim && ok {
                let vi = v_row[i];
                if vi.is_finite() {
                    let prod = w * vi;
                    acc[i] = acc[i].fma(alpha, prod);
                } else {
                    ok = false;
                }
                i += 1;
            }
            if ok {
                m = m_next;
            }
        } else {
            ok = false;
        }
        s += 1;
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    if den.le(F32::ZERO) || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = F32::ONE / den;
    let mut i = 0usize;
    while i < head_dim && ok {
        let res = acc[i] * inv_den;
        if res.is_finite() {
            out[i] = res;
        } else {
            ok = false;
        }
        i += 1;
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    Ok(())
}

/// Key positions scored together by [`attn_tile8_avx2`].
const TILE: usize = 8;

/// Score, softmax-update and accumulate the `TILE` key positions `s..s + TILE` (`s >= 1`).
///
/// Bit-identical to `TILE` consecutive iterations of the single-position loop in
/// [`attn_core_f32_strided_avx2_inner`]: each score is the same 8-lane FMA chain (the eight chains
/// run interleaved, so the latency of one is hidden by the others), the running maximum is the
/// same sequential recurrence, the two `exp` calls per position are the lane-wise
/// [`exp_f32_avx2`] (equal to [`exp_f32_scalar`] bit for bit), the denominator is the same
/// sequential `mul_add` chain, and each accumulator chunk takes the same `fma(acc, alpha_j, w_j *
/// v_j)` sequence in position order (chunk-outer, position-inner, so the chunk stays in a register
/// across the tile). A non-finite score refuses with [`AttnError::Overflow`] before any of the
/// tile's state is written.
///
/// # Complexity
///
/// `O(TILE · head_dim)` FMAs and `O(1)` vector `exp` calls per tile: per position `head_dim / 8`
/// FMAs for the score and `2 · head_dim / 8` vector ops for the accumulator. No allocation.
///
/// # Safety
///
/// Requires AVX2 and FMA; `k` and `v` hold rows `0..s + TILE` at `stride`, each `head_dim` wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "explicit strided geometry, lane loops pin the operation order"
)]
unsafe fn attn_tile8_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    s: usize,
    scale: f32,
    m: &mut f32,
    den: &mut f32,
    acc_vec: &mut [__m256],
) -> Result<(), AttnError> {
    let chunks = head_dim / 8;
    let kp = k.as_ptr().add(s * stride);
    let vp = v.as_ptr().add(s * stride);
    let qp = q.as_ptr();
    let mut dots = [_mm256_setzero_ps(); TILE];
    for c in 0..chunks {
        let vq = _mm256_loadu_ps(qp.add(c * 8));
        for j in 0..TILE {
            let vk = _mm256_loadu_ps(kp.add(j * stride + c * 8));
            dots[j] = _mm256_fmadd_ps(vq, vk, dots[j]);
        }
    }
    let mut score = [0.0f32; TILE];
    for j in 0..TILE {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), dots[j]);
        let raw =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        score[j] = raw * scale;
        if !score[j].is_finite() {
            return Err(AttnError::Overflow);
        }
    }
    let mut prev = [0.0f32; TILE];
    let mut next = [0.0f32; TILE];
    let mut run = *m;
    for j in 0..TILE {
        prev[j] = run;
        if score[j] > run {
            run = score[j];
        }
        next[j] = run;
    }
    let vnext = _mm256_loadu_ps(next.as_ptr());
    let valpha = exp_f32_avx2(_mm256_sub_ps(_mm256_loadu_ps(prev.as_ptr()), vnext));
    let vw = exp_f32_avx2(_mm256_sub_ps(_mm256_loadu_ps(score.as_ptr()), vnext));
    let mut alpha = [0.0f32; TILE];
    let mut w = [0.0f32; TILE];
    _mm256_storeu_ps(alpha.as_mut_ptr(), valpha);
    _mm256_storeu_ps(w.as_mut_ptr(), vw);
    let mut d = *den;
    for j in 0..TILE {
        d = d.mul_add(alpha[j], w[j]);
    }
    for c in 0..chunks {
        let mut a = acc_vec[c];
        for j in 0..TILE {
            let vv = _mm256_loadu_ps(vp.add(j * stride + c * 8));
            let prod = _mm256_mul_ps(_mm256_set1_ps(w[j]), vv);
            a = _mm256_fmadd_ps(a, _mm256_set1_ps(alpha[j]), prod);
        }
        acc_vec[c] = a;
    }
    *m = run;
    *den = d;
    Ok(())
}

/// AVX2+FMA SIMD implementation of the binary32 online-softmax attention core with strided K/V layout.
///
/// Processes 8 value components per vector. Matches [`attn_core_f32_strided_scalar`] bit for bit.
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time in 8-lane AVX2+FMA vector instructions and `O(pos + 1)`
/// scalar exponential evaluations. Zero heap allocations for `head_dim <= 256` (register/stack accumulators).
///
/// # Safety
///
/// Requires the target processor to support AVX2 and FMA feature sets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, vector variables and explicit strided geometry"
)]
unsafe fn attn_core_f32_strided_avx2_inner(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    let count = pos.checked_add(1).ok_or(AttnError::Overflow)?;
    let chunks = head_dim / 8;

    let score0 = dot_f32_avx2(q, &k[..head_dim], scale);
    if !score0.is_finite() {
        return Err(AttnError::Overflow);
    }

    let mut m = score0;
    let mut den = 1.0f32;

    let mut acc_stack = [_mm256_setzero_ps(); 32];
    let mut acc_heap;
    let acc_vec: &mut [__m256] = if chunks <= 32 {
        &mut acc_stack[..chunks]
    } else {
        acc_heap = vec![_mm256_setzero_ps(); chunks];
        &mut acc_heap[..]
    };

    for c in 0..chunks {
        let v_val = _mm256_loadu_ps(v.as_ptr().add(c * 8));
        acc_vec[c] = v_val;
    }

    // The two in-loop early exits become one `ok` flag and one `err`, both consumed after the
    // position loops, for the reason asserted on the scalar twin: the pinned Aeneas model refuses
    // this body with `Returns inside of nested loops are not supported yet`, and the `?` on
    // [`attn_tile8_avx2`] and the `return Err(AttnError::Overflow)` on a non-finite tail score were
    // both early exits from a loop body. No `zip` appears in this body, so that refusal does not
    // apply here.
    //
    // Neither loop is reassociated and no operation is reordered: the tile call still runs at
    // positions `1, 1 + TILE, ...` under the same `s + TILE <= count` bound, the tail still runs at
    // `s` up to `count - 1`, and the per-position chain is the same `mul_add` on `den`, the same
    // `_mm256_set1_ps` broadcasts of `alpha` and `w`, and the same `_mm256_fmadd_ps` chain over
    // `0..chunks` in ascending order. The `err` slot preserves the tile call's error, which is
    // always [`AttnError::Overflow`] because [`attn_tile8_avx2`] writes no tile state on its refusal
    // path, and its `if let Some(e) = err` return precedes the `den` test, so the refusal order is
    // the old one. The two `while` guards keep each loop from re-entering after a refusal, which is
    // what the old `?` and `return` did.
    let mut ok = true;
    let mut err: Option<AttnError> = None;

    let mut s = 1usize;
    while s + TILE <= count && ok {
        match attn_tile8_avx2(
            q, k, v, head_dim, stride, s, scale, &mut m, &mut den, acc_vec,
        ) {
            Ok(()) => s += TILE,
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
    }
    while s < count && ok {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_ptr = v.as_ptr().add(base);

        let score_s = dot_f32_avx2(q, k_row, scale);
        if score_s.is_finite() {
            let m_prev = m;
            let m_next = if score_s > m_prev { score_s } else { m_prev };
            let alpha = nat(exp_f32_scalar(fp(m_prev - m_next)));
            let w = nat(exp_f32_scalar(fp(score_s - m_next)));

            den = den.mul_add(alpha, w);

            let v_alpha = _mm256_set1_ps(alpha);
            let v_w = _mm256_set1_ps(w);

            for c in 0..chunks {
                let v_val = _mm256_loadu_ps(v_ptr.add(c * 8));
                let prod = _mm256_mul_ps(v_w, v_val);
                acc_vec[c] = _mm256_fmadd_ps(acc_vec[c], v_alpha, prod);
            }
            m = m_next;
        } else {
            ok = false;
        }
        s += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    if den <= 0.0 || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = 1.0f32 / den;
    let v_inv_den = _mm256_set1_ps(inv_den);

    for c in 0..chunks {
        let res = _mm256_mul_ps(acc_vec[c], v_inv_den);
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), res);
    }

    // The `Overflow` refusal that the old `for item in out.iter().take(head_dim)` raised from
    // inside its own loop body now leaves as a counter tested after it, for the same asserted
    // reason. The check reads `out` and writes nothing, so `out` carries the same `head_dim` values
    // on the error path as it did when the loop returned at the first non-finite element: the vector
    // store loop above has already written them all. The index range `0..head_dim` and the ascending
    // order are the `take(head_dim)` range and order.
    let mut bad = 0u32;
    let mut i = 0usize;
    while i < head_dim {
        bad += u32::from(!out[i].is_finite());
        i += 1;
    }
    if bad != 0 {
        return Err(AttnError::Overflow);
    }

    Ok(())
}

/// Dispatches strided attention core execution to AVX2+FMA when supported and `head_dim % 8 == 0`,
/// falling back to the scalar implementation.
///
/// # Errors
///
/// [`AttnError`] on a geometry mismatch or a non-finite output.
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time and zero heap allocation for `head_dim <= 256`.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit strided geometry and buffers passed to core kernel"
)]
pub fn attn_core_f32_strided(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || out.len() != head_dim || q.len() != head_dim || stride < head_dim {
        return Err(AttnError::BadGeometry);
    }
    let required_len = pos
        .checked_mul(stride)
        .and_then(|p| p.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required_len || v.len() < required_len {
        return Err(AttnError::BadGeometry);
    }

    #[cfg(target_arch = "x86_64")]
    if head_dim.is_multiple_of(8) && crate::engine::num::fp::prim::cpu_avx2_fma() {
        // SAFETY: `cpu_avx2_fma` checked AVX2 and FMA support on this processor.
        return unsafe {
            attn_core_f32_strided_avx2_inner(
                native(q),
                native(k),
                native(v),
                head_dim,
                stride,
                pos,
                nat(scale),
                native_mut(out),
            )
        };
    }
    attn_core_f32_strided_scalar(q, k, v, head_dim, stride, pos, scale, out)
}

/// Causal scaled dot-product attention for a tile of query positions over one K/V plane: each key
/// and value row is loaded once per tile and folded against every query head of every position in
/// the tile.
///
/// Query head `g` of position `j` occupies `q[(j·query_heads + g)·head_dim..][..head_dim]`, has
/// last position `pos[j]` (strictly ascending, as consecutive causal columns are) and writes the
/// same slot of `out`. The tile is llama.cpp's `flash_attn_ext_tiled` loop shape
/// (`ggml/src/ggml-cpu/ops.cpp:8641`): a [`KV_TILE`]-position key/value tile is packed transposed,
/// the tile's scores are one GEMM against the tile's query rows, one row maximum over the tile's
/// causal prefix rescales the denominator and the value accumulator, the tile's probabilities are
/// one vector `exp`, and the value update is one GEMM. The score and value arithmetic per element
/// is the per-column [`attn_core_f32_strided`]'s, but the tile rescales once per tile instead of
/// once per position, so the rounding differs and the per-column core is the oracle the tests
/// bound, not a word-for-word identity. The one `exp` whose argument is exactly `+0` is still
/// evaluated by the shared `exp` kernel.
///
/// # Errors
///
/// [`AttnError::BadGeometry`] for a shape mismatch or positions that are not strictly ascending,
/// [`AttnError::Overflow`] for a non-finite score, value or output, and [`AttnError::ZeroSum`] for
/// a non-positive or non-finite denominator. A refusal on the tile aborts the whole tile. Every
/// value row up to the last position is tested, so a non-finite value row at a position no query
/// in the tile reaches still refuses the tile where those queries' single-query calls would
/// succeed; where a query does read the row, the tile may report a different query's fault first,
/// because its score pass covers the whole tile before the per-query denominators are tested. The
/// step discards the block on any refusal.
///
/// # Complexity
///
/// `O((pos[nq-1] + 1) · nq · query_heads · head_dim)` floating-point operations for the tile,
/// against `Σ_j (pos[j] + 1) · query_heads · head_dim` for the per-query calls — the same
/// arithmetic, with each K and V row read once per tile (`O((pos[nq-1] + 1) · head_dim)` words)
/// instead of once per query. Scratch `O(nq · query_heads · (KV_TILE + head_dim))` words.
/// The K/V row source of the tiled core: native binary32 rows, or signed-8-bit codes with one
/// binary32 scale per row (the [`KvCarrier::I8`](crate::engine::attn::KvCarrier) plane).
///
/// Both arms place row `s` at `s · stride`. The i8 plane is KV-head-major with contiguous
/// `head_dim` rows, so it is read with `stride = head_dim` and its scale plane indexed by the same
/// row number.
#[derive(Clone, Copy)]
enum TileRows<'a> {
    F32 {
        k: &'a [F32],
        v: &'a [F32],
        stride: usize,
    },
    I8 {
        k: &'a [i8],
        v: &'a [i8],
        ks: &'a [F32],
        vs: &'a [F32],
        stride: usize,
    },
}

impl TileRows<'_> {
    /// The row pitch of both planes.
    fn stride(&self) -> usize {
        match self {
            Self::F32 { stride, .. } | Self::I8 { stride, .. } => *stride,
        }
    }

    /// Checks that both planes hold every word of row `last`, and the i8 scale planes row `last`.
    fn check(&self, last: usize, head_dim: usize) -> Result<(), AttnError> {
        let required = last
            .checked_mul(self.stride())
            .and_then(|b| b.checked_add(head_dim))
            .ok_or(AttnError::Overflow)?;
        match self {
            Self::F32 { k, v, .. } => {
                if k.len() < required || v.len() < required {
                    return Err(AttnError::BadGeometry);
                }
            }
            Self::I8 { k, v, ks, vs, .. } => {
                if k.len() < required || v.len() < required || ks.len() <= last || vs.len() <= last
                {
                    return Err(AttnError::BadGeometry);
                }
            }
        }
        Ok(())
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "explicit strided geometry and the four tile operands"
)]
pub fn attn_core_f32_tile(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_tile(
        q,
        TileRows::F32 { k, v, stride },
        head_dim,
        query_heads,
        pos,
        scale,
        out,
    )
}

/// [`attn_core_f32_tile`] over signed-8-bit code rows: the int8 carrier's tiled entry.
///
/// `k`/`v` hold the code rows at `row · stride` and `ks`/`vs` one binary32 scale per row. The key
/// scale multiplies the score after the dot (`dot · (ks[row] · scale)`), and the value scale
/// multiplies the probability before the value dot (`(w · vs[row]) · code`); that is the per-column
/// int8 group core's product order. The tiled recurrence otherwise matches
/// [`attn_core_f32_tile`]'s and carries the same bound.
///
/// # Errors
///
/// [`AttnError::BadGeometry`], [`AttnError::Overflow`] and [`AttnError::ZeroSum`] as
/// [`attn_core_f32_tile`], plus a refusal for a non-finite scale.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit strided geometry and the four tile operands"
)]
pub fn attn_core_i8_tile(
    q: &[F32],
    k: &[i8],
    v: &[i8],
    ks: &[F32],
    vs: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_tile(
        q,
        TileRows::I8 {
            k,
            v,
            ks,
            vs,
            stride,
        },
        head_dim,
        query_heads,
        pos,
        scale,
        out,
    )
}

/// Dispatches the tiled core over a row source: the AVX2 body when the host supports it and the
/// head width is a whole number of vectors, else the scalar twin.
fn attn_core_tile(
    q: &[F32],
    kv: TileRows<'_>,
    head_dim: usize,
    query_heads: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || query_heads == 0 || kv.stride() < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    #[cfg(target_arch = "x86_64")]
    if head_dim.is_multiple_of(8) && crate::engine::num::fp::prim::cpu_avx2_fma() {
        // SAFETY: `cpu_avx2_fma` checked AVX2 and FMA support on this processor; the geometry and
        // the row planes are checked again by the body it calls.
        return unsafe { attn_core_tile_avx2_inner(q, kv, head_dim, query_heads, pos, scale, out) };
    }
    attn_core_tile_scalar(q, kv, head_dim, query_heads, pos, scale, out)
}

/// Key positions per score/value tile in [`attn_core_f32_tile`]: llama.cpp's `GGML_FA_TILE_KV`
/// (`ggml/src/ggml-cpu/common.h`, 64). One tile holds `rows · KV_TILE` scores and reads and writes
/// the value accumulator once, so the accumulator stays in registers across the tile instead of
/// being round-tripped once per position.
const KV_TILE: usize = 64;

/// Scalar twin and specification of [`attn_core_f32_tile`]. Performs the same geometry checks.
///
/// Per key tile of [`KV_TILE`] positions: pack K transposed and V row-major, score every
/// `(row, key)` pair by a sequential FMA over `head_dim`, scale, take the row's causal maximum,
/// rescale `den` and the value accumulator by `exp(m_prev − m_next)` when the maximum moved, write
/// `exp(score − m_next)` as the tile's probabilities, and accumulate `Σ_t P[row][t] · V[t][·]`
/// sequentially over the tile. The AVX2 body runs the two GEMMs with a 6×2 microkernel in the same
/// accumulation order, so the two produce the same words.
///
/// The per-tile rescale is llama.cpp's `flash_attn_ext_tiled` (`ggml/src/ggml-cpu/ops.cpp`): one
/// rescale per tile instead of one per position. That is the same real-valued recurrence with
/// different rounding from [`attn_core_f32_strided`]'s per-position form; the per-column core
/// remains the oracle, and the tests assert the bound between them.
///
/// # Errors
///
/// As [`attn_core_f32_tile`].
///
/// # Complexity
///
/// `O((last + 1) · rows · head_dim)` time and
/// `O(rows · (KV_TILE + head_dim) + KV_TILE · head_dim)` scratch words.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "mathematical recurrence symbols, explicit strided geometry, and the pass structure"
)]
#[allow(dead_code, reason = "public scalar twin specification and test target")]
pub fn attn_core_f32_tile_scalar(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_tile_scalar(
        q,
        TileRows::F32 { k, v, stride },
        head_dim,
        query_heads,
        pos,
        scale,
        out,
    )
}

/// [`attn_core_f32_tile_scalar`] over signed-8-bit code rows; see [`attn_core_i8_tile`].
#[allow(
    dead_code,
    clippy::too_many_arguments,
    reason = "public scalar twin specification and test target; explicit strided geometry"
)]
pub fn attn_core_i8_tile_scalar(
    q: &[F32],
    k: &[i8],
    v: &[i8],
    ks: &[F32],
    vs: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_tile_scalar(
        q,
        TileRows::I8 {
            k,
            v,
            ks,
            vs,
            stride,
        },
        head_dim,
        query_heads,
        pos,
        scale,
        out,
    )
}

/// The tiled scalar body, shared by the `f32` and `i8` row sources.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "mathematical recurrence symbols, explicit strided geometry, and the pass structure"
)]
fn attn_core_tile_scalar(
    q: &[F32],
    kv: TileRows<'_>,
    head_dim: usize,
    query_heads: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || query_heads == 0 || kv.stride() < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    let nq = pos.len();
    let rows = nq.checked_mul(query_heads).ok_or(AttnError::Overflow)?;
    let need = rows.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if q.len() != need || out.len() != need {
        return Err(AttnError::BadGeometry);
    }
    let mut j = 1usize;
    while j < nq {
        if pos[j] <= pos[j - 1] {
            return Err(AttnError::BadGeometry);
        }
        j += 1;
    }
    kv.check(pos[nq - 1], head_dim)?;

    let score_len = rows.checked_mul(KV_TILE).ok_or(AttnError::Overflow)?;
    let tile_len = KV_TILE.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    let mut m = vec![F32::NEG_INFINITY; rows];
    let mut den = vec![F32::ZERO; rows];
    let mut acc = vec![F32::ZERO; need];
    let mut kt = vec![F32::ZERO; tile_len];
    let mut vt = vec![F32::ZERO; tile_len];
    let mut p = vec![F32::ZERO; score_len];
    // The per-row key scale (already times `scale`) and value scale; the f32 arm leaves them at
    // `scale` and one, so its score and value products are unchanged.
    let mut kscale = vec![scale; KV_TILE];
    let mut vscale = vec![F32::ONE; KV_TILE];
    let scaled_values = matches!(kv, TileRows::I8 { .. });
    let stride = kv.stride();

    let last = pos[nq - 1];
    let mut at0 = 0usize;
    while at0 <= last {
        let kv_tile = (last + 1 - at0).min(KV_TILE);
        if let TileRows::I8 { ks, vs, .. } = kv {
            let mut t = 0usize;
            while t < kv_tile {
                let kk = ks[at0 + t] * scale;
                let vv = vs[at0 + t];
                if !kk.is_finite() || !vv.is_finite() {
                    return Err(AttnError::Overflow);
                }
                kscale[t] = kk;
                vscale[t] = vv;
                t += 1;
            }
        }
        // Pack K transposed (`kt[d · KV_TILE + t]`) and V row-major. Every V word up to the tile
        // end is checked here, as the per-position tile checked each row it read: a non-finite
        // value refuses the tile even at a position no query in it reaches.
        let mut t = 0usize;
        while t < kv_tile {
            let base = (at0 + t) * stride;
            let mut d = 0usize;
            match kv {
                TileRows::F32 { k, v, .. } => {
                    while d < head_dim {
                        let vi = v[base + d];
                        if !vi.is_finite() {
                            return Err(AttnError::Overflow);
                        }
                        kt[d * KV_TILE + t] = k[base + d];
                        vt[t * head_dim + d] = vi;
                        d += 1;
                    }
                }
                TileRows::I8 { k, v, .. } => {
                    // Codes are exact in binary32; the per-row scale is checked above.
                    while d < head_dim {
                        kt[d * KV_TILE + t] = F32::from_i64(i64::from(k[base + d]));
                        vt[t * head_dim + d] = F32::from_i64(i64::from(v[base + d]));
                        d += 1;
                    }
                }
            }
            t += 1;
        }
        while t < KV_TILE {
            let mut d = 0usize;
            while d < head_dim {
                kt[d * KV_TILE + t] = F32::ZERO;
                vt[t * head_dim + d] = F32::ZERO;
                d += 1;
            }
            t += 1;
        }

        // Columns ascend with the row, so the rows whose prefix reaches `at0` are a suffix.
        let mut lo = 0usize;
        while lo < rows && pos[lo / query_heads] < at0 {
            lo += 1;
        }
        let mut r = lo;
        while r < rows {
            let cut = pos[r / query_heads];
            let end = kv_tile.min(cut + 1 - at0);
            let qrow = &q[r * head_dim..(r + 1) * head_dim];
            let pbase = r * KV_TILE;
            let mut tmax = F32::NEG_INFINITY;
            let mut t = 0usize;
            while t < end {
                let mut dot = F32::ZERO;
                let mut d = 0usize;
                while d < head_dim {
                    dot = qrow[d].fma(kt[d * KV_TILE + t], dot);
                    d += 1;
                }
                let score = if scaled_values {
                    dot * kscale[t]
                } else {
                    dot * scale
                };
                if !score.is_finite() {
                    return Err(AttnError::Overflow);
                }
                p[pbase + t] = score;
                if score.gt(tmax) {
                    tmax = score;
                }
                t += 1;
            }
            let m_prev = m[r];
            let m_next = if tmax.gt(m_prev) { tmax } else { m_prev };
            if m_prev.to_bits() == F32::NEG_INFINITY.to_bits() {
                // The row's first tile: the denominator and accumulator start at zero.
                den[r] = F32::ZERO;
                let mut i = 0usize;
                while i < head_dim {
                    acc[r * head_dim + i] = F32::ZERO;
                    i += 1;
                }
            } else if m_next.gt(m_prev) {
                let ms = exp_f32_scalar(m_prev - m_next);
                den[r] *= ms;
                let mut i = 0usize;
                while i < head_dim {
                    acc[r * head_dim + i] *= ms;
                    i += 1;
                }
            }
            let mut sum = F32::ZERO;
            let mut t = 0usize;
            while t < end {
                let w = exp_f32_scalar(p[pbase + t] - m_next);
                p[pbase + t] = w;
                sum += w;
                t += 1;
            }
            while t < KV_TILE {
                p[pbase + t] = F32::ZERO;
                t += 1;
            }
            den[r] += sum;
            if scaled_values {
                let mut t = 0usize;
                while t < end {
                    p[pbase + t] *= vscale[t];
                    t += 1;
                }
            }
            m[r] = m_next;
            r += 1;
        }

        // Values: `acc += P · V`, accumulated over the tile in ascending position order.
        let mut r = lo;
        while r < rows {
            let pbase = r * KV_TILE;
            let mut i = 0usize;
            while i < head_dim {
                let mut a = acc[r * head_dim + i];
                let mut t = 0usize;
                while t < KV_TILE {
                    a = p[pbase + t].fma(vt[t * head_dim + i], a);
                    t += 1;
                }
                acc[r * head_dim + i] = a;
                i += 1;
            }
            r += 1;
        }
        at0 += kv_tile;
    }

    let mut row = 0usize;
    while row < rows {
        if den[row].le(F32::ZERO) || !den[row].is_finite() {
            return Err(AttnError::ZeroSum);
        }
        let inv_den = F32::ONE / den[row];
        let mut i = 0usize;
        while i < head_dim {
            let res = acc[row * head_dim + i] * inv_den;
            if !res.is_finite() {
                return Err(AttnError::Overflow);
            }
            out[row * head_dim + i] = res;
            i += 1;
        }
        row += 1;
    }
    Ok(())
}

/// `C[RM][RN·8] += A[RM][K] · B[K][RN·8]` for one register tile: llama.cpp's `simd_gemm_ukernel`
/// (`ggml/src/ggml-cpu/simd-gemm.h`) with `GEMM_RM`/`GEMM_RN` as const parameters.
///
/// # Safety
///
/// Requires AVX2 and FMA; `a` holds `RM · K`, `b` holds `K · n` and `c` holds `RM · n` words,
/// with `n >= RN · 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::many_single_char_names,
    reason = "the microkernel's register-tile indices are RM, RN, K, N, i, r and kk"
)]
unsafe fn gemm_ukernel<const RM: usize, const RN: usize>(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize,
) {
    let mut acc = [[_mm256_setzero_ps(); RN]; RM];
    let mut i = 0usize;
    while i < RM {
        let mut r = 0usize;
        while r < RN {
            acc[i][r] = _mm256_loadu_ps(c.add(i * n + r * 8));
            r += 1;
        }
        i += 1;
    }
    let mut kk = 0usize;
    while kk < k {
        let mut bv = [_mm256_setzero_ps(); RN];
        let mut r = 0usize;
        while r < RN {
            bv[r] = _mm256_loadu_ps(b.add(kk * n + r * 8));
            r += 1;
        }
        let mut i = 0usize;
        while i < RM {
            let p = _mm256_broadcast_ss(&*a.add(i * k + kk));
            let mut r = 0usize;
            while r < RN {
                acc[i][r] = _mm256_fmadd_ps(bv[r], p, acc[i][r]);
                r += 1;
            }
            i += 1;
        }
        kk += 1;
    }
    let mut i = 0usize;
    while i < RM {
        let mut r = 0usize;
        while r < RN {
            _mm256_storeu_ps(c.add(i * n + r * 8), acc[i][r]);
            r += 1;
        }
        i += 1;
    }
}

/// `C[M][N] += A[M][K] · B[K][N]` by [`gemm_ukernel`] with llama.cpp's AVX2 register tile: six
/// rows by two vectors (16 columns), a one-row tail and a scalar column tail for `N` not a
/// multiple of eight. The accumulation order is the scalar twin's: ascending `kk`, one FMA per
/// element.
///
/// # Safety
///
/// Requires AVX2 and FMA; `a` holds `M · K`, `b` holds `K · N` and `c` holds `M · N` words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::many_single_char_names,
    reason = "the microkernel's register-tile indices are M, N, K, i, r and kk"
)]
unsafe fn gemm_tile(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    let cp = c.as_mut_ptr();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut ii = 0usize;
    while ii + 6 <= m {
        let mut jj = 0usize;
        while jj + 16 <= n {
            gemm_ukernel::<6, 2>(cp.add(ii * n + jj), ap.add(ii * k), bp.add(jj), k, n);
            jj += 16;
        }
        while jj + 8 <= n {
            gemm_ukernel::<6, 1>(cp.add(ii * n + jj), ap.add(ii * k), bp.add(jj), k, n);
            jj += 8;
        }
        while jj < n {
            let mut i = 0usize;
            while i < 6 {
                let mut acc = *cp.add((ii + i) * n + jj);
                let mut kk = 0usize;
                while kk < k {
                    acc = (*bp.add(kk * n + jj)).mul_add(*ap.add((ii + i) * k + kk), acc);
                    kk += 1;
                }
                *cp.add((ii + i) * n + jj) = acc;
                i += 1;
            }
            jj += 1;
        }
        ii += 6;
    }
    while ii < m {
        let mut jj = 0usize;
        while jj + 16 <= n {
            gemm_ukernel::<1, 2>(cp.add(ii * n + jj), ap.add(ii * k), bp.add(jj), k, n);
            jj += 16;
        }
        while jj + 8 <= n {
            gemm_ukernel::<1, 1>(cp.add(ii * n + jj), ap.add(ii * k), bp.add(jj), k, n);
            jj += 8;
        }
        while jj < n {
            let mut acc = *cp.add(ii * n + jj);
            let mut kk = 0usize;
            while kk < k {
                acc = (*bp.add(kk * n + jj)).mul_add(*ap.add(ii * k + kk), acc);
                kk += 1;
            }
            *cp.add(ii * n + jj) = acc;
            jj += 1;
        }
        ii += 1;
    }
}

/// AVX2+FMA body of [`attn_core_f32_tile`]; the caller has checked the geometry and the feature.
///
/// The score and value stages are [`gemm_tile`] with llama.cpp's 6×2 AVX2 microkernel: six rows
/// share each broadcast of the A operand and each 8-lane load of the B operand. Every arithmetic
/// operation is the scalar twin's in the same order, and [`exp_f32_avx2`] equals eight
/// [`exp_f32_scalar`] calls lane for lane, so the words equal the twin's.
///
/// # Safety
///
/// Requires AVX2 and FMA; `q`, `k` and `v` hold the geometry the body checks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "mathematical recurrence symbols, explicit strided geometry, and the pass structure"
)]
unsafe fn attn_core_tile_avx2_inner(
    q: &[F32],
    kv: TileRows<'_>,
    head_dim: usize,
    query_heads: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    let q = native(q);
    let scale = nat(scale);
    let out = native_mut(out);
    if head_dim == 0 || query_heads == 0 || kv.stride() < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    let nq = pos.len();
    let rows = nq.checked_mul(query_heads).ok_or(AttnError::Overflow)?;
    let need = rows.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if q.len() != need || out.len() != need || !head_dim.is_multiple_of(8) {
        return Err(AttnError::BadGeometry);
    }
    let mut j = 1usize;
    while j < nq {
        if pos[j] <= pos[j - 1] {
            return Err(AttnError::BadGeometry);
        }
        j += 1;
    }
    kv.check(pos[nq - 1], head_dim)?;

    let score_len = rows.checked_mul(KV_TILE).ok_or(AttnError::Overflow)?;
    let tile_len = KV_TILE.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    let mut m = vec![f32::NEG_INFINITY; rows];
    let mut den = vec![0.0f32; rows];
    let mut acc = vec![0.0f32; need];
    let mut kt = vec![0.0f32; tile_len];
    let mut vt = vec![0.0f32; tile_len];
    let mut p = vec![0.0f32; score_len];
    let mut kscale = vec![scale; KV_TILE];
    let mut vscale = vec![1.0f32; KV_TILE];
    let scaled_values = matches!(kv, TileRows::I8 { .. });
    let stride = kv.stride();

    let last = pos[nq - 1];
    let mut at0 = 0usize;
    while at0 <= last {
        let kv_tile = (last + 1 - at0).min(KV_TILE);
        if let TileRows::I8 { ks, vs, .. } = kv {
            let ksn = native(ks);
            let vsn = native(vs);
            let mut t = 0usize;
            while t < kv_tile {
                let kk = ksn[at0 + t] * scale;
                let vv = vsn[at0 + t];
                if !kk.is_finite() || !vv.is_finite() {
                    return Err(AttnError::Overflow);
                }
                kscale[t] = kk;
                vscale[t] = vv;
                t += 1;
            }
        }
        let mut t = 0usize;
        while t < kv_tile {
            let base = (at0 + t) * stride;
            let mut d = 0usize;
            match kv {
                TileRows::F32 { k, v, .. } => {
                    let k = native(k);
                    let v = native(v);
                    while d < head_dim {
                        let vi = v[base + d];
                        if !vi.is_finite() {
                            return Err(AttnError::Overflow);
                        }
                        kt[d * KV_TILE + t] = k[base + d];
                        vt[t * head_dim + d] = vi;
                        d += 1;
                    }
                }
                TileRows::I8 { k, v, .. } => {
                    while d < head_dim {
                        kt[d * KV_TILE + t] = f32::from(k[base + d]);
                        vt[t * head_dim + d] = f32::from(v[base + d]);
                        d += 1;
                    }
                }
            }
            t += 1;
        }
        while t < KV_TILE {
            let mut d = 0usize;
            while d < head_dim {
                kt[d * KV_TILE + t] = 0.0;
                vt[t * head_dim + d] = 0.0;
                d += 1;
            }
            t += 1;
        }

        let mut lo = 0usize;
        while lo < rows && pos[lo / query_heads] < at0 {
            lo += 1;
        }
        let active = rows - lo;
        // Scores: `p = q · kt`, then scale and the per-row causal maximum, in the twin's order.
        p[lo * KV_TILE..rows * KV_TILE].fill(0.0);
        gemm_tile(
            &mut p[lo * KV_TILE..rows * KV_TILE],
            &q[lo * head_dim..rows * head_dim],
            &kt,
            active,
            head_dim,
            KV_TILE,
        );
        let mut r = lo;
        while r < rows {
            let cut = pos[r / query_heads];
            let end = kv_tile.min(cut + 1 - at0);
            let pbase = r * KV_TILE;
            let mut tmax = f32::NEG_INFINITY;
            let mut t = 0usize;
            while t < end {
                let score = if scaled_values {
                    p[pbase + t] * kscale[t]
                } else {
                    p[pbase + t] * scale
                };
                if !score.is_finite() {
                    return Err(AttnError::Overflow);
                }
                p[pbase + t] = score;
                if score > tmax {
                    tmax = score;
                }
                t += 1;
            }
            let m_prev = m[r];
            let m_next = if tmax > m_prev { tmax } else { m_prev };
            if m_prev == f32::NEG_INFINITY {
                den[r] = 0.0;
                let mut i = 0usize;
                while i < head_dim {
                    acc[r * head_dim + i] = 0.0;
                    i += 1;
                }
            } else if m_next > m_prev {
                let ms = nat(exp_f32_scalar(fp(m_prev - m_next)));
                den[r] *= ms;
                let v_ms = _mm256_set1_ps(ms);
                let mut c = 0usize;
                while c + 8 <= head_dim {
                    let a = _mm256_loadu_ps(acc.as_ptr().add(r * head_dim + c));
                    _mm256_storeu_ps(
                        acc.as_mut_ptr().add(r * head_dim + c),
                        _mm256_mul_ps(a, v_ms),
                    );
                    c += 8;
                }
                while c < head_dim {
                    acc[r * head_dim + c] *= ms;
                    c += 1;
                }
            }
            // P = exp(score − m_next), eight lanes at a time; the sum is sequential, as in the
            // twin.
            let v_next = _mm256_set1_ps(m_next);
            let mut t = 0usize;
            while t + 8 <= end {
                let vv = _mm256_loadu_ps(p.as_ptr().add(pbase + t));
                let e = exp_f32_avx2(_mm256_sub_ps(vv, v_next));
                _mm256_storeu_ps(p.as_mut_ptr().add(pbase + t), e);
                t += 8;
            }
            while t < end {
                p[pbase + t] = nat(exp_f32_scalar(fp(p[pbase + t] - m_next)));
                t += 1;
            }
            while t < KV_TILE {
                p[pbase + t] = 0.0;
                t += 1;
            }
            let mut sum = 0.0f32;
            let mut t = 0usize;
            while t < end {
                sum += p[pbase + t];
                t += 1;
            }
            den[r] += sum;
            if scaled_values {
                let vsn: &[f32] = &vscale;
                let mut t = 0usize;
                while t + 8 <= end {
                    let w = _mm256_loadu_ps(p.as_ptr().add(pbase + t));
                    let s = _mm256_loadu_ps(vsn.as_ptr().add(t));
                    _mm256_storeu_ps(p.as_mut_ptr().add(pbase + t), _mm256_mul_ps(w, s));
                    t += 8;
                }
                while t < end {
                    p[pbase + t] *= vsn[t];
                    t += 1;
                }
            }
            m[r] = m_next;
            r += 1;
        }
        // Values: `acc += p · vt`, one accumulator read and write per tile.
        gemm_tile(
            &mut acc[lo * head_dim..rows * head_dim],
            &p[lo * KV_TILE..rows * KV_TILE],
            &vt,
            active,
            KV_TILE,
            head_dim,
        );
        at0 += kv_tile;
    }

    let mut row = 0usize;
    while row < rows {
        if den[row] <= 0.0 || !den[row].is_finite() {
            return Err(AttnError::ZeroSum);
        }
        let inv_den = 1.0f32 / den[row];
        let v_inv = _mm256_set1_ps(inv_den);
        let mut c = 0usize;
        while c + 8 <= head_dim {
            let a = _mm256_loadu_ps(acc.as_ptr().add(row * head_dim + c));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(row * head_dim + c),
                _mm256_mul_ps(a, v_inv),
            );
            c += 8;
        }
        while c < head_dim {
            out[row * head_dim + c] = acc[row * head_dim + c] * inv_den;
            c += 1;
        }
        // The twin scans each row before the next; this scan keeps the same refusal order.
        let mut i = 0usize;
        while i < head_dim {
            if !out[row * head_dim + i].is_finite() {
                return Err(AttnError::Overflow);
            }
            i += 1;
        }
        row += 1;
    }
    Ok(())
}

/// Dispatches contiguous attention core execution (`stride = head_dim`).
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time.
#[allow(
    dead_code,
    reason = "public contiguous core specification and test target"
)]
pub fn attn_core_f32(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_f32_strided(q, k, v, head_dim, head_dim, pos, scale, out)
}

/// Scalar twin for contiguous attention core (`stride = head_dim`).
#[allow(
    dead_code,
    reason = "public scalar twin specification for contiguous layout"
)]
pub fn attn_core_f32_scalar(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_f32_strided_scalar(q, k, v, head_dim, head_dim, pos, scale, out)
}

#[cfg(test)]
#[path = "f32_core/tests.rs"]
pub(crate) mod tests;
