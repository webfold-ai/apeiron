//! Tests for `f32_core`: the scalar twin against the AVX2 kernels and the oracle.

#![allow(
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::manual_is_multiple_of,
    clippy::semicolon_if_nothing_returned,
    clippy::items_after_statements,
    reason = "test harnesses, random generators, timing loops and numerical verification"
)]

use super::*;
use crate::engine::attn::KvCarrier;
use crate::engine::attn::KvState;
use crate::engine::num::fp::F64;

fn fv(x: &[f32]) -> Vec<F32> {
    x.iter().map(|&v| F32::from_bits(v.to_bits())).collect()
}

fn fo(out: &mut [f32], src: &[F32]) {
    for (o, x) in out.iter_mut().zip(src) {
        *o = f32::from_bits(x.to_bits());
    }
}

fn f(x: F32) -> f32 {
    f32::from_bits(x.to_bits())
}

/// The bound between [`attn_core_f32_tile`] and the per-position [`attn_core_f32_strided`]
/// oracle. The two evaluate the same real recurrence; the tile rescales the accumulator once per
/// [`KV_TILE`] positions and the oracle once per position, so their words differ by the
/// roundoff the oracle accumulates over `prefix` rescales of a value of magnitude `|want|`:
/// `2 · (prefix + KV_TILE) · f32::EPSILON`, with the factor two for the two paths.
fn tile_gap(prefix: usize, want: f32) -> f32 {
    2.0 * (prefix + KV_TILE) as f32 * f32::EPSILON * want.abs().max(1.0)
}

fn scalar_n(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    let mut of = vec![F32::ZERO; out.len()];
    let r = attn_core_f32_strided_scalar(
        &fv(q),
        &fv(k),
        &fv(v),
        hd,
        stride,
        pos,
        F32::from_bits(scale.to_bits()),
        &mut of,
    );
    fo(out, &of);
    r
}

fn strided_n(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    let mut of = vec![F32::ZERO; out.len()];
    let r = attn_core_f32_strided(
        &fv(q),
        &fv(k),
        &fv(v),
        hd,
        stride,
        pos,
        F32::from_bits(scale.to_bits()),
        &mut of,
    );
    fo(out, &of);
    r
}

fn core_n(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    let mut of = vec![F32::ZERO; out.len()];
    let r = attn_core_f32(
        &fv(q),
        &fv(k),
        &fv(v),
        hd,
        pos,
        F32::from_bits(scale.to_bits()),
        &mut of,
    );
    fo(out, &of);
    r
}

/// The f64 body that `F32::to_q32` replaces, copied from the pre-conversion `f32_to_q32_checked`.
fn old_f32_to_q32(v: f32) -> Option<i64> {
    if !v.is_finite() {
        return None;
    }
    let q = (f64::from(v) * 4_294_967_296.0).round();
    if !(q.is_finite() && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&q))
    {
        return None;
    }
    Some(q as i64)
}

#[test]
fn test_f32_to_q32_equals_old_f64_formula() {
    let mut bits: Vec<u32> = vec![
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x2f00_0000,
        0xaf00_0000,
        0x2f40_0000,
        0x2e00_0000,
        0x3fa0_0000,
        0xc020_0000,
        0x4effffff,
        0x4f00_0000,
        0xcf00_0000,
        0x5f00_0000,
        0xdf00_0000,
        0x7f00_0000,
    ];
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..200_000 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        bits.push((s >> 32) as u32);
    }
    for b in bits {
        let want = old_f32_to_q32(f32::from_bits(b));
        assert_eq!(F32::from_bits(b).to_q32(), want, "bits {b:#010x}");
        assert_eq!(
            f32_to_q32_checked(F32::from_bits(b)).ok(),
            want,
            "bits {b:#010x}"
        );
    }
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn next_f32_range(&mut self, low: f32, high: f32) -> f32 {
        low + self.next_f32() * (high - low)
    }
}

#[test]
fn test_shared_exp_accuracy_in_attention() {
    let steps = 100_000;
    let mut max_rel_err = 0.0f64;
    let mut max_err_at = 0.0f32;

    for step in 0..=steps {
        let x = -80.0f32 * (step as f32 / steps as f32);
        let approx = f64::from(f32::from_bits(
            exp_f32_scalar(F32::from_bits(x.to_bits())).to_bits(),
        ));
        let exact = f64::from(x).exp();

        if exact > 1e-35 {
            let err = (approx - exact).abs() / exact;
            if err > max_rel_err {
                max_rel_err = err;
                max_err_at = x;
            }
        }
    }

    // Relative error of shared exp must be bounded by 3.0e-7 across [-80.0, 0.0]
    assert!(
        max_rel_err < 3.0e-7,
        "max rel err {max_rel_err:e} at {max_err_at} exceeds 3.0e-7 bound"
    );
}

#[test]
fn test_attn_core_avx2_equals_scalar() {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let mut rng = SimpleRng::new(0x1337_C0DE_5EED_0001);
        let test_configs = [
            (64, 64, 0),
            (64, 128, 3),
            (128, 128, 7),
            (128, 256, 31),
            (256, 512, 15),
        ];

        for &(hd, stride, pos) in &test_configs {
            let count = pos + 1;
            let mut q = vec![0.0f32; hd];
            for item in q.iter_mut().take(hd) {
                *item = rng.next_f32_range(-1.0, 1.0);
            }
            let total_len = pos * stride + hd;
            let mut k = vec![0.0f32; total_len];
            let mut v = vec![0.0f32; total_len];
            for s in 0..count {
                let base = s * stride;
                for i in 0..hd {
                    k[base + i] = rng.next_f32_range(-1.0, 1.0);
                    v[base + i] = rng.next_f32_range(-4.0, 4.0);
                }
            }
            let scale = 1.0f32 / (hd as f32).sqrt();

            let mut out_scalar = vec![0.0f32; hd];
            let mut out_avx2 = vec![0.0f32; hd];

            scalar_n(&q, &k, &v, hd, stride, pos, scale, &mut out_scalar).expect("scalar");
            unsafe {
                attn_core_f32_strided_avx2_inner(&q, &k, &v, hd, stride, pos, scale, &mut out_avx2)
                    .expect("avx2");
            }

            for i in 0..hd {
                assert_eq!(
                    out_scalar[i].to_bits(),
                    out_avx2[i].to_bits(),
                    "hd {hd} stride {stride} pos {pos} idx {i}: scalar={:e} ({:08x}) vs avx2={:e} ({:08x})",
                    out_scalar[i],
                    out_scalar[i].to_bits(),
                    out_avx2[i],
                    out_avx2[i].to_bits()
                );
            }
        }
    }
}

#[test]
fn test_attn_core_query_tile_agrees_with_the_per_query_core() {
    let mut rng = SimpleRng::new(0x1337_C0DE_5EED_0002);
    // First queries at and across the 64-position key-tile boundary, and short prefixes too.
    let configs = [
        (8usize, 8usize, 0usize),
        (8, 16, 1),
        (16, 16, 7),
        (16, 24, 255),
        (64, 64, 256),
        (128, 128, 257),
        (128, 256, 600),
    ];
    let scale_of = |hd: usize| F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
    for &(hd, stride, first) in &configs {
        // 1, 2 and 4 query heads share one K/V row: the group sizes the block dispatch uses.
        for query_heads in [1usize, 2, 4] {
            for nq in [1usize, 2, 3, 8, 33] {
                let last = first + nq - 1;
                let total = last * stride + hd;
                let width = query_heads * hd;
                let queries: Vec<f32> = (0..nq * width)
                    .map(|_| rng.next_f32_range(-1.0, 1.0))
                    .collect();
                let mut k = vec![0.0f32; total];
                let mut v = vec![0.0f32; total];
                for s in 0..=last {
                    for i in 0..hd {
                        k[s * stride + i] = rng.next_f32_range(-1.0, 1.0);
                        v[s * stride + i] = rng.next_f32_range(-4.0, 4.0);
                    }
                }
                let positions: Vec<usize> = (first..=last).collect();
                let qf = fv(&queries);
                let kf = fv(&k);
                let vf = fv(&v);
                let mut tiled = vec![F32::ZERO; nq * width];
                attn_core_f32_tile(
                    &qf,
                    &kf,
                    &vf,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale_of(hd),
                    &mut tiled,
                )
                .expect("tile");
                let mut scalar = vec![F32::ZERO; nq * width];
                attn_core_f32_tile_scalar(
                    &qf,
                    &kf,
                    &vf,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale_of(hd),
                    &mut scalar,
                )
                .expect("tile scalar");
                for j in 0..nq {
                    for g in 0..query_heads {
                        let row = j * query_heads + g;
                        let mut single = vec![F32::ZERO; hd];
                        attn_core_f32_strided(
                            &qf[row * hd..(row + 1) * hd],
                            &kf,
                            &vf,
                            hd,
                            stride,
                            positions[j],
                            scale_of(hd),
                            &mut single,
                        )
                        .expect("single");
                        for i in 0..hd {
                            let at = row * hd + i;
                            let (got, want) = (f(tiled[at]), f(single[i]));
                            assert!(
                                (got - want).abs() <= tile_gap(positions[j] + 1, want),
                                "tile hd {hd} stride {stride} heads {query_heads} nq {nq}                                  query {j} head {g} idx {i}: {got:e} vs {want:e}"
                            );
                            assert_eq!(
                                scalar[at].to_bits(),
                                tiled[at].to_bits(),
                                "twin hd {hd} stride {stride} heads {query_heads} nq {nq} query {j} head {g} idx {i}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The int8 tiled core is its scalar twin's words, and both reproduce the per-column int8 group
/// core's attention over the same codes and scales.
///
/// They are not bit-equal to that core: it accumulates each score dot in eight independent lanes
/// and the tile's GEMM accumulates `head_dim` terms sequentially, so the scores (and through the
/// softmax the outputs) differ by the dot-order roundoff. On this fixture the largest relative
/// output gap is 1.3e-3; the assertion allows 2e-3, which still refuses a wrong scale, index or
/// dot. The tile-versus-twin comparison is word-exact. The model-level quality check is the PPL
/// A/B in the receipt, not this bound.
#[test]
fn test_attn_core_i8_query_tile_agrees_with_the_per_query_core() {
    let mut rng = SimpleRng::new(0x1337_C0DE_5EED_0004);
    let configs = [
        (8usize, 8usize, 0usize),
        (8, 16, 1),
        (16, 16, 7),
        (16, 24, 255),
        (64, 64, 256),
        (128, 128, 257),
    ];
    let scale_of = |hd: usize| F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
    let scale = |rng: &mut SimpleRng| F32::from_bits(rng.next_f32_range(0.001, 2.0).to_bits());
    for &(hd, stride, first) in &configs {
        // 1, 2 and 4 query heads share one K/V row.
        for query_heads in [1usize, 2, 4] {
            for nq in [1usize, 3, 8] {
                let last = first + nq - 1;
                let total = last * stride + hd;
                let width = query_heads * hd;
                let queries: Vec<f32> = (0..nq * width)
                    .map(|_| rng.next_f32_range(-1.0, 1.0))
                    .collect();
                let mut kc = vec![0i8; total];
                let mut vc = vec![0i8; total];
                for s in 0..=last {
                    for i in 0..hd {
                        kc[s * stride + i] = ((rng.next_u64() % 255) as i16 - 127) as i8;
                        vc[s * stride + i] = ((rng.next_u64() % 255) as i16 - 127) as i8;
                    }
                }
                let ks: Vec<F32> = (0..=last).map(|_| scale(&mut rng)).collect();
                let vs: Vec<F32> = (0..=last).map(|_| scale(&mut rng)).collect();
                // The carrier's rows are contiguous (`stride = head_dim`); the oracle takes that
                // layout, so compact a padded fixture before calling it.
                let mut kcompact = vec![0i8; (last + 1) * hd];
                let mut vcompact = vec![0i8; (last + 1) * hd];
                for s in 0..=last {
                    kcompact[s * hd..(s + 1) * hd]
                        .copy_from_slice(&kc[s * stride..s * stride + hd]);
                    vcompact[s * hd..(s + 1) * hd]
                        .copy_from_slice(&vc[s * stride..s * stride + hd]);
                }
                let positions: Vec<usize> = (first..=last).collect();
                let qf = fv(&queries);
                let mut tiled = vec![F32::ZERO; nq * width];
                attn_core_i8_tile(
                    &qf,
                    &kc,
                    &vc,
                    &ks,
                    &vs,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale_of(hd),
                    &mut tiled,
                )
                .expect("tile");
                let mut scalar = vec![F32::ZERO; nq * width];
                attn_core_i8_tile_scalar(
                    &qf,
                    &kc,
                    &vc,
                    &ks,
                    &vs,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale_of(hd),
                    &mut scalar,
                )
                .expect("tile scalar");
                for j in 0..nq {
                    for g in 0..query_heads {
                        let row = j * query_heads + g;
                        let mut one = vec![F32::ZERO; hd];
                        crate::engine::attn::i8::attn_core_i8_group(
                            &qf[row * hd..(row + 1) * hd],
                            1,
                            &kcompact,
                            &vcompact,
                            &ks,
                            &vs,
                            hd,
                            positions[j] + 1,
                            scale_of(hd),
                            &mut one,
                        )
                        .expect("per column");
                        for i in 0..hd {
                            let at = row * hd + i;
                            assert_eq!(
                                scalar[at].to_bits(),
                                tiled[at].to_bits(),
                                "i8 twin hd {hd} stride {stride} heads {query_heads} nq {nq}                                  query {j} head {g} idx {i}"
                            );
                            let (got, want) = (f(tiled[at]), f(one[i]));
                            assert!(
                                (got - want).abs() <= 2.0e-3 * want.abs().max(1.0),
                                "i8 tile hd {hd} stride {stride} heads {query_heads} nq {nq}                                  query {j} head {g} idx {i}: {got:e} vs {want:e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Rate of one [`attn_core_f32_tile`] task at the 0.6B geometry: `QUERY_TILE = 32` columns,
/// `group = 2` query heads per KV head, `head_dim = 128`, eight KV heads, a cold tile at position
/// zero and a tail tile behind the full 1,827-position prefix.
/// `cargo test --release --lib --locked attn_tile_rate -- --ignored --nocapture`.
#[test]
#[ignore = "timing harness: run under the exclusive host lock"]
#[allow(
    clippy::cast_precision_loss,
    reason = "the timing report is a rate over counts far below 2^53"
)]
fn attn_tile_rate() {
    let (hd, group, n_kv) = (128usize, 2usize, 8usize);
    let stride = n_kv * hd;
    let nq = 32usize;
    let width = group * hd;
    let scale = F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
    for first in [0usize, 1_795] {
        let total = (first + nq) * stride;
        let mut rng = SimpleRng::new(0xA77E_1234);
        let q: Vec<f32> = (0..nq * width)
            .map(|_| rng.next_f32_range(-1.0, 1.0))
            .collect();
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        for x in &mut k {
            *x = rng.next_f32_range(-1.0, 1.0);
        }
        for x in &mut v {
            *x = rng.next_f32_range(-4.0, 4.0);
        }
        let positions: Vec<usize> = (first..first + nq).collect();
        let qf = fv(&q);
        let kf = fv(&k);
        let vf = fv(&v);
        let mut out = vec![F32::ZERO; nq * width];
        attn_core_f32_tile(
            &qf, &kf, &vf, hd, group, stride, &positions, scale, &mut out,
        )
        .expect("warm");
        let reps = 3u32;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            attn_core_f32_tile(
                &qf, &kf, &vf, hd, group, stride, &positions, scale, &mut out,
            )
            .expect("rep");
        }
        let dt = t0.elapsed().as_secs_f64() / f64::from(reps);
        let pairs: u64 = (0..nq).map(|j| ((first + j + 1) * group) as u64).sum();
        let flops = 4.0 * pairs as f64 * hd as f64;
        println!("first {first}: {dt:.4} s, {:.3} GFLOP/s", flops / dt / 1e9);
    }
}

#[test]
fn test_attn_core_refusal_on_non_finite() {
    let hd = 16;
    let pos = 2;
    let count = pos + 1;
    let scale = 1.0 / (hd as f32).sqrt();

    // 1. NaN in query
    let mut q = vec![0.5f32; hd];
    q[3] = f32::NAN;
    let k = vec![0.5f32; count * hd];
    let v = vec![0.5f32; count * hd];
    let mut out = vec![0.0f32; hd];
    assert_eq!(
        core_n(&q, &k, &v, hd, pos, scale, &mut out),
        Err(AttnError::Overflow)
    );

    // 2. Inf in key
    let q = vec![0.5f32; hd];
    let mut k = vec![0.5f32; count * hd];
    k[hd + 2] = f32::INFINITY;
    assert_eq!(
        core_n(&q, &k, &v, hd, pos, scale, &mut out),
        Err(AttnError::Overflow)
    );

    // 3. NaN in value
    let k = vec![0.5f32; count * hd];
    let mut v = vec![0.5f32; count * hd];
    v[2 * hd + 1] = f32::NAN;
    assert_eq!(
        core_n(&q, &k, &v, hd, pos, scale, &mut out),
        Err(AttnError::Overflow)
    );

    // 4. Bad geometry: query width mismatch
    let v = vec![0.5f32; count * hd];
    assert_eq!(
        core_n(&q[..hd - 1], &k, &v, hd, pos, scale, &mut out),
        Err(AttnError::BadGeometry)
    );

    // 5. Bad geometry: stride < head_dim
    assert_eq!(
        strided_n(&q, &k, &v, hd, hd - 1, pos, scale, &mut out),
        Err(AttnError::BadGeometry)
    );
}

#[test]
fn test_attn_core_vs_q32_oracle_deviation() {
    let mut rng = SimpleRng::new(0xCAFE_BABE_9876_5432);
    let test_positions = [0, 1, 5, 20, 50];
    let hd = 128;
    let n_head = 4;
    let n_head_kv = 2;
    let max_cap = 64;

    let mut kv_q32 =
        KvState::with_carrier(1, n_head, n_head_kv, hd, max_cap, KvCarrier::Q32).expect("kv q32");
    let mut kv_f32 =
        KvState::with_carrier(1, n_head, n_head_kv, hd, max_cap, KvCarrier::F32).expect("kv f32");

    // Fill KV caches with realistic Q32.32 values
    let scale_q32 = 1i64 << 32;
    for _ in 0..max_cap {
        let mut k_rows = Vec::with_capacity(n_head_kv * hd);
        let mut v_rows = Vec::with_capacity(n_head_kv * hd);
        for _ in 0..n_head_kv * hd {
            let k_f = rng.next_f32_range(-1.0, 1.0);
            let v_f = rng.next_f32_range(-2.0, 2.0);
            k_rows.push((f64::from(k_f) * (scale_q32 as f64)).round() as i64);
            v_rows.push((f64::from(v_f) * (scale_q32 as f64)).round() as i64);
        }
        kv_q32.append(0, &k_rows, &v_rows).expect("append q32");
        kv_f32.append(0, &k_rows, &v_rows).expect("append f32");
    }

    let mut max_abs_diff = 0i64;
    let mut max_rel_err = 0.0f64;

    for &pos in &test_positions {
        for h in 0..n_head {
            let mut q = Vec::with_capacity(hd);
            for _ in 0..hd {
                let q_f = rng.next_f32_range(-1.0, 1.0);
                q.push((f64::from(q_f) * (scale_q32 as f64)).round() as i64);
            }

            let mut oracle_out = vec![0i64; hd];
            let mut f32_out = vec![0i64; hd];

            // Run Q32 oracle
            kv_q32
                .attn_head(0, h, &q, pos, &mut oracle_out)
                .expect("q32 oracle");

            // Run f32 core on native f32 KV cache
            kv_f32
                .attn_head(0, h, &q, pos, &mut f32_out)
                .expect("f32 core");

            for i in 0..hd {
                let o_val = oracle_out[i];
                let f_val = f32_out[i];
                let abs_diff = (o_val - f_val).abs();
                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                }
                if o_val != 0 {
                    let rel = (abs_diff as f64) / (o_val.abs() as f64);
                    if rel > max_rel_err {
                        max_rel_err = rel;
                    }
                }
            }
        }
    }

    let max_abs_q32_float = (max_abs_diff as f64) / (scale_q32 as f64);
    eprintln!(
        "Observed max abs diff in Q32.32: {max_abs_diff} ({max_abs_q32_float:e}), max rel err: {max_rel_err:e}"
    );
    assert!(
        max_abs_q32_float < 1.0e-3,
        "max abs deviation {max_abs_q32_float:e} exceeds 1.0e-3"
    );
}

/// Benchmark comparing Q32 path vs native f32 KV cache attention at pos = 200, `head_dim` = 256.
/// Run with:
/// `taskset -c 2 cargo test --release --lib attn_f32_vs_q32_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing benchmark: run with cargo test --release --lib attn_f32_vs_q32_timing -- --ignored --nocapture"]
fn attn_f32_vs_q32_timing() {
    let mut rng = SimpleRng::new(0x4242_1337_BEEF_0001);
    let hd = 256;
    let pos = 200;
    let n_head = 4;
    let n_head_kv = 2;
    let cap = pos + 1;

    let mut kv_q32 =
        KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::Q32).expect("kv q32");
    let mut kv_f32 =
        KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::F32).expect("kv f32");
    let scale_q32 = 1i64 << 32;

    for _ in 0..cap {
        let mut k_rows = Vec::with_capacity(n_head_kv * hd);
        let mut v_rows = Vec::with_capacity(n_head_kv * hd);
        for _ in 0..n_head_kv * hd {
            let k_f = rng.next_f32_range(-1.0, 1.0);
            let v_f = rng.next_f32_range(-2.0, 2.0);
            k_rows.push((f64::from(k_f) * (scale_q32 as f64)).round() as i64);
            v_rows.push((f64::from(v_f) * (scale_q32 as f64)).round() as i64);
        }
        kv_q32.append(0, &k_rows, &v_rows).expect("append q32");
        kv_f32.append(0, &k_rows, &v_rows).expect("append f32");
    }

    let mut q = Vec::with_capacity(hd);
    for _ in 0..hd {
        let q_f = rng.next_f32_range(-1.0, 1.0);
        q.push((f64::from(q_f) * (scale_q32 as f64)).round() as i64);
    }

    let reps = 200;
    let mut out_q32 = vec![0i64; hd];
    let mut out_f32 = vec![0i64; hd];

    // Warm up:
    for _ in 0..20 {
        kv_q32
            .attn_head(0, 0, &q, pos, &mut out_q32)
            .expect("warm q32");
        kv_f32
            .attn_head(0, 0, &q, pos, &mut out_f32)
            .expect("warm f32");
    }

    let elem_steps = (pos * hd) as f64;
    let total_elem_steps = elem_steps * f64::from(reps);

    let mut q32_times = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            kv_q32.attn_head(0, 0, &q, pos, &mut out_q32).expect("q32");
        }
        let el = t0.elapsed();
        let ns_per_elem = (el.as_nanos() as f64) / total_elem_steps;
        q32_times.push(ns_per_elem);
    }

    let mut f32_times = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            kv_f32.attn_head(0, 0, &q, pos, &mut out_f32).expect("f32");
        }
        let el = t0.elapsed();
        let ns_per_elem = (el.as_nanos() as f64) / total_elem_steps;
        f32_times.push(ns_per_elem);
    }

    let best_q32 = q32_times.iter().copied().fold(f64::INFINITY, f64::min);
    let best_f32 = f32_times.iter().copied().fold(f64::INFINITY, f64::min);
    let speedup = best_q32 / best_f32;

    println!("--- TIMING RESULTS (best of 5, pos={pos}, hd={hd}) ---");
    println!("Q32.32 path:            {best_q32:.4} ns per (pos × hd)");
    println!("f32 native KV cache:    {best_f32:.4} ns per (pos × hd) (speedup: {speedup:.2}x)");
}

#[test]
fn test_attn_core_tile_exp_group_edges_agrees_with_the_per_query_core() {
    // Row counts below, at and across the six-row microkernel tile and the eight-lane exp group,
    // against a fixture where every position ties the first maximum (no row rescales after its
    // first tile), one where every position moves it (every row rescales every tile), one where a
    // single group mixes moving, tying and below-max rows, and a random fixture in between.
    let scale_of = |hd: usize| F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
    let mut rng = SimpleRng::new(0x1337_C0DE_5EED_0003);
    for &hd in &[8usize, 128usize] {
        for &(query_heads, nq) in &[
            (1usize, 1usize),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (2, 3),
            (2, 4),
            (2, 9),
            (4, 2),
            (4, 3),
        ] {
            let first = 3usize;
            let last = first + nq - 1;
            for fixture in 0..4 {
                let width = query_heads * hd;
                let mut q = vec![0.0f32; nq * width];
                let mut k = vec![0.0f32; (last + 1) * hd];
                let mut v = vec![0.0f32; (last + 1) * hd];
                match fixture {
                    // Every score is +0: each position ties the start maximum, alpha = 1.
                    0 => {
                        for s in 0..=last {
                            for i in 0..hd {
                                k[s * hd + i] = rng.next_f32_range(-1.0, 1.0);
                                v[s * hd + i] = rng.next_f32_range(-1.0, 1.0);
                            }
                        }
                    }
                    // Positive keys of growing amplitude: each position is a new maximum.
                    1 => {
                        q.fill(1.0);
                        for s in 0..=last {
                            for i in 0..hd {
                                k[s * hd + i] = (s as f32 + 1.0) / 64.0;
                                v[s * hd + i] = rng.next_f32_range(-1.0, 1.0);
                            }
                        }
                    }
                    // A mixed group at key 1: rows 0/3/6 move the maximum, 1/4/7 tie it and
                    // 2/5 stay below, so the vector alpha branch runs with unmoved lanes in the
                    // same group.
                    2 => {
                        let pattern = [1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
                        for j in 0..nq {
                            for g in 0..query_heads {
                                let row = j * query_heads + g;
                                q[row * hd] = pattern[row % 8];
                            }
                        }
                        for s in 1..=last {
                            k[s * hd] = 1.0;
                        }
                        for s in 0..=last {
                            for i in 0..hd {
                                v[s * hd + i] = rng.next_f32_range(-1.0, 1.0);
                            }
                        }
                    }
                    // Both branches, plus magnitudes that make exp underflow.
                    _ => {
                        for x in &mut q {
                            *x = rng.next_f32_range(-2.0, 2.0);
                        }
                        for s in 0..=last {
                            for i in 0..hd {
                                k[s * hd + i] = rng.next_f32_range(-4.0, 4.0);
                                v[s * hd + i] = rng.next_f32_range(-4.0, 4.0);
                            }
                        }
                    }
                }
                let positions: Vec<usize> = (first..=last).collect();
                let qf = fv(&q);
                let kf = fv(&k);
                let vf = fv(&v);
                let mut tiled = vec![F32::ZERO; nq * width];
                attn_core_f32_tile(
                    &qf,
                    &kf,
                    &vf,
                    hd,
                    query_heads,
                    hd,
                    &positions,
                    scale_of(hd),
                    &mut tiled,
                )
                .expect("tile");
                for j in 0..nq {
                    for g in 0..query_heads {
                        let row = j * query_heads + g;
                        let mut single = vec![F32::ZERO; hd];
                        attn_core_f32_strided(
                            &qf[row * hd..(row + 1) * hd],
                            &kf,
                            &vf,
                            hd,
                            hd,
                            positions[j],
                            scale_of(hd),
                            &mut single,
                        )
                        .expect("single");
                        for i in 0..hd {
                            let (got, want) = (f(tiled[row * hd + i]), f(single[i]));
                            assert!(
                                (got - want).abs() <= tile_gap(positions[j] + 1, want),
                                "hd {hd} heads {query_heads} nq {nq} fixture {fixture} \
                                 query {j} head {g} idx {i}: {got:e} vs {want:e}"
                            );
                        }
                    }
                }
                // The scalar twin and the AVX2 body must agree word for word.
                let mut twin = vec![F32::ZERO; nq * width];
                attn_core_f32_tile_scalar(
                    &qf,
                    &kf,
                    &vf,
                    hd,
                    query_heads,
                    hd,
                    &positions,
                    scale_of(hd),
                    &mut twin,
                )
                .expect("tile scalar");
                for at in 0..tiled.len() {
                    assert_eq!(
                        tiled[at].to_bits(),
                        twin[at].to_bits(),
                        "hd {hd} heads {query_heads} nq {nq} fixture {fixture} idx {at}"
                    );
                }
            }
        }
    }
}

/// The tiled AVX2 core equals the scalar twin on every tile boundary, at score magnitudes that
/// move the running maximum often and underflow `exp`, and refuses a non-finite key at any
/// position with the scalar twin's error.
#[test]
fn test_attn_core_tiled_equals_scalar_on_tile_edges() {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let mut rng = SimpleRng::new(0x7117_ED00_0000_0001);
        let cfgs = [
            (256usize, 512usize, 0usize),
            (256, 512, 1),
            (256, 512, 8),
            (256, 512, 9),
            (256, 512, 10),
            (256, 512, 16),
            (256, 512, 17),
            (256, 512, 100),
            (256, 512, 199),
            (128, 128, 40),
            (64, 192, 63),
            (8, 8, 33),
        ];
        for amp in [0.5f32, 2.0, 6.0, 20.0] {
            for &(hd, stride, pos) in &cfgs {
                let total = pos * stride + hd;
                let mut q = vec![0.0f32; hd];
                let mut k = vec![0.0f32; total];
                let mut v = vec![0.0f32; total];
                for x in &mut q {
                    *x = rng.next_f32_range(-amp, amp);
                }
                for x in &mut k {
                    *x = rng.next_f32_range(-amp, amp);
                }
                for x in &mut v {
                    *x = rng.next_f32_range(-4.0, 4.0);
                }
                let scale = 1.0f32 / (hd as f32).sqrt();
                let mut a = vec![0.0f32; hd];
                let mut b = vec![0.0f32; hd];
                let ra = scalar_n(&q, &k, &v, hd, stride, pos, scale, &mut a);
                let rb = unsafe {
                    attn_core_f32_strided_avx2_inner(&q, &k, &v, hd, stride, pos, scale, &mut b)
                };
                assert_eq!(ra, rb, "hd {hd} pos {pos} amp {amp}");
                if ra.is_ok() {
                    for i in 0..hd {
                        assert_eq!(
                            a[i].to_bits(),
                            b[i].to_bits(),
                            "hd {hd} pos {pos} amp {amp} i {i}"
                        );
                    }
                }
                if pos >= 2 {
                    for bad in [1usize, pos / 2, pos] {
                        let mut k2 = k.clone();
                        k2[bad * stride + 3] = f32::INFINITY;
                        let ra = scalar_n(&q, &k2, &v, hd, stride, pos, scale, &mut a);
                        let rb = unsafe {
                            attn_core_f32_strided_avx2_inner(
                                &q, &k2, &v, hd, stride, pos, scale, &mut b,
                            )
                        };
                        assert_eq!(ra, Err(AttnError::Overflow));
                        assert_eq!(rb, Err(AttnError::Overflow), "bad {bad} pos {pos}");
                    }
                }
                // A non-finite value refuses on both bodies at position 0 (the first row, copied
                // before the key loop) and inside the pass. The AVX2 body checks position 0 before
                // its vector copy, so the refusal is not left to a lane test it does not run.
                for bad in [0usize, pos / 2, pos] {
                    let mut v2 = v.clone();
                    v2[bad * stride + 3] = f32::INFINITY;
                    let ra = scalar_n(&q, &k, &v2, hd, stride, pos, scale, &mut a);
                    let rb = unsafe {
                        attn_core_f32_strided_avx2_inner(
                            &q, &k, &v2, hd, stride, pos, scale, &mut b,
                        )
                    };
                    assert_eq!(ra, Err(AttnError::Overflow), "value bad {bad} pos {pos}");
                    assert_eq!(rb, Err(AttnError::Overflow), "value bad {bad} pos {pos}");
                }
            }
        }
    }
}

/// Segment attribution of one prefill attention layer call (`k = 200`, `pos0 = 0`, the
/// Qwen3.5-0.8B geometry: 8 query heads, 2 KV heads, head dim 256, rotary width 64, gated
/// query) over the pool, mirroring the three stages of `softmax_core_block`.
/// `cargo test --release --lib --locked attn_layer_segments -- --ignored --nocapture`.
#[test]
#[ignore = "timing harness: run under the exclusive host lock"]
fn attn_layer_segments_k200() {
    use crate::engine::chart::ffn::attn_heads;
    use crate::engine::chart::task::FnHead;
    use crate::engine::loader::qk_norm_heads;
    use crate::engine::num::value::mul_q32;
    use crate::engine::ops::rope::RopeTable;
    use crate::engine::step::attn_block::{rotate_pair_prefixes, sigmoid_gate};
    use crate::numeric::fixed_ln::fixed_sigmoid;
    use std::time::Instant;

    let (k, nh, nkv, hd, rot) = (200usize, 8usize, 2usize, 256usize, 64usize);
    let (q_out, kv_out) = (nh * hd, nkv * hd);
    let stride = 2 * q_out + 2 * kv_out;
    let one = 1i64 << 32;
    let mut rng = SimpleRng::new(0xA77E_0001);
    let mut word =
        |lo: f32, hi: f32| (f64::from(rng.next_f32_range(lo, hi)) * 4_294_967_296.0) as i64;
    // Per column `[q | gate]` interleaved per head as the gated `attn_q` output, then k, v.
    let qfull: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..2 * q_out).map(|_| word(-2.0, 2.0)).collect())
        .collect();
    let kraw: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..kv_out).map(|_| word(-2.0, 2.0)).collect())
        .collect();
    let vraw: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..kv_out).map(|_| word(-2.0, 2.0)).collect())
        .collect();
    let gain: Vec<i64> = (0..hd).map(|_| one + word(-0.1, 0.1)).collect();
    let eps = one / 1_000_000;
    let rope = RopeTable::new(F64::from_f64(10_000_000.0), rot).expect("rope");
    let mut kv = KvState::with_carrier(1, nh, nkv, hd, 256, KvCarrier::F32).expect("kv");

    let mut prep = vec![0i64; k * stride];
    let mut att = vec![0i64; k * q_out];
    let mut last = vec![0usize; k];
    use crate::engine::chart::task::ChunkTask;
    // Pool task contexts (FORM-PARALLEL D5): shared borrows and `Copy` values only; each `run`
    // delegates to a named body that takes its pieces as separate parameters.
    #[derive(Clone, Copy)]
    struct Geom {
        k: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        stride: usize,
        q_out: usize,
        kv_out: usize,
    }
    fn fold_slot(n: usize, k: usize) -> usize {
        if n % 2 == 0 {
            n / 2
        } else {
            k - 1 - n / 2
        }
    }
    struct AttnAll<'a> {
        kv: &'a KvState,
        prep: &'a [i64],
        last: &'a [usize],
        g: Geom,
        gated: bool,
    }
    impl ChunkTask<i64> for AttnAll<'_> {
        fn run(&self, t: usize, _s: usize, dst: &mut [i64]) {
            attn_all_task(self.kv, self.prep, self.last, self.g, self.gated, t, dst);
        }
    }
    fn attn_all_task(
        kv: &KvState,
        prep: &[i64],
        last: &[usize],
        g: Geom,
        gated: bool,
        t: usize,
        dst: &mut [i64],
    ) {
        let (n, h) = (t / g.nh, t % g.nh);
        let c = fold_slot(n, g.k);
        let b = c * g.stride;
        kv.attn_head(0, h, &prep[b + h * g.hd..b + (h + 1) * g.hd], last[c], dst)
            .unwrap();
        if gated {
            let g0 = b + 2 * g.kv_out + g.q_out + h * g.hd;
            sigmoid_gate(dst, &prep[g0..g0 + g.hd]).unwrap();
        }
    }
    struct GateOnly<'a> {
        prep: &'a [i64],
        g: Geom,
        fast: bool,
    }
    impl ChunkTask<i64> for GateOnly<'_> {
        fn run(&self, t: usize, _s: usize, dst: &mut [i64]) {
            gate_only_task(self.prep, self.g, self.fast, t, dst);
        }
    }
    fn gate_only_task(prep: &[i64], g: Geom, fast: bool, t: usize, dst: &mut [i64]) {
        let (n, h) = (t / g.nh, t % g.nh);
        let b = fold_slot(n, g.k) * g.stride;
        let g0 = b + 2 * g.kv_out + g.q_out + h * g.hd;
        if fast {
            sigmoid_gate(dst, &prep[g0..g0 + g.hd]).unwrap();
        } else {
            for i in 0..g.hd {
                let s = fixed_sigmoid(prep[g0 + i]).unwrap();
                dst[i] = mul_q32(dst[i], s).unwrap();
            }
        }
    }
    struct CoreOnly<'a> {
        kp: &'a [F32],
        vp: &'a [F32],
        prep: &'a [i64],
        last: &'a [usize],
        g: Geom,
        readout: bool,
    }
    impl ChunkTask<i64> for CoreOnly<'_> {
        fn run(&self, t: usize, _s: usize, dst: &mut [i64]) {
            core_only_task(
                self.kp,
                self.vp,
                self.prep,
                self.last,
                self.g,
                self.readout,
                t,
                dst,
            );
        }
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the two cached planes, the activation, the last-position table, the geometry, the readout flag, and the task index and piece"
    )]
    fn core_only_task(
        kp: &[F32],
        vp: &[F32],
        prep: &[i64],
        last: &[usize],
        g: Geom,
        readout: bool,
        t: usize,
        dst: &mut [i64],
    ) {
        let (n, h) = (t / g.nh, t % g.nh);
        let c = fold_slot(n, g.k);
        let b = c * g.stride;
        let mut qf = [F32::ZERO; 256];
        for i in 0..g.hd {
            qf[i] = crate::engine::mixer::f32_carrier::q32_to_f32(prep[b + h * g.hd + i]);
        }
        let first = (h / (g.nh / g.nkv)) * g.hd;
        let mut of = [F32::ZERO; 256];
        attn_core_f32_strided(
            &qf[..g.hd],
            &kp[first..],
            &vp[first..],
            g.hd,
            g.nkv * g.hd,
            last[c],
            F32::from_bits(0x3d80_0000),
            &mut of[..g.hd],
        )
        .unwrap();
        if readout {
            for i in 0..g.hd {
                dst[i] = f32_to_q32_checked(of[i]).unwrap();
            }
        } else {
            dst[0] = of[0].to_bits() as i64;
        }
    }
    struct EmptyDispatch;
    impl ChunkTask<i64> for EmptyDispatch {
        fn run(&self, t: usize, _s: usize, dst: &mut [i64]) {
            dst[0] = t as i64;
        }
    }
    let geom = Geom {
        k,
        nh,
        nkv,
        hd,
        stride,
        q_out,
        kv_out,
    };
    let fold = |n: usize| fold_slot(n, k);

    let stage1 = |prep: &mut [i64]| {
        let f = |c: usize, dst: &mut [i64]| -> Result<(), ()> {
            let (mut query, mut key, value) = (Vec::new(), kraw[c].clone(), vraw[c].clone());
            let mut gate = Vec::new();
            for h in 0..nh {
                query.extend_from_slice(&qfull[c][h * 2 * hd..h * 2 * hd + hd]);
                gate.extend_from_slice(&qfull[c][h * 2 * hd + hd..(h + 1) * 2 * hd]);
            }
            qk_norm_heads(Some(&gain), &mut query, hd, eps).map_err(|_| ())?;
            qk_norm_heads(Some(&gain), &mut key, hd, eps).map_err(|_| ())?;
            rotate_pair_prefixes(&rope, &mut query, &mut key, hd, c as u32).map_err(|_| ())?;
            let (dq, rest) = dst.split_at_mut(q_out);
            let (dk, rest) = rest.split_at_mut(kv_out);
            let (dv, dg) = rest.split_at_mut(kv_out);
            dq.copy_from_slice(&query);
            dk.copy_from_slice(&key);
            dv.copy_from_slice(&value);
            dg.copy_from_slice(&gate);
            Ok(())
        };
        assert!(attn_heads(k, stride, prep, true, &FnHead(f)).is_none());
    };
    // Split of stage 1: norm only and rope only (same pool dispatch, same clones).
    let stage1_part = |prep: &mut [i64], do_norm: bool, do_rope: u8| {
        let f = |c: usize, dst: &mut [i64]| -> Result<(), ()> {
            let (mut query, mut key) = (qfull[c][..q_out].to_vec(), kraw[c].clone());
            if do_norm {
                qk_norm_heads(Some(&gain), &mut query, hd, eps).map_err(|_| ())?;
                qk_norm_heads(Some(&gain), &mut key, hd, eps).map_err(|_| ())?;
            }
            if do_rope == 1 {
                let pos = c as u32;
                for base in (0..q_out).step_by(hd) {
                    rope.rotate_one(&mut query[base..base + rot], pos)
                        .map_err(|_| ())?;
                }
                for base in (0..kv_out).step_by(hd) {
                    rope.rotate_one(&mut key[base..base + rot], pos)
                        .map_err(|_| ())?;
                }
            } else if do_rope == 2 {
                rotate_pair_prefixes(&rope, &mut query, &mut key, hd, c as u32).map_err(|_| ())?;
            }
            dst[..q_out].copy_from_slice(&query);
            dst[q_out..q_out + kv_out].copy_from_slice(&key);
            Ok(())
        };
        assert!(attn_heads(k, stride, prep, true, &FnHead(f)).is_none());
    };
    let append_all = |kv: &mut KvState, prep: &[i64], last: &mut [usize]| {
        kv.truncate(0).unwrap();
        for c in 0..k {
            let b = c * stride;
            last[c] = kv
                .append(
                    0,
                    &prep[b + q_out..b + q_out + kv_out],
                    &prep[b + q_out + kv_out..b + q_out + 2 * kv_out],
                )
                .unwrap();
        }
    };
    let attn_all = |kv: &KvState, prep: &[i64], last: &[usize], att: &mut [i64], gated: bool| {
        crate::engine::chart::pool::chunks_mut_global(
            att,
            hd,
            &AttnAll {
                kv,
                prep,
                last,
                g: geom,
                gated,
            },
        );
    };
    let gate_only = |prep: &[i64], att: &mut [i64], fast: bool| {
        crate::engine::chart::pool::chunks_mut_global(
            att,
            hd,
            &GateOnly {
                prep,
                g: geom,
                fast,
            },
        );
    };
    // core only: q cvt + online softmax, result discarded into a per-task stack buffer.
    let core_only = |kv: &KvState, prep: &[i64], last: &[usize], att: &mut [i64], readout: bool| {
        let (kp, vp) = match &kv.storage {
            crate::engine::attn::KvStorage::F32 { k, v } => (&k[0][..], &v[0][..]),
            crate::engine::attn::KvStorage::Q32 { .. }
            | crate::engine::attn::KvStorage::I8 { .. } => unreachable!(),
        };
        crate::engine::chart::pool::chunks_mut_global(
            att,
            hd,
            &CoreOnly {
                kp,
                vp,
                prep,
                last,
                g: geom,
                readout,
            },
        );
    };
    let empty_dispatch = |att: &mut [i64]| {
        crate::engine::chart::pool::chunks_mut_global(att, hd, &EmptyDispatch);
    };
    let tail = |att: &[i64]| -> Vec<Vec<i64>> {
        let mut atts: Vec<Vec<i64>> = (0..k).map(|_| Vec::new()).collect();
        for n in 0..k {
            atts[fold(n)] = att[n * q_out..(n + 1) * q_out].to_vec();
        }
        atts
    };

    // One full composed call, in the stage order of `softmax_core_block`.
    let full =
        |kv: &mut KvState, prep: &mut Vec<i64>, att: &mut Vec<i64>, last: &mut Vec<usize>| {
            let mut p = vec![0i64; k * stride];
            stage1(&mut p);
            append_all(kv, &p, last);
            let mut a = vec![0i64; k * q_out];
            attn_all(kv, &p, last, &mut a, true);
            let t = tail(&a);
            std::hint::black_box(&t);
            *prep = p;
            *att = a;
        };

    fn med(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
    let reps = 25usize;
    let mut rows: Vec<(&str, f64)> = Vec::new();
    macro_rules! seg {
        ($name:expr, $body:block) => {{
            let mut ts = Vec::new();
            for r in 0..reps + 3 {
                let t0 = Instant::now();
                $body;
                if r >= 3 {
                    ts.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            rows.push(($name, med(&mut ts)));
        }};
    }
    stage1(&mut prep);
    append_all(&mut kv, &prep, &mut last);
    seg!("full composed call", {
        full(&mut kv, &mut prep, &mut att, &mut last)
    });
    stage1(&mut prep);
    append_all(&mut kv, &prep, &mut last);
    seg!("stage1 (clone+split+qk norm+rope), pooled 200 tasks", {
        stage1(&mut prep)
    });
    seg!("  stage1 norm only", { stage1_part(&mut prep, true, 0) });
    seg!("  stage1 rope only (per-head fixed_sin_cos, old)", {
        stage1_part(&mut prep, false, 1)
    });
    seg!("  stage1 rope only (shared sin/cos, new)", {
        stage1_part(&mut prep, false, 2)
    });
    seg!("  stage1 clones/copies only", {
        stage1_part(&mut prep, false, 0)
    });
    seg!("prep calloc 200x5120 i64", {
        std::hint::black_box(vec![0i64; k * stride]);
    });
    seg!("stage2 kv append x200 (serial)", {
        append_all(&mut kv, &prep, &mut last)
    });
    seg!("stage3 attn (q cvt+core+readout), no gate", {
        attn_all(&kv, &prep, &last, &mut att, false)
    });
    seg!("  core only (q cvt + softmax), no readout", {
        core_only(&kv, &prep, &last, &mut att, false)
    });
    seg!("  core + readout", {
        core_only(&kv, &prep, &last, &mut att, true)
    });
    seg!("stage3 output gate only, oracle loop (old)", {
        gate_only(&prep, &mut att, false)
    });
    seg!("stage3 output gate only, sigmoid_gate (new)", {
        gate_only(&prep, &mut att, true)
    });
    seg!("stage3 attn + gate", {
        attn_all(&kv, &prep, &last, &mut att, true)
    });
    seg!("empty 1600-task dispatch", { empty_dispatch(&mut att) });
    seg!("att calloc 200x2048 i64", {
        std::hint::black_box(vec![0i64; k * q_out]);
    });
    seg!("tail: 200 to_vec gather", {
        std::hint::black_box(tail(&att));
    });
    println!("--- attn layer segments, k={k} pos0=0 (median of {reps}, ms/call) ---");
    let total = rows[0].1;
    for (n, ms) in &rows {
        println!("{ms:8.3} ms  {:5.1}%  {n}", 100.0 * ms / total);
    }
}
