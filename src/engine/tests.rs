use super::*;
use crate::engine::num::dequant::dequant_row_fixed;
use crate::engine::num::fp::F64;
use crate::engine::testdata::model_path;
use crate::format::quant::{
    self, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_0_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_trig::turn_to_radians;

#[test]
fn f16_scale_conversion_is_exact_and_total_on_finite_values() {
    assert_eq!(f16_to_fixed(0x0000).expect("+0"), 0);
    assert_eq!(f16_to_fixed(0x8000).expect("-0"), 0);
    assert_eq!(f16_to_fixed(0x3C00).expect("1.0"), ONE);
    assert_eq!(f16_to_fixed(0x3800).expect("0.5"), ONE >> 1);
    assert_eq!(f16_to_fixed(0xC000).expect("-2.0"), -(ONE << 1));
    // Smallest positive subnormal: 2^-24 -> 2^8 units.
    assert_eq!(f16_to_fixed(0x0001).expect("min subnormal"), 1 << 8);
    // Largest finite f16, 65504 -> 65504 * 2^32.
    assert_eq!(f16_to_fixed(0x7BFF).expect("max finite"), 65504 * ONE);
    // Every finite bit pattern converts; only the Inf/NaN exponent refuses.
    let mut h = 0u32;
    while h < 0x1_0000 {
        let bits = u16::try_from(h).expect("h < 2^16");
        let finite = (bits >> 10) & 0x1f != 0x1f;
        assert_eq!(f16_to_fixed(bits).is_ok(), finite, "bits {bits:#06x}");
        h += 1;
    }
}

#[test]
fn q8_0_block_is_scale_times_code() {
    let mut raw = vec![0u8; Q8_0_BLOCK_BYTES];
    raw[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    raw[2] = 2u8; // +2
    raw[3] = 0xFEu8; // -2
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q8_0, &raw, 2, &mut out).expect("q8_0 row");
    assert_eq!(out, vec![ONE, -ONE]);
}

#[test]
fn q4_k_block_applies_scale_and_min() {
    // d = 1.0, dmin = 1.0, pair 0 = (sc 5, min 7), code 3 -> 5*3 - 7 = 8.
    let mut raw = vec![0u8; Q4_K_BLOCK_BYTES];
    raw[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    raw[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
    raw[4] = 5;
    raw[8] = 7;
    raw[16] = 0x03;
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q4_K, &raw, 256, &mut out).expect("q4_k row");
    assert_eq!(out[0], 8 * ONE);
    assert_eq!(out.len(), 256);
}

#[test]
fn q5_0_block_is_scale_times_code() {
    let mut raw = vec![0u8; Q5_0_BLOCK_BYTES];
    raw[..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    raw[4] = 0x01; // qh[2]: fifth bit of output 16
    raw[6] = 0x21; // qs[0]: low nibble 1 -> -15, high nibble 2
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q5_0, &raw, 32, &mut out).expect("q5_0 row");
    assert_eq!(out.len(), 32);
    assert_eq!(out[0], -15 * ONE);
    assert_eq!(out[1], -16 * ONE);
    assert_eq!(out[16], 2 * ONE); // (2 | 16) - 16
    assert_eq!(out[17], -16 * ONE);
}

#[test]
fn q2_k_block_applies_scale_and_min() {
    // d = 1.0, dmin = 0.5; group 0 = (scale 1, min 5), group 1 = (scale 1, min 2).
    let mut raw = vec![0u8; Q2_K_BLOCK_BYTES];
    raw[0] = 0x51;
    raw[1] = 0x21;
    raw[16] = 0x03; // output 0 code 3
    raw[32] = 0x03; // output 16 code 3
    raw[80..82].copy_from_slice(&0x3C00u16.to_le_bytes());
    raw[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q2_K, &raw, 256, &mut out).expect("q2_k row");
    assert_eq!(out.len(), 256);
    assert_eq!(out[0], ONE / 2); // 1*3 - 0.5*5
    assert_eq!(out[16], 2 * ONE); // 1*3 - 0.5*2
    assert_eq!(out[17], -ONE); // 1*0 - 0.5*2
}

#[test]
fn q3_k_block_applies_signed_scale() {
    // d = 1.0; group 0 scale code 40 (effective 8), group 1 scale code 0 (effective -32),
    // every code -4 (`hmask` clear).
    let mut raw = vec![0u8; Q3_K_BLOCK_BYTES];
    raw[96] = 0x08; // low nibble of the shuffled scale word
    raw[96 + 8] = 0x02; // (2 & 3) << 4 = 32, so scale 0 = 8 | 32 = 40
    raw[108..110].copy_from_slice(&0x3C00u16.to_le_bytes());
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q3_K, &raw, 256, &mut out).expect("q3_k row");
    assert_eq!(out.len(), 256);
    assert_eq!(out[0], -32 * ONE); // 8 * (-4)
    assert_eq!(out[16], 128 * ONE); // -32 * (-4)
}

#[test]
fn q6_k_block_applies_signed_scale() {
    // d = 1.0; group 0 scale 2, group 1 scale 3, every code 0 - 32.
    let mut raw = vec![0u8; Q6_K_BLOCK_BYTES];
    raw[192] = 2;
    raw[193] = 3;
    raw[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    let mut out = Vec::new();
    dequant_row_fixed(quant::GGML_TYPE_Q6_K, &raw, 256, &mut out).expect("q6_k row");
    assert_eq!(out.len(), 256);
    assert_eq!(out[0], -64 * ONE); // 2 * (-32)
    assert_eq!(out[16], -96 * ONE); // 3 * (-32)
}

#[test]
fn bf16_row_agrees_with_the_fold_converter_on_every_pattern() {
    use crate::learn::fold::run::convert::{decode_bf16, ConvertError};
    let mut mismatched: Vec<u16> = Vec::new();
    let mut h = 0u32;
    while h < 0x1_0000 {
        let bits = u16::try_from(h).expect("h < 2^16");
        let raw = bits.to_le_bytes();
        let mut out = Vec::new();
        let got = dequant_row_fixed(quant::GGML_TYPE_BF16, &raw, 1, &mut out);
        let want: Result<Vec<i64>, ConvertError> = decode_bf16(&raw);
        let same = match (&got, &want) {
            (Ok(()), Ok(w)) => out == *w,
            (Err(_), Err(_)) => true,
            _ => false,
        };
        if !same {
            mismatched.push(bits);
        }
        h += 1;
    }
    assert!(
        mismatched.is_empty(),
        "{} bf16 patterns differ, first {:?}",
        mismatched.len(),
        &mismatched[..mismatched.len().min(8)]
    );
}

#[test]
fn unsupported_embedding_quant_refuses() {
    let mut out = Vec::new();
    let e = dequant_row_fixed(quant::GGML_TYPE_F32, &[], 1, &mut out)
        .expect_err("F32 has no fixed-point reader");
    assert_eq!(e, EngineError::UnsupportedEmbeddingQuant { ggml_type: 0 });
}

#[test]
fn rms_norm_fixed_has_an_exact_expected_vector() {
    // x = [1, -1, 1, -1]; sum x^2 = 4*2^64, mean = 4*2^64 / (4*2^32) = 2^32 = ONE.
    // eps = 3*ONE gives arg = 4*ONE and fixed_rsqrt(4*ONE) = ONE/2 exactly.
    let x = [ONE, -ONE, ONE, -ONE];
    let y = rms_norm_fixed(&x, 3 * ONE).expect("4*ONE is inside the rsqrt window");
    assert_eq!(y, [ONE / 2, -(ONE / 2), ONE / 2, -(ONE / 2)]);
}

#[test]
fn rms_norm_fixed_refuses_empty_and_out_of_window() {
    assert_eq!(rms_norm_fixed(&[], ONE), Err(NormError::EmptyRow));
    // Zero mean with zero epsilon is below the rsqrt window.
    assert_eq!(rms_norm_fixed(&[0, 0], 0), Err(NormError::OutOfRange));
    // Mean 2^54 exceeds the window upper bound 2^52.
    assert_eq!(rms_norm_fixed(&[ONE << 11], 0), Err(NormError::OutOfRange));
}

#[test]
fn rope_table_matches_the_load_time_boundary() {
    // theta = 4: turn_0 = round(2^64 / 2pi) and turn_1 = round((1/2) * 2^64 / 2pi).
    let t = RopeTable::new(F64::from_f64(4.0), 4).expect("theta 4, head_dim 4");
    assert_eq!(t.head_dim(), 4);
    assert_eq!(
        t.turns(),
        [2_935_890_503_282_001_408, 1_467_945_251_641_000_704]
    );
}

#[test]
fn rope_zero_angle_is_the_identity() {
    let t = RopeTable::new(F64::from_f64(10_000.0), 8).expect("head_dim 8");
    assert_eq!(t.turns().len(), 4);
    let q0 = [ONE, -3 * ONE, 5, -7 * ONE, 11, 13, ONE / 3, -(ONE / 3)];
    let k0 = [2, 3, 5, 7, 11, 13, 17, 19];
    let (mut q, mut k) = (q0, k0);
    rope_rotate_fixed(&mut q, &mut k, 0, &t).expect("turn 0 is the identity");
    assert_eq!(q, q0);
    assert_eq!(k, k0);
}

#[test]
fn rope_quarter_turn_is_hand_computed() {
    // head_dim 4, turns = [2^62, 0]: at pos 1 the pair (j=0) rotates by a quarter turn
    // (cos 0, sin ONE) and (j=1) by zero. turn_to_radians(2^62) = round(pi/2 * 2^32), so every
    // output is exact.
    const QUARTER_TURN: u64 = 1 << 62;
    let t = RopeTable::from_turns(4, vec![QUARTER_TURN, 0]).expect("explicit table");
    let (mut q, mut k) = (
        [ONE, 2 * ONE, 3 * ONE, 4 * ONE],
        [5 * ONE, 6 * ONE, 7 * ONE, 8 * ONE],
    );
    rope_rotate_fixed(&mut q, &mut k, 1, &t).expect("a quarter turn is in the window");
    assert_eq!(q, [-3 * ONE, 2 * ONE, ONE, 4 * ONE]);
    assert_eq!(k, [-7 * ONE, 6 * ONE, 5 * ONE, 8 * ONE]);
}

#[test]
fn rope_turn_phase_wraps_mod_2_to_64_in_rotate() {
    // turn = 2^62 (quarter turn), pos 5: 5 * 2^62 = 1.25 * 2^64 wraps to 2^62, a quarter turn.
    // A phase that saturates at 2^64 instead of wrapping would leave the pair unrotated, so this
    // exercises the `& u64::MAX` mask in `rotate_head`, not a test-local copy of it.
    let t = RopeTable::from_turns(2, vec![1 << 62]).expect("one rotary dim");
    let (mut q, mut k) = ([ONE, 0], [2 * ONE, 0]);
    rope_rotate_fixed(&mut q, &mut k, 5, &t).expect("wrapped quarter turn");
    assert_eq!(q, [0, ONE]);
    assert_eq!(k, [0, 2 * ONE]);
}

#[test]
fn rope_turn_to_radians_maps_the_named_phases() {
    // The half-turn phase (-2^63 after the two's-complement lift) maps to -13_493_037_704, the
    // round-to-nearest result within 1 unit of -round(pi * 2^32) = -13_493_037_705; the exact
    // value pins the `+ 2^63` rounding bias. 2^62 is a quarter turn at round(pi/2 * 2^32).
    assert_eq!(turn_to_radians(0), 0);
    assert_eq!(turn_to_radians(1u64 << 63), -13_493_037_704);
    assert_eq!(turn_to_radians(1u64 << 62), 6_746_518_852);
}

#[test]
fn rope_admits_positions_past_the_old_window() {
    // The radian path refused above position 4096; the turn path is total on every u32 position.
    let t = RopeTable::new(F64::from_f64(10_000.0), 8).expect("head_dim 8");
    let q0 = [ONE, -3 * ONE, 5, -7 * ONE, 11, 13, ONE / 3, -(ONE / 3)];
    let k0 = [2, 3, 5, 7, 11, 13, 17, 19];
    for pos in [4096u32, 4097, 32_767, u32::MAX] {
        let (mut q, mut k) = (q0, k0);
        rope_rotate_fixed(&mut q, &mut k, pos, &t)
            .unwrap_or_else(|e| panic!("pos {pos} refused: {e:?}"));
        let mut x = q0;
        t.rotate_one(&mut x, pos)
            .unwrap_or_else(|e| panic!("rotate_one pos {pos} refused: {e:?}"));
    }
}

#[test]
fn rope_turn_phase_is_additive_mod_2_to_64() {
    // The phase is a group homomorphism Z -> Z/2^64: phase(p + q) = phase(p) + phase(q).
    let t = RopeTable::new(F64::from_f64(1.0e6), 64).expect("qwen rotary table");
    let (p, q) = (32_767u32, 4097u32);
    for &turn in t.turns() {
        let phase = |pos: u32| turn.wrapping_mul(u64::from(pos));
        assert_eq!(
            phase(p + q),
            phase(p).wrapping_add(phase(q)),
            "turn {turn} is not additive"
        );
        // The same identity in the u128 form `rotate_head` masks.
        let wide = |pos: u32| (u128::from(turn) * u128::from(pos)) & u128::from(u64::MAX);
        assert_eq!(wide(p + q), (wide(p) + wide(q)) & u128::from(u64::MAX));
    }
}

#[test]
fn rope_score_depends_on_the_relative_offset_only() {
    // faces R1 H1 falsifier on the shipped rotation: <R(m)q, R(n)k> = g(n - m). Prediction written
    // before the run: the shifted score differs from the base score by fixed-point rounding only
    // (<= 2^-20 of |q||k|), while moving q alone changes the score by more than 2^-6 of |q||k|.
    let t = RopeTable::new(F64::from_f64(1.0e6), 64).expect("qwen rotary table");
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut draw = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        i64::try_from(state >> 40).expect("24 bits") - (1 << 23)
    };
    let q0: Vec<i64> = (0..64).map(|_| draw() << 8).collect();
    let k0: Vec<i64> = (0..64).map(|_| draw() << 8).collect();
    let dot = |a: &[i64], b: &[i64]| -> i128 {
        a.iter()
            .zip(b)
            .map(|(x, y)| i128::from(*x) * i128::from(*y))
            .sum()
    };
    let scale = dot(&q0, &q0).isqrt() * dot(&k0, &k0).isqrt();
    let score = |m: u32, n: u32| {
        let (mut q, mut k) = (q0.clone(), k0.clone());
        t.rotate_one(&mut q, m).expect("q in window");
        t.rotate_one(&mut k, n).expect("k in window");
        dot(&q, &k)
    };
    let (mut worst_shift, mut worst_move) = (0i128, 0i128);
    for (m, n) in [(0u32, 5u32), (3, 100), (17, 17), (200, 1), (1000, 4096)] {
        for s in [1u32, 7, 513, 20_000] {
            worst_shift = worst_shift.max((score(m + s, n + s) - score(m, n)).abs());
            worst_move = worst_move.max((score(m + s, n) - score(m, n)).abs());
        }
    }
    eprintln!("rope relative: scale {scale} worst_shift {worst_shift} worst_move {worst_move}");
    assert!(
        worst_shift <= scale >> 20,
        "relative-offset identity fails: {worst_shift} vs {scale}"
    );
    assert!(
        worst_move > scale >> 6,
        "control: moving q alone must change the score"
    );
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the f64 oracle and the Q32.32 angle are compared in floating point; the 2^-31 \
              tolerance absorbs the lossy i64->f64 cast and the small j/head_dim casts, and the \
              u128 mask makes the final u64 cast exact"
)]
#[test]
fn rope_turn_angle_tracks_theta_within_2_to_the_minus_31() {
    use core::f64::consts::PI;
    const THETA: f64 = 1.0e6;
    const HEAD_DIM: usize = 64;
    let t = RopeTable::new(F64::from_f64(THETA), HEAD_DIM).expect("qwen rotary table");
    let hd = HEAD_DIM as f64;
    let tau = 2.0 * PI;
    let tol = 2.0f64.powi(-31);
    let mut worst = 0.0f64;
    for (j, &turn) in t.turns().iter().enumerate() {
        let theta_j = THETA.powf(-2.0 * (j as f64) / hd);
        let mut pos = 0u32;
        while pos <= 32_768 {
            let phase = ((u128::from(turn) * u128::from(pos)) & u128::from(u64::MAX)) as u64;
            let executed = turn_to_radians(phase) as f64 / 4_294_967_296.0;
            let mut ideal = (f64::from(pos) * theta_j).rem_euclid(tau);
            if ideal > PI {
                ideal -= tau;
            }
            let err = (executed - ideal).abs();
            worst = worst.max(err.min(tau - err));
            pos += 1;
        }
    }
    assert!(worst <= tol, "worst turn angle error {worst} > {tol}");
}

#[test]
fn rope_refuses_malformed_tables_and_short_vectors() {
    assert_eq!(
        RopeTable::new(F64::from_f64(1e4), 3),
        Err(RopeError::BadHeadDim)
    );
    assert_eq!(
        RopeTable::new(F64::from_f64(0.0), 4),
        Err(RopeError::BadTheta)
    );
    assert_eq!(
        RopeTable::new(F64::from_f64(f64::NAN), 4),
        Err(RopeError::BadTheta)
    );
    assert_eq!(
        RopeTable::from_turns(4, vec![0]),
        Err(RopeError::TableLength)
    );

    // A vector that is not a whole number of heads refuses.
    let t = RopeTable::from_turns(2, vec![0]).expect("one rotary dim");
    let (mut q3, mut k3) = ([ONE, ONE, ONE], [ONE, ONE, ONE]);
    assert_eq!(
        rope_rotate_fixed(&mut q3, &mut k3, 0, &t),
        Err(RopeError::VectorLength)
    );
}

#[test]
fn checksum_is_order_sensitive_and_stable() {
    let a = fixed_vector_checksum(&[1, 2, 3]);
    assert_eq!(a, fixed_vector_checksum(&[1, 2, 3]));
    assert_ne!(a, fixed_vector_checksum(&[3, 2, 1]));
    assert_ne!(a, fixed_vector_checksum(&[1, 2]));
}

#[test]
fn matvec_fixed_hand_computed_two_by_two_with_a_rounding_tie() {
    // Two Q8_0 rows with d = 0.5, so each weight is `code / 2`: row 0 = [1.5, 0],
    // row 1 = [1.0, 0], exact Q32.32 weights.
    let mut raw = vec![0u8; 2 * Q8_0_BLOCK_BYTES];
    let mut r = 0usize;
    while r < 2 {
        let base = r * Q8_0_BLOCK_BYTES;
        raw[base..base + 2].copy_from_slice(&0x3800u16.to_le_bytes());
        r += 1;
    }
    raw[2] = 3;
    raw[Q8_0_BLOCK_BYTES + 2] = 2;
    // x = [1, 1]: y0 = round(1.5) = 2, a rounding tie upward; y1 = round(1.0) = 1 exactly.
    let x = [1, 1];
    let y = matvec_fixed(quant::GGML_TYPE_Q8_0, &raw, 2, 2, &x).expect("2x2 projection");
    assert_eq!(y, vec![2, 1]);
    assert_eq!(row_dot_fixed(&[3 * ONE / 2, 0], &x), Ok(2));
    assert_eq!(row_dot_fixed(&[ONE, 0], &x), Ok(1));
}

#[test]
fn matvec_fixed_refuses_shape_short_payload_and_bad_type() {
    let x = [ONE, ONE];
    assert_eq!(
        matvec_fixed(quant::GGML_TYPE_Q8_0, &[0u8; 68], 2, 2, &[ONE]),
        Err(MatvecError::ShapeMismatch { x: 1, n_in: 2 })
    );
    // 34 bytes hold one Q8_0 row; two output rows need 68.
    assert_eq!(
        matvec_fixed(quant::GGML_TYPE_Q8_0, &[0u8; 34], 2, 2, &x),
        Err(MatvecError::ShortPayload { need: 68, have: 34 })
    );
    // An unknown ggml type has no row geometry at all.
    assert_eq!(
        matvec_fixed(999, &[], 1, 1, &[ONE]),
        Err(MatvecError::UnsupportedQuant { ggml_type: 999 })
    );
    // F32 has geometry but no fixed-point row reader.
    match matvec_fixed(quant::GGML_TYPE_F32, &[0u8; 4], 1, 1, &[ONE]) {
        Err(MatvecError::Dequant(EngineError::UnsupportedEmbeddingQuant { ggml_type })) => {
            assert_eq!(ggml_type, quant::GGML_TYPE_F32);
        }
        other => panic!("expected an F32 quant refusal, got {other:?}"),
    }
}

#[test]
fn row_dot_fixed_rounds_ties_and_refuses_out_of_range() {
    // The output is `round(Σ w·x / 2^32)`; half-integers round upward (toward +infinity).
    assert_eq!(row_dot_fixed(&[ONE / 2], &[3]), Ok(2)); // 1.5 -> 2
    assert_eq!(row_dot_fixed(&[ONE / 2], &[-3]), Ok(-1)); // -1.5 -> -1
    assert_eq!(row_dot_fixed(&[ONE / 2], &[1]), Ok(1)); // 0.5 -> 1
                                                        // Exact: 1.0 + 2.0 - 1.0 = 2.0.
    assert_eq!(
        row_dot_fixed(&[ONE, ONE, -ONE], &[ONE, 2 * ONE, ONE]),
        Ok(2 * ONE)
    );
    assert_eq!(
        row_dot_fixed(&[ONE, ONE], &[ONE]),
        Err(MatvecError::RowLength {
            row: 0,
            weights: 2,
            n_in: 1,
        })
    );
    // Three maximal products leave i128; the accumulator refuses, never wraps.
    assert_eq!(
        row_dot_fixed(&[i64::MAX; 3], &[i64::MAX; 3]),
        Err(MatvecError::AccumulatorOverflow)
    );
    // A rounded value past i64::MAX refuses rather than clamping.
    assert_eq!(
        row_dot_fixed(&[i64::MAX], &[2 * ONE]),
        Err(MatvecError::OutputOverflow)
    );
}

/// Default model named by the task; the real-table test skips only when it is absent.
const QWEN05B_PATH: &str = "models/gguf/qwen05b-Q4_K_M.gguf";

#[test]
fn token_embd_row_dot_through_the_real_tensor_table() {
    let Some(path) = model_path(QWEN05B_PATH) else {
        eprintln!("skip token_embd_row_dot_through_the_real_tensor_table: {QWEN05B_PATH} absent");
        return;
    };
    let engine = Engine::open(&path).expect("load qwen05b");
    let tensor = engine.token_embd();
    let n_in = engine.config.n_embd;
    let act = engine.token_embedding(0).expect("token 0 embedding");
    let (qk, bb) = quant::block_geom(tensor.ggml_type).expect("token_embd type has geometry");
    let row_bytes = n_in.div_ceil(qk) * bb;
    let raw = engine
        .read_range(
            tensor.byte_range.start,
            u64::try_from(row_bytes).expect("row bytes fit u64"),
        )
        .expect("token_embd row 0 payload");

    let dot = matvec_fixed(tensor.ggml_type, &raw, 1, n_in, &act).expect("row-dot row 0");
    assert_eq!(dot.len(), 1);

    // Independent recomputation of the same row with a fresh i128 accumulator.
    let mut weights = Vec::new();
    dequant_row_fixed(tensor.ggml_type, &raw, n_in, &mut weights).expect("dequant row 0");
    let mut acc: i128 = 0;
    let mut idx = 0usize;
    while idx < n_in {
        acc += i128::from(weights[idx]) * i128::from(act[idx]);
        idx += 1;
    }
    let expected = i64::try_from((acc + (1i128 << 31)) >> 32).expect("dot fits i64");
    assert_eq!(dot[0], expected);

    // Bounded and deterministic across an independent load.
    assert!(
        dot[0].abs() < (1i64 << 40),
        "dot {} left the Q8_0 embedding band",
        dot[0]
    );
    let second = Engine::open(&path).expect("second load");
    let act_reload = second
        .token_embedding(0)
        .expect("token 0 embedding (second)");
    let dot_reload =
        matvec_fixed(tensor.ggml_type, &raw, 1, n_in, &act_reload).expect("row-dot row 0");
    assert_eq!(dot, dot_reload);
}

#[test]
fn attn_head_hand_computed_two_key_softmax() {
    // head_dim 4 gives scale = fixed_rsqrt(4*ONE) = ONE/2 exactly. q = [ONE, 0, 0, 0] makes
    // each raw dot the key's first component, and both keys carry 2*ONE there, so the scaled
    // scores are both ONE. Max shift [0, 0] -> exp [ONE, ONE] -> total 2*ONE -> p = ONE/2 each,
    // so out = (v0 + v1) / 2, exact.
    let mut kv = KvState::new(1, 1, 1, 4, 4).expect("geometry");
    assert_eq!(
        kv.append(0, &[2 * ONE, 0, 0, 0], &[ONE, 2 * ONE, 0, -ONE]),
        Ok(0)
    );
    assert_eq!(
        kv.append(0, &[2 * ONE, 0, 0, 0], &[3 * ONE, 0, 2 * ONE, ONE]),
        Ok(1)
    );
    let q = [ONE, 0, 0, 0];
    let mut out = [0i64; 4];
    kv.attn_head(0, 0, &q, 1, &mut out).expect("attend pos 1");
    assert_eq!(out, [2 * ONE, ONE, ONE, 0]);
    // The scale is the load-time fixed_rsqrt, not a restated constant.
    assert_eq!(kv.scale(), ONE / 2);
}

#[test]
fn attn_head_distinct_scores_pin_the_scale() {
    // head_dim 4 gives scale ONE/2. q = [ONE, 0, 0, 0] makes each raw dot its key's first
    // component; keys 0 and 2*LN2_Q32 give scores 0 and LN2_Q32. The max shift is then
    // [-LN2_Q32, 0] and fixed_exp(-LN2_Q32) = ONE/2 exactly, so e = [ONE/2, ONE] and
    // total = 1.5*ONE. With v0 = 0 and v1 = [ONE, 0, 0, 0],
    // p1 = round(ONE*2^32 / (1.5*ONE)) = round(2^32/3*2) = 2863311531, and out[0] = p1.
    const LN2_Q32: i64 = 2_977_044_472;
    let mut kv =
        KvState::with_carrier(1, 1, 1, 4, 4, super::attn::KvCarrier::Q32).expect("geometry");
    assert_eq!(kv.append(0, &[0, 0, 0, 0], &[0, 0, 0, 0]), Ok(0));
    assert_eq!(
        kv.append(0, &[2 * LN2_Q32, 0, 0, 0], &[ONE, 0, 0, 0]),
        Ok(1)
    );
    let q = [ONE, 0, 0, 0];
    let mut out = [0i64; 4];
    kv.attn_head(0, 0, &q, 1, &mut out).expect("attend pos 1");
    assert_eq!(out, [2_863_311_531, 0, 0, 0]);
}

#[test]
fn attn_head_gqa_groups_query_heads() {
    // n_head 4 over n_head_kv 2 gives group 2: query heads 0,1 read KV head 0 and 2,3 read
    // KV head 1. head_dim 1 gives scale = ONE, so each output is its KV head's value verbatim.
    let mut kv = KvState::new(1, 4, 2, 1, 2).expect("geometry");
    assert_eq!(kv.append(0, &[2 * ONE, 2 * ONE], &[ONE, 3 * ONE]), Ok(0));
    let q = [ONE];
    let mut out0 = [0i64; 1];
    let mut out1 = [0i64; 1];
    let mut out2 = [0i64; 1];
    let mut out3 = [0i64; 1];
    kv.attn_head(0, 0, &q, 0, &mut out0).expect("head 0");
    kv.attn_head(0, 1, &q, 0, &mut out1).expect("head 1");
    kv.attn_head(0, 2, &q, 0, &mut out2).expect("head 2");
    kv.attn_head(0, 3, &q, 0, &mut out3).expect("head 3");
    assert_eq!(out0, [ONE]);
    assert_eq!(out1, [ONE]);
    assert_eq!(out2, [3 * ONE]);
    assert_eq!(out3, [3 * ONE]);
}

#[test]
fn attn_head_causal_mask_hides_later_positions() {
    // Two equal-score keys, values ONE and 64*ONE. Position 0 sees only the first (out ONE);
    // position 1 sees both with p = ONE/2 each, so out = (ONE + 64*ONE)/2 = 32.5*ONE.
    let mut kv = KvState::new(1, 1, 1, 1, 2).expect("geometry");
    assert_eq!(kv.append(0, &[ONE], &[ONE]), Ok(0));
    assert_eq!(kv.append(0, &[ONE], &[64 * ONE]), Ok(1));
    let q = [ONE];
    let mut out = [0i64; 1];
    kv.attn_head(0, 0, &q, 0, &mut out).expect("pos 0");
    assert_eq!(out, [ONE]);
    kv.attn_head(0, 0, &q, 1, &mut out).expect("pos 1");
    assert_eq!(out, [32 * ONE + ONE / 2]);
}

#[test]
fn attn_refuses_bad_geometry_shapes_positions_and_capacity() {
    assert!(matches!(
        KvState::new(0, 1, 1, 1, 1),
        Err(AttnError::BadGeometry)
    ));
    assert!(matches!(
        KvState::new(1, 3, 2, 1, 1),
        Err(AttnError::BadGeometry)
    ));
    assert!(matches!(
        KvState::new(1, 0, 1, 1, 1),
        Err(AttnError::BadGeometry)
    ));

    let mut kv = KvState::new(1, 2, 1, 2, 1).expect("geometry");
    assert_eq!(
        kv.append(1, &[ONE, ONE], &[ONE, ONE]),
        Err(AttnError::LayerOutOfRange {
            layer: 1,
            n_layer: 1,
        })
    );
    assert_eq!(
        kv.append(0, &[ONE], &[ONE, ONE]),
        Err(AttnError::RowLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(kv.append(0, &[ONE, ONE], &[ONE, ONE]), Ok(0));
    assert_eq!(
        kv.append(0, &[ONE, ONE], &[ONE, ONE]),
        Err(AttnError::CapacityExceeded { cap: 1 })
    );

    let q = [ONE, ONE];
    let mut out = [0i64; 2];
    assert_eq!(
        kv.attn_head(0, 0, &q, 1, &mut out),
        Err(AttnError::PositionOutOfRange { pos: 1, len: 1 })
    );
    assert_eq!(
        kv.attn_head(0, 2, &q, 0, &mut out),
        Err(AttnError::HeadOutOfRange { head: 2, n_head: 2 })
    );
    assert_eq!(
        kv.attn_head(0, 0, &[ONE], 0, &mut out),
        Err(AttnError::RowLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        kv.attn_head(0, 0, &q, 0, &mut out[..1]),
        Err(AttnError::RowLength {
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn attn_saturates_out_of_window_exp_and_refuses_dot_overflow() {
    // head_dim 1, scale ONE. q = 64*ONE against keys 64*ONE and -64*ONE gives shifted scores
    // 0 and -8192*ONE; the second saturates its numerator to 0, so the query attends the first
    // key exactly.
    let mut kv =
        KvState::with_carrier(1, 1, 1, 1, 2, super::attn::KvCarrier::Q32).expect("geometry");
    assert_eq!(kv.append(0, &[64 * ONE], &[ONE]), Ok(0));
    assert_eq!(kv.append(0, &[-64 * ONE], &[ONE]), Ok(1));
    let q = [64 * ONE];
    let mut out = [0i64; 1];
    kv.attn_head(0, 0, &q, 1, &mut out)
        .expect("saturated softmax");
    assert_eq!(out, [ONE]);
    assert!(kv.attn_head(0, 0, &q, 0, &mut out).is_ok());

    // A dot whose i128 accumulation leaves range refuses rather than wrapping.
    let mut big =
        KvState::with_carrier(1, 1, 1, 3, 1, super::attn::KvCarrier::Q32).expect("geometry");
    assert_eq!(big.append(0, &[i64::MAX; 3], &[0, 0, 0]), Ok(0));
    assert_eq!(
        big.attn_head(0, 0, &[i64::MAX; 3], 0, &mut [0i64; 3]),
        Err(AttnError::Overflow)
    );
}

#[test]
fn attn_head_is_deterministic_across_calls() {
    let mut kv = KvState::new(2, 4, 2, 2, 3).expect("geometry");
    let keys = [0i64, ONE / 4, ONE / 2];
    let mut l = 0usize;
    while l < 2 {
        let mut s = 0usize;
        while s < 3 {
            let k = keys[s];
            let sep = i64::try_from(s + 1).expect("small") * ONE;
            let pos = kv
                .append(l, &[k, 0, k, 0], &[sep, -sep, 2 * sep, -2 * sep])
                .expect("append");
            assert_eq!(pos, s);
            s += 1;
        }
        l += 1;
    }
    assert_eq!(kv.occupied(0), Ok(3));
    assert_eq!(
        kv.occupied(2),
        Err(AttnError::LayerOutOfRange {
            layer: 2,
            n_layer: 2,
        })
    );

    let q = [ONE, 0];
    let mut a = [0i64; 2];
    let mut b = [0i64; 2];
    kv.attn_head(0, 1, &q, 2, &mut a).expect("first call");
    kv.attn_head(0, 1, &q, 2, &mut b).expect("second call");
    assert_eq!(a, b);
}

/// A deterministic Q32.32 word in `[-4·ONE, 4·ONE)` from a 64-bit LCG state: reproducible
/// non-trivial attention keys, queries and values without an RNG dependency.
fn attn_word(state: &mut u64) -> i64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let wide = i64::try_from(*state >> 1).expect("the top 63 bits fit i64");
    wide % (8 * ONE) - 4 * ONE
}

#[test]
fn attn_fused_den_num_are_independent_of_tile_count() {
    let mut kv =
        KvState::with_carrier(1, 2, 1, 4, 8, super::attn::KvCarrier::Q32).expect("geometry");
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut s = 0usize;
    while s < 8 {
        let k = [
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
        ];
        let v = [
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
        ];
        assert_eq!(kv.append(0, &k, &v), Ok(s));
        s += 1;
    }
    let q = [ONE, ONE / 2, -ONE / 4, ONE / 8];
    let pos = 7usize;
    let m = kv.attn_max(0, 0, &q, pos).expect("max over the prefix");
    let reference = kv
        .attn_fused_state(0, 0, &q, pos, m, 1)
        .expect("unit tiles");
    assert_eq!(reference.m, m);
    let mut tile = 1usize;
    while tile <= 9 {
        let got = kv
            .attn_fused_state(0, 0, &q, pos, m, tile)
            .expect("tiled fold");
        assert_eq!(got.den, reference.den, "den changed with tile {tile}");
        assert_eq!(got.num, reference.num, "num changed with tile {tile}");
        tile += 1;
    }
}

#[test]
fn attn_head_fused_matches_per_position_oracle_within_bound() {
    let mut kv = KvState::new(1, 2, 1, 4, 8).expect("geometry");
    let mut seed = 0x1234_5678_9ABC_DEF0_u64;
    let mut value_rows: Vec<[i64; 4]> = Vec::new();
    let mut s = 0usize;
    while s < 8 {
        let k = [
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
        ];
        let v = [
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
        ];
        assert_eq!(kv.append(0, &k, &v), Ok(s));
        value_rows.push(v);
        s += 1;
    }
    let one = i128::from(ONE);
    let mut pos = 0usize;
    while pos < 8 {
        let q = [
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
            attn_word(&mut seed),
        ];
        let mut head = 0usize;
        while head < 2 {
            let mut oracle = [0i64; 4];
            kv.attn_head(0, head, &q, pos, &mut oracle)
                .expect("per-position oracle");
            let mut fused = [0i64; 4];
            kv.attn_head_fused(0, head, &q, pos, 1 + pos % 3, &mut fused)
                .expect("fused read");
            let mut i = 0usize;
            while i < 4 {
                let mut sum_abs: i128 = 0;
                let mut j = 0usize;
                while j <= pos {
                    sum_abs += i128::from(value_rows[j][i].abs());
                    j += 1;
                }
                let dev = (i128::from(oracle[i]) - i128::from(fused[i])).abs();
                // Each oracle weight rounds by < 1/2 LSB of `2^-32`, accumulated over the prefix,
                // and the one-normalisation readout adds at most 2 LSB; the scaled integer form
                // of `|oracle - fused| <= Σ_j |v_j|/(2·2^32) + 2` is tested here.
                assert!(
                    2 * one * dev <= sum_abs + 4 * one,
                    "pos {pos} head {head} i {i}: oracle {} fused {} dev {dev}",
                    oracle[i],
                    fused[i],
                );
                i += 1;
            }
            head += 1;
        }
        pos += 1;
    }
}

#[test]
fn attn_head_fused_uses_the_global_max_across_tiles() {
    // Two keys of scores 0 and 22·ONE. The gap exceeds `16·ONE`, so at the true maximum key 0's
    // numerator saturates to 0, and it exceeds the `fixed_exp` window under key 0's smaller
    // shift; a fold that fixed the first tile's maximum only could not reproduce the oracle.
    let mut kv = KvState::new(1, 1, 1, 1, 2).expect("geometry");
    assert_eq!(kv.append(0, &[0], &[5 * ONE]), Ok(0));
    assert_eq!(kv.append(0, &[22 * ONE], &[-3 * ONE]), Ok(1));
    let q = [ONE];
    let mut oracle = [0i64; 1];
    kv.attn_head(0, 0, &q, 1, &mut oracle).expect("oracle");
    assert_eq!(oracle, [-3 * ONE]);
    let mut fused = [0i64; 1];
    kv.attn_head_fused(0, 0, &q, 1, 1, &mut fused)
        .expect("fused read");
    assert_eq!(fused, oracle);
}

#[test]
fn attn_fused_refuses_zero_sum_and_zero_tile() {
    let mut kv =
        KvState::with_carrier(1, 1, 1, 1, 2, super::attn::KvCarrier::Q32).expect("geometry");
    assert_eq!(kv.append(0, &[ONE], &[ONE]), Ok(0));
    assert_eq!(kv.append(0, &[ONE], &[ONE]), Ok(1));
    let q = [ONE];
    let m = kv.attn_max(0, 0, &q, 1).expect("max");
    assert_eq!(m, ONE);
    assert_eq!(
        kv.attn_fused_state(0, 0, &q, 1, m, 1).expect("state").den,
        2 * i128::from(ONE)
    );
    // Every shift below the window saturates to zero, so the fused denominator is empty.
    assert_eq!(
        kv.attn_fused_state(0, 0, &q, 1, m + 17 * ONE, 1),
        Err(AttnError::ZeroSum)
    );
    assert_eq!(
        kv.attn_fused_state(0, 0, &q, 1, m, 0),
        Err(AttnError::BadGeometry)
    );
    assert_eq!(
        kv.attn_head_fused(0, 0, &q, 1, 0, &mut [0i64; 1]),
        Err(AttnError::BadGeometry)
    );
}

#[test]
fn attn_head_fused_refuses_bad_geometry_and_shapes() {
    let mut kv = KvState::new(1, 2, 1, 2, 1).expect("geometry");
    assert_eq!(kv.append(0, &[ONE, ONE], &[ONE, ONE]), Ok(0));
    let q = [ONE, ONE];
    assert_eq!(
        kv.attn_head_fused(1, 0, &q, 0, 1, &mut [0i64; 2]),
        Err(AttnError::LayerOutOfRange {
            layer: 1,
            n_layer: 1,
        })
    );
    assert_eq!(
        kv.attn_head_fused(0, 2, &q, 0, 1, &mut [0i64; 2]),
        Err(AttnError::HeadOutOfRange { head: 2, n_head: 2 })
    );
    assert_eq!(
        kv.attn_head_fused(0, 0, &[ONE], 0, 1, &mut [0i64; 2]),
        Err(AttnError::RowLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        kv.attn_head_fused(0, 0, &q, 0, 1, &mut [0i64; 1]),
        Err(AttnError::RowLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        kv.attn_head_fused(0, 0, &q, 1, 1, &mut [0i64; 2]),
        Err(AttnError::PositionOutOfRange { pos: 1, len: 1 })
    );
}

/// One `Q8_0` block per row, `d = 1.0` and the first `row.len()` codes, for synthetic FFN
/// payloads; the unused codes stay `0`.
fn q8_0_rows(rows: &[&[u8]]) -> Vec<u8> {
    let mut raw = vec![0u8; rows.len() * Q8_0_BLOCK_BYTES];
    let mut r = 0usize;
    while r < rows.len() {
        let base = r * Q8_0_BLOCK_BYTES;
        raw[base..base + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        let mut c = 0usize;
        while c < rows[r].len() {
            raw[base + 2 + c] = rows[r][c];
            c += 1;
        }
        r += 1;
    }
    raw
}

/// A `Q8_0` projection over `raw`, the test-side shorthand for [`ProjWeights`].
fn q8_proj(raw: &[u8]) -> ProjWeights<'_> {
    ProjWeights {
        ggml_type: quant::GGML_TYPE_Q8_0,
        raw,
    }
}

#[test]
fn silu_fixed_matches_hand_computed_values() {
    const LN2_Q32: i64 = 2_977_044_472;
    // silu(0) = 0 / (1 + exp(0)) = 0, exact with exp(0) = ONE.
    assert_eq!(silu_fixed(0), Ok(0));
    // exp(-ln 2) = 1/2 is exact in fixed_exp, so silu(ln 2) = ln 2 / (3/2) = round(2·LN2_Q32/3).
    assert_eq!(fixed_exp(-LN2_Q32), Ok(ONE / 2));
    assert_eq!(silu_fixed(LN2_Q32), Ok(1_984_696_315));
    // exp(ln 2) = 2 is exact, so silu(-ln 2) = -ln 2 / 3 = -round(LN2_Q32/3).
    assert_eq!(fixed_exp(LN2_Q32), Ok(2 * ONE));
    assert_eq!(silu_fixed(-LN2_Q32), Ok(-992_348_157));
    // The SiLU domain boundaries saturate: at and above `16·ONE` the result is `v`, at and below
    // `-21·ONE` it is `0`; the in-window quotient is unchanged.
    assert_eq!(silu_fixed(16 * ONE), Ok(16 * ONE));
    assert_eq!(silu_fixed(16 * ONE + 1), Ok(16 * ONE + 1));
    assert_eq!(silu_fixed(-21 * ONE), Ok(0));
    assert_eq!(silu_fixed(-21 * ONE - 1), Ok(0));
    assert_eq!(silu_fixed(i64::MIN), Ok(0));
    assert_eq!(silu_fixed(i64::MAX), Ok(i64::MAX));
}

#[test]
fn ffn_fixed_swiglu_matches_hand_computed_values() {
    const LN2_Q32: i64 = 2_977_044_472;
    // n_embd 2, n_ff 1. W_gate = [1, 0], W_up = [0, 1], W_down rows [1], [0], x = [ln 2, 1].
    // gate = round(ONE·ln2/2^32) = LN2_Q32; up = round(ONE·ONE/2^32) = ONE; h =
    // round(silu(LN2_Q32)·ONE/2^32) = 1_984_696_315; y = [round(h·ONE/2^32), 0] = [h, 0].
    let gate_raw = q8_0_rows(&[&[1, 0]]);
    let up_raw = q8_0_rows(&[&[0, 1]]);
    let down_raw = q8_0_rows(&[&[1], &[0]]);
    let x = [LN2_Q32, ONE];
    let y = ffn_fixed(
        q8_proj(&gate_raw),
        q8_proj(&up_raw),
        q8_proj(&down_raw),
        1,
        &x,
    )
    .expect("synthetic SwiGLU");
    assert_eq!(y, vec![1_984_696_315, 0]);
}

#[test]
fn ffn_fixed_refuses_geometry_shape_type_and_saturates_window() {
    let row = q8_0_rows(&[&[1, 0]]);
    let x = [ONE, ONE];
    assert_eq!(
        ffn_fixed(q8_proj(&row), q8_proj(&row), q8_proj(&row), 0, &x),
        Err(FfnError::BadGeometry)
    );
    assert_eq!(
        ffn_fixed(q8_proj(&row), q8_proj(&row), q8_proj(&row), 1, &[]),
        Err(FfnError::BadGeometry)
    );
    // The down payload must hold n_embd = 2 rows of 34 bytes; 34 bytes is one row short.
    assert_eq!(
        ffn_fixed(q8_proj(&row), q8_proj(&row), q8_proj(&row), 1, &x),
        Err(FfnError::Matvec(MatvecError::ShortPayload {
            need: 2 * Q8_0_BLOCK_BYTES,
            have: Q8_0_BLOCK_BYTES,
        }))
    );
    // A type with no fixed-point reader refuses in the projection.
    assert_eq!(
        ffn_fixed(
            ProjWeights {
                ggml_type: 999,
                raw: &row,
            },
            q8_proj(&row),
            q8_proj(&row),
            1,
            &x,
        ),
        Err(FfnError::Matvec(MatvecError::UnsupportedQuant {
            ggml_type: 999
        }))
    );
    // gate = round(ONE·17·ONE/2^32) = 17·ONE saturates `SiLU` to `17·ONE` instead of refusing; the
    // projection then completes (`down` provides `n_embd` rows for this call).
    let down = q8_0_rows(&[&[1], &[0]]);
    let x_big = [17 * ONE, 0];
    assert!(
        ffn_fixed(q8_proj(&row), q8_proj(&row), q8_proj(&down), 1, &x_big).is_ok(),
        "a gate logit past 16·ONE saturates, never refuses"
    );
}

#[test]
fn argmax_fixed_takes_the_max_and_prefers_the_lowest_tie() {
    assert_eq!(argmax_fixed(&[]), Err(SamplerError::EmptyLogits));
    assert_eq!(argmax_fixed(&[5]), Ok(0));
    assert_eq!(argmax_fixed(&[1, 3, 3, 2]), Ok(1)); // exact tie -> lowest index
    assert_eq!(argmax_fixed(&[-5, -1, -9]), Ok(1));
    // i64 extremes are in the domain: no negation, no overflow.
    assert_eq!(argmax_fixed(&[i64::MIN, i64::MAX, i64::MAX]), Ok(1));
    assert_eq!(argmax_fixed(&[i64::MAX, i64::MIN]), Ok(0));
    // Determinism on a longer vector.
    let v: Vec<i64> = (0i64..64).map(|i| (i * 37) % 11).collect();
    assert_eq!(argmax_fixed(&v), argmax_fixed(&v));
}

#[test]
fn top_p_fixed_selects_the_nucleus_prefix_and_refuses_windows() {
    const LN2_Q32: i64 = 2_977_044_472;
    // logits [0, ln 2]: shifted [-ln 2, 0] -> exp [ONE/2, ONE], total 1.5·ONE.
    // weight(1) = round(2·ONE/3) = 2_863_311_531, weight(0) = round(ONE/3) = 1_431_655_765.
    // Order is descending logit: [1, 0]. p = ONE/2 stops after index 1; p = ONE admits both.
    let logits = [0, LN2_Q32];
    assert_eq!(top_p_fixed(&logits, ONE / 2), Ok(vec![1]));
    assert_eq!(top_p_fixed(&logits, ONE), Ok(vec![1, 0]));
    assert_eq!(top_p_fixed(&logits, ONE - 1), Ok(vec![1, 0]));

    // Exact tie: order keeps the lowest index, and a half threshold admits it alone.
    assert_eq!(top_p_fixed(&[5, 5], ONE / 2), Ok(vec![0]));

    // Empty and out-of-range thresholds refuse.
    assert_eq!(top_p_fixed(&[], ONE), Err(SamplerError::EmptyLogits));
    assert_eq!(
        top_p_fixed(&[ONE], 0),
        Err(SamplerError::BadThreshold { p: 0 })
    );
    assert_eq!(
        top_p_fixed(&[ONE], ONE + 1),
        Err(SamplerError::BadThreshold { p: ONE + 1 })
    );
    assert_eq!(
        top_p_fixed(&[ONE], -ONE),
        Err(SamplerError::BadThreshold { p: -ONE })
    );
    // max = 0; the second logit is 17·ONE below it, past the fixed_exp lower window.
    assert_eq!(
        top_p_fixed(&[0, -17 * ONE], ONE / 2),
        Err(SamplerError::ExpOutOfRange)
    );
    // -16·ONE is admitted exactly (the window boundary is not clamped).
    assert!(top_p_fixed(&[0, -16 * ONE], ONE).is_ok());
}

#[test]
fn lm_head_and_argmax_through_the_real_tensor_table() {
    let Some(path) = model_path(QWEN05B_PATH) else {
        eprintln!("skip lm_head_and_argmax_through_the_real_tensor_table: {QWEN05B_PATH} absent");
        return;
    };
    let engine = Engine::open(&path).expect("load qwen05b");
    // This fixture is the untied Qwen2.5-0.5B-Instruct export (sha256
    // 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db), so the loader
    // prefers `output.weight` over the embedding. The tied fallback is the loader's other branch.
    let head = engine.lm_head();
    assert_eq!(head.name, "output.weight");

    let x = engine.token_embedding(0).expect("token 0 embedding");
    let logits = engine.lm_head_fixed(&x).expect("token 0 logits");
    assert_eq!(logits.len(), engine.config.vocab_size);
    let a0 = argmax_fixed(&logits).expect("argmax over token 0 logits");
    let c0 = fixed_vector_checksum(&logits);
    // Recorded from this model/source identity: token 0's own embedding row is the argmax.
    assert_eq!(a0, 0, "recorded argmax for token 0");
    assert_eq!(c0, 0xa8cd_ee90_a4e2_faad, "recorded token 0 logit checksum");

    // A short activation is refused before any read.
    assert_eq!(
        engine.lm_head_fixed(&x[..x.len() - 1]),
        Err(LmHeadError::ShapeMismatch {
            x: x.len() - 1,
            n_embd: engine.config.n_embd,
        })
    );

    // Determinism across an independent load.
    let second = Engine::open(&path).expect("second load");
    let x2 = second
        .token_embedding(0)
        .expect("token 0 embedding (second)");
    let logits2 = second.lm_head_fixed(&x2).expect("token 0 logits (second)");
    assert_eq!(argmax_fixed(&logits2), Ok(a0));
    assert_eq!(fixed_vector_checksum(&logits2), c0);
}

// ── synthetic decode model (rows 0–8 composed) ──────────────────────────────────────────────

/// Little-endian writer for the synthetic GGUF fixture.
struct Gw(Vec<u8>);

impl Gw {
    fn new() -> Self {
        Gw(Vec::new())
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn string(&mut self, s: &str) {
        self.u64(u64::try_from(s.len()).expect("small"));
        self.bytes(s.as_bytes());
    }
}

fn push_t(
    ts: &mut Vec<(String, u32, Vec<u64>, Vec<u8>)>,
    omit: Option<&str>,
    name: String,
    ggml_type: u32,
    dims: Vec<u64>,
    payload: Vec<u8>,
) {
    if omit == Some(name.as_str()) {
        return;
    }
    ts.push((name, ggml_type, dims, payload));
}

fn d1(n: usize) -> Vec<u64> {
    vec![u64::try_from(n).expect("small")]
}

fn d2(a: usize, b: usize) -> Vec<u64> {
    vec![
        u64::try_from(a).expect("small"),
        u64::try_from(b).expect("small"),
    ]
}

/// A `[-2, 2]` deterministic code varying with the row and column.
fn synth_code(r: usize, c: usize, salt: usize) -> i8 {
    let rr = i64::try_from(r).expect("small");
    let cc = i64::try_from(c).expect("small");
    let ss = i64::try_from(salt).expect("small");
    i8::try_from((rr * 3 + cc * 5 + ss) % 5 - 2).expect("in [-2, 2]")
}

/// One `Q8_0` weight of `n_out` rows over `n_in = 32` (one block per row) with scale `d`.
fn synth_q8_weight(n_out: usize, d_bits: u16, salt: usize) -> Vec<u8> {
    const N_IN: usize = 32;
    let mut out: Vec<u8> = Vec::with_capacity(n_out * (2 + N_IN));
    let mut r = 0usize;
    while r < n_out {
        out.extend_from_slice(&d_bits.to_le_bytes());
        let mut c = 0usize;
        while c < N_IN {
            out.push(synth_code(r, c, salt).to_le_bytes()[0]);
            c += 1;
        }
        r += 1;
    }
    out
}

fn synth_f32_vec(n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&f(i).to_le_bytes());
        i += 1;
    }
    b
}

/// A tiny dense `qwen2` GGUF: 2 layers, `n_embd 32`, `n_ff 32`, 4 query heads over 2 KV
/// heads of width 8, vocabulary 16. Embedding weights carry `d = 1.0` (so activations are
/// O(ONE)); projection weights carry `d = 2^-5` (so the FFN gate stays inside the `SiLU`
/// window). `omit` drops one named tensor, `f32_q` stores layer 0's Q weight `F32`, and
/// `bad_k_shape` swaps layer 0's K weight dims.
#[allow(
    clippy::too_many_lines,
    reason = "one tabular fixture builder: the two-layer tensor list is a flat sequence and \
                  splitting it would hide which tensor a stage reads"
)]
fn synth_gguf_bytes(omit: Option<&str>, f32_q: bool, bad_k_shape: bool) -> Vec<u8> {
    const NL: usize = 2;
    const NE: usize = 32;
    const NF: usize = 32;
    const NH: usize = 4;
    const NKV: usize = 2;
    const HD: usize = 8;
    const V: usize = 16;
    const Q_OUT: usize = NH * HD;
    const KV_OUT: usize = NKV * HD;
    const D_PROJ: u16 = 0x2800; // f16 2^-5
    const D_EMB: u16 = 0x3C00; // f16 1.0
    let gain = |_: usize| 1.0_f32;
    let bias = |i: usize| match i % 3 { 0 => -1.0_f32, 1 => 0.0, _ => 1.0 } / 64.0;

    let mut ts: Vec<(String, u32, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut l = 0usize;
    while l < NL {
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_norm.weight"),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        let qname = format!("blk.{l}.attn_q.weight");
        if f32_q && l == 0 {
            push_t(
                &mut ts,
                omit,
                qname,
                quant::GGML_TYPE_F32,
                d2(NE, Q_OUT),
                vec![0u8; NE * Q_OUT * 4],
            );
        } else {
            push_t(
                &mut ts,
                omit,
                qname,
                quant::GGML_TYPE_Q8_0,
                d2(NE, Q_OUT),
                synth_q8_weight(Q_OUT, D_PROJ, l + 1),
            );
        }
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_q.bias"),
            quant::GGML_TYPE_F32,
            d1(Q_OUT),
            synth_f32_vec(Q_OUT, bias),
        );
        let kdims = if bad_k_shape && l == 0 {
            d2(KV_OUT, NE)
        } else {
            d2(NE, KV_OUT)
        };
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_k.weight"),
            quant::GGML_TYPE_Q8_0,
            kdims,
            synth_q8_weight(KV_OUT, D_PROJ, l + 2),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_k.bias"),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_v.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, KV_OUT),
            synth_q8_weight(KV_OUT, D_PROJ, l + 3),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_v.bias"),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.attn_output.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NE),
            synth_q8_weight(NE, D_PROJ, l + 4),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.ffn_norm.weight"),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.ffn_gate.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, l + 5),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.ffn_up.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, l + 6),
        );
        push_t(
            &mut ts,
            omit,
            format!("blk.{l}.ffn_down.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NF, NE),
            synth_q8_weight(NE, D_PROJ, l + 7),
        );
        l += 1;
    }
    push_t(
        &mut ts,
        omit,
        "output_norm.weight".to_string(),
        quant::GGML_TYPE_F32,
        d1(NE),
        synth_f32_vec(NE, gain),
    );
    push_t(
        &mut ts,
        omit,
        "token_embd.weight".to_string(),
        quant::GGML_TYPE_Q8_0,
        d2(NE, V),
        synth_q8_weight(V, D_EMB, 9),
    );

    let mut h = Gw::new();
    h.bytes(b"GGUF");
    h.u32(3);
    h.u64(u64::try_from(ts.len()).expect("small"));
    h.u64(8);
    h.string("general.architecture");
    h.u32(8);
    h.string("qwen2");
    for (k, v) in [
        ("qwen2.block_count", u32::try_from(NL).expect("small")),
        (
            "qwen2.attention.head_count",
            u32::try_from(NH).expect("small"),
        ),
        (
            "qwen2.attention.head_count_kv",
            u32::try_from(NKV).expect("small"),
        ),
        ("qwen2.embedding_length", u32::try_from(NE).expect("small")),
        (
            "qwen2.feed_forward_length",
            u32::try_from(NF).expect("small"),
        ),
    ] {
        h.string(k);
        h.u32(4);
        h.u32(v);
    }
    h.string("qwen2.rope.freq_base");
    h.u32(6);
    h.u32(10_000.0_f32.to_bits());
    h.string("qwen2.attention.layer_norm_rms_epsilon");
    h.u32(6);
    h.u32(1e-5_f32.to_bits());

    let mut off = 0u64;
    for (name, ty, dims, payload) in &ts {
        h.string(name);
        h.u32(u32::try_from(dims.len()).expect("small"));
        for &d in dims {
            h.u64(d);
        }
        h.u32(*ty);
        h.u64(off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let header_len = h.0.len();
    let pad = (32 - header_len % 32) % 32;
    let mut i = 0usize;
    while i < pad {
        h.u8(0);
        i += 1;
    }
    for (_, _, _, payload) in &ts {
        h.bytes(payload);
    }
    h.0
}

/// Write the synthetic fixture to the OS temp dir, unique per process and test name.
fn write_synth_gguf(name: &str, omit: Option<&str>, f32_q: bool, bad_k_shape: bool) -> String {
    let path = std::env::temp_dir().join(format!("apeiron_{}_{name}", std::process::id()));
    std::fs::write(&path, synth_gguf_bytes(omit, f32_q, bad_k_shape)).expect("write fixture");
    path.to_str().expect("utf8 temp path").to_string()
}

#[test]
fn engine_step_greedy_decode_is_deterministic() {
    const INITIAL: u32 = 3;
    const STEPS: usize = 8;
    let path = write_synth_gguf("engine_step_decode.gguf", None, false, false);
    let a = Engine::open(&path).expect("load synthetic A");
    let b = Engine::open(&path).expect("load synthetic B");
    let mut ka = KvState::new(
        a.config.n_layer,
        a.config.n_head,
        a.config.n_head_kv,
        a.config.head_dim,
        STEPS,
    )
    .expect("kv A");
    let mut kb = KvState::new(
        b.config.n_layer,
        b.config.n_head,
        b.config.n_head_kv,
        b.config.head_dim,
        STEPS,
    )
    .expect("kv B");

    let mut ids: Vec<u32> = Vec::new();
    let mut checksums: Vec<u64> = Vec::new();
    let mut tok = INITIAL;
    let mut i = 0usize;
    while i < STEPS {
        let ra = a.step_with_kv(&mut ka, tok).expect("step A");
        let rb = b.step_with_kv(&mut kb, tok).expect("step B");
        assert_eq!(ra, rb, "independent loads disagree at step {i}");
        ids.push(ra.token);
        checksums.push(ra.logits_checksum);
        tok = ra.token;
        i += 1;
    }
    // The KV prefix grows each step, so the forward is not static across steps.
    assert!(
        checksums.iter().any(|&c| c != checksums[0]),
        "the logits checksum never changed across {STEPS} steps"
    );
    eprintln!(
        "engine_step_decode initial {INITIAL} steps {STEPS} ids {ids:?} first_checksum {:#018x} \
             checksums {checksums:?}",
        checksums[0]
    );
    assert_eq!(ids, vec![0, 0, 0, 0, 0, 0, 0, 0], "recorded generated ids");
    assert_eq!(
        checksums[0],
        match ka.carrier() {
            super::attn::KvCarrier::Q32 => 0x483c_6d3e_ce96_8247,
            super::attn::KvCarrier::F32 => 12_301_219_398_411_439_768,
            super::attn::KvCarrier::I8 => panic!("this test builds an F32 or Q32 cache"),
        },
        "recorded first-step logits checksum"
    );

    // `step` (fresh one-position cache) reproduces the first stateful step exactly.
    let mut c = Engine::open(&path).expect("load synthetic C");
    let s0 = c.step(INITIAL).expect("stateless step");
    assert_eq!(s0.token, ids[0]);
    assert_eq!(s0.logits_checksum, checksums[0]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn engine_step_refuses_bad_token_capacity_and_shapes() {
    let path = write_synth_gguf("engine_step_refusals.gguf", None, false, false);
    let mut e = Engine::open(&path).expect("load refusals");
    let vocab = u32::try_from(e.config.vocab_size).expect("small vocab");
    match e.step(vocab) {
        Err(StepError::Embedding(EngineError::TokenOutOfRange { id, .. })) => {
            assert_eq!(id, vocab);
        }
        other => panic!("expected an out-of-vocab embedding refusal, got {other:?}"),
    }

    // A one-position cache admits position 0 and refuses the next append.
    let mut kv = KvState::new(
        e.config.n_layer,
        e.config.n_head,
        e.config.n_head_kv,
        e.config.head_dim,
        1,
    )
    .expect("kv cap 1");
    e.step_with_kv(&mut kv, 1).expect("first position");
    match e.step_with_kv(&mut kv, 1) {
        Err(StepError::Kv(AttnError::CapacityExceeded { cap })) => assert_eq!(cap, 1),
        other => panic!("expected a KV capacity refusal, got {other:?}"),
    }

    // A cache whose geometry is not the config's refuses before any read.
    let mut wrong =
        KvState::new(e.config.n_layer, 2, 2, e.config.head_dim, 1).expect("valid foreign kv");
    assert!(matches!(
        e.step_with_kv(&mut wrong, 1),
        Err(StepError::KvGeometry { .. })
    ));

    // A missing tensor refuses typed, at its own stage.
    let missing = write_synth_gguf(
        "engine_step_missing.gguf",
        Some("blk.0.ffn_gate.weight"),
        false,
        false,
    );
    let mut m = Engine::open(&missing).expect("load missing");
    match m.step(1) {
        Err(StepError::MissingTensor(n)) => assert_eq!(n.to_string(), "blk.0.ffn_gate.weight"),
        other => panic!("expected a missing-tensor refusal, got {other:?}"),
    }

    // A present `F32` projection weight reaches the projection kernel and refuses there, not at
    // a step-local guard.
    let f32q = write_synth_gguf("engine_step_f32q.gguf", None, true, false);
    let mut f = Engine::open(&f32q).expect("load f32 q");
    match f.step(1) {
        Err(StepError::Matvec(MatvecError::Dequant(EngineError::UnsupportedEmbeddingQuant {
            ggml_type,
        }))) => assert_eq!(ggml_type, quant::GGML_TYPE_F32),
        other => panic!("expected an F32 weight refusal, got {other:?}"),
    }

    // A swapped K-weight dims refuses typed.
    let bad = write_synth_gguf("engine_step_badshape.gguf", None, false, true);
    let mut bs = Engine::open(&bad).expect("load bad shape");
    assert!(matches!(bs.step(1), Err(StepError::BadShape { .. })));

    for p in [path, missing, f32q, bad] {
        let _ = std::fs::remove_file(p);
    }
}

const QWEN35_08B_PATH: &str = "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf";

#[test]
fn schedule_is_read_from_metadata_not_architecture_name() {
    let (Some(dense_path), Some(hybrid_path)) =
        (model_path(QWEN05B_PATH), model_path(QWEN35_08B_PATH))
    else {
        eprintln!("skip schedule test: a model file is absent");
        return;
    };
    let dense = Engine::open(&dense_path).expect("load qwen05b");
    let c = &dense.config;
    assert!(c.mixers.iter().all(|m| *m == MixerKind::Softmax));
    assert_eq!((c.rot_dims, c.head_dim, c.attn_gate), (64, 64, false));
    assert!(c.affine.is_none());

    let hybrid = Engine::open(&hybrid_path).expect("load qwen35");
    let c = &hybrid.config;
    assert_eq!(c.n_layer, 24, "the MTP block is not a trunk layer");
    for (l, m) in c.mixers.iter().enumerate() {
        let want = if (l + 1) % 4 == 0 {
            MixerKind::Softmax
        } else {
            MixerKind::Affine
        };
        assert_eq!(*m, want, "layer {l}");
    }
    assert_eq!((c.head_dim, c.rot_dims, c.attn_gate), (256, 64, true));
    let g = c.affine.expect("affine geometry");
    assert_eq!(
        (g.conv_kernel, g.n_k_heads, g.n_v_heads, g.head_k, g.head_v),
        (4, 16, 16, 128, 128)
    );
    let v = hybrid.decode_vectors();
    assert!(
        v.norms_of(0).and_then(|n| n[1].as_ref()).is_some(),
        "pre-FFN norm alias"
    );
    assert!(
        v.qk_norms_of(3).and_then(|n| n[0].as_ref()).is_some(),
        "q_norm at key_length"
    );
}

/// Greedy ids of `prompt` then `n` generated tokens through the one executor.
fn greedy_state(engine: &Engine, prompt: &[u32], n: usize, mode: KernelMode) -> Vec<u32> {
    let mut st = engine.state(prompt.len() + n + 1).expect("state");
    let mut next = 0u32;
    for &t in prompt {
        next = engine
            .step_state(&mut st, t, mode)
            .expect("prompt step")
            .token;
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(next);
        next = engine
            .step_state(&mut st, next, mode)
            .expect("gen step")
            .token;
    }
    out
}

#[test]
fn qwen35_on_the_one_executor_decodes_llama_cpp_ids() {
    let Some(path) = model_path(QWEN35_08B_PATH) else {
        eprintln!("skip qwen35 executor test: {QWEN35_08B_PATH} absent");
        return;
    };
    let engine = Engine::open(&path).expect("load qwen35");
    // "The capital of France is"; llama.cpp ece963f41 greedy ids.
    let prompt = [760u32, 6511, 314, 9338, 369];
    let llama = [11751u32, 13, 198, 760, 6511, 314, 9338, 369];
    assert_eq!(greedy_state(&engine, &prompt, 8, KernelMode::Q8Act), llama);
    assert_eq!(
        greedy_state(&engine, &prompt, 1, KernelMode::Exact),
        llama[..1]
    );
}

/// The 200-token README prompt (ids from `tools/qwen35/reference.py (unpublished) get_prompt2_ids`) against
/// llama.cpp's 32 greedy ids (`LLAMA_PROMPT_2_IDS`). Release only: 232 decode steps.
#[test]
#[ignore = "release-only: cargo test --release -- --ignored qwen35_long_prompt"]
fn qwen35_long_prompt_on_the_one_executor() {
    const PROMPT_IDS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35-prompt2.ids"
    );
    let (Some(model), Some(ids)) = (model_path(QWEN35_08B_PATH), model_path(PROMPT_IDS)) else {
        eprintln!("skip qwen35_long_prompt: model or prompt ids absent");
        return;
    };
    let prompt: Vec<u32> = std::fs::read_to_string(&ids)
        .expect("prompt ids")
        .split_whitespace()
        .map(|t| t.parse().expect("id"))
        .collect();
    assert_eq!(prompt.len(), 200);
    let llama = [
        12182u32, 9251, 2349, 1074, 5039, 877, 14, 1825, 42427, 14, 1825, 42427, 33673, 14905, 14,
        52838, 20668, 553, 271, 760, 1510, 49, 8065, 9477, 7, 87, 61, 17, 471, 19638, 44646, 17189,
    ];
    let engine = Engine::open(&model).expect("load qwen35");
    assert_eq!(greedy_state(&engine, &prompt, 32, KernelMode::Q8Act), llama);
}

/// The greedy bit-plane head path returns exactly the dense path's token at every step of a real
/// decode, and both decode llama.cpp's ids.
#[test]
fn qwen35_bitplane_head_equals_dense_head_in_lockstep() {
    let Some(path) = model_path(QWEN35_08B_PATH) else {
        eprintln!("skip bitplane lockstep: {QWEN35_08B_PATH} absent");
        return;
    };
    let engine = Engine::open(&path).expect("load qwen35");
    assert!(
        engine.bitplane_head().is_some(),
        "Q8_0 head gets bit planes"
    );
    let prompt = [760u32, 6511, 314, 9338, 369];
    let llama = [11751u32, 13, 198, 760, 6511, 314, 9338, 369];
    let mut dense = engine.state(32).expect("state");
    let mut fast = engine.state(32).expect("state");
    let mut ids = Vec::new();
    let mut tok = 0u32;
    for (i, step) in prompt
        .iter()
        .copied()
        .chain(std::iter::repeat(0))
        .take(prompt.len() + 8)
        .enumerate()
    {
        let input = if i < prompt.len() { step } else { tok };
        let d = engine
            .step_state(&mut dense, input, KernelMode::Q8Act)
            .expect("dense")
            .token;
        let xf = engine
            .compose_final(
                &mut fast.kv,
                fast.affine.as_mut(),
                input,
                super::step::Sched::Resident,
                KernelMode::Q8Act,
                true,
            )
            .expect("trunk");
        let q = super::chart::ffn::quant_act_pooled(&xf).expect("quant");
        assert!(super::chart::q8act_fast_bound_ok(xf.len(), 128, &q, true));
        let (f, _) = engine
            .bitplane_head()
            .expect("planes")
            .argmax(&q)
            .expect("bitplane");
        assert_eq!(f, d, "step {i}");
        tok = f;
        if i + 1 >= prompt.len() {
            ids.push(f);
        }
    }
    assert_eq!(&ids[..8], &llama);
}

/// Head timing on real final states: the dense `Q8Act` head vs the bit-plane argmax, with the
/// survivor count. `cargo test --release --lib engine::tests::bench_qwen35_head -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
#[allow(clippy::cast_precision_loss, reason = "a 24-sample count")]
fn bench_qwen35_head() {
    let Some(path) = model_path(QWEN35_08B_PATH) else {
        return;
    };
    let engine = Engine::open(&path).expect("load qwen35");
    let bp = engine.bitplane_head().expect("planes");
    let mut st = engine.state(64).expect("state");
    let mut states = Vec::new();
    let mut tok = 760u32;
    for _ in 0..24 {
        let xf = engine
            .compose_final(
                &mut st.kv,
                st.affine.as_mut(),
                tok,
                super::step::Sched::Resident,
                KernelMode::Q8Act,
                true,
            )
            .expect("trunk");
        let q = super::chart::ffn::quant_act_pooled(&xf).expect("quant");
        let (next, _) = bp.argmax(&q).expect("bitplane");
        states.push(q);
        tok = next;
    }
    let t = engine.lm_head();
    let raw = engine
        .payload(t.byte_range.start, t.n_bytes.expect("bytes"))
        .expect("payload");
    let (vocab, n) = (engine.config.vocab_size, engine.config.n_embd);
    let budget = crate::engine::ops::par::row_chunk_budget(vocab);
    let t0 = std::time::Instant::now();
    for q in &states {
        let l =
            super::chart::ffn::matvec_q8act_quant_chunked(t.ggml_type, raw, vocab, n, q, budget)
                .expect("dense");
        std::hint::black_box(l);
    }
    let dense = t0.elapsed().as_secs_f64() * 1e3 / states.len() as f64;
    let t1 = std::time::Instant::now();
    let mut surv = Vec::new();
    for q in &states {
        surv.push(bp.argmax(q).expect("bitplane").1);
    }
    let fast = t1.elapsed().as_secs_f64() * 1e3 / states.len() as f64;
    surv.sort_unstable();
    println!(
        "head dense {dense:.3} ms, bitplane {fast:.3} ms, survivors median {} max {}",
        surv[surv.len() / 2],
        surv[surv.len() - 1]
    );
}

/// A k-column pass equals k single steps bit for bit: every column's final state, and the state it
/// leaves behind (checked by the next step's output).
fn cols_equal_singles(path: &str, tokens: &[u32]) {
    let engine = Engine::open(path).expect("load");
    let mut a = engine.state(64).expect("state");
    let mut b = engine.state(64).expect("state");
    for &t in &[1u32, 2, 3] {
        engine
            .next_token_state(&mut a, t, KernelMode::Q8Act)
            .expect("warm a");
        engine
            .next_token_state(&mut b, t, KernelMode::Q8Act)
            .expect("warm b");
    }
    let cols = engine
        .compose_cols(
            &mut a.kv,
            a.affine.as_mut(),
            tokens,
            super::step::Sched::Resident,
            KernelMode::Q8Act,
            true,
            None,
        )
        .expect("cols");
    for (c, &t) in tokens.iter().enumerate() {
        let single = engine
            .compose_final(
                &mut b.kv,
                b.affine.as_mut(),
                t,
                super::step::Sched::Resident,
                KernelMode::Q8Act,
                true,
            )
            .expect("single");
        assert_eq!(cols[c], single, "{path}: column {c}");
    }
    let na = engine
        .step_logits_state(&mut a, 9, KernelMode::Q8Act)
        .expect("after a");
    let nb = engine
        .step_logits_state(&mut b, 9, KernelMode::Q8Act)
        .expect("after b");
    assert_eq!(na, nb, "{path}: state after the block");
}

#[test]
fn qwen35_two_columns_equal_two_single_steps() {
    if let Some(path) = model_path(QWEN35_08B_PATH) {
        cols_equal_singles(&path, &[760, 6511]);
        cols_equal_singles(&path, &[11751, 13, 198]);
        // A 40-column block: five GEMM column groups and the block attention stages.
        let block: Vec<u32> = (0..40u32).map(|i| 100 + 37 * i).collect();
        cols_equal_singles(&path, &block);
    }
}

/// One block pass's per-column output-normed states with the exact-scan attention flag forced `on`
/// or `off`; the warm-up runs under the same override, so the two blocks start from equal state.
fn exact_scan_block_states(engine: &Engine, tokens: &[u32], on: bool) -> Vec<Vec<i64>> {
    super::attn::set_exact_scan_override(Some(on));
    let mut st = engine.state(64).expect("state");
    for &t in &[1u32, 2, 3] {
        engine
            .next_token_state(&mut st, t, KernelMode::Q8Act)
            .expect("warm");
    }
    let states = engine
        .compose_cols(
            &mut st.kv,
            st.affine.as_mut(),
            tokens,
            super::step::Sched::Resident,
            KernelMode::Q8Act,
            true,
            None,
        )
        .expect("cols");
    super::attn::set_exact_scan_override(None);
    states
}

/// The greedy token `state` selects through the engine's bit-plane head, which returns exactly
/// `argmax_fixed` of the dense logits (`qwen35_bitplane_head_equals_dense_head_in_lockstep`).
fn bitplane_greedy(engine: &Engine, state: &[i64]) -> u32 {
    let q = super::chart::ffn::quant_act_pooled(state).expect("quant");
    assert!(super::chart::q8act_fast_bound_ok(
        state.len(),
        128,
        &q,
        true
    ));
    engine
        .bitplane_head()
        .expect("bit planes")
        .argmax(&q)
        .expect("bitplane")
        .0
}

/// The exact-scan fused softmax read (`APEIRON_EXACT_SCAN=1`) changes no greedy token and keeps
/// the logits within the epic's 256 LSB bound: a 40-column block with the flag forced on vs off.
/// Greedy is checked at every column through the exact bit-plane argmax; the dense logit deviation
/// is measured on sampled columns (the full 40 dense reads exceed the test's wall-clock budget).
#[test]
fn qwen35_exact_scan_attention_block_matches_baseline_within_bound() {
    let Some(path) = model_path(QWEN35_08B_PATH) else {
        eprintln!("skip exact-scan attention block: {QWEN35_08B_PATH} absent");
        return;
    };
    let engine = Engine::open(&path).expect("load qwen35");
    assert!(
        engine.bitplane_head().is_some(),
        "Q8_0 head gets bit planes"
    );
    let block: Vec<u32> = (0..40u32).map(|i| 100 + 37 * i).collect();
    let off = exact_scan_block_states(&engine, &block, false);
    let on = exact_scan_block_states(&engine, &block, true);
    for c in 0..block.len() {
        assert_eq!(
            bitplane_greedy(&engine, &off[c]),
            bitplane_greedy(&engine, &on[c]),
            "column {c}"
        );
    }
    let mut max_dev = 0u64;
    for &c in &[0usize, 13, 26, 39] {
        let a = engine.lm_head_fixed(&off[c]).expect("baseline logits");
        let b = engine.lm_head_fixed(&on[c]).expect("fused logits");
        assert_eq!(
            argmax_fixed(&a).expect("argmax baseline"),
            argmax_fixed(&b).expect("argmax fused"),
            "column {c}"
        );
        for (x, y) in a.iter().zip(&b) {
            max_dev = max_dev.max(x.abs_diff(*y));
        }
    }
    eprintln!("exact-scan attention block: max sampled logit deviation {max_dev} LSB");
    assert!(
        max_dev <= 256,
        "max logit deviation {max_dev} LSB exceeds the epic's 256 LSB bound"
    );
}

#[test]
fn qwen25_two_columns_equal_two_single_steps() {
    if let Some(path) = model_path(QWEN05B_PATH) {
        cols_equal_singles(&path, &[785, 6722]);
    }
}

/// Speculative decoding emits exactly the greedy ids: the full-vocabulary draft (mostly accepted)
/// and a tiny draft head (mostly rejected, so every round exercises the rollback) both reproduce
/// llama.cpp's ids.
#[test]
fn qwen35_speculative_equals_greedy() {
    let Some(path) = model_path(QWEN35_08B_PATH) else {
        return;
    };
    let engine = Engine::open(&path).expect("load");
    let prompt = [760u32, 6511, 314, 9338, 369];
    let llama = [
        11751u32, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369,
    ];
    for rows in [engine.config.vocab_size, 64] {
        let (ids, stats) = engine
            .generate_speculative(&prompt, 16, rows)
            .expect("speculative");
        eprintln!("draft rows {rows}: {stats:?}");
        assert_eq!(ids, llama, "draft rows {rows}");
        assert!(stats.rounds > 0);
    }
}

/// The 200-token prompt through the speculative decoder against llama.cpp's 32 greedy ids.
#[test]
#[ignore = "release-only: cargo test --release -- --ignored qwen35_speculative_long"]
fn qwen35_speculative_long_prompt() {
    const PROMPT_IDS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35-prompt2.ids"
    );
    let (Some(model), Some(ids)) = (model_path(QWEN35_08B_PATH), model_path(PROMPT_IDS)) else {
        return;
    };
    let prompt: Vec<u32> = std::fs::read_to_string(&ids)
        .expect("prompt ids")
        .split_whitespace()
        .map(|t| t.parse().expect("id"))
        .collect();
    let llama = [
        12182u32, 9251, 2349, 1074, 5039, 877, 14, 1825, 42427, 14, 1825, 42427, 33673, 14905, 14,
        52838, 20668, 553, 271, 760, 1510, 49, 8065, 9477, 7, 87, 61, 17, 471, 19638, 44646, 17189,
    ];
    let engine = Engine::open(&model).expect("load");
    let (ids, stats) = engine
        .generate_speculative(&prompt, 32, 32_768)
        .expect("speculative");
    eprintln!("long prompt: {stats:?}");
    assert_eq!(ids, llama);
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the reference softmax weights are transcendental (exp), so the fixed-point result is widened to f64 for the tolerance comparison"
)]
#[test]
fn f32_append_index_loop_pins_plane_values_and_writes_nothing_on_err() {
    // The `KvStorage::F32` arm of `KvState::append` is the extraction fragment's index-`while`
    // shape with a `bad` accumulator: the two `iter_mut().zip` loops became index loops and the
    // two in-loop `return Err` became one `Err` after both loops. These pins read the planes
    // through the only reader of them, `attn_head`, so a moved slot, a dropped slot or an
    // off-by-one in the index loop is visible. They cannot observe the `bad != 0` branch: over
    // the `i64` domain `q32_to_f32(x) = (x as f32) / 2^32` is finite for every `x`, since
    // `|x as f32| <= 2^63 < f32::MAX` and the divisor is normal, so that `Err` is unreachable
    // from `append` and the doc clause "contents unspecified on `Err`" has no witness here.
    let carrier = super::attn::KvCarrier::F32;
    let mut kv = KvState::with_carrier(1, 1, 1, 2, 4, carrier).expect("geometry");
    // One attended position makes the softmax denominator exactly 1, so `out` is the stored
    // value row bit for bit. This pins value-plane slots 0 and 1 exactly.
    assert_eq!(kv.append(0, &[ONE, 0], &[0, ONE]), Ok(0));
    let mut out = [i64::MIN; 2];
    kv.attn_head(0, 0, &[ONE, 0], 0, &mut out)
        .expect("attend pos 0");
    assert_eq!(out, [0, ONE]);
    // A second key row makes the softmax weights depend on the stored key row: with
    // `scale = 1/sqrt(2)` the scores are `1/sqrt(2)` and `0`, so the weights are
    // `e^0 / (1 + e^(-1/sqrt2))` and `e^(-1/sqrt2) / (1 + e^(-1/sqrt2))` over value row `[0,1]`.
    // A key-plane slot swap moves each component by about 1.14e9, far outside the tolerance.
    assert_eq!(kv.append(0, &[0, ONE], &[ONE, 0]), Ok(1));
    let mut out = [i64::MIN; 2];
    kv.attn_head(0, 0, &[ONE, 0], 1, &mut out)
        .expect("attend pos 1");
    let e = (-1.0f64 / 2.0f64.sqrt()).exp();
    let want = [(e / (1.0 + e)) * ONE as f64, (1.0 / (1.0 + e)) * ONE as f64];
    for (i, w) in want.iter().enumerate() {
        let got = out[i] as f64;
        assert!(
            (got - w).abs() <= 1.0e-3 * ONE as f64,
            "out[{i}] = {got}, want {w}, delta {}",
            got - w
        );
    }
    // A rejected append must not write. Every rejected row below is distinct from the stored
    // one, so a write on either `Err` path would change the exact reading at position 0.
    let mut kv = KvState::with_carrier(1, 1, 1, 2, 1, carrier).expect("geometry");
    assert_eq!(kv.append(0, &[ONE, 0], &[0, ONE]), Ok(0));
    assert_eq!(
        kv.append(0, &[ONE, 0, 0], &[0, ONE, 0]),
        Err(AttnError::RowLength {
            expected: 2,
            got: 3
        })
    );
    assert_eq!(
        kv.append(0, &[-ONE, -ONE], &[-ONE, -ONE]),
        Err(AttnError::CapacityExceeded { cap: 1 })
    );
    assert_eq!(kv.occupied(0), Ok(1));
    let mut out = [i64::MIN; 2];
    kv.attn_head(0, 0, &[ONE, 0], 0, &mut out)
        .expect("attend pos 0");
    assert_eq!(out, [0, ONE]);
}
