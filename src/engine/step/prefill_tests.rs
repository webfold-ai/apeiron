#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test-only fixture generation"
)]

use super::*;
use crate::format::quant::{GGML_TYPE_F16, GGML_TYPE_F32};

const F_E: usize = 8;
const F_FF: usize = 128;
const F_Q: usize = 8;
const F_KO: usize = 4;

/// Bound between the tiled attention and the per-position oracle in Q32.32 words: the tile
/// rescales the accumulator once per key tile, the oracle once per position, so the two differ by
/// the f32 roundoff the per-position recurrence accumulates over the prefix. Relative to the
/// larger word, with a small absolute floor.
fn assert_words_close(a: &[i64], b: &[i64], prefix: usize, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let m = i128::from(x.abs().max(y.abs()));
        let tol = 256
            + i64::try_from((m * 512 * (prefix as i128 + 64)) >> 32)
                .expect("the tile bound fits i64");
        assert!(
            (x - y).abs() <= tol,
            "{what} word {i}: {x} vs {y} (tol {tol})"
        );
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    put_u32(b, 4);
    put_u32(b, v);
}

fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    put_u32(b, 8);
    put_str(b, v);
}

fn f16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits << 1 == 0 {
        return 0;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = i32::try_from((bits >> 23) & 0xff).expect("8 bits") - 127 + 15;
    let mut m = i32::try_from((bits >> 13) & 0x3ff).expect("10 bits");
    let rem = bits & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let carry = m >> 10;
    m &= 0x3ff;
    sign | (((e + carry) as u16) << 10) | (m as u16)
}

fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
        }
    }
    b
}

fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    for _ in 0..n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    b
}

fn build_layer_gguf() -> Vec<u8> {
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "blk.0.attn_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.attn_q.weight",
            vec![F_E as u64, F_Q as u64],
            GGML_TYPE_F16,
            f16_weight(F_Q, F_E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![F_E as u64, F_KO as u64],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![F_E as u64, F_KO as u64],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![F_E as u64, F_E as u64],
            GGML_TYPE_F16,
            f16_weight(F_E, F_E, 4),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![F_FF as u64, F_E as u64],
            GGML_TYPE_F16,
            f16_weight(F_E, F_FF, 7),
        ),
        (
            "output_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "token_embd.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, ts.len() as u64);
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(&mut b, "qwen2.attention.head_count", 2);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
    kv_u32(&mut b, "qwen2.embedding_length", F_E as u32);
    kv_u32(&mut b, "qwen2.feed_forward_length", F_FF as u32);
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, dims.len() as u32);
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += payload.len() as u64;
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

fn open_bytes(tag: &str, bytes: &[u8]) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!("apeiron_prefill_{}_{tag}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

#[test]
fn prefill_state_matches_two_phase_computation() {
    let (engine, path) = open_bytes("prefill_test", &build_layer_gguf());
    let prompt_fixture = [
        0u32, 1, 2, 5, 9, 3, 7, 4, 8, 6, 0, 2, 4, 1, 3, 5, 7, 9, 8, 6, 1, 0, 5, 2, 8, 4, 9, 7, 3,
        6, 2, 1, 4, 0, 7, 5, 6, 8, 9, 3, 4, 7, 1, 8, 0, 9, 2, 6, 5, 3, 8, 1, 6, 0, 4, 9, 7, 2, 5,
        3, 9, 4, 2, 7, 1, 8, 5, 0, 6, 3, 5, 8, 0, 4, 7, 1, 9, 3, 6, 2, 3, 6, 9, 1, 4, 7, 0, 8, 2,
        5, 7, 0, 3, 6, 9, 2, 5, 8, 1, 4, 2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 6, 9, 2, 5, 8, 1, 4, 7, 0,
        3, 0, 3, 6, 9, 2, 5, 8, 1, 4, 7, 1, 4, 7, 0, 3, 6, 9, 2, 5, 8,
    ];

    for &w in &[1usize, 3, 64] {
        let lens = [1usize, 2, w.saturating_sub(1).max(1), w, w + 1, 2 * w + 3];
        for &len in &lens {
            let prompt = &prompt_fixture[..len];

            // 1. Two-phase computation (reference):
            // prefill_cols over head + next_token_state over last
            let mut st_ref = engine.state(len + 2).expect("state ref");
            let (last, head) = prompt.split_last().expect("non-empty");
            // Run blocks of width w
            for chunk in head.chunks(w) {
                engine
                    .compose_cols(
                        &mut st_ref.kv,
                        st_ref.affine.as_mut(),
                        chunk,
                        Sched::Resident,
                        KernelMode::Exact,
                        false,
                        None,
                    )
                    .expect("compose_cols");
            }
            let tok_ref = engine
                .next_token_state(&mut st_ref, *last, KernelMode::Exact)
                .expect("next_token_state");
            let dec_ref = engine
                .step_state(&mut st_ref, tok_ref, KernelMode::Exact)
                .expect("step_state ref")
                .token;

            // 2. prefill_state (new single-pass block computation):
            // Override prefill block width to w
            std::env::set_var("APEIRON_PREFILL_COLS", w.to_string());
            let mut st_act = engine.state(len + 2).expect("state act");
            let tok_act = engine
                .prefill_state_with_width(&mut st_act, prompt, KernelMode::Exact, w)
                .expect("prefill_state");
            let dec_act = engine
                .step_state(&mut st_act, tok_act, KernelMode::Exact)
                .expect("step_state act")
                .token;

            assert_eq!(
                tok_act, tok_ref,
                "token mismatch at w={w}, len={len}: got {tok_act}, expected {tok_ref}"
            );
            assert_eq!(
                dec_act, dec_ref,
                "decode token mismatch at w={w}, len={len}: got {dec_act}, expected {dec_ref}"
            );
        }
    }
    std::env::remove_var("APEIRON_PREFILL_COLS");
    let _ = std::fs::remove_file(&path);
}

/// A `NanoQuant` section of the fixture model's shape for `kind`: a deterministic sign plane and
/// the served pack's reader geometry (exact-power-of-two scales, no salient correction, no dense
/// residual layer).
fn packed_kind(kind: usize, out: usize, inn: usize) -> crate::format::nq::PackedLinear {
    let rank = 8usize;
    let u_words = rank.div_ceil(32);
    let v_words = inn.div_ceil(32);
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (kind as u64) << 32 ^ out as u64;
    let mut word = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u32
    };
    let u: Vec<u32> = (0..out * u_words).map(|_| word()).collect();
    let v: Vec<u32> = (0..rank * v_words).map(|_| word()).collect();
    // bfloat16 bits of 1/4, 1/2 and 1/2: exact powers of two, so every scale is one Q32.32 word.
    let s_pre: Vec<u16> = vec![0x3e80; inn];
    let s_mid: Vec<u16> = vec![0x3f00; rank];
    let s_post: Vec<u16> = vec![0x3f00; out];
    crate::format::nq::PackedLinear {
        layer: 0,
        kind,
        out,
        inn,
        rank,
        u: u.into(),
        v: v.into(),
        s_pre: s_pre.into(),
        s_mid: s_mid.into(),
        s_post: s_post.into(),
        salient_idx: Vec::new().into(),
        salient: Vec::new().into(),
        res_layers: 0,
        shared: false,
        res: Vec::new().into(),
        s_res: Vec::new().into(),
    }
}

/// The whole executor on a packed model: one column per block (the per-column projection path)
/// and `len` columns per block (the pooled path) produce the same prefill token, the same next
/// token's logits word for word, and the same cached rows. The kernel test owns the projection's
/// arithmetic; this owns the wiring — the reader choice, the group split and the fallback.
#[test]
fn the_nq_prefill_agrees_with_the_per_column_prefill() {
    let (engine, path) = open_bytes("prefill_nq", &build_layer_gguf());
    let c = &engine.config;
    let shapes = [
        (0, c.n_head * c.head_dim, c.n_embd),
        (1, c.n_head_kv * c.head_dim, c.n_embd),
        (2, c.n_head_kv * c.head_dim, c.n_embd),
        (3, c.n_embd, c.n_head * c.head_dim),
        (4, c.n_ff, c.n_embd),
        (5, c.n_ff, c.n_embd),
        (6, c.n_embd, c.n_ff),
    ];
    let linears: Vec<_> = shapes
        .iter()
        .map(|&(kind, out, inn)| packed_kind(kind, out, inn))
        .collect();
    let pack = crate::format::nq::NqPack::new(1, linears).expect("pack");
    let engine = Engine::open_with_nq(&path, pack).expect("open packed");
    assert!(
        engine.nq_linear(super::part_key(0, 0)).is_some(),
        "the pack must replace the layer's `q` linear, or this test runs the dense path twice"
    );

    let prompt = &[
        0u32, 1, 2, 5, 9, 3, 7, 4, 8, 6, 0, 2, 4, 1, 3, 5, 7, 9, 8, 6, 1, 0, 5, 2, 8, 4, 9, 7, 3,
        6, 2, 1, 4, 0, 7, 5, 6, 8, 9, 3, 4, 7, 1, 8, 0, 9, 2, 6, 5, 3, 8, 1, 6, 0, 4, 9, 7, 2, 5,
    ];
    for mode in [KernelMode::Exact, KernelMode::Q8Act] {
        // Every prompt column's hidden state: the block path's tiled attention writes each
        // column's `att` through the fold gather, so this is the comparison that sees it. (The
        // next-token logits below read only the KV rows, which attention never writes.)
        let mut one = engine.state(prompt.len() + 2).expect("state one");
        let mut block = engine.state(prompt.len() + 2).expect("state block");
        let sa = engine
            .prefill_cols_width(&mut one, prompt, mode, 1)
            .expect("prefill one");
        let sb = engine
            .prefill_cols_width(&mut block, prompt, mode, prompt.len())
            .expect("prefill block");
        assert_eq!(sa.len(), sb.len(), "prefill state rows, {mode:?}");
        for (c, (a, b)) in sa.iter().zip(&sb).enumerate() {
            assert_words_close(a, b, prompt.len(), &format!("prefill column {c}, {mode:?}"));
        }
        assert_eq!(
            one.kv.occupied(0).expect("occupied"),
            block.kv.occupied(0).expect("occupied"),
            "cache length, {mode:?}"
        );
        let mut one = engine.state(prompt.len() + 2).expect("state one");
        let mut block = engine.state(prompt.len() + 2).expect("state block");
        let a = engine
            .prefill_state_with_width(&mut one, prompt, mode, 1)
            .expect("prefill one");
        let b = engine
            .prefill_state_with_width(&mut block, prompt, mode, prompt.len())
            .expect("prefill block");
        assert_eq!(a, b, "prefill token, {mode:?}");
        let la = engine
            .step_logits_with_mode(&mut one.kv, a, mode)
            .expect("logits one");
        let lb = engine
            .step_logits_with_mode(&mut block.kv, b, mode)
            .expect("logits block");
        assert_words_close(
            &la,
            &lb,
            prompt.len(),
            &format!("next-token logits, {mode:?}"),
        );
    }
    let _ = std::fs::remove_file(&path);
}
