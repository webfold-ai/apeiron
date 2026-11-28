//! Board .
//! The engine's `Q8Act` activation Gram against its own `Exact` tape.
//!
//! One dense `qwen2` layer is written to a GGUF fixture and loaded by the engine. The reference
//! SyrkAcc are [`SyrkAcc::accumulate`] over the tape of
//! `hocapture::engine_tape_mode(.., KernelMode::Exact)`: Q32.32 words, `i128` products each
//! rounded once. The compared SyrkAcc are [`SyrkAcc::accumulate_tap`] over `Engine::prefill_tap_cols`
//! under [`KernelMode::Q8Act`], the production limbed path.
//!
//! The two modes are not asserted equal a priori: the test measures the per-entry and
//! matrix-scaled relative difference of each of the four SyrkAcc and reports it. Under
//! [`KernelMode::Q8Act`] the projections read an 8-bit quantised copy of each activation, so the
//! SyrkAcc differ through the hidden state the quantisation moves; `Q8_BOUND` states that
//! difference. The same measurement carries the fold path's own residual, the limb adapter against
//! the exact `i128` fold: `1.6e-6` of scale on `qkv`, whose input is the same word under both
//! modes. The tap rows themselves are the words the projections read, so a mutated tap word is the
//! control that the comparison can still fail.
//!
//! Grade: `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::type_complexity,
    reason = "test-only fixture arithmetic and Q32.32 word comparisons on bounded small values; \
              the single-letter names are the geometry's symbols"
)]

use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::{testdata, KernelMode};
use apeiron::format::quant;
use apeiron::learn::fold::stats::capture::{LayerSyrk, Normalisation, SyrkAcc};
use apeiron::learn::fold::stats::hocapture;
use apeiron::learn::grad::model::{Config, LayerTape, Tape, Weights};

/// Residual width.
const E: usize = 8;
/// Query heads.
const NH: usize = 2;
/// KV heads.
const NKV: usize = 1;
/// Per-head width.
const HD: usize = 4;
/// `q_out = NH·HD`.
const Q: usize = NH * HD;
/// `kv_out = NKV·HD`.
const KV: usize = NKV * HD;
/// Feed-forward width, and the fixture vocabulary (`token_embd` rows).
const FF: usize = 16;
/// Trunk layers.
const L: usize = 1;
/// Tokens the fixture runs.
const TOKENS: [u32; 4] = [1, 2, 3, 4];

/// Relative bound for the [`KernelMode::Q8Act`] SyrkAcc, on the matrix-scale metric: each projection
/// input is quantised to an 8-bit block code, so one activation word moves by at most `2⁻⁸` of its
/// block scale; the hidden state composes one layer of those projections and a Gram entry sees the
/// input twice. `2⁻⁵` is 2.8× the measured worst (`down`, `1.1e-2`).
const Q8_BOUND: f64 = 1.0 / 32.0;

/// Stage slots of the tap, as [`SyrkAcc::accumulate_tap`] reads them.
const STAGE_LN1_OUT: usize = 1;

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

/// Round-to-nearest-even `f16` bits for a finite normal `f32` in `(2^-14, 65504)`.
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

fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// A deterministic `F16` payload of `rows · cols` weights in `{-2, -1, 0, 1, 2}·2⁻⁵`.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    let mut r = 0usize;
    while r < rows {
        let mut c = 0usize;
        while c < cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
            c += 1;
        }
        r += 1;
    }
    b
}

/// An all-`1.0` `F32` vector of `n` gains.
fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        i += 1;
    }
    b
}

/// One dense `qwen2` layer with a tied head, every matrix `F16` and every norm gain exactly one.
///
/// The writer mirrors `tests/engine_tap.rs`; a `F16` value of `code·2⁻⁵` is exact, so the engine's
/// `f16_to_fixed` and [`Weights::from_engine`]'s dequantiser produce the same Q32.32 word.
fn build_gguf() -> Vec<u8> {
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "blk.0.attn_norm.weight",
            vec![u(E)],
            quant::GGML_TYPE_F32,
            f32_gain(E),
        ),
        (
            "blk.0.attn_q.weight",
            vec![u(E), u(Q)],
            quant::GGML_TYPE_F16,
            f16_weight(Q, E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![u(E), u(KV)],
            quant::GGML_TYPE_F16,
            f16_weight(KV, E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![u(E), u(KV)],
            quant::GGML_TYPE_F16,
            f16_weight(KV, E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![u(Q), u(E)],
            quant::GGML_TYPE_F16,
            f16_weight(E, Q, 4),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![u(E)],
            quant::GGML_TYPE_F32,
            f32_gain(E),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![u(E), u(FF)],
            quant::GGML_TYPE_F16,
            f16_weight(FF, E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![u(E), u(FF)],
            quant::GGML_TYPE_F16,
            f16_weight(FF, E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![u(FF), u(E)],
            quant::GGML_TYPE_F16,
            f16_weight(E, FF, 7),
        ),
        (
            "output_norm.weight",
            vec![u(E)],
            quant::GGML_TYPE_F32,
            f32_gain(E),
        ),
        (
            "token_embd.weight",
            vec![u(E), u(FF)],
            quant::GGML_TYPE_F16,
            f16_weight(FF, E, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(
        &mut b,
        "qwen2.block_count",
        u32::try_from(L).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.attention.head_count",
        u32::try_from(NH).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.attention.head_count_kv",
        u32::try_from(NKV).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(E).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(FF).expect("small"),
    );
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank"));
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

/// Write the fixture to a per-process temp file and open it.
fn open(tag: &str) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!(
        "apeiron_capture_engine_{}_{tag}",
        std::process::id()
    ));
    std::fs::write(&path, build_gguf()).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// The reference SyrkAcc of the fixture: the engine's own `Exact` tape folded by the exact
/// accumulator. The `cfg` comes from [`Weights::from_engine`], the same dimensions the engine
/// holds.
fn reference(engine: &Engine) -> (Config, Vec<LayerSyrk>) {
    let (_, cfg) = Weights::from_engine(engine).expect("weights");
    let (_, tape) =
        hocapture::engine_tape_mode(engine, &cfg, &TOKENS, KernelMode::Exact).expect("exact tape");
    let mut grams = SyrkAcc::new(&cfg).expect("grams");
    grams.accumulate(&tape, &cfg).expect("reference accumulate");
    (cfg, grams.finish(Normalisation::Sum).expect("finish"))
}

/// The four finished SyrkAcc of the same reference tape, as a tuple, for a compact comparison.
fn quads(g: &[LayerSyrk]) -> Vec<(&[f64], &[f64], &[f64], &[f64])> {
    g.iter()
        .map(|l| {
            (
                l.qkv.as_slice(),
                l.o.as_slice(),
                l.gate_up.as_slice(),
                l.down.as_slice(),
            )
        })
        .collect()
}

/// `max |a - b| / max(|a|, |b|)` over the buffer, `0` for two zeros.
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "gram length {} != {}", a.len(), b.len());
    let mut m = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        let s = x.abs().max(y.abs());
        let r = if s > 0.0 { d / s } else { 0.0 };
        if r > m {
            m = r;
        }
    }
    m
}

/// `max |a - b| / max|a|` over the buffer: the entry difference against the matrix's own scale,
/// the metric a Gram comparison uses (an entry near zero has no meaningful relative error).
fn scaled_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "gram length {} != {}", a.len(), b.len());
    let mut scale = 0.0f64;
    let mut diff = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        scale = scale.max(x.abs());
        diff = diff.max((x - y).abs());
    }
    if scale > 0.0 {
        diff / scale
    } else {
        0.0
    }
}

/// The per-Gram entry-wise and matrix-scale relative differences, layer by layer.
fn diff(want: &[LayerSyrk], got: &[LayerSyrk]) -> ([f64; 4], [f64; 4]) {
    assert_eq!(want.len(), got.len());
    let mut m = [0.0f64; 4];
    let mut s = [0.0f64; 4];
    for (a, b) in quads(want).iter().zip(quads(got)) {
        m[0] = m[0].max(max_rel(a.0, b.0));
        m[1] = m[1].max(max_rel(a.1, b.1));
        m[2] = m[2].max(max_rel(a.2, b.2));
        m[3] = m[3].max(max_rel(a.3, b.3));
        s[0] = s[0].max(scaled_rel(a.0, b.0));
        s[1] = s[1].max(scaled_rel(a.1, b.1));
        s[2] = s[2].max(scaled_rel(a.2, b.2));
        s[3] = s[3].max(scaled_rel(a.3, b.3));
    }
    (m, s)
}

/// The engine tap SyrkAcc of `engine` under `mode`.
fn engine_syrks(
    engine: &Engine,
    cfg: &Config,
    mode: KernelMode,
) -> (Vec<LayerSyrk>, Vec<Vec<Vec<Vec<i64>>>>) {
    let mut st = engine.state(TOKENS.len()).expect("state");
    let (_, rows) = engine
        .prefill_tap_cols(&mut st, &TOKENS, mode, true)
        .expect("tapped prefill");
    let mut grams = SyrkAcc::new(cfg).expect("grams");
    grams.accumulate_tap(&rows).expect("tap accumulate");
    let fin = grams.finish(Normalisation::Sum).expect("finish");
    (fin, rows)
}

/// The production `Q8Act` path: the Gram difference against the exact reference is measured and
/// bounded. The `ln1_out` Gram is the residual norm of the embedding, before the first quantised
/// projection runs; the other three carry the quantiser's effect through the hidden state.
#[test]
fn engine_q8act_syrks_are_within_the_stated_bound() {
    let (engine, path) = open("q8act");
    let (cfg, want) = reference(&engine);
    let (got, _) = engine_syrks(&engine, &cfg, KernelMode::Q8Act);
    let _ = std::fs::remove_file(&path);
    let (m, s) = diff(&want, &got);
    println!("q8act max entry rel err qkv/o/gate_up/down = {m:?}");
    println!("q8act matrix-scaled rel err qkv/o/gate_up/down = {s:?}");
    for (name, v) in ["qkv", "o", "gate_up", "down"].iter().zip(s) {
        assert!(
            v <= Q8_BOUND,
            "{name} Gram scaled error {v} > bound {Q8_BOUND}"
        );
    }
    for (a, b) in want.iter().zip(&got) {
        assert_eq!(a.tokens, TOKENS.len() as u64);
        assert_eq!(b.tokens, a.tokens);
    }
}

/// The control: one mutated tap word must move its Gram past the exact comparison. A check that
/// cannot fail has no weight.
#[test]
fn mutated_tap_row_breaks_the_comparison() {
    let (engine, path) = open("control");
    let (cfg, want) = reference(&engine);
    let (clean, mut mut_rows) = engine_syrks(&engine, &cfg, KernelMode::Q8Act);
    let _ = std::fs::remove_file(&path);
    let (_, clean_s) = diff(&want, &clean);
    // One word of the `ln1_out` stage, by `2⁻⁸` real (Q32.32 `2²⁴`), well inside the quantiser's
    // own scale: the comparison must still see it.
    mut_rows[0][STAGE_LN1_OUT][0][1] += 1 << 24;
    let mut grams = SyrkAcc::new(&cfg).expect("grams");
    grams.accumulate_tap(&mut_rows).expect("mutated accumulate");
    let got = grams.finish(Normalisation::Sum).expect("finish");
    let (m, s) = diff(&want, &got);
    println!(
        "control clean qkv scaled {:.3e}; mutated entry {:.3e} scaled {:.3e}",
        clean_s[0], m[0], s[0]
    );
    assert!(
        s[0] > 1e-4 && s[0] > 100.0 * clean_s[0],
        "the mutated qkv Gram stayed within the comparison: {m:?} / {s:?}"
    );
}

/// The old exact tape fold against the limb-adapter tap fold, at the production down-projection
/// shape (`t = 512`, `w = 4864`), five paired runs of one layer's Gram. Time in seconds; the
/// per-run ratio is the Gram phase's, not the forward's.
#[test]
#[ignore = "measured report: five paired runs at the production Gram shape; run with --ignored --nocapture"]
fn syrk_fold_old_vs_new_at_production_shape() {
    const T: usize = 512;
    const W: usize = 4864;
    const RUNS: usize = 5;
    let e = 1usize;
    let cfg = Config {
        n_layer: 1,
        n_head: 1,
        n_head_kv: 1,
        n_embd: e,
        n_ff: W,
        vocab: 1,
        head_dim: 1,
        eps: 1,
        rope_theta: 1.0e6,
    };
    // One stage's `t × W` words: a deterministic LCG in `[-1, 1)`.
    let mut s = 7u64;
    let mut x = vec![0i64; T * W];
    for v in &mut x {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *v = i64::try_from((s >> 33) % 2_147_483_648).expect("small") - 1_073_741_824;
    }
    let cols: Vec<Vec<i64>> = (0..T).map(|r| x[r * W..(r + 1) * W].to_vec()).collect();
    let tape = Tape {
        t: T,
        tokens: Vec::new(),
        layers: vec![LayerTape {
            ln1_in: Vec::new(),
            ln1_out: vec![0; T * e],
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            p: Vec::new(),
            attn_out: vec![0; T * e],
            h2: Vec::new(),
            ln2_out: vec![0; T * e],
            gate: Vec::new(),
            up: Vec::new(),
            m: x.clone(),
        }],
        final_hidden: Vec::new(),
        final_normed: Vec::new(),
    };
    // `n_embd = 1` keeps the three other SyrkAcc trivial; the down Gram is the production shape.
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for _ in 0..RUNS {
        let t0 = Instant::now();
        let mut old = SyrkAcc::new(&cfg).expect("grams");
        old.accumulate(&tape, &cfg).expect("old accumulate");
        let old_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut new = SyrkAcc::new(&cfg).expect("grams");
        let ones: Vec<Vec<i64>> = vec![vec![0i64; e]; T];
        let rows = vec![vec![
            Vec::new(),
            ones.clone(),
            ones.clone(),
            ones,
            cols.clone(),
        ]];
        new.accumulate_tap(&rows).expect("new accumulate");
        let new_s = t1.elapsed().as_secs_f64();
        pairs.push((old_s, new_s));
    }
    let old_mean = pairs.iter().map(|p| p.0).sum::<f64>() / RUNS as f64;
    let new_mean = pairs.iter().map(|p| p.1).sum::<f64>() / RUNS as f64;
    for (o, n) in &pairs {
        println!("t={T} w={W} old {o:.3}s new {n:.3}s ratio {:.1}x", o / n);
    }
    println!(
        "mean old {old_mean:.3}s new {new_mean:.3}s ratio {:.1}x",
        old_mean / new_mean
    );
    assert!(
        new_mean < old_mean,
        "the limb path must beat the scalar exact fold"
    );
}

/// The engine capture's wall on the teacher, 512 tokens: prefill with the tap plus the Gram
/// accumulation. Skips when the fixture is absent.
#[test]
#[ignore = "measured report: reads models/gguf/qwen05b-f16.gguf; run with --ignored --nocapture"]
fn capture_qwen05b_512_wall() {
    const T: usize = 512;
    let Some(path) = testdata::model_path("models/gguf/qwen05b-f16.gguf") else {
        println!("qwen05b-f16.gguf absent; skipped");
        return;
    };
    let engine = Engine::open(&path).expect("open teacher");
    let ec = &engine.config;
    let cfg = Config {
        n_layer: ec.n_layer,
        n_head: ec.n_head,
        n_head_kv: ec.n_head_kv,
        n_embd: ec.n_embd,
        n_ff: ec.n_ff,
        vocab: ec.vocab_size,
        head_dim: ec.head_dim,
        eps: (ec.rms_eps.to_f64() * 4_294_967_296.0).round() as i64,
        rope_theta: ec.rope_theta.to_f64(),
    };
    let tokens: Vec<u32> = (0..T)
        .map(|i| u32::try_from(i % ec.vocab_size).expect("vocab"))
        .collect();
    for run in 0..5 {
        let t0 = Instant::now();
        let mut st = engine.state(T).expect("state");
        let (_, rows) = engine
            .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, true)
            .expect("tapped prefill");
        let t_fwd = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut grams = SyrkAcc::new(&cfg).expect("grams");
        grams.accumulate_tap(&rows).expect("accumulate");
        let t_syrk = t1.elapsed().as_secs_f64();
        println!(
            "qwen05b t={T} run {run}: prefill+tap {t_fwd:.2}s gram {t_syrk:.2}s total {:.2}s",
            t_fwd + t_syrk
        );
    }
}
