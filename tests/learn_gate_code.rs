//! Board . Lane (gate, up) and the `ffn_down` arm.
//!
//! The FFN gate as a hyperplane code on `Qwen3.5-0.8B`: under the production `fold_fit` arm,
//! per layer, (1) the off/near/on split of `g·x`, (2) the sign-flip rate of `ĝ·x` against
//! `arccos(ρ_H)/π` (Sheppard 1899; Charikar 2002), (3) the gate/up split of `E‖δh‖²`, and (4) the
//! held-out KL with only `ffn_gate`, only `ffn_up` or only `ffn_down` replaced.
//!
//! Every statistic is a Q32.32 word computed from the engine's own tap stage 3 (`STAGE_LN2_OUT`,
//! the gate/up input) and the teacher's own `ffn_gate`/`ffn_up` rows, so the flip is the weight's
//! and not the activation quantiser's. The engine is read on its one inference path
//! (`Engine::prefill_tap_cols`, `KernelMode::Q8Act`).
//!
//! Stages, selected by `APEIRON_GATE_STAGE`:
//! - `probe`: one 512-token tapped prefill and one production `fold_fit`, so the cost equation is
//!   measured before a campaign is launched.
//! - `fit`: calibration `H` from chunks `0..CALIB`, then the production fit of
//!   `ffn_gate` and `ffn_up` for `APEIRON_GATE_LAYERS`, cached as Q32.32 words.
//! - `stats`: the per-layer table over the held-out chunks.
//! - `kl`: held-out KL of one arm against the llama.cpp `--kl-divergence-base` file.
//!
//! Grade `measured`: every number is the print of a command that ran.
//!
//! Token disjointness: `H` is accumulated over chunk indices `0..calib` of the base file's token
//! block; every statistic and every KL row is scored on chunk indices `calib..`, a distinct text
//! range of the same corpus, so no token is in both sets.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the Q32.32 ↔ f64 boundary is range-checked by the grid it came from; the \
              single-letter names are the geometry's symbols (`g`, `u`, `h`, `x`, `t`, `o`, `n`) \
              and every loop is one linear composition of the stage it measures"
)]

use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant;
use apeiron::learn::fold::fit::eigh::eigh;
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFit, FoldFitConfig};
use apeiron::learn::fold::run::convert::{encode_f16, encode_f32};
use apeiron::learn::fold::stats::basis::solve_water_fill;
use apeiron::learn::fold::stats::capture::{LayerSyrk, Normalisation, SyrkAcc};
use apeiron::learn::fold::stats::heldout::{held_out, held_out_kldbase_chunk};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: the `Qwen3.5-0.8B` artefact, binary16. The engine refuses an `F32`
/// embedding row (`src/engine/mod.rs:29`), so `qwen08b-f32.gguf` is transcoded to binary16 with
/// `llama-quantize`; every other tensor keeps the `f32` value within binary16's 11-bit mantissa.
const MODEL_DEFAULT: &str = "~/kld08/qwen08b-f16.gguf";

/// The teacher image path, overridable by `APEIRON_GATE_MODEL`.
fn model_path() -> String {
    std::env::var("APEIRON_GATE_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}
/// llama.cpp's `--kl-divergence-base` file for the binary32 teacher, 40 chunks of `n_ctx = 512`.
const BASE: &str = "~/kld08/calib.c40.kld";

/// Calibration chunks `0..CALIB`: 8 chunks of 512 tokens give 4096 rows, above the `n_ff = 3584`
/// width of the down Gram; 2 chunks bound its rank by 1024 and `fold_fit` refuses it as not
/// positive-definite. Every held-out read starts at chunk `CALIB` or later.
/// Fixed with the production SyrkAcc under board .
const CALIB: usize = 8;

/// The base file path, overridable by `APEIRON_GATE_BASE` (the binary16 teacher's own base).
fn base_path() -> String {
    std::env::var("APEIRON_GATE_BASE").unwrap_or_else(|_| String::from(BASE))
}
/// Where the fitted rows and the calibration Gram are cached as Q32.32 words.
const CACHE: &str = "~/kld08/gatecode-tok";

/// Tap slot of the FFN-side `RMSNorm` output: the gate/up input (`STAGE_LN2_OUT`).
const TAP_LN2: usize = 3;
/// Tap slot of the gate pre-activation (`STAGE_GATE`).
const TAP_GATE: usize = 8;
/// Tap slot of the `SwiGLU` product `m = silu(g·x)⊙(u·x)`, the `ffn_down` input (`STAGE_M`).
const TAP_M: usize = 4;

/// The production call-site configuration (`fold_transfer.py:1733-1736`). The search knobs cost no
/// bits and are overridable for a sweep (board row ): `APEIRON_FIT_BEAM`,
/// `APEIRON_FIT_SWEEPS`, `APEIRON_FIT_INNER`, and `APEIRON_FIT_RIDGE_PPM`, `APEIRON_FIT_DAMP_PPM`,
/// `APEIRON_FIT_ICM_PPM` in millionths (defaults 1000, 10000, 200000).
fn production_cfg() -> FoldFitConfig {
    let q = |key: &str, ppm: usize| ppm_word(env_usize(key, ppm));
    FoldFitConfig {
        iters: 1,
        ridge: q("APEIRON_FIT_RIDGE_PPM", 1_000),
        damp: q("APEIRON_FIT_DAMP_PPM", 10_000),
        inner_iters: env_usize("APEIRON_FIT_INNER", 3),
        e8_beam: env_usize("APEIRON_FIT_BEAM", 4),
        icm_ridge: q("APEIRON_FIT_ICM_PPM", 200_000),
        sweeps: env_usize("APEIRON_FIT_SWEEPS", 2),
    }
}

/// `ppm` millionths as a Q32.32 word, rounded to nearest.
fn ppm_word(ppm: usize) -> i64 {
    ((((ppm as u128) << 32) + 500_000) / 1_000_000) as i64
}

/// Post-pack fit selection (board row ): one `fold_fit` per `APEIRON_FIT_SELECT`
/// entry (`gptq` sets `inner_iters = 0`; a number sets the icm ridge in millionths), keeping the least
/// calibration [`energy`] of packed `Wp − W`; `fold_fit` selects on `J` before the pack. Unset: production.
/// Cost: per entry one `fold_fit` and one [`energy`] (`O(o·n²)`); space one extra fit.
fn fit_select(w: &[i64], h: &[f64], o: usize, n: usize) -> (FoldFit, String) {
    let hq = qv(h);
    let set = std::env::var("APEIRON_FIT_SELECT").unwrap_or_default();
    if set.is_empty() {
        let fit = fold_fit(w, &hq, o, n, None, &production_cfg()).expect("fit");
        return (fit, String::from("production"));
    }
    let mut best: Option<(f64, FoldFit, String)> = None;
    for arm in set.split(',') {
        let mut c = production_cfg();
        if arm == "gptq" {
            c.inner_iters = 0;
        } else {
            c.icm_ridge = ppm_word(arm.parse().expect("APEIRON_FIT_SELECT entry"));
        }
        let fit = fold_fit(w, &hq, o, n, None, &c).expect("fit");
        let e: Vec<i64> = fit.wp.iter().zip(w).map(|(a, b)| a - b).collect();
        let en = energy(h, &e, o, n);
        println!(
            "SELECT {arm} tag {} calib energy {en:.6e}",
            fit.tag.as_str()
        );
        if best.as_ref().is_none_or(|b| en < b.0) {
            best = Some((en, fit, arm.to_string()));
        }
    }
    let (_, fit, arm) = best.expect("APEIRON_FIT_SELECT has an entry");
    (fit, arm)
}

/// The cached fit name of `kind` on calibration `calib`/`extra`, `APEIRON_GATE_TAG` appended.
fn fit_name(kind: &str, calib: usize, extra: usize) -> String {
    let tag = std::env::var("APEIRON_GATE_TAG").unwrap_or_default();
    match (calib != CALIB, extra > 0) {
        (true, _) => format!("{kind}-c{calib}x{extra}{tag}"),
        (false, true) => format!("{kind}-x{extra}{tag}"),
        _ => format!("{kind}{tag}"),
    }
}

/// An environment knob with a default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The layer list from `APEIRON_GATE_LAYERS` (`0,3,7`), or every layer.
fn env_layers(n_layer: usize) -> Vec<usize> {
    match std::env::var("APEIRON_GATE_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => (0..n_layer).collect(),
    }
}

/// `words` as reals, one conversion at the boundary.
fn unq(words: &[i64]) -> Vec<f64> {
    words.iter().map(|&w| from_grid(w)).collect()
}

/// `x` as Q32.32 words, the one rounding into the grid.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
}

/// Write `words` as a little-endian `i64` payload with an `(o, n)` header, through a
/// temporary file and a rename: a run killed mid-write leaves no short payload behind.
fn write_words(path: &str, o: usize, n: usize, words: &[i64]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    let mut b = [(o as u64).to_le_bytes(), (n as u64).to_le_bytes()].concat();
    b.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    std::fs::write(&tmp, b)?;
    std::fs::rename(&tmp, path)
}

/// Read a [`write_words`] payload back as `(o, n, words)`.
fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("8")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("8")) as usize;
    let words = b[16..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect::<Vec<i64>>();
    assert_eq!(words.len(), o * n, "{path}: payload is not o*n words");
    (o, n, words)
}

/// `Y = W·X` in `f64`: `w` is `o × n` row-major, `x` is `t × n` token-major, the result `o × t`
/// unit-major. `O(o·n·t)` multiply-adds; the inner loop runs over the contiguous token axis.
fn project(w: &[f64], x: &[f64], o: usize, n: usize, t: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; o * t];
    for u in 0..o {
        let row = &w[u * n..(u + 1) * n];
        let dst = &mut out[u * t..(u + 1) * t];
        for (k, &wk) in row.iter().enumerate() {
            let xk = &x[k * t..(k + 1) * t];
            for (d, &x) in dst.iter_mut().zip(xk) {
                *d += wk * x;
            }
        }
    }
    out
}

/// The logistic function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(x) = x·σ(x)`.
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `d/dx silu(x) = σ(x)·(1 + x·σ(−x))`.
fn dsilu(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// The chunk token block of a base file, `n_chunk` rows of `n_ctx` tokens.
fn base_tokens(path: &str) -> (Vec<Vec<u32>>, usize) {
    let mut base = KldBase::open(Path::new(path)).expect("open base file");
    let n = base.n_chunk();
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut n_ctx = 0usize;
    for _ in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        n_ctx = chunk.tokens.len();
        out.push(chunk.tokens);
    }
    (out, n_ctx)
}

/// One tapped `Q8Act` prefill of `tokens`, returning the `[layer][stage][column][width]` rows.
fn tapped(engine: &Engine, tokens: &[u32]) -> Vec<Vec<Vec<Vec<i64>>>> {
    let mut st = engine.state(tokens.len()).expect("decode state");
    let (_out, rows) = engine
        .prefill_tap_cols(&mut st, tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    rows
}

/// `vᵀ·H·v` for the symmetric `h`, `n × n`.
fn quad(h: &[f64], v: &[f64], n: usize) -> f64 {
    let mut hv = vec![0.0f64; n];
    for i in 0..n {
        let row = &h[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for (a, b) in row.iter().zip(v) {
            s += a * b;
        }
        hv[i] = s;
    }
    let mut out = 0.0f64;
    for (a, b) in hv.iter().zip(v) {
        out += a * b;
    }
    out
}

/// The gate and up row of one layer as the teacher's Q32.32 words, with the layer's mixer name.
fn gate_up(engine: &Engine, cfg: &Config, l: usize) -> (Vec<i64>, Vec<i64>, String) {
    let ll = layer_from_engine(engine, cfg, l).expect("layer linears");
    let take = |name: &str| {
        ll.linears
            .iter()
            .find(|x| x.name == name)
            .unwrap_or_else(|| panic!("blk.{l}.{name}"))
            .clone()
    };
    let g = take("ffn_gate");
    let u = take("ffn_up");
    assert_eq!(g.rows, cfg.n_ff, "layer {l}: n_ff");
    assert_eq!(g.cols, cfg.n_embd, "layer {l}: n_embd");
    (g.w, u.w, format!("{:?}", ll.mixer))
}

/// The teacher's `ffn_down` words (`n_embd × n_ff`) of one layer, with the layer's mixer name.
fn ffn_down(engine: &Engine, cfg: &Config, l: usize) -> (Vec<i64>, String) {
    let ll = layer_from_engine(engine, cfg, l).expect("layer linears");
    let d = ll
        .linears
        .iter()
        .find(|x| x.name == "ffn_down")
        .unwrap_or_else(|| panic!("blk.{l}.ffn_down"))
        .clone();
    assert_eq!(d.rows, cfg.n_embd, "layer {l}: down rows");
    assert_eq!(d.cols, cfg.n_ff, "layer {l}: down cols");
    (d.w, format!("{:?}", ll.mixer))
}

/// The per-layer geometric piece of the metric.
#[derive(Debug, Clone, Copy)]
struct LayerFit {
    /// `arccos(ρ_H)/π`.
    angle: f64,
    /// `‖g − ĝ‖_H / ‖g‖_H`.
    rel_g: f64,
    /// `‖u − û‖_H / ‖u‖_H`.
    rel_u: f64,
}

/// `ρ_H(g, ĝ)` and the two relative `H`-errors for one layer.
fn layer_fit(h: &[f64], g: &[f64], gh: &[f64], u: &[f64], uh: &[f64], n: usize) -> LayerFit {
    let (ghg, ghgh) = (quad(h, g, n), quad(h, gh, n));
    let (uhu, _uhuh) = (quad(h, u, n), quad(h, uh, n));
    let mut dot = 0.0f64;
    for i in 0..n {
        let row = &h[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for (a, b) in row.iter().zip(gh) {
            s += a * b;
        }
        dot += s * g[i];
    }
    let rho = dot / (ghg.max(0.0).sqrt() * ghgh.max(0.0).sqrt());
    let dg: Vec<f64> = g.iter().zip(gh).map(|(a, b)| a - b).collect();
    let du: Vec<f64> = u.iter().zip(uh).map(|(a, b)| a - b).collect();
    LayerFit {
        angle: rho.clamp(-1.0, 1.0).acos() / std::f64::consts::PI,
        rel_g: (quad(h, &dg, n) / ghg).max(0.0).sqrt(),
        rel_u: (quad(h, &du, n) / uhu).max(0.0).sqrt(),
    }
}

/// Calibration chunks `0..calib` plus `extra` chunks from `CALIB + 3` on, so the held-out chunks
/// 8, 9, 10 of [`gate_code_kl`] stay out of `H`. `O(chunks)` token copies.
fn calib_set(chunks: &[Vec<u32>], calib: usize, extra: usize) -> Vec<Vec<u32>> {
    let from = CALIB + 3;
    assert!(
        extra == 0 || calib <= CALIB,
        "extra chunks need calib <= {CALIB}"
    );
    assert!(
        from + extra <= chunks.len(),
        "extra {extra}: {} chunks",
        chunks.len()
    );
    chunks[..calib]
        .iter()
        .chain(&chunks[from..from + extra])
        .cloned()
        .collect()
}

/// The production calibration SyrkAcc over chunks `0..calib`: `SyrkAcc::accumulate_tap` on each tapped
/// chunk, then `Normalisation::Mean`. `gate_up` is the `ffn_gate`/`ffn_up` input Gram (stage 3) and
/// `down` the `ffn_down` input Gram (stage 4, the engine's own `SwiGLU` product), both `Σ_t x_t x_tᵀ / T`
/// over tokens, so this file restates no layout. Layout control, per layer in `layers`: the diagonal of each Gram against `Σ_t x_t[k]² / T`
/// summed straight from the tap columns, relative error below `1e-4`. A feature-major buffer read as
/// token rows (the defect this replaces) misses it by `O(1)`. Cost:
/// `O(calib·n_layer·t·(n_embd² + q_out² + n_ff²)/2)` multiply-adds in the accumulator plus
/// `O(calib·|layers|·t·(n_embd + n_ff))` for the control; space the accumulator's
/// `O(n_layer·(n_embd² + q_out² + n_ff²))` words and one `f64` copy.
fn calib_syrks(
    engine: &Engine,
    cfg: &Config,
    chunks: &[Vec<u32>],
    calib: usize,
    layers: &[usize],
) -> Vec<LayerSyrk> {
    let (n, f) = (cfg.n_embd, cfg.n_ff);
    let mut grams = SyrkAcc::new(cfg).expect("grams");
    let mut diag: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![0.0; n], vec![0.0; f]); cfg.n_layer];
    let mut seen = 0usize;
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate().take(calib) {
        let rows = tapped(engine, tokens);
        grams.accumulate_tap(&rows).expect("accumulate");
        for &l in layers {
            for col in &rows[l][TAP_LN2] {
                for (k, v) in unq(col).iter().enumerate() {
                    diag[l].0[k] += v * v;
                }
            }
            for col in &rows[l][TAP_M] {
                for (k, v) in unq(col).iter().enumerate() {
                    diag[l].1[k] += v * v;
                }
            }
        }
        seen += tokens.len();
        println!(
            "GRAMS calib chunk {c} at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let done = grams.finish(Normalisation::Mean).expect("finish");
    for &l in layers {
        let checks = [
            ("gate_up", &done[l].gate_up, &diag[l].0, n),
            ("down", &done[l].down, &diag[l].1, f),
        ];
        for (name, g, d, w) in checks {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for k in 0..w {
                let e = d[k] / seen as f64;
                num += (g[k * w + k] - e).powi(2);
                den += e * e;
            }
            let rel = (num / den).sqrt();
            println!("GRAMS layer {l} {name} diag rel {rel:.3e}");
            assert!(
                rel < 1e-4,
                "layer {l} {name}: Gram diagonal off the tap columns by {rel}"
            );
        }
    }
    done
}

/// [`calib_syrks`] over all of `set`, cached per layer as `g{l}.{tag}.{o,gate_up,down}.i64` in Q32.32
/// words, the precision every consumer reads them at. When every file exists they are read and the
/// layout control is not re-run (it ran when they were written); `qkv` and `tokens` are not cached.
/// Layers outside `layers` hold empty SyrkAcc.
///
/// `O(|layers|·(q_out² + n_embd² + n_ff²))` words read or written, plus one [`calib_syrks`] on a miss.
fn syrks_cached(
    engine: &Engine,
    cfg: &Config,
    set: &[Vec<u32>],
    layers: &[usize],
    tag: &str,
) -> Vec<LayerSyrk> {
    let path = |l: usize, k: &str| format!("{CACHE}/g{l}.{tag}.{k}.i64");
    let kinds = ["o", "gate_up", "down"];
    let hit = layers.iter().all(|&l| {
        kinds
            .iter()
            .all(|k| std::path::Path::new(&path(l, k)).exists())
    });
    if !hit {
        let grams = calib_syrks(engine, cfg, set, set.len(), layers);
        for &l in layers {
            for (k, g) in kinds
                .iter()
                .zip([&grams[l].o, &grams[l].gate_up, &grams[l].down])
            {
                let w = g.len().isqrt();
                write_words(&path(l, k), w, w, &qv(g)).expect("cache Gram");
            }
        }
        return grams;
    }
    let read = |l: usize, k: &str| unq(&read_words(&path(l, k)).2);
    (0..cfg.n_layer)
        .map(|l| {
            let on = layers.contains(&l);
            let get = |k: &str| if on { read(l, k) } else { Vec::new() };
            LayerSyrk {
                qkv: Vec::new(),
                o: get("o"),
                gate_up: get("gate_up"),
                down: get("down"),
                tokens: 0,
            }
        })
        .collect()
}

/// The cost equation and the two fixtures: one 512-token tapped prefill, one production `fold_fit`.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher; run explicitly with --ignored"]
fn gate_code_probe() {
    let t = Instant::now();
    let engine = Engine::open(&model_path()).expect("open teacher");
    println!("PROBE load {:.1} s", t.elapsed().as_secs_f64());
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    println!(
        "PROBE geometry n_embd {} n_ff {} n_layer {} chunks {} n_ctx {}",
        cfg.n_embd,
        cfg.n_ff,
        cfg.n_layer,
        chunks.len(),
        n_ctx
    );
    let t = Instant::now();
    let rows = tapped(&engine, &chunks[0]);
    let prefill = t.elapsed().as_secs_f64();
    for l in 0..cfg.n_layer {
        assert_eq!(
            rows[l][TAP_LN2].len(),
            n_ctx,
            "layer {l}: stage 3 records every column"
        );
        assert_eq!(
            rows[l][TAP_LN2][0].len(),
            cfg.n_embd,
            "layer {l}: stage 3 width"
        );
    }
    println!("PROBE tap {prefill:.2} s per {n_ctx}-token chunk");
    let n = cfg.n_embd;
    let t = Instant::now();
    let grams = calib_syrks(&engine, &cfg, &chunks, 1, &[0]);
    let gram = t.elapsed().as_secs_f64();
    println!("PROBE grams {gram:.2} s per chunk, all layers, tap included");
    let (g, u, mixer) = gate_up(&engine, &cfg, 0);
    let hw = qv(&grams[0].gate_up);
    let t = Instant::now();
    let fit = fold_fit(&g, &hw, cfg.n_ff, n, None, &production_cfg()).expect("fit");
    let fw = t.elapsed().as_secs_f64();
    println!(
        "PROBE fit layer 0 gate {mixer} {fw:.2} s, {} words, bits r/c/d {}/{}/{}",
        fit.wp.len(),
        fit.bits_r,
        fit.bits_c,
        fit.bits_d
    );
    let _ = u;
    let layers2 = 2.0 * f64::from(u32::try_from(cfg.n_layer).expect("24"));
    println!(
        "PROBE equation: fit {:.0} s for {layers2} tensors; H {gram:.0} s per calib chunk; \
         stats {prefill:.0} s per layer-chunk",
        fw * layers2
    );
}

/// The [`fit_select`] fit of each kind in `APEIRON_GATE_KIND` (comma list of `down`, `o`, `gate`,
/// `up`; default `down`) for `APEIRON_GATE_LAYERS`, under one calibration `H` over [`calib_set`]
/// (`APEIRON_GATE_CALIB`, `APEIRON_GATE_EXTRA`), written as `l{l}.{name}.i64` by [`fit_name`]; an
/// existing file is skipped. `APEIRON_GATE_CALIB=6` drops chunks 6–7, the start of held-out 8–10's
/// article. `APEIRON_GATE_COPY=1` writes the `APEIRON_GATE_MODEL` tensor itself, no fit (with
/// `APEIRON_GATE_TAG=-q4km`: same-scope `Q4_K_M` arm). Cost: [`calib_syrks`], [`fit_select`].
#[test]
#[ignore = "opens the teacher and runs production fold_fit per tensor; run explicitly"]
fn gate_code_fit_kind() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let kinds = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("down"));
    let extra = env_usize("APEIRON_GATE_EXTRA", 0);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let set = calib_set(&chunks, calib, extra);
    std::fs::create_dir_all(CACHE).expect("cache dir");
    let t0 = Instant::now();
    let grams = syrks_cached(&engine, &cfg, &set, &layers, &format!("c{calib}x{extra}"));
    for &l in &layers {
        for kind in kinds.split(',') {
            let name = fit_name(kind, calib, extra);
            let path = format!("{CACHE}/l{l}.{name}.i64");
            if std::path::Path::new(&path).exists() {
                println!("FITK layer {l:02} {name} exists, skipped");
                continue;
            }
            let (w, o, n, h) = teacher(&engine, &cfg, &grams, kind, l);
            if std::env::var("APEIRON_GATE_COPY").is_ok() {
                write_words(&path, o, n, &w).expect("copy teacher");
                continue;
            }
            let t = Instant::now();
            let (fit, arm) = fit_select(&w, &h, o, n);
            write_words(&path, o, n, &fit.wp).expect("cache fit");
            println!(
                "FITK layer {l:02} {name} {o}x{n} {:.2} s bits r/c/d {}/{}/{} tag {} arm {arm}",
                t.elapsed().as_secs_f64(),
                fit.bits_r,
                fit.bits_c,
                fit.bits_d,
                fit.tag.as_str()
            );
        }
    }
    println!(
        "FITK done {} layers, {} calibration tokens, {:.1} s",
        layers.len(),
        set.len() * n_ctx,
        t0.elapsed().as_secs_f64()
    );
}

/// `tr(E H Eᵀ)` for the `o × n` word matrix `e` under the symmetric `n × n` Gram `h`: one [`quad`]
/// per row, `Σ_u e_uᵀ H e_u`.
///
/// `O(o·n²)` multiply-adds for the `o` quadratic forms, the dominant per-layer term beside
/// `fold_fit`; space `O(n)` for the converted row.
fn energy(h: &[f64], e: &[i64], o: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for u in 0..o {
        acc += quad(h, &unq(&e[u * n..(u + 1) * n]), n);
    }
    acc
}

/// The teacher words `W` (`o × n`) of `kind` in layer `l` and its calibration Gram `H` (`n × n`):
/// for gate/up the cached `g{l}.c8x0.gate_up.i64` ([`syrks_cached`]) when `grams` is empty, else `grams`' Gram of that kind.
///
/// `O(o·n)` to read the tensor; space `O(o·n + n²)`.
fn teacher(
    engine: &Engine,
    cfg: &Config,
    grams: &[LayerSyrk],
    kind: &str,
    l: usize,
) -> (Vec<i64>, usize, usize, Vec<f64>) {
    match kind {
        "down" => {
            let (d, _mixer) = ffn_down(engine, cfg, l);
            (d, cfg.n_embd, cfg.n_ff, unq(&qv(&grams[l].down)))
        }
        "o" => {
            let ll = layer_from_engine(engine, cfg, l).expect("layer linears");
            let w = ll
                .linears
                .into_iter()
                .find(|x| matches!(x.name, "attn_output" | "ssm_out"))
                .unwrap_or_else(|| panic!("blk.{l} output projection"));
            let (o, q) = (w.rows, w.cols);
            (w.w, o, q, unq(&qv(&grams[l].o)))
        }
        _ => {
            let n = cfg.n_embd;
            let h = if grams.is_empty() {
                let (ho, hn, hw) = read_words(&format!("{CACHE}/g{l}.c8x0.gate_up.i64"));
                assert_eq!((ho, hn, hw.len()), (n, n, n * n), "layer {l}: cached H");
                unq(&hw)
            } else {
                unq(&qv(&grams[l].gate_up))
            };
            let (g, up, _mixer) = gate_up(engine, cfg, l);
            (if kind == "gate" { g } else { up }, cfg.n_ff, n, h)
        }
    }
}

/// **Measurement 8** (board ). A second production `fold_fit` on the
/// residual of `ffn_up` or `ffn_gate`: `R = W − Ŵ`, `out = Ŵ + R̂`, so the added bits are one more
/// fit of the same tensor: its sign plane `o·n` plus `bits_r + bits_c + bits_d`; the metric is
/// unchanged.
///
/// `APEIRON_GATE_KIND` (`up` | `gate` | `down` | `o`, default `up`), `APEIRON_GATE_LAYERS` (default:
/// every layer), `APEIRON_GATE_PASSES` (`P`, default 2: the cached fit plus `P − 1` residual passes).
/// `H` is the calibration Gram of chunks `0..CALIB`: the cached `g{l}.c8x0.gate_up.i64` for gate/up, the
/// [`calib_syrks`] `down` or `o` Gram for down/o; no held-out chunk is read. `APEIRON_GATE_EXTRA`
/// widens `H` by [`calib_set`] (every kind) and starts from `l{l}.{kind}-x{extra}.i64`; with
/// `APEIRON_GATE_CALIB` set off `CALIB` it starts from `l{l}.{kind}-c{calib}x{extra}.i64`, the
/// [`fit_name`] (board row ); each pass fits through [`fit_select`].
/// Writes `{CACHE}/l{l}.{base}-res{P}.i64`, the summed words, for [`gate_code_kl`]
/// (`APEIRON_GATE_ARM=up-res2`); `P = 1` writes the cached fit back unchanged.
///
/// Cost: `(P−1)` `fold_fit` per layer (~33 s each for gate/up, ~52–75 s down, measured) plus `2(P−1)`
/// [`energy`] at `O(o·n²)`, and for down/o one [`calib_syrks`] (~70 s); space `O(n² + o·n)` per layer
/// for the Gram, the words and the residual. **Measurement 11** adds `o` and the `down`/`o` passes.
#[test]
#[ignore = "opens the teacher and runs production fold_fit on the residual; run explicitly"]
fn ffn_residual_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("up"));
    assert!(
        matches!(kind.as_str(), "gate" | "up" | "down" | "o"),
        "APEIRON_GATE_KIND {kind}"
    );
    let passes = env_usize("APEIRON_GATE_PASSES", 2).max(1);
    let extra = env_usize("APEIRON_GATE_EXTRA", 0);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let base = fit_name(&kind, calib, extra);
    println!("RES layers {layers:?} kind {kind} base {base} passes {passes}");
    let t0 = Instant::now();
    let grams = if matches!(kind.as_str(), "down" | "o") || extra > 0 || calib != CALIB {
        let (chunks, _) = base_tokens(&base_path());
        let set = calib_set(&chunks, calib, extra);
        syrks_cached(&engine, &cfg, &set, &layers, &format!("c{calib}x{extra}"))
    } else {
        Vec::new()
    };
    for &l in &layers {
        let (w, o, n, h) = teacher(&engine, &cfg, &grams, &kind, l);
        let (wo, wn, cached) = read_words(&format!("{CACHE}/l{l}.{base}.i64"));
        assert_eq!((wo, wn, cached.len()), (o, n, o * n), "layer {l}: {base}");
        let mut what = cached;
        for p in 1..passes {
            let e: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let before = energy(&h, &e, o, n);
            let r: Vec<i64> = w.iter().zip(&what).map(|(a, b)| a - b).collect();
            let t = Instant::now();
            let (fit, arm) = fit_select(&r, &h, o, n);
            println!("RES layer {l:02} kind {kind} pass {p} arm {arm}");
            let wall = t.elapsed().as_secs_f64();
            for (dst, add) in what.iter_mut().zip(&fit.wp) {
                *dst += add;
            }
            let e2: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let after = energy(&h, &e2, o, n);
            let ratio = if before > 0.0 { after / before } else { 0.0 };
            let bits = (o * n) as u64 + fit.bits_r + fit.bits_c + fit.bits_d;
            println!(
                "RES layer {l:02} kind {kind} pass {p} energy before {before:.6e} after {after:.6e} \
                 ratio {ratio:.4} fit {wall:.1} s bits sign/r/c/d {}/{}/{}/{} (+{:.4} b/w)",
                o * n,
                fit.bits_r,
                fit.bits_c,
                fit.bits_d,
                bits as f64 / (o * n) as f64
            );
        }
        let path = format!("{CACHE}/l{l}.{base}-res{passes}.i64");
        write_words(&path, o, n, &what).expect("write residual fit");
        println!("RES wrote {path}");
    }
    println!(
        "RES done layers {} kind {kind} passes {passes} wall {:.1} s",
        layers.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// The eigenvalues of the symmetric `n × n` row-major `a`, ascending: Householder reduction to
/// tridiagonal form (`A ← PAP`, `P = I − uuᵀ/h`, full storage so every access is a row), then the
/// implicit-shift QL iteration (Wilkinson and Reinsch, *Handbook for Automatic Computation* II
/// (1971), `tred2`/`tql1`), values only. A measurement instrument for [`gate_code_floor`]; its
/// control is the floor of [`eigh`] on layer 0's `H`.
///
/// `O(4n³/3)` multiply-adds for the reduction (per step `i` a `i × i` matrix–vector product and a
/// rank-2 update), `O(n²)` for QL at a bounded iteration count per value; space `O(n)` besides `a`.
fn eigvalsh(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; n];
    for i in (1..n).rev() {
        let mut u = a[i * n..i * n + i].to_vec();
        let scale: f64 = u.iter().map(|x| x.abs()).sum();
        if i == 1 || scale == 0.0 {
            e[i] = a[i * n + i - 1];
            continue;
        }
        for x in &mut u {
            *x /= scale;
        }
        let mut h: f64 = u.iter().map(|x| x * x).sum();
        let f = u[i - 1];
        let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
        e[i] = scale * g;
        h -= f * g;
        u[i - 1] = f - g;
        let p: Vec<f64> = (0..i)
            .map(|j| {
                a[j * n..j * n + i]
                    .iter()
                    .zip(&u)
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
                    / h
            })
            .collect();
        let k = p.iter().zip(&u).map(|(x, y)| x * y).sum::<f64>() / (h + h);
        let q: Vec<f64> = p.iter().zip(&u).map(|(x, y)| x - k * y).collect();
        for j in 0..i {
            let (uj, qj) = (u[j], q[j]);
            for (x, (uk, qk)) in a[j * n..j * n + i].iter_mut().zip(u.iter().zip(&q)) {
                *x -= uj * qk + qj * uk;
            }
        }
    }
    let mut d: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n && e[m].abs() > f64::EPSILON * (d[m].abs() + d[m + 1].abs()) {
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter < 64, "eigvalsh: QL did not converge at {l}");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0f64, 1.0f64, 0.0f64);
            let mut split = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    split = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if split {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d.sort_by(|x, y| x.partial_cmp(y).expect("finite eigenvalue"));
    d
}

/// **Measurement 12** (board ). The rate–distortion floor of each
/// fitted tensor against its fit. Rows `w ~ N(0, s²I)` under the metric `H` (the isotropic Gaussian
/// model of the onebit-w13 note, grade `model`): reverse water-filling over `s²λ_i(H)` at
/// `b·n` bits per row gives `ρ_floor(b) = Σ min(θ, λ_i) / Σ λ_i` (`s²` cancels). Per cached plane
/// count `P` (`l{l}.{kind}.i64`, `-res2`, `-res3`) it prints the fit's `ρ = E_H(Ŵ − W) / E_H(W)`, its
/// b/w `P·(o·n + scale)/(o·n)`, `ρ_floor` at that b/w, and `b*`, the floor's rate at the fit's `ρ`.
/// `APEIRON_GATE_KIND` and `APEIRON_GATE_LAYERS` as in [`ffn_residual_fit`].
///
/// Cost: one [`eigvalsh`] of `H` per layer, `O(4n³/3)` (`APEIRON_FLOOR_EIGH` selects the crate [`eigh`],
/// `O(I·n³)`, 120 s at `n = 1024`, measured); `1 + P` [`energy`] at `O(o·n²)`; per plane a
/// 50-step bisection over [`solve_water_fill`] at `O(n·S)` each; for down/o one [`calib_syrks`].
#[test]
#[ignore = "opens the teacher and decomposes each calibration Gram; run explicitly"]
fn gate_code_floor() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("up"));
    let grams = if matches!(kind.as_str(), "down" | "o") {
        let (chunks, _) = base_tokens(&base_path());
        calib_syrks(&engine, &cfg, &chunks, CALIB, &layers)
    } else {
        Vec::new()
    };
    for &l in &layers {
        let (w, o, n, h) = teacher(&engine, &cfg, &grams, &kind, l);
        let t = Instant::now();
        let lam: Vec<i64> = if std::env::var("APEIRON_FLOOR_EIGH").is_ok() {
            let (evals, _) = eigh(&qv(&h), n).expect("eigh H");
            evals.iter().map(|&x| x.max(0)).collect()
        } else {
            qv(&eigvalsh(h.clone(), n)
                .iter()
                .map(|&x| x.max(0.0))
                .collect::<Vec<_>>())
        };
        let total: f64 = lam.iter().map(|&x| from_grid(x)).sum();
        let rho = |b: f64| {
            let budget = to_grid(b * n as f64, GRID_BOUND).expect("budget");
            let wf = solve_water_fill(&lam, budget, 0);
            assert!(wf.theta > 0, "layer {l}: water-fill refused at {b} b/w");
            from_grid(wf.distortion) / total
        };
        let ew = energy(&h, &w, o, n);
        println!(
            "FLOOR layer {l:02} kind {kind} eigh {:.1} s E_H(W) {ew:.4e} floor@1.0 {:.4} @1.18 {:.4} \
             @2.0 {:.4} @3.0 {:.4}",
            t.elapsed().as_secs_f64(),
            rho(1.0),
            rho(1.18),
            rho(2.0),
            rho(3.0)
        );
        let plane = (o * n) as f64 + plane_scale_bits(o, n) as f64;
        for p in 1..=3usize {
            let path = if p == 1 {
                format!("{CACHE}/l{l}.{kind}.i64")
            } else {
                format!("{CACHE}/l{l}.{kind}-res{p}.i64")
            };
            if !Path::new(&path).exists() {
                continue;
            }
            let (_, _, what) = read_words(&path);
            let e: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let rf = energy(&h, &e, o, n) / ew;
            let b = p as f64 * plane / (o * n) as f64;
            let (mut lo, mut hi) = (0.0f64, 16.0f64);
            for _ in 0..50 {
                let mid = f64::midpoint(lo, hi);
                if rho(mid) > rf {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            println!(
                "FLOOR layer {l:02} kind {kind} planes {p} b/w {b:.4} rho fit {rf:.4} floor {:.4} \
                 b* {hi:.4} gap {:.4} b/w",
                rho(b),
                b - hi
            );
        }
    }
}

/// `A·Bᵀ` for row-major `a` (`m × k`) and `b` (`p × k`): every entry a row–row dot product, rows of
/// the output split across the available threads.
///
/// `O(m·p·k)` multiply-adds; space `O(m·p)` for the output.
fn gemm_nt(a: &[f64], b: &[f64], m: usize, k: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * p];
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let rows = m.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(rows * p).enumerate() {
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(p).enumerate() {
                    let ar = &a[(c * rows + i) * k..(c * rows + i + 1) * k];
                    for (j, x) in row.iter_mut().enumerate() {
                        *x = ar
                            .iter()
                            .zip(&b[j * k..(j + 1) * k])
                            .map(|(u, v)| u * v)
                            .sum();
                    }
                }
            });
        }
    });
    out
}

/// The lower Cholesky factor `L` of the symmetric `n × n` `h` (`H = L Lᵀ`, Cholesky–Banachiewicz); a
/// non-positive pivot is clamped to `1e-12·tr(H)/n`.
///
/// `O(n³/3)` multiply-adds; space `O(n²)`.
fn cholesky(h: &[f64], n: usize) -> Vec<f64> {
    let floor = 1e-12 * (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|q| l[i * n + q] * l[j * n + q]).sum();
            if i == j {
                l[i * n + i] = (h[i * n + i] - s).max(floor).sqrt();
            } else {
                l[i * n + j] = (h[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    l
}

/// `x` with `L Lᵀ x = b` for the lower Cholesky factor `l` (`k × k`): forward then back substitution.
///
/// `O(k²)` multiply-adds; space `O(k)`.
fn chol_solve(l: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..k {
        let s: f64 = (0..i).map(|q| l[i * k + q] * y[q]).sum();
        y[i] = (y[i] - s) / l[i * k + i];
    }
    for i in (0..k).rev() {
        let s: f64 = (i + 1..k).map(|q| l[q * k + i] * y[q]).sum();
        y[i] = (y[i] - s) / l[i * k + i];
    }
    y
}

/// **Measurement 14** (board ). A residual plane on `k` input columns
/// only: the variable-rate step of the floor (`gate_code_floor`) spent where the error is. For a
/// correction `D` supported on columns `S`, `tr((E + D) H (E + D)ᵀ)` is least at
/// `D_S = −(E H)_{:,S} H_SS⁻¹` (the complement's error carried into `S`, as GPTQ's feedback);
/// one production `fold_fit` codes that target under `H_SS` and is added on `S`. Column `j`'s score
/// is its exact single-column gain `‖(E H)_{:,j}‖² / H_jj`; the top `k` are gathered ascending.
/// `APEIRON_GATE_KIND`, `APEIRON_GATE_LAYERS` as in [`ffn_residual_fit`]; `APEIRON_COLS_K`
/// (multiple of 128, default 128); `APEIRON_COLS_BASE` (default the kind) names the starting fit
/// `l{l}.{base}.i64`; `APEIRON_COLS_PASSES` (`P`, default 1) repeats selection and fit on the
/// updated residual, a greedy water-filling over columns. `APEIRON_COLS_RIDGE` (`λ`, default 0)
/// adds `λ·tr(H_SS)/k` to the transfer's diagonal; `APEIRON_COLS_TARGET=resid` drops the transfer
/// (target `−E_S`). Each pass also prints the held-out energy ([`heldout_h`]). Writes
/// `l{l}.{base}-cols{k}x{P}.i64` (`r{λ}` / `e` appended for a ridge / residual target).
///
/// Cost per layer: `O(o·n²)` for `E H`, `O(k³/3 + o·k²)` for the solve, one `fold_fit` on `o × k`,
/// two [`energy`] at `O(o·n²)`; space `O(o·n + n²)`.
#[test]
#[ignore = "opens the teacher and runs production fold_fit on selected columns; run explicitly"]
fn cols_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("o"));
    let base = std::env::var("APEIRON_COLS_BASE").unwrap_or_else(|_| kind.clone());
    let k = env_usize("APEIRON_COLS_K", 128);
    assert!(k.is_multiple_of(128) && k > 0, "APEIRON_COLS_K {k}");
    let (chunks, _) = base_tokens(&base_path());
    let grams = syrks_cached(&engine, &cfg, &chunks[..CALIB], &layers, "c8x0");
    let ho = syrks_cached(&engine, &cfg, &chunks[CALIB..CALIB + 3], &layers, "ho");
    let passes = env_usize("APEIRON_COLS_PASSES", 1).max(1);
    let ridge: f64 = std::env::var("APEIRON_COLS_RIDGE").map_or(0.0, |v| v.parse().expect("ridge"));
    let resid = std::env::var("APEIRON_COLS_TARGET").is_ok_and(|v| v == "resid");
    let r = if ridge > 0.0 {
        format!("r{ridge}")
    } else {
        String::new()
    };
    let tag = format!("cols{k}x{passes}{r}{}", if resid { "e" } else { "" });
    let mut added = 0u64;
    for &l in &layers {
        let (w, o, n, h) = teacher(&engine, &cfg, &grams, &kind, l);
        let hh = heldout_h(&ho, &kind, l);
        let (_, _, mut what) = read_words(&format!("{CACHE}/l{l}.{base}.i64"));
        let mut failed = false;
        for p in 1..=passes {
            let ei: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let (before, ho_before) = (energy(&h, &ei, o, n), energy(&hh, &ei, o, n));
            let eh = gemm_nt(&unq(&ei), &h, o, n, n);
            let mut idx: Vec<usize> = (0..n).collect();
            let gain = |j: usize| (0..o).map(|u| eh[u * n + j].powi(2)).sum::<f64>() / h[j * n + j];
            idx.sort_by(|&a, &b| gain(b).partial_cmp(&gain(a)).expect("finite"));
            let mut s = idx[..k].to_vec();
            s.sort_unstable();
            let mut hss: Vec<f64> = s
                .iter()
                .flat_map(|&a| s.iter().map(move |&b| (a, b)))
                .map(|(a, b)| h[a * n + b])
                .collect();
            let lam = ridge * (0..k).map(|c| hss[c * k + c]).sum::<f64>() / k as f64;
            (0..k).for_each(|c| hss[c * k + c] += lam);
            let lss = cholesky(&hss, k);
            let mut target = Vec::with_capacity(o * k);
            for u in 0..o {
                if resid {
                    target.extend(s.iter().map(|&j| -unq(&ei[u * n + j..=u * n + j])[0]));
                    continue;
                }
                let row: Vec<f64> = s.iter().map(|&j| eh[u * n + j]).collect();
                target.extend(chol_solve(&lss, &row, k).iter().map(|x| -x));
            }
            let t = Instant::now();
            let fit = match fold_fit(&qv(&target), &qv(&hss), o, k, None, &production_cfg()) {
                Ok(fit) => fit,
                Err(err) => {
                    println!("COLS layer {l:02} kind {kind} pass {p} fit error {err:?}");
                    failed = true;
                    break;
                }
            };
            let wall = t.elapsed().as_secs_f64();
            for u in 0..o {
                for (c, &j) in s.iter().enumerate() {
                    what[u * n + j] += fit.wp[u * k + c];
                }
            }
            let e2: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let (after, ho_after) = (energy(&h, &e2, o, n), energy(&hh, &e2, o, n));
            let bits = (o * k) as u64 + fit.bits_r + fit.bits_c + fit.bits_d;
            added += bits;
            let bw = bits as f64 / (o * n) as f64;
            println!(
                "COLS layer {l:02} kind {kind} base {base} k {k} pass {p} energy before \
                 {before:.6e} after {after:.6e} ratio {:.4} heldout ratio {:.4} ln/b-w {:.3} fit \
                 {wall:.1} s bits sign/r/c/d {}/{}/{}/{} (+{bw:.4} b/w)",
                after / before,
                ho_after / ho_before,
                (after / before).ln() / bw,
                o * k,
                fit.bits_r,
                fit.bits_c,
                fit.bits_d
            );
        }
        if !failed {
            let path = format!("{CACHE}/l{l}.{base}-{tag}.i64");
            write_words(&path, o, n, &what).expect("write column fit");
        }
    }
    println!("COLS added {added} bits ({:.3} Mbit)", added as f64 / 1e6);
}

/// The held-out metric of `kind` at layer `l`: its Gram over chunks 8, 9, 10, the KL chunks of
/// [`gate_code_kl`] (`ho` is [`calib_syrks`] on that slice).
fn heldout_h(ho: &[LayerSyrk], kind: &str, l: usize) -> Vec<f64> {
    let g = match kind {
        "o" => &ho[l].o,
        "down" => &ho[l].down,
        _ => &ho[l].gate_up,
    };
    unq(&qv(g))
}

/// **Measurement 15** (board ). Is held-out energy the fast proxy of
/// held-out KL? For cached fits `l{l}.{arm}.i64`, prints `E_H(E)` under the calibration Gram
/// (chunks 0..8) and the held-out Gram ([`heldout_h`]), and the sums over layers against the first
/// arm. `APEIRON_GATE_KIND`, `APEIRON_GATE_LAYERS` as in [`ffn_residual_fit`]; `APEIRON_HE_ARMS`
/// comma-separated (default `o,o-res2,o-cols256x8`); a layer without an arm's file skips it, so
/// sums compare only over a common layer set.
///
/// Cost: the SyrkAcc over 11 chunks (~7 s each), two [`energy`] per arm and layer at `O(o·n²)`.
#[test]
#[ignore = "opens the teacher and reads the cached fits; run explicitly"]
fn heldout_energy() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("o"));
    let arms =
        std::env::var("APEIRON_HE_ARMS").unwrap_or_else(|_| String::from("o,o-res2,o-cols256x8"));
    let arms: Vec<&str> = arms.split(',').collect();
    let (chunks, _) = base_tokens(&base_path());
    let grams = syrks_cached(&engine, &cfg, &chunks[..CALIB], &layers, "c8x0");
    let ho = syrks_cached(&engine, &cfg, &chunks[CALIB..CALIB + 3], &layers, "ho");
    let mut tot = vec![(0.0f64, 0.0f64); arms.len()];
    for &l in &layers {
        let (w, o, n, h) = teacher(&engine, &cfg, &grams, &kind, l);
        let hh = heldout_h(&ho, &kind, l);
        for (a, arm) in arms.iter().enumerate() {
            let path = format!("{CACHE}/l{l}.{arm}.i64");
            if !Path::new(&path).exists() {
                continue;
            }
            let (_, _, what) = read_words(&path);
            let e: Vec<i64> = what.iter().zip(&w).map(|(x, y)| x - y).collect();
            let (ec, eh) = (energy(&h, &e, o, n), energy(&hh, &e, o, n));
            tot[a] = (tot[a].0 + ec, tot[a].1 + eh);
            println!("HELDOUT layer {l:02} kind {kind} arm {arm} calib {ec:.6e} heldout {eh:.6e}");
        }
    }
    for (a, arm) in arms.iter().enumerate() {
        println!(
            "HELDOUT sum arm {arm} calib {:.6e} heldout {:.6e} vs {} calib {:.4} heldout {:.4}",
            tot[a].0,
            tot[a].1,
            arms[0],
            tot[a].0 / tot[0].0,
            tot[a].1 / tot[0].1
        );
    }
}

/// The per-layer table: off/near/on, flip against `arccos(ρ_H)/π`, the `H` errors and the split.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher and reads the cached fits; run explicitly with --ignored"]
fn gate_code_stats() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let held = env_usize("APEIRON_GATE_HELD", 1);
    let (n, o) = (cfg.n_embd, cfg.n_ff);
    let t = held * n_ctx;
    let mut xs: Vec<Vec<i64>> = vec![vec![0i64; n * t]; cfg.n_layer];
    let mut g8: Vec<Vec<i64>> = vec![vec![0i64; o * t]; cfg.n_layer];
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate().skip(calib).take(held) {
        let rows = tapped(&engine, tokens);
        let at = (c - calib) * n_ctx;
        for l in 0..cfg.n_layer {
            for (ti, col) in rows[l][TAP_LN2].iter().enumerate() {
                for (k, &w) in col.iter().enumerate() {
                    xs[l][k * t + at + ti] = w;
                }
            }
            for (ti, col) in rows[l][TAP_GATE].iter().enumerate() {
                g8[l][(at + ti) * o..(at + ti + 1) * o].copy_from_slice(col);
            }
        }
        println!(
            "STATS held chunk {c} at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let mut tot = [0.0f64; 6];
    let mut cnt = 0.0f64;
    for &l in &layers {
        let (gw, uw, mixer) = gate_up(&engine, &cfg, l);
        let (_, _, ghw) = read_words(&format!("{CACHE}/l{l}.gate.i64"));
        let (_, _, uhw) = read_words(&format!("{CACHE}/l{l}.up.i64"));
        let (_, _, hw) = read_words(&format!("{CACHE}/g{l}.c8x0.gate_up.i64"));
        let (gf, uf, ghat, uhat, hf) = (unq(&gw), unq(&uw), unq(&ghw), unq(&uhw), unq(&hw));
        let ti = Instant::now();
        let xf = unq(&xs[l]);
        let gx = project(&gf, &xf, o, n, t);
        let ghx = project(&ghat, &xf, o, n, t);
        let ux = project(&uf, &xf, o, n, t);
        let uhx = project(&uhat, &xf, o, n, t);
        let fit = layer_fit(&hf, &gf, &ghat, &uf, &uhat, n);
        let (mut off, mut near, mut on, mut flip, mut nz) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut tg2, mut tu2, mut dh2, mut tg2f) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for u in 0..o {
            for s in 0..t {
                let a = gx[u * t + s];
                let b = ghx[u * t + s];
                let c = ux[u * t + s];
                let d = uhx[u * t + s];
                if a < -2.0 {
                    off += 1.0;
                } else if a > 2.0 {
                    on += 1.0;
                } else {
                    near += 1.0;
                }
                let mut flipped = false;
                if a != 0.0 {
                    nz += 1.0;
                    if (a > 0.0) != (b > 0.0) {
                        flip += 1.0;
                        flipped = true;
                    }
                }
                let dh = silu(b) * d - silu(a) * c;
                let tg = dsilu(a) * (b - a) * c;
                let tu = silu(a) * (d - c);
                dh2 += dh * dh;
                tg2 += tg * tg;
                tu2 += tu * tu;
                if flipped {
                    tg2f += tg * tg;
                }
            }
        }
        // The engine's own Q8Act gate against `W_gate·x` from the same stage-3 row.
        let (mut en, mut ed) = (0.0f64, 0.0f64);
        for s in 0..t {
            for u in 0..o {
                let e = from_grid(g8[l][s * o + u]) - gx[u * t + s];
                en += e * e;
                ed += gx[u * t + s] * gx[u * t + s];
            }
        }
        let cells = (o * t) as f64;
        println!(
            "LAYER {l:02} {mixer} off {:.4} near {:.4} on {:.4} flip {:.6} acos {:.6} relg {:.3e} relu {:.3e} share_g {:.4} share_u {:.4} cross {:+.4} flipshare {:.4} q8 {:.2e} proj {:.1} s",
            off / cells,
            near / cells,
            on / cells,
            flip / nz,
            fit.angle,
            fit.rel_g,
            fit.rel_u,
            tg2 / dh2,
            tu2 / dh2,
            1.0 - tg2 / dh2 - tu2 / dh2,
            tg2f / tg2,
            (en / ed).sqrt(),
            ti.elapsed().as_secs_f64()
        );
        for (a, b) in tot.iter_mut().zip([
            off / cells,
            near / cells,
            on / cells,
            flip / nz,
            tg2 / dh2,
            tu2 / dh2,
        ]) {
            *a += b;
        }
        cnt += 1.0;
        if l == 0 {
            // Positive control: `ĝ = g` on the same words and tokens.
            let mut p = 0.0f64;
            let mut pnz = 0.0f64;
            for u in 0..o {
                for s in 0..t {
                    let a = gx[u * t + s];
                    if a != 0.0 {
                        pnz += 1.0;
                        if (a > 0.0) != (a > 0.0) {
                            p += 1.0;
                        }
                    }
                }
            }
            println!(
                "CONTROL positive flip {:.9} over {:.0} nonzero",
                p / pnz,
                pnz
            );
            // Negative control: with `ĝ = g`, negating one row must flip every token
            // (`sign(g·x) != sign(-g·x)` on every column whose `g·x` is nonzero).
            let mut flips = 0.0f64;
            let mut nzt = 0.0f64;
            for s in 0..t {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc -= gf[k] * xf[k * t + s];
                }
                if gx[s] != 0.0 {
                    nzt += 1.0;
                    if (gx[s] > 0.0) != (acc > 0.0) {
                        flips += 1.0;
                    }
                }
            }
            println!(
                "CONTROL negative unit-0 flip {:.6} over {:.0} tokens",
                flips / nzt,
                nzt
            );
            assert_eq!(p, 0.0, "the positive control must be exactly zero");
        }
    }
    println!(
        "MEAN over {:.0} layers: off {:.4} near {:.4} on {:.4} flip {:.6} share_g {:.4} share_u {:.4}",
        cnt,
        tot[0] / cnt,
        tot[1] / cnt,
        tot[2] / cnt,
        tot[3] / cnt,
        tot[4] / cnt,
        tot[5] / cnt
    );
}

/// The measured layers: every layer with a cached `key` fit.
fn fitted_layers(key: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for l in 0..1024 {
        if Path::new(&format!("{CACHE}/l{l}.{key}.i64")).exists() {
            out.push(l);
        }
    }
    out
}

/// The teacher image with FFN tensors replaced by cached words, every other byte the teacher's.
/// The arm is one or more parts joined by `+` (`gate+up-rows-sal+down-spqr9-salience`); each part's
/// tensor kind is its prefix before the first `-`, and it replaces that kind in every layer holding
/// `{CACHE}/l{l}.{part}.i64`. Returns the image and the number of tensors replaced.
///
/// `O(file bytes)` to read plus `O(tensors·n_ff·n_embd)` written; the whole image is held once.
fn student_bytes(arm: &str) -> (Vec<u8>, usize) {
    let mut data = std::fs::read(model_path()).expect("read teacher");
    let gguf = Gguf::open(&model_path()).expect("teacher directory");
    let mut kinds = Vec::new();
    let mut replaced = 0;
    for part in arm.split('+') {
        let kind = part.split('-').next().unwrap_or(part);
        assert!(
            matches!(kind, "gate" | "up" | "down" | "o"),
            "APEIRON_GATE_ARM {arm}: tensor kind {kind} is not gate | up | down | o"
        );
        assert!(
            !kinds.contains(&kind),
            "APEIRON_GATE_ARM {arm}: kind {kind} twice"
        );
        kinds.push(kind);
        let layers = fitted_layers(part);
        assert!(
            !layers.is_empty(),
            "APEIRON_GATE_ARM {arm}: no cached {part}"
        );
        replaced += layers.len();
        replace_kind(&mut data, &gguf, kind, part, &layers);
    }
    (data, replaced)
}

/// Overwrites `blk.{l}.ffn_{kind}.weight` (kind `o`: `attn_output`, or `ssm_out` on a Gated
/// `DeltaNet` layer) in `data` with `{CACHE}/l{l}.{part}.i64` for each layer.
fn replace_kind(data: &mut [u8], gguf: &Gguf, kind: &str, part: &str, layers: &[usize]) {
    for &l in layers {
        let names = if kind == "o" {
            vec![
                format!("blk.{l}.attn_output.weight"),
                format!("blk.{l}.ssm_out.weight"),
            ]
        } else {
            vec![format!("blk.{l}.ffn_{kind}.weight")]
        };
        let (_, _, words) = read_words(&format!("{CACHE}/l{l}.{part}.i64"));
        let t = gguf
            .header
            .tensors
            .iter()
            .find(|t| {
                names
                    .iter()
                    .any(|n| String::from_utf8_lossy(&t.name) == n.as_str())
            })
            .unwrap_or_else(|| panic!("{names:?}"));
        let name = String::from_utf8_lossy(&t.name).into_owned();
        let off = usize::try_from(gguf.header.data0 + t.offset).expect("offset");
        let bytes = match t.ggml_type {
            quant::GGML_TYPE_F16 => encode_f16(&words).expect("binary16 encode"),
            quant::GGML_TYPE_F32 => encode_f32(&words),
            other => panic!("{name}: ggml type {other}"),
        };
        data[off..off + bytes.len()].copy_from_slice(&bytes);
    }
}

/// Held-out KL of one arm against llama.cpp's own base distribution.
#[test]
#[ignore = "reads the ~5 GB base file and a 3.1 GB student image; run explicitly with --ignored"]
fn gate_code_kl() {
    let arm = std::env::var("APEIRON_GATE_ARM").unwrap_or_else(|_| String::from("teacher"));
    let skip = env_usize("APEIRON_GATE_SKIP", CALIB);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    assert!(
        skip >= calib,
        "held-out chunk {skip} lies inside calibration 0..{calib}"
    );
    let count = env_usize("APEIRON_GATE_CHUNKS", 10);
    let t0 = Instant::now();
    let (data, replaced) = if arm == "teacher" {
        (std::fs::read(model_path()).expect("read teacher"), 0)
    } else {
        student_bytes(&arm)
    };
    println!(
        "KL arm {arm} tensors {replaced} image {:.1} GB in {:.1} s",
        data.len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let engine = Engine::from_bytes(&format!("student-{arm}"), data).expect("student engine");
    let mut base = KldBase::open(Path::new(&base_path())).expect("base file");
    for _ in 0..skip.min(base.n_chunk()) {
        let _ = base.next_chunk().expect("skip");
    }
    let n = base.n_chunk().saturating_sub(skip).min(count);
    let (mut kl, mut top1, mut nll, mut tokens) = (0.0f64, 0.0f64, 0.0f64, 0u64);
    let mut per = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        let t = Instant::now();
        let s = held_out_kldbase_chunk(&engine, &chunk).expect("score chunk");
        let (k, p, m) = (from_grid(s.kl), from_grid(s.top1), from_grid(s.nll_student));
        per.push(k);
        kl += k;
        top1 += p;
        nll += m;
        tokens += s.tokens;
        println!(
            "KL {arm} chunk {} kl {k:.6} top1 {p:.6} nll {m:.4} wall {:.1} s",
            skip + i,
            t.elapsed().as_secs_f64()
        );
    }
    let nf = n as f64;
    let mean = kl / nf;
    let sd = (per.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0)).sqrt();
    println!(
        "KL {arm} MEAN chunks {skip}..{} KL {mean:.6} +- {sd:.6} (sd) top1 {:.6} nll {:.4} tokens {tokens} wall {:.1} s",
        skip + n,
        top1 / nf,
        nll / nf,
        t0.elapsed().as_secs_f64()
    );
}

/// The positive control: the teacher's own bytes as the student must give KL exactly zero.
#[test]
#[ignore = "opens two 3.1 GB images; run explicitly with --ignored"]
fn gate_code_control_self() {
    let teacher = Engine::open(&model_path()).expect("teacher");
    let data = std::fs::read(model_path()).expect("read teacher");
    let student = Engine::from_bytes("roundtrip", data).expect("student engine");
    let (chunks, _n_ctx) = base_tokens(&base_path());
    let at = env_usize("APEIRON_GATE_CALIB", CALIB);
    let len = env_usize("APEIRON_GATE_T", 128);
    let seq: &[u32] = chunks[at].get(..len).expect("column prefix");
    let h = held_out(&teacher, &student, &[seq]).expect("held out");
    println!(
        "CONTROL self KL {} top1 {} tokens {}",
        from_grid(h.kl),
        from_grid(h.top1),
        h.tokens
    );
    assert_eq!(h.kl, 0, "identical images must score KL 0");
    assert_eq!(
        h.top1,
        apeiron::engine::ONE,
        "identical images agree on top-1"
    );
}

/// The scale bits `bits_r + bits_c + bits_d` of one production [`fold_fit`] on `o × n` words:
/// `quant_r_hier`'s `4·o·(n/128) + 8·o + 64` (128 is [`fold_fit`]'s `GROUP`), `quant_c_u8`'s
/// `6·n + 32` for the row carrier and `5·n + 32` for the column carrier. `154752` at `3584 × 1024`,
/// the measured figure of T07's layer-23 residual fit.
///
/// `O(1)`; space none.
fn plane_scale_bits(o: usize, n: usize) -> u64 {
    let ou = u64::try_from(o).expect("o fits u64");
    let nu = u64::try_from(n).expect("n fits u64");
    (4 * ou * (nu / 128) + 8 * ou + 64) + (6 * nu + 32) + (5 * nu + 32)
}
