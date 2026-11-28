//! Board . Lane : where `ffn_down`'s
//! held-out KL lives on `Qwen3.5-0.8B`.
//!
//! Two measurements, both on the engine's one inference path (`KernelMode::Q8Act`):
//!
//! - `down_one_layer_kl`: the teacher image with `blk.<l>.ffn_down.weight` replaced by the cached
//!   production fit of ONE layer, held-out KL against llama.cpp's own base distribution on chunk
//!   `CALIB` (8). One image per listed layer, scored in turn, so the per-layer KL costs are additive
//!   by construction (`APEIRON_DOWN_LAYERS`, `APEIRON_DOWN_SKIP`, `APEIRON_DOWN_CHUNKS`).
//! - `down_unit_energy`: the exact decomposition of the hidden-layer error energy
//!   `tr(E H Eᵀ) = Σ_i c_i` with `E = D̂ − D` (`n_embd × n_ff`) and `H` the held-out down Gram
//!   `E[m mᵀ]` (`n_ff × n_ff`, the production `SyrkAcc::down` over tap stage 4, the engine's `SwiGLU`
//!   product `m = silu(g·x)⊙(u·x)`).
//!
//! Token disjointness: `H` for the energy is the mean Gram over chunk `CALIB` (8) of the base file's
//! token block — the same held-out chunk the KL is scored on, and no calibration chunk. The cached fits
//! were fitted on chunks `0..CALIB` (`gate_code_fit_down`, `tests/learn_gate_code.rs`).
//!
//! Grade `measured`: every number is the print of a command that ran.
//!
//! Controls: an image with no tensor replaced (KL 0 and top-1 1 exactly); `E = 0` (zero energy and
//! every `c_i = 0`); `E` one column of `D` zeroed (`c_i = 0` off that unit and
//! `c_k = H_kk ‖D[:,k]‖²`).

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
              single-letter names are the geometry's symbols (`D`, `E`, `H`, `g`, `u`, `h`, `c`) \
              and every loop is one linear composition of the stage it measures"
)]

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant;
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig};
use apeiron::learn::fold::run::convert::{encode_f16, encode_f32};
use apeiron::learn::fold::stats::capture::{Normalisation, SyrkAcc};
use apeiron::learn::fold::stats::heldout::{held_out, held_out_kldbase_chunk};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: the `Qwen3.5-0.8B` artefact, binary16 (`llama-quantize f32 f16`), the image
/// the cached fits were scored against.
const MODEL_DEFAULT: &str = "~/kld08/qwen08b-f16.gguf";

/// The teacher image path, overridable by `APEIRON_DOWN_MODEL`.
fn model_path() -> String {
    std::env::var("APEIRON_DOWN_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}

/// llama.cpp's `--kl-divergence-base` file for the binary16 teacher, 40 chunks of `n_ctx = 512`.
const BASE: &str = "~/kld08/calib.c40.kld";

/// The calibration chunk count of `gate_code_fit_down` (`tests/learn_gate_code.rs`): the cached fits
/// saw chunks `0..CALIB`, so every held-out read here starts at chunk `CALIB` or later.
const CALIB: usize = 8;

/// The base file path, overridable by `APEIRON_DOWN_BASE`.
fn base_path() -> String {
    std::env::var("APEIRON_DOWN_BASE").unwrap_or_else(|_| String::from(BASE))
}

/// Where the fitted rows and the calibration SyrkAcc are cached as Q32.32 words.
const CACHE: &str = "~/kld08/gatecode-tok";

/// Tap slot of the `SwiGLU` product `m`, the `ffn_down` input (`STAGE_M`), `n_ff` wide.
const TAP_M: usize = 4;

/// An environment knob with a default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The layer list from `APEIRON_DOWN_LAYERS`, or every layer with a cached `ffn_down` fit.
fn down_layers() -> Vec<usize> {
    match std::env::var("APEIRON_DOWN_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => fitted_layers("down"),
    }
}

/// `words` as reals, one conversion at the boundary.
fn unq(words: &[i64]) -> Vec<f64> {
    words.iter().map(|&w| from_grid(w)).collect()
}

/// Read the `(o, n)`-headed little-endian `i64` payload back as `(o, n, words)`.
fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("8")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("8")) as usize;
    let words = b[16..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect();
    (o, n, words)
}

/// Write `words` as a little-endian `i64` payload with an `(o, n)` header — the exact bytes
/// [`read_words`] reads back.
///
/// `O(o·n)` 8-byte writes, one unbuffered `write_all` per word; space `O(1)` beyond `words`.
fn write_words(path: &str, o: usize, n: usize, words: &[i64]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(o as u64).to_le_bytes())?;
    f.write_all(&(n as u64).to_le_bytes())?;
    for w in words {
        f.write_all(&w.to_le_bytes())?;
    }
    Ok(())
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

/// Add `Σ_r x_r x_rᵀ` into `acc`, `n × n`, from `x` as `rows × n` row-major words.
///
/// `O(rows·n²/2)` multiply-adds, the upper triangle only; the row loop is outer so each `O(n)`
/// row is read once per row index.
fn syrk_add(x: &[i64], rows: usize, n: usize, acc: &mut [f64]) {
    for r in 0..rows {
        let row = unq(&x[r * n..(r + 1) * n]);
        for i in 0..n {
            let xi = row[i];
            let base = i * n;
            for j in i..n {
                acc[base + j] += xi * row[j];
            }
        }
    }
}

/// Mirror the upper triangle of an `n × n` Gram. `O(n²)` copies.
fn mirror(acc: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            acc[j * n + i] = acc[i * n + j];
        }
    }
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

/// The production call-site configuration (`fold_transfer.py:1733-1736`), copied from
/// `tests/learn_gate_code.rs`.
fn production_cfg() -> FoldFitConfig {
    FoldFitConfig {
        iters: 1,
        ridge: 4_294_967,
        damp: 42_949_673,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 858_993_459,
        sweeps: 2,
    }
}

/// `x` as Q32.32 words, the one rounding into the grid; copied from `tests/learn_gate_code.rs`.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
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

/// The teacher image with `blk.<l>.ffn_down.weight` replaced by the cached fitted words for the
/// listed layers, every other byte the teacher's.
///
/// `O(file bytes)` to read plus `O(layers·n_ff·n_embd)` written; the whole image is held once.
fn student_bytes(key: &str, layers: &[usize]) -> Vec<u8> {
    let mut data = std::fs::read(model_path()).expect("read teacher");
    let gguf = Gguf::open(&model_path()).expect("teacher directory");
    for &l in layers {
        let name = format!("blk.{l}.ffn_down.weight");
        let (_, _, words) = read_words(&format!("{CACHE}/l{l}.{key}.i64"));
        let t = gguf
            .header
            .tensors
            .iter()
            .find(|t| String::from_utf8_lossy(&t.name) == name)
            .unwrap_or_else(|| panic!("{name}"));
        let off = usize::try_from(gguf.header.data0 + t.offset).expect("offset");
        let bytes = match t.ggml_type {
            quant::GGML_TYPE_F16 => encode_f16(&words).expect("binary16 encode"),
            quant::GGML_TYPE_F32 => encode_f32(&words),
            other => panic!("{name}: ggml type {other}"),
        };
        data[off..off + bytes.len()].copy_from_slice(&bytes);
    }
    data
}

/// Score the held-out chunks `skip..skip+count` of the base file on `engine`, `O(cost(prefill) +
/// rows·cost(head))` each; returns `(kl, top1)` means and the per-chunk rows.
fn held_out_kl(
    engine: &Engine,
    path: &str,
    skip: usize,
    count: usize,
) -> (f64, f64, Vec<(usize, f64, f64)>) {
    let mut base = KldBase::open(Path::new(path)).expect("base file");
    for _ in 0..skip.min(base.n_chunk()) {
        let _ = base.next_chunk().expect("skip");
    }
    let n = base.n_chunk().saturating_sub(skip).min(count).max(1);
    let (mut kl, mut top1) = (0.0f64, 0.0f64);
    let mut per = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        let t = Instant::now();
        let s = held_out_kldbase_chunk(engine, &chunk).expect("score chunk");
        let (k, p) = (from_grid(s.kl), from_grid(s.top1));
        println!(
            "  chunk {} kl {k:.6} top1 {p:.6} nll {:.4} tokens {} wall {:.1} s",
            skip + i,
            from_grid(s.nll_student),
            s.tokens,
            t.elapsed().as_secs_f64()
        );
        kl += k;
        top1 += p;
        per.push((skip + i, k, p));
    }
    (kl / n as f64, top1 / n as f64, per)
}

/// **Measurement 1.** Held-out KL of the teacher with only `ffn_down` replaced in one layer at a
/// time, chunk `CALIB` (8) by default.
///
/// `APEIRON_DOWN_LAYERS` names the layers (default: every cached fit); one image is built and one
/// engine opened per layer, so the wall is additive and each invocation stays inside its budget.
#[test]
#[ignore = "reads the 1.6 GB teacher image per layer and scores ~174 s of held-out KL each"]
fn down_one_layer_kl() {
    let layers = down_layers();
    let skip = env_usize("APEIRON_DOWN_SKIP", CALIB);
    let count = env_usize("APEIRON_DOWN_CHUNKS", 1);
    println!(
        "DOWNKL layers {layers:?} chunks {skip}..{} (all-seven down arm 0.102195 on chunk 8 at \
         f13cf1a5)",
        skip + count
    );
    let mut per = Vec::new();
    let mut sum = 0.0f64;
    for &l in &layers {
        let t0 = Instant::now();
        let data = student_bytes("down", &[l]);
        let engine = Engine::from_bytes(&format!("down-l{l}"), data).expect("student engine");
        println!(
            "DOWNKL layer {l} image ready at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let (kl, top1, _rows) = held_out_kl(&engine, &base_path(), skip, count);
        println!(
            "DOWNKL layer {l:02} kl {kl:.6} top1 {top1:.6} wall {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        per.push((l, kl, top1));
        sum += kl;
    }
    for (l, kl, top1) in &per {
        println!(
            "DOWNKL share layer {l:02} kl {kl:.6} top1 {top1:.6} share {:.4}",
            kl / sum
        );
    }
    println!(
        "DOWNKL sum over {} layers {sum:.6} mean {:.6}",
        per.len(),
        sum / per.len() as f64
    );
}

/// **Control, positive.** The teacher's own bytes as the student, scored against the teacher's own
/// engine: held-out KL exactly 0 and top-1 exactly 1. The image is unmodified, so the only
/// difference the scorer can see is none.
///
/// The engine's floor against llama.cpp's own base file is a separate, larger number — measured
/// `0.000735` against `calib.c40.f16.kld` and `0.000907` against `calib.c40.kld` on chunk 2 — and
/// is the numerical distance between the Q32.32 path and llama.cpp's, not between two of our
/// images. It is why the base-file KL of an unmodified image is not zero.
#[test]
#[ignore = "opens two 1.6 GB teacher images; run explicitly with --ignored"]
fn down_control_self() {
    let teacher = Engine::open(&model_path()).expect("teacher");
    let data = std::fs::read(model_path()).expect("read teacher");
    let student = Engine::from_bytes("down-roundtrip", data).expect("student engine");
    let (chunks, _n_ctx) = base_tokens(&base_path());
    let at = env_usize("APEIRON_DOWN_SKIP", CALIB);
    let len = env_usize("APEIRON_DOWN_T", 128);
    let seq: &[u32] = chunks[at].get(..len).expect("column prefix");
    let t = Instant::now();
    let h = held_out(&teacher, &student, &[seq]).expect("held out");
    println!(
        "CONTROL positive kl {} top1 {} tokens {} wall {:.1} s",
        from_grid(h.kl),
        from_grid(h.top1),
        h.tokens,
        t.elapsed().as_secs_f64()
    );
    assert_eq!(h.kl, 0, "identical images must score KL 0");
    assert_eq!(
        h.top1,
        apeiron::engine::ONE,
        "identical images agree on top-1"
    );
}

/// The exact energy of `E` (`o × n`, Q32.32 words) under the `n × n` Gram `h`:
/// `(tr(E H Eᵀ), Σ_i H_ii‖E[:,i]‖², c)` with `c_i = Σ_j (EᵀE)_ij H_ji`.
///
/// `O(o·n²/2 + n²)` multiply-adds: one `o × n × n` symmetric Gram `EᵀE`, then the `n²` contraction
/// with `h`. Space `O(n²)` for the Gram.
fn energy_of(e: &[i64], h: &[f64], o: usize, n: usize) -> (f64, f64, Vec<f64>) {
    let mut g = vec![0.0f64; n * n];
    syrk_add(e, o, n, &mut g);
    mirror(&mut g, n);
    let mut c = vec![0.0f64; n];
    for i in 0..n {
        let gi = &g[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for j in 0..n {
            s += gi[j] * h[j * n + i];
        }
        c[i] = s;
    }
    let diag: f64 = (0..n).map(|i| h[i * n + i] * g[i * n + i]).sum();
    (c.iter().sum(), diag, c)
}

/// The share of `v`'s sum carried by its `frac` largest entries (at least one), descending sort so
/// `O(n log n)`; `0` for a zero or non-finite sum.
fn top_share(v: &[f64], frac: f64) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let k = (((v.len() as f64) * frac).ceil() as usize).clamp(1, v.len());
    let total: f64 = s.iter().sum();
    if total == 0.0 || !total.is_finite() {
        return 0.0;
    }
    s[..k].iter().sum::<f64>() / total
}

/// The median of `v`, `O(n log n)`; the mean of the two middle entries for even `n`.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        f64::midpoint(s[n / 2 - 1], s[n / 2])
    }
}

/// **Measurement 2.** The per-unit decomposition of `tr(E H Eᵀ)` for each layer's `ffn_down` fit,
/// with `H` the held-out down Gram of chunk `APEIRON_DOWN_HELD` (default [`CALIB`]) — the held-out chunk,
/// not the calibration chunks the fit used.
#[test]
#[ignore = "opens the 1.6 GB teacher and rebuilds the n_ff x n_ff down Gram per layer"]
fn down_unit_energy() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let held = env_usize("APEIRON_DOWN_HELD", CALIB);
    let layers = down_layers();
    let (o, n) = (cfg.n_embd, cfg.n_ff);
    println!(
        "DOWNEN layers {layers:?} held chunk {held} tokens {n_ctx} n_embd {o} n_ff {n} \
         (H = held-out chunk {held} mean Gram, no calibration chunk)"
    );
    let t0 = Instant::now();
    let rows = tapped(&engine, &chunks[held]);
    let mut grams = SyrkAcc::new(&cfg).expect("grams");
    grams.accumulate_tap(&rows).expect("accumulate");
    let held_syrks = grams.finish(Normalisation::Mean).expect("finish");
    drop(grams);
    println!("DOWNEN tap and SyrkAcc {:.1} s", t0.elapsed().as_secs_f64());
    let mut control_done = false;
    let mut best_c = (0usize, 0.0f64);
    for &l in &layers {
        let t = Instant::now();
        let h: &[f64] = &held_syrks[l].down;
        let mut dm = vec![0.0f64; n];
        for col in &rows[l][TAP_M] {
            for (k, v) in unq(col).iter().enumerate() {
                dm[k] += v * v / n_ctx as f64;
            }
        }
        let (num, den) = (0..n).fold((0.0f64, 0.0f64), |(a, b), k| {
            (a + (h[k * n + k] - dm[k]).powi(2), b + dm[k] * dm[k])
        });
        let layout = (num / den).sqrt();
        assert!(
            layout < 1e-4,
            "layer {l}: Gram diagonal off the tap columns by {layout}"
        );
        println!("DOWNEN layer {l} Gram diag layout rel {layout:.3e}");
        let (d, mixer) = ffn_down(&engine, &cfg, l);
        let (ro, co, dhw) = read_words(&format!("{CACHE}/l{l}.down.i64"));
        assert_eq!((ro, co), (o, n), "layer {l}: cached down shape");
        let e: Vec<i64> = dhw.iter().zip(&d).map(|(a, b)| a - b).collect();
        let (total, diag, c) = energy_of(&e, h, o, n);
        let mut gd = vec![0.0f64; n * n];
        syrk_add(&d, o, n, &mut gd);
        mirror(&mut gd, n);
        let mut base = 0.0f64;
        for i in 0..n {
            let gi = &gd[i * n..(i + 1) * n];
            for j in 0..n {
                base += gi[j] * h[j * n + i];
            }
        }
        let hi: Vec<f64> = (0..n).map(|i| h[i * n + i]).collect();
        let sum_h: f64 = hi.iter().sum();
        let sum_h2: f64 = hi.iter().map(|v| v * v).sum();
        let pr = sum_h * sum_h / ((n as f64) * sum_h2);
        let absfrac: f64 = c.iter().map(|v| v.abs()).sum::<f64>() / total;
        let hmin = hi.iter().copied().fold(f64::INFINITY, f64::min);
        let top_c = (0..n)
            .max_by(|&a, &b| c[a].partial_cmp(&c[b]).expect("finite"))
            .expect("n>0");
        if c[top_c] / total > best_c.1 {
            best_c = (l, c[top_c] / total);
        }
        println!(
            "LAYER {l:02} {mixer} total {total:.6e} rel {:.6e} diag {diag:.6e} diagshare {:.4} \
             c_top1 {:.4} c_top10 {:.4} h_top1 {:.4} h_top10 {:.4} pr {:.4} hmax {:.4e} hmed {:.4e} \
             hmin {:.4e} absfrac {:.4} topc_unit {top_c} topc_share {:.4} wall {:.1} s",
            total / base,
            diag / total,
            top_share(&c, 0.01),
            top_share(&c, 0.10),
            top_share(&hi, 0.01),
            top_share(&hi, 0.10),
            pr,
            hi.iter().copied().fold(0.0f64, f64::max),
            median(&hi),
            hmin,
            absfrac,
            c[top_c] / total,
            t.elapsed().as_secs_f64()
        );
        if !control_done {
            control_done = true;
            // Negative: `D̂ = D`, so `E = 0`: zero energy and every `c_i = 0`.
            let (zt, zd, zc) = energy_of(&vec![0i64; o * n], h, o, n);
            println!(
                "CONTROL negative total {zt:.1e} diag {zd:.1e} max|c| {:.1e}",
                zc.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
            );
            assert_eq!(zt, 0.0, "E = 0 must give zero energy");
            assert_eq!(zd, 0.0, "E = 0 must give zero diagonal energy");
            assert!(
                zc.iter().all(|v| *v == 0.0),
                "E = 0 must give every c_i = 0"
            );
            // Attribution: `D̂ = D` with unit `k`'s column zeroed, so `E` is `-D[:,k]` in column `k`.
            let k = n / 2;
            let mut ek = vec![0i64; o * n];
            for u in 0..o {
                ek[u * n + k] = -d[u * n + k];
            }
            let (kt, _kd, kc) = energy_of(&ek, h, o, n);
            let want = h[k * n + k] * gd[k * n + k];
            let off: f64 = kc
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != k)
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);
            let rel = (kc[k] - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            println!(
                "CONTROL attribution unit {k} c_k {:.6e} H_kk|D_k|^2 {want:.6e} rel {rel:.2e} \
                 total {kt:.6e} max|c_i| i!=k {off:.1e}",
                kc[k]
            );
            assert_eq!(kt, kc[k], "one column carries the whole energy");
            assert!(off <= 1e-9 * want.abs(), "c_i = 0 off unit k");
            assert!(rel <= 1e-9, "c_k = H_kk ‖D[:,k]‖^2");
        }
    }
    println!(
        "DOWNEN done {} layers, largest single-unit share {:.4} at layer {:02}, {:.1} s",
        layers.len(),
        best_c.1,
        best_c.0,
        t0.elapsed().as_secs_f64()
    );
}

/// One xorshift64 step (Marsaglia 2003): `O(1)`, a pure function of `state`.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// The `k` retained `n_ff` columns under `select`: the `k` largest entries of `score` for `energy`
/// (the per-unit energy `c_i`) and `salience` (the input saliency), `random` `k` distinct indices
/// from [`xorshift64`] seeded `0x9E3779B97F4A7C15 ^ l`.
///
/// `O(n log n)` for the sort. The random walk draws `O(k / (1 − k/n))` in expectation (the
/// coupon-collector tail over `n` distinct indices); space `O(n)` for the index vector and the seen
/// map.
fn keep_indices(score: &[f64], k: usize, l: usize, select: &str) -> Vec<usize> {
    let n = score.len();
    let k = k.min(n);
    if select != "random" {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).expect("finite"));
        idx.truncate(k);
        return idx;
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (l as u64);
    let mut seen = vec![false; n];
    let mut out: Vec<usize> = Vec::with_capacity(k);
    while out.len() < k {
        let i = (xorshift64(&mut state) % n as u64) as usize;
        if !seen[i] {
            seen[i] = true;
            out.push(i);
        }
    }
    out
}

/// The calibration `ffn_down` input SyrkAcc (`SyrkAcc::down`, tap stage [`TAP_M`]) over chunks `0..CALIB`
/// of `chunks`, mean-normalised, returned per layer index over `0..cfg.n_layer` — empty for a layer
/// not in `layers`.
///
/// Layout control per listed layer: the Gram diagonal against `Σ_t m_t[k]² / T` summed straight from
/// the tap columns, relative error below `1e-4`. A feature-major buffer read as token rows (the
/// defect this replaces) misses it by `O(1)`.
///
/// `SyrkAcc::accumulate_tap` carries the accumulation, `O(CALIB·tokens·(n_embd² + q_out² + n_ff²))`;
/// this block adds `O(CALIB·|layers|·tokens·n_ff)` for the control. Space `O(n_layer·n_ff²)` for the
/// finished SyrkAcc plus `O(|layers|·n_ff)`.
fn calib_down(
    engine: &Engine,
    cfg: &Config,
    chunks: &[Vec<u32>],
    layers: &[usize],
) -> Vec<Vec<f64>> {
    let n = cfg.n_ff;
    let mut grams = SyrkAcc::new(cfg).expect("grams");
    let mut dm = vec![vec![0.0f64; n]; layers.len()];
    let mut tokens = 0usize;
    let t0 = Instant::now();
    for c in 0..CALIB.min(chunks.len()) {
        let rows = tapped(engine, &chunks[c]);
        grams.accumulate_tap(&rows).expect("accumulate");
        let t = rows.first().map_or(0, |st| st[TAP_M].len());
        for (i, &l) in layers.iter().enumerate() {
            for col in &rows[l][TAP_M] {
                for (u, v) in unq(col).iter().enumerate() {
                    dm[i][u] += v * v;
                }
            }
        }
        tokens += t;
        println!(
            "CALIB chunk {c} tokens {t} {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let done = grams.finish(Normalisation::Mean).expect("finish");
    drop(grams);
    println!(
        "CALIB calibration SyrkAcc {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(done.len());
    for (l, g) in done.into_iter().enumerate() {
        if let Some(i) = layers.iter().position(|&x| x == l) {
            let h = &g.down;
            let (num, den) = (0..n).fold((0.0f64, 0.0f64), |(a, b), u| {
                let m = dm[i][u] / tokens as f64;
                (a + (h[u * n + u] - m).powi(2), b + m * m)
            });
            let layout = (num / den).sqrt();
            assert!(
                layout < 1e-4,
                "layer {l}: Gram diagonal off the tap columns by {layout}"
            );
            println!("CALIB layer {l} Gram diag layout rel {layout:.3e}");
            out.push(g.down);
        } else {
            out.push(Vec::new());
        }
    }
    out
}

/// The input saliency of each `ffn_down` column (LLM.int8, Dettmers et al. 2022; `SpQR`, Dettmers et al.
/// 2023, arXiv 2306.03078): `s_i = H_ii · Σ_u D_ui²`, the metric weight on a column times the
/// calibration activation energy it multiplies. Needs no fit.
///
/// `O(o·n)` multiply-adds, one pass over `d`; space `O(n)`.
fn salience(h: &[f64], d: &[i64], o: usize, n: usize) -> Vec<f64> {
    let mut sq = vec![0.0f64; n];
    for u in 0..o {
        for i in 0..n {
            let v = from_grid(d[u * n + i]);
            sq[i] += v * v;
        }
    }
    (0..n).map(|i| h[i * n + i] * sq[i]).collect()
}

/// **Measurement 3** (board ). What keeping `K` teacher `ffn_down` columns per layer buys on the calibration
/// error energy `tr(E H Eᵀ) = Σ_i c_i`, against `K` random columns.
///
/// `APEIRON_DOWN_LAYERS` (default: every cached `down` fit), `APEIRON_DOWN_KEEP` (`K`, default 36,
/// the top 1% of `n_ff`), `APEIRON_DOWN_SELECT` (`energy` | `random`). `H` is the mean Gram of calibration
/// chunks `0..CALIB`, never a held-out chunk; no chunk `≥ CALIB` is read. Writes
/// `{CACHE}/l{l}.down-keep{K}-{select}.i64`, the fit with the selected columns taken from the
/// teacher, for [`down_arm_kl`].
///
/// Cost: the calibration SyrkAcc is `O(CALIB·tokens·(n_ff² + n_embd²))`; per layer two `energy_of` at
/// `O(n_embd·n_ff²)` each, one symmetric `n_ff × n_ff` Gram of `E`, the dominant term. Space
/// `O(n_layer·n_ff²)` for the SyrkAcc plus `O(n_embd·n_ff)` for one layer's words.
#[test]
#[ignore = "builds the 8-chunk calibration n_ff x n_ff SyrkAcc and two Gram energies per layer"]
fn down_keep_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = down_layers();
    let k = env_usize("APEIRON_DOWN_KEEP", 36);
    let select = std::env::var("APEIRON_DOWN_SELECT").unwrap_or_else(|_| String::from("energy"));
    let (o, n) = (cfg.n_embd, cfg.n_ff);
    let bits = k * o * 16 + k * 12;
    println!(
        "DOWNKEEP layers {layers:?} K {k} select {select} calib chunks 0..{CALIB} tokens/chunk \
         {n_ctx} n_embd {o} n_ff {n} (H = calibration chunks 0..{CALIB} mean Gram)"
    );
    let cal = calib_down(&engine, &cfg, &chunks, &layers);
    for &l in &layers {
        let t = Instant::now();
        let h: &[f64] = &cal[l];
        let (d, _mixer) = ffn_down(&engine, &cfg, l);
        let (ro, co, dhw) = read_words(&format!("{CACHE}/l{l}.down.i64"));
        assert_eq!((ro, co), (o, n), "layer {l}: cached down shape");
        let e: Vec<i64> = dhw.iter().zip(&d).map(|(a, b)| a - b).collect();
        let (before, _diag, c) = energy_of(&e, h, o, n);
        let keep = keep_indices(&c, k, l, &select);
        let mut kept = dhw.clone();
        for &i in &keep {
            for u in 0..o {
                kept[u * n + i] = d[u * n + i];
            }
        }
        let e2: Vec<i64> = kept.iter().zip(&d).map(|(a, b)| a - b).collect();
        let after = energy_of(&e2, h, o, n).0;
        let removed = if before > 0.0 {
            (before - after) / before
        } else {
            0.0
        };
        // Closed form of the drop: zeroing error column `k` also removes the two cross blocks, so the
        // energy removed by a set `S` is `2Σ_{k∈S} c_k − Σ_{i,j∈S} (EᵀE)_ij H_ji`, not `Σ c_k`. A
        // positive `c_k` does not imply a drop.
        let sum_c: f64 = keep.iter().map(|&i| c[i]).sum();
        let mut within = 0.0f64;
        for q in 0..o {
            let xs: Vec<f64> = keep.iter().map(|&i| from_grid(e[q * n + i])).collect();
            for (a, &i) in keep.iter().enumerate() {
                for (b, &j) in keep.iter().enumerate() {
                    within += xs[a] * xs[b] * h[j * n + i];
                }
            }
        }
        let mut drop_max = f64::NEG_INFINITY;
        let mut drop_arg = 0usize;
        for i in 0..n {
            let mut s = 0.0f64;
            for u in 0..o {
                let v = from_grid(e[u * n + i]);
                s += v * v;
            }
            let d = 2.0 * c[i] - s * h[i * n + i];
            if d > drop_max {
                drop_max = d;
                drop_arg = i;
            }
        }
        let drop_pred = 2.0 * sum_c - within;
        let bperw = bits as f64 / (o * n) as f64;
        println!(
            "KEEP layer {l:02} select {select} K {k} before {before:.6e} after {after:.6e} \
             removed {removed:.4} bits +{bits} (+{bperw:.4} b/w on this tensor) wall {:.1} s",
            t.elapsed().as_secs_f64()
        );
        println!(
            "KEEPIDENT layer {l:02} sum_c {sum_c:.6e} c_top1 {:.4} within {within:.6e} \
             drop_pred {drop_pred:.6e} drop_meas {:.6e} rel {:.2e} drop_max {drop_max:.6e} \
             at {drop_arg}",
            top_share(&c, 0.01),
            before - after,
            (after - (before - drop_pred)) / before
        );
        let path = format!("{CACHE}/l{l}.down-keep{k}-{select}.i64");
        write_words(&path, o, n, &kept).expect("write kept fit");
        println!("KEEP wrote {path}");
    }
}

/// **Measurement 5** (board ). The `SpQR` form of the `ffn_down` fit
/// (Dettmers et al. 2023, arXiv 2306.03078): `K` teacher columns `S` are restored exactly after a fit
/// of the other columns that already knows they will be, so the fit minimises its error on the masked
/// metric `Hm` — `H` with `S`'s rows and columns zeroed and `ε` on `S`'s diagonal — against the target
/// `D' = zeroCols(S, D)`.
///
/// The identity (`QuantVerdicts.ColumnRestore.masked_energy`) is
/// `tr(E Hm Eᵀ) = J(zeroCols_S E) + ε Σ_S ‖e_i‖²`, so the masked fit minimises the restored energy
/// plus an `ε` penalty on `S` alone; `Hm`'s off-`S` blocks are `H`'s, so the fit spends no error on
/// columns that will be overwritten. `ε = 1e-2 · mean(diag H)` keeps `Hm`'s `S`-diagonal above the
/// rounding grid.
///
/// `APEIRON_DOWN_LAYERS`, `APEIRON_DOWN_KEEP` (`K`, default 36), `APEIRON_DOWN_SELECT` (`energy` |
/// `salience` | `random`). `H` is the mean Gram of calibration chunks `0..CALIB`, never a held-out
/// chunk. Writes `{CACHE}/l{l}.down-spqr{K}-{select}.i64`, the fit with `S`'s columns taken from the
/// teacher, for [`down_arm_kl`] (`APEIRON_DOWN_KEY=down-spqr{K}-{select}`).
///
/// A `random` mask whose masked fit refuses `NotPositiveDefinite` is redrawn (at most twice, seed
/// `0x9E3779B97F4A7C15 ^ (l + 1024·attempt)`), with each refusal printed: the control's subject is a
/// random `K`-set, not one particular draw. `energy` and `salience` are never redrawn.
///
/// Cost: `fold_fit` dominates at ~52 s per layer measured on this box (the count, not the wall, is
/// the claim: `O(o·n² + gptq + inner_iters·(o·n² + n³ + e8_icm))`); three `energy_of` calls at
/// `O(n_embd·n_ff²)` each — the `before`/`c` one, the masked `obj` one and the restored `after` one —
/// plus `O(n²)` to mask `Hm` and `O(o·n)` to zero `D'` and restore `S`. Space `O(n² + o·n)` per
/// layer.
#[test]
#[ignore = "builds the 8-chunk calibration SyrkAcc and runs production fold_fit on the masked metric"]
fn down_spqr_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = down_layers();
    let k = env_usize("APEIRON_DOWN_KEEP", 36);
    let select = std::env::var("APEIRON_DOWN_SELECT").unwrap_or_else(|_| String::from("energy"));
    let (o, n) = (cfg.n_embd, cfg.n_ff);
    let bits = k * o * 16 + k * 12;
    println!(
        "SPQR layers {layers:?} K {k} select {select} calib chunks 0..{CALIB} tokens/chunk {n_ctx} \
         n_embd {o} n_ff {n} eps 1e-2*mean(diag H)"
    );
    let t0 = Instant::now();
    let cal = calib_down(&engine, &cfg, &chunks, &layers);
    let (mut sum_before, mut sum_after) = (0.0f64, 0.0f64);
    for &l in &layers {
        let h: &[f64] = &cal[l];
        let (d, _mixer) = ffn_down(&engine, &cfg, l);
        let (ro, co, dhw) = read_words(&format!("{CACHE}/l{l}.down.i64"));
        assert_eq!((ro, co), (o, n), "layer {l}: cached down shape");
        let e: Vec<i64> = dhw.iter().zip(&d).map(|(a, b)| a - b).collect();
        let (before, _diag, c) = energy_of(&e, h, o, n);
        let score = if select == "salience" {
            salience(h, &d, o, n)
        } else {
            c
        };
        let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
        let eps = 1e-2 * mean_diag;
        let mut keep = keep_indices(&score, k, l, &select);
        let mut attempt = 0usize;
        let (fit, hm, dp, wall) = loop {
            let mut mask = vec![false; n];
            for &i in &keep {
                mask[i] = true;
            }
            let mut hm = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    hm[i * n + j] = if mask[i] || mask[j] {
                        if i == j {
                            eps
                        } else {
                            0.0
                        }
                    } else {
                        h[i * n + j]
                    };
                }
            }
            let mut dp = d.clone();
            for &i in &keep {
                for u in 0..o {
                    dp[u * n + i] = 0;
                }
            }
            let t = Instant::now();
            match fold_fit(&dp, &qv(&hm), o, n, None, &production_cfg()) {
                Ok(fit) => break (fit, hm, dp, t.elapsed().as_secs_f64()),
                Err(err) => {
                    // A random mask can leave the masked metric too near singular for the reduced
                    // normal systems; the control's purpose is a random `K`-set, not this one, so
                    // it is redrawn with `l + 1024·attempt` as the seed argument.
                    println!(
                        "SPQR layer {l:02} select {select} K {k} attempt {attempt} refused \
                         {err:?}"
                    );
                    if select != "random" || attempt >= 2 {
                        println!("SPQR layer {l:02} select {select} K {k} fit error {err:?}");
                        return;
                    }
                    attempt += 1;
                    keep = keep_indices(&score, k, l + 1024 * attempt, &select);
                }
            }
        };
        let mut out = fit.wp.clone();
        for &i in &keep {
            for u in 0..o {
                out[u * n + i] = d[u * n + i];
            }
        }
        let e2: Vec<i64> = out.iter().zip(&d).map(|(a, b)| a - b).collect();
        let after = energy_of(&e2, h, o, n).0;
        let ego: Vec<i64> = fit.wp.iter().zip(&dp).map(|(a, b)| a - b).collect();
        let (obj, _od, _oc) = energy_of(&ego, &hm, o, n);
        let mut pen = 0.0f64;
        for &i in &keep {
            for u in 0..o {
                let v = from_grid(ego[u * n + i]);
                pen += v * v;
            }
        }
        let rhs = after + eps * pen;
        let rel = (obj - rhs).abs() / rhs.abs().max(f64::MIN_POSITIVE);
        let ratio = if before > 0.0 { after / before } else { 0.0 };
        let bperw = bits as f64 / (o * n) as f64;
        println!(
            "SPQR layer {l:02} select {select} K {k} before {before:.6e} after {after:.6e} ratio \
             {ratio:.4} ident rel {rel:.2e} fit {wall:.1} s bits +{bits} (+{bperw:.4} b/w)"
        );
        let path = format!("{CACHE}/l{l}.down-spqr{k}-{select}.i64");
        write_words(&path, o, n, &out).expect("write spqr fit");
        println!(
            "SPQRIDENT layer {l:02} obj {obj:.6e} after {after:.6e} eps {eps:.6e} pen {pen:.6e} \
             max|wp| on S {:.3e}",
            keep.iter().fold(0.0f64, |m, &i| (0..o)
                .map(|u| from_grid(fit.wp[u * n + i]).abs())
                .fold(m, f64::max))
        );
        println!("SPQR wrote {path}");
        sum_before += before;
        sum_after += after;
    }
    println!(
        "SPQR sum layers {} select {select} K {k} before {sum_before:.6e} after {sum_after:.6e} \
         ratio {:.4} wall {:.1} s",
        layers.len(),
        sum_after / sum_before.max(f64::MIN_POSITIVE),
        t0.elapsed().as_secs_f64()
    );
}

/// **Measurement 4.** Held-out KL of one image whose `blk.<l>.ffn_down.weight` is the cached `KEY`
/// fit in every layer that has one. `APEIRON_DOWN_KEY` (default `down`), `APEIRON_DOWN_SKIP`
/// (default [`CALIB`]), `APEIRON_DOWN_CHUNKS` (default 1).
///
/// Cost: `O(file bytes)` to read and replace, plus `O(chunks·(prefill + tokens·head))`; one 512-token
/// prefill dominates at ~175 s per chunk. Space the image once.
#[test]
#[ignore = "builds a 1.6 GB image and scores held-out KL chunks (~175 s each)"]
fn down_arm_kl() {
    let key = std::env::var("APEIRON_DOWN_KEY").unwrap_or_else(|_| String::from("down"));
    let skip = env_usize("APEIRON_DOWN_SKIP", CALIB);
    let count = env_usize("APEIRON_DOWN_CHUNKS", 1);
    let layers = fitted_layers(&key);
    assert_eq!(
        layers,
        fitted_layers("down"),
        "the {key} arm must cover every fitted down layer"
    );
    let t0 = Instant::now();
    let data = student_bytes(&key, &layers);
    let engine = Engine::from_bytes(&format!("down-{key}"), data).expect("student engine");
    println!(
        "DOWNARM key {key} layers {layers:?} chunks {skip}..{} image ready at {:.1} s",
        skip + count,
        t0.elapsed().as_secs_f64()
    );
    let (kl, top1, _rows) = held_out_kl(&engine, &base_path(), skip, count);
    println!(
        "DOWNARM key {key} kl {kl:.6} top1 {top1:.6} wall {:.1} s",
        t0.elapsed().as_secs_f64()
    );
}
