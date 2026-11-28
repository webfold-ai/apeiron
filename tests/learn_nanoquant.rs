//! Board . `apeiron::learn::nanoquant`: the `NanoQuant` ADMM factor
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `core/admm_nq.py`).
//!
//! Positive tests on small Gaussian matrices (rank formula against the bit budget, the export's
//! dense form, the wide-matrix transpose path, loss falling with the iteration count), and one
//! ignored stage that fits the teacher's linears of `Qwen3.5-0.8B` under the cached calibration
//! SyrkAcc and writes `l{l}.{kind}-nq{tag}.i64` for `learn_gate_code`'s `gate_code_kl`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "test geometry: single letters are the reference's symbols, loops are index maps; \
              `nq_fit` is one campaign (load, per-seed fit and tune, report)"
)]

use std::fmt::Write as _;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::format::bpe::Bpe;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::{Config, Weights};
use apeiron::learn::nanoquant::{
    factorize, factorize_floor, h_sketch, rank_for_bits, stored_bits, tune, tune_sketch, NqConfig,
    NqFactor, Schedule, TuneConfig,
};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: `Qwen3.5-0.8B` binary16. Overridable by `APEIRON_NQ_MODEL`, so a second
/// model (`Qwen3-0.6B-Base`, 28 layers) fits against its own SyrkAcc without editing this default.
const MODEL_DEFAULT: &str = "~/kld08/qwen08b-f16.gguf";
/// Where the calibration SyrkAcc, the factors and the fitted words are cached. Overridable by
/// `APEIRON_NQ_CACHE`; the producer is `learn_gate_code`'s `syrks_cached`, whose directory is its
/// own `CACHE` const, so a cache for a second model is those `g{l}.{tag}.*` files copied under
/// the tag it wrote.
const CACHE_DEFAULT: &str = "~/kld08/gatecode-tok";

/// The teacher image path, `APEIRON_NQ_MODEL` or [`MODEL_DEFAULT`].
fn model_path() -> String {
    std::env::var("APEIRON_NQ_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}

/// The cache directory, `APEIRON_NQ_CACHE` or [`CACHE_DEFAULT`].
fn cache_dir() -> String {
    std::env::var("APEIRON_NQ_CACHE").unwrap_or_else(|_| String::from(CACHE_DEFAULT))
}

/// The factor tag: `APEIRON_NQ_TAG` if set, else `-{model stem}` (`-q3-0p6b-f16`), so no run
/// overwrites another model's `l{l}.{kind}-nq{tag}.{nqf,i64}`.
fn factor_tag() -> String {
    if let Ok(t) = std::env::var("APEIRON_NQ_TAG") {
        return t;
    }
    let path = model_path();
    let stem = path.rsplit('/').next().unwrap_or(&path);
    let stem = stem.strip_suffix(".gguf").unwrap_or(stem);
    let stem: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("-{stem}")
}

/// The calibration Gram tag in the cache, `APEIRON_NQ_GRAM_TAG` (default `c6x29`): the
/// `calib`/`extra` pair `learn_gate_code`'s `syrks_cached` writes as `g{l}.c{calib}x{extra}.*.i64`.
/// Board : the model, cache and tag knobs let a second model be fitted.
fn syrk_tag() -> String {
    std::env::var("APEIRON_NQ_GRAM_TAG").unwrap_or_else(|_| String::from("c6x29"))
}

fn gauss(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / 9_007_199_254_740_992.0
    };
    (0..n)
        .map(|_| {
            let (a, b) = (next().max(1e-300), next());
            scale * (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect()
}

fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
}

/// The dense form by definition: `s_post[i]·Σ_k U[i][k]·V[k][j]·s_pre[j]`.
fn dense(f: &NqFactor) -> Vec<f64> {
    let (o, n, r) = (f.out, f.inn, f.rank);
    let mut w = vec![0.0; o * n];
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0i64;
            for k in 0..r {
                acc += i64::from(f.u[i * r + k]) * i64::from(f.v[k * n + j]);
            }
            w[i * n + j] = f.s_post[i] * acc as f64 * f.s_pre[j];
        }
    }
    w
}

#[test]
fn rank_fits_budget_and_is_maximal() {
    // Qwen3-0.6B projections and the 0.8B proxy's, at 1.00 and 1.18 bits per weight.
    for &(a, b) in &[
        (2048, 1024),
        (1024, 2048),
        (3072, 1024),
        (1024, 3072),
        (1024, 3584),
        (3584, 1024),
    ] {
        for &ppm in &[1_000_000u64, 1_180_000] {
            let r = rank_for_bits(a, b, ppm, 32);
            assert_eq!(r % 32, 0);
            let budget = (a * b) as u128 * u128::from(ppm);
            assert!(
                u128::from(stored_bits(a, b, r)) * 1_000_000 <= budget,
                "{a}x{b} r {r}"
            );
            assert!(
                u128::from(stored_bits(a, b, r + 32)) * 1_000_000 > budget,
                "{a}x{b} r {r}+32"
            );
        }
    }
    // A budget below the scale vectors falls back to the minimum rank.
    assert_eq!(rank_for_bits(64, 64, 100_000, 32), 32);
}

#[test]
fn export_matches_its_dense_form_and_loss_falls() {
    let (o, n, r) = (96, 64, 32);
    let w = gauss(o * n, 1, 0.05);
    let i_norm: Vec<f64> = gauss(n, 2, 1.0).iter().map(|x| 0.5 + x * x).collect();
    let o_norm = vec![1.0; o];
    let ww = words(&w);
    let mut cfg = NqConfig::reference(r);
    cfg.outer_iters = 2;
    let short = factorize(&ww, o, n, &i_norm, &o_norm, &cfg).expect("2 iterations");
    cfg.outer_iters = 60;
    let long = factorize(&ww, o, n, &i_norm, &o_norm, &cfg).expect("60 iterations");
    assert!(
        long.admm_loss < short.admm_loss,
        "{} !< {}",
        long.admm_loss,
        short.admm_loss
    );
    assert!(long.admm_loss < 0.9, "loss {}", long.admm_loss);
    assert!(long.u.iter().chain(&long.v).all(|&s| s == 1 || s == -1));
    let kernel = long.weight().expect("weight");
    let defn = dense(&long);
    let max_w = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (k, d) in kernel.iter().zip(&defn) {
        assert!(
            (from_grid(*k) - d).abs() <= 1e-6 * max_w,
            "{} vs {d}",
            from_grid(*k)
        );
    }
}

#[test]
fn wide_matrix_runs_on_the_transpose() {
    let (o, n, r) = (48, 80, 32);
    let w = words(&gauss(o * n, 3, 0.05));
    let cfg = NqConfig {
        outer_iters: 20,
        ..NqConfig::reference(r)
    };
    let f = factorize(&w, o, n, &vec![1.0; n], &vec![1.0; o], &cfg).expect("wide");
    assert_eq!((f.out, f.inn, f.u.len(), f.v.len()), (o, n, o * r, r * n));
    assert_eq!((f.s_post.len(), f.s_pre.len()), (o, n));
    assert_eq!(f.weight().expect("weight").len(), o * n);
}

/// `H = XᵀX/t` of `t` Gaussian rows with a few loud columns, as Q32.32 words.
fn gram(n: usize, t: usize, seed: u64) -> Vec<i64> {
    let mut x = gauss(t * n, seed, 1.0);
    for row in x.chunks_mut(n) {
        for j in (0..n).step_by(9) {
            row[j] *= 6.0;
        }
    }
    let mut h = vec![0.0; n * n];
    for row in x.chunks(n) {
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] += row[i] * row[j] / t as f64;
            }
        }
    }
    words(&h)
}

#[test]
fn latents_carry_the_signs_and_tune_lowers_the_h_energy() {
    let (o, n, r) = (96, 64, 32);
    let w = words(&gauss(o * n, 4, 0.05));
    let h = gram(n, 256, 5);
    let i_norm: Vec<f64> = (0..n).map(|i| from_grid(h[i * n + i])).collect();
    let cfg = NqConfig {
        outer_iters: 40,
        ..NqConfig::reference(r)
    };
    let mut f = factorize(&w, o, n, &i_norm, &vec![1.0; o], &cfg).expect("factorize");
    let agree =
        f.u.iter()
            .zip(&f.u_latent)
            .chain(f.v.iter().zip(&f.v_latent))
            .filter(|(&s, &l)| (l < 0.0) == (s < 0))
            .count();
    assert!(agree * 100 >= 99 * (o * r + r * n), "agree {agree}");
    let before = f.weight().expect("weight");
    // The latents at the reference rate, the scales faster: on this matrix the scales alone
    // converge to 0.987x of the ADMM energy; latent steps of 1e-3 flip 135 signs and raise it.
    let tc = TuneConfig {
        lr_scale: 1e-3,
        ..TuneConfig::reference(60)
    };
    let tr = tune(&mut f, &w, &h, &tc).expect("tune");
    assert!(
        tr.rel_after < 0.99 * tr.rel_before,
        "{} !< 0.99 x {}",
        tr.rel_after,
        tr.rel_before
    );
    assert!(f.u.iter().chain(&f.v).all(|&s| s == 1 || s == -1));
    assert_ne!(before, f.weight().expect("weight"));
}

/// A coordinate dead in the calibration Gram: `diag = 0` exactly, as `Qwen3-0.6B-Base` layer 0
/// `gate_up` has at index 1 (its neighbours 2.2, 2.0, 1.7). At `EPS = 1e-12` the scaled column is
/// `1e-12` of the others and the limb kernel refuses it (`LimbError::Range`); the relative floor
/// keeps the column inside the block-exponent range. The coordinate is dead, so it carries no
/// weight in `tr(EHEᵀ)`.
#[test]
fn dead_syrk_coordinate_does_not_refuse() {
    let (o, n, r) = (96, 64, 32);
    let w = words(&gauss(o * n, 7, 0.05));
    let h = gram(n, 256, 5);
    let mut i_norm: Vec<f64> = (0..n).map(|i| from_grid(h[i * n + i])).collect();
    i_norm[1] = 0.0;
    let cfg = NqConfig {
        outer_iters: 40,
        ..NqConfig::reference(r)
    };
    let f = factorize(&w, o, n, &i_norm, &vec![1.0; o], &cfg).expect("a dead coordinate fits");
    let wq = f.weight().expect("weight");
    let hf: Vec<f64> = h.iter().map(|&x| from_grid(x)).collect();
    let wf: Vec<f64> = w.iter().map(|&x| from_grid(x)).collect();
    let e: Vec<f64> = wq
        .iter()
        .zip(&w)
        .map(|(&q, &x)| from_grid(q) - from_grid(x))
        .collect();
    let rel = energy(&hf, &e, o, n) / energy(&hf, &wf, o, n);
    assert!(
        f.admm_loss.is_finite() && rel.is_finite() && rel < 1.0,
        "admm_loss {} relH {rel}",
        f.admm_loss
    );
}

/// At `s = n` the sketch spans the input space, `Q·H_s·Qᵀ = H`, and [`tune_sketch`] is [`tune`] in
/// a rotated frame: the sketched energy equals `tr(EHEᵀ)` and both tunes lower it alike.
#[test]
fn full_width_sketch_tune_matches_the_full_tune() {
    let (o, n, r) = (96, 64, 32);
    let w = words(&gauss(o * n, 4, 0.05));
    let h = gram(n, 256, 5);
    let i_norm: Vec<f64> = (0..n).map(|i| from_grid(h[i * n + i])).collect();
    let cfg = NqConfig {
        outer_iters: 40,
        ..NqConfig::reference(r)
    };
    let f0 = factorize(&w, o, n, &i_norm, &vec![1.0; o], &cfg).expect("factorize");
    let sk = h_sketch(&h, n, n, 3, 0).expect("sketch");
    let e: Vec<f64> = f0
        .weight()
        .expect("weight")
        .iter()
        .zip(&w)
        .map(|(&a, &b)| from_grid(a) - from_grid(b))
        .collect();
    let full: f64 = (0..o)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..n)
                        .map(|k| e[i * n + j] * from_grid(h[j * n + k]) * e[i * n + k])
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum();
    let sketched = sk.energy(&e, o).expect("energy");
    // 3-limb products carry about `1e-4` relative (`orth`'s bound); a lost direction costs O(1).
    assert!(
        (sketched - full).abs() <= 1e-4 * full,
        "{sketched} vs {full}"
    );
    let tc = TuneConfig {
        lr_scale: 1e-3,
        ..TuneConfig::reference(60)
    };
    let (mut fa, mut fb) = (f0.clone(), f0);
    let ta = tune(&mut fa, &w, &h, &tc).expect("tune");
    let tb = tune_sketch(&mut fb, &w, &h, &sk, &tc).expect("tune_sketch");
    // The descent's size is a property of the ADMM start (60 scale-dominated steps saturate near 1%);
    // a tune that moves nothing fails here, and the next assertion ties it to the full tune.
    assert!(
        tb.rel_after < tb.rel_before && ta.rel_after < ta.rel_before,
        "sketch {} -> {}, full {} -> {}",
        tb.rel_before,
        tb.rel_after,
        ta.rel_before,
        ta.rel_after
    );
    assert!(
        (tb.rel_after - ta.rel_after).abs() <= 0.02 * ta.rel_after,
        "{} vs {}",
        tb.rel_after,
        ta.rel_after
    );
    assert!(fb.u.iter().chain(&fb.v).all(|&s| s == 1 || s == -1));
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Calibration-base builder: `APEIRON_NQ_TEXT` (natural text) tokenized by `APEIRON_NQ_TOKENIZER`
/// (`Qwen3-0.6B-Base`'s `tokenizer.json`, vocabulary 151936) and written to `APEIRON_NQ_BASE` as
/// llama.cpp's `_logits_` base file — signature, `n_ctx`, `n_vocab`, `n_chunk`, tokens, then the per
/// chunk row blocks of `src/learn/fold/stats/kldbase.rs`. `APEIRON_NQ_CHUNKS` (15) is the chunk
/// count and `n_ctx` is 512. The row blocks are a sparse zero tail: `learn_gate_code`'s Gram stage
/// reads the tokens and no row. Cost: `O(tokens)` encode over the first `n_ctx·n_chunk` ids, one
/// `O(n_ctx·n_chunk)` write and one `set_len`.
#[test]
#[ignore = "writes a calibration base file; run explicitly"]
fn nq_calib_base() {
    let get = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
    let (n_ctx, n_chunk) = (512usize, env_usize("APEIRON_NQ_CHUNKS", 15));
    let bpe = Bpe::from_tokenizer_json(&get("APEIRON_NQ_TOKENIZER")).expect("tokenizer");
    let text = std::fs::read_to_string(get("APEIRON_NQ_TEXT")).expect("text");
    let head: String = text.chars().take(24 * n_ctx * n_chunk).collect();
    let mut ids = bpe.encode(&head).expect("encode");
    assert!(
        ids.len() >= n_ctx * n_chunk,
        "{} tokens for {} chunks",
        ids.len(),
        n_chunk
    );
    ids.truncate(n_ctx * n_chunk);
    let n_vocab = bpe.vocab_size();
    let mut out = Vec::with_capacity(20 + 4 * ids.len());
    out.extend_from_slice(b"_logits_");
    for v in [n_ctx, n_vocab, n_chunk] {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for id in &ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    let path = get("APEIRON_NQ_BASE");
    std::fs::write(&path, &out).expect("write base");
    let rows = n_ctx - 1 - n_ctx / 2;
    let nv = 2 * n_vocab.div_ceil(2) + 4;
    let total = out.len() + rows * nv * 2 * n_chunk;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open base")
        .set_len(total as u64)
        .expect("sparse rows");
    println!(
        "BASE {path} n_ctx {n_ctx} n_vocab {n_vocab} n_chunk {n_chunk} tokens {} bytes {total} max_id {}",
        ids.len(),
        ids.iter().max().expect("a token")
    );
}

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// The `tr(EHEᵀ)/tr(WHWᵀ)` screen runs unless `APEIRON_NQ_RELH=0`; the `.nqf` and `.i64` writes
/// never depend on it. Board : the screen is two dense `n×n` passes per
/// tensor, a build of the student needs neither.
fn relh_enabled() -> bool {
    !matches!(std::env::var("APEIRON_NQ_RELH").as_deref(), Ok("0"))
}

/// `out, inn, rank` (u64), `admm_loss`, `s_post`, `s_pre`, `u_latent`, `v_latent` (f64), `u`, `v`
/// (i8), little-endian.
fn save_factor(path: &str, f: &NqFactor) {
    let mut b = Vec::new();
    for x in [f.out, f.inn, f.rank] {
        b.extend_from_slice(&(x as u64).to_le_bytes());
    }
    b.extend_from_slice(&f.admm_loss.to_le_bytes());
    for v in [&f.s_post, &f.s_pre, &f.u_latent, &f.v_latent] {
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b.extend(f.u.iter().chain(&f.v).map(|&s| s.cast_unsigned()));
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn load_factor(path: &str) -> NqFactor {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().expect("word"));
    let (o, n, r) = (word(0) as usize, word(1) as usize, word(2) as usize);
    let mut at = 4;
    let mut take = |len: usize| -> Vec<f64> {
        let v = (at..at + len).map(|i| f64::from_bits(word(i))).collect();
        at += len;
        v
    };
    let (s_post, s_pre, u_latent, v_latent) = (take(o), take(n), take(o * r), take(r * n));
    let signs = &b[8 * (4 + o + n + o * r + r * n)..];
    assert_eq!(signs.len(), o * r + r * n, "{path}: sign bytes");
    NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: signs[..o * r].iter().map(|&s| s.cast_signed()).collect(),
        v: signs[o * r..].iter().map(|&s| s.cast_signed()).collect(),
        s_pre,
        s_post,
        admm_loss: f64::from_bits(word(3)),
        u_latent,
        v_latent,
    }
}

/// The `block_loop` slot names in file order (`tests/learn_block_tune.rs:95-103` `SLOT_KEYS`).
const LOOP_KEYS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// One `block_loop` slot as `Q32.32` words, `out × inn` row-major, read from the `.param` file
/// `save_param` writes (`tests/learn_block_tune.rs:686-710`: `[out, inn, rank, kind]` as `u64`,
/// then a dense `w` for `kind` 0, or a factor's `u` (`out × r`), `vt` (`inn × r`), `s_post`,
/// `s_pre` for `kind` 1). A factor is `W = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)` with
/// `sign(0) = +1` (`modules/linear.py:44-56`, `:120-123`).
///
/// # Complexity
/// Time `O(out·inn·⌈r/64⌉)` word operations: each sign row packs into `⌈r/64⌉` `u64` words and an
/// entry's sign dot is `r − 2·popcount(a ⊕ b)`. Space `O(out·inn)` words for the result plus
/// `O((out + inn)·⌈r/64⌉)` for the packed signs.
fn loop_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word =
        |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().expect("word")) as usize;
    let (out, inn, r, kind) = (word(0), word(1), word(2), word(3));
    let f = |at: usize, n: usize| -> Vec<f32> {
        b[32 + 4 * at..32 + 4 * (at + n)]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let q32 = |v: f64| (v * 4_294_967_296.0).round() as i64;
    if kind == 0 {
        assert_eq!(b.len(), 32 + 4 * out * inn, "{path}: dense bytes");
        return (
            out,
            inn,
            f(0, out * inn).iter().map(|&v| q32(f64::from(v))).collect(),
        );
    }
    assert_eq!(kind, 1, "{path}: kind {kind}");
    assert_eq!(
        b.len(),
        32 + 4 * ((out + inn) * r + out + inn),
        "{path}: factor bytes"
    );
    let (u, vt) = (f(0, out * r), f(out * r, inn * r));
    let (s_post, s_pre) = (f((out + inn) * r, out), f((out + inn) * r + out, inn));
    let nw = r.div_ceil(64);
    let pack = |m: &[f32], rows: usize| -> Vec<u64> {
        let mut p = vec![0u64; rows * nw];
        for i in 0..rows {
            for k in 0..r {
                if m[i * r + k] < 0.0 {
                    p[i * nw + k / 64] |= 1 << (k % 64);
                }
            }
        }
        p
    };
    let (pu, pv) = (pack(&u, out), pack(&vt, inn));
    let mut w = vec![0i64; out * inn];
    let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let rows = out.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in w.chunks_mut(rows * inn).enumerate() {
            let (pu, pv, s_post, s_pre) = (&pu, &pv, &s_post, &s_pre);
            s.spawn(move || {
                for (di, row) in chunk.chunks_mut(inn).enumerate() {
                    let i = t * rows + di;
                    let a = &pu[i * nw..(i + 1) * nw];
                    for (c, x) in row.iter_mut().enumerate() {
                        let bv = &pv[c * nw..(c + 1) * nw];
                        let neg: u32 = a.iter().zip(bv).map(|(p, q)| (p ^ q).count_ones()).sum();
                        let dot = r as f64 - 2.0 * f64::from(neg);
                        *x = q32(f64::from(s_post[i]) * dot * f64::from(s_pre[c]));
                    }
                }
            });
        }
    });
    (out, inn, w)
}

/// `loop_words` against the factor's definition summed entry by entry, at a rank (70) that
/// spans two packed words, with exact zeros in the latents (`sign(0) = +1`); and a dense file
/// read back word for word.
#[test]
fn loop_words_matches_the_factor() {
    let (o, n, r) = (5usize, 7usize, 70usize);
    let mut u: Vec<f32> = gauss(o * r, 1, 1.0).iter().map(|&v| v as f32).collect();
    let vt: Vec<f32> = gauss(n * r, 2, 1.0).iter().map(|&v| v as f32).collect();
    u[3] = 0.0;
    let s_post: Vec<f32> = gauss(o, 3, 0.1).iter().map(|&v| v as f32).collect();
    let s_pre: Vec<f32> = gauss(n, 4, 0.1).iter().map(|&v| v as f32).collect();
    let dir = std::env::temp_dir().join(format!("loop-words-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let write = |name: &str, h: [u64; 4], vals: &[&[f32]]| {
        let mut b: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
        for v in vals {
            b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        let p = dir.join(name).to_string_lossy().into_owned();
        std::fs::write(&p, b).expect("write");
        p
    };
    let p = write(
        "f.param",
        [o as u64, n as u64, r as u64, 1],
        &[&u, &vt, &s_post, &s_pre],
    );
    let (ro, cn, w) = loop_words(&p);
    assert_eq!((ro, cn), (o, n));
    let sg = |v: f32| if v < 0.0 { -1.0 } else { 1.0 };
    for i in 0..o {
        for c in 0..n {
            let dot: f64 = (0..r).map(|k| sg(u[i * r + k]) * sg(vt[c * r + k])).sum();
            let want = f64::from(s_post[i]) * dot * f64::from(s_pre[c]);
            assert_eq!(
                w[i * n + c],
                (want * 4_294_967_296.0).round() as i64,
                "({i}, {c})"
            );
        }
    }
    let d: Vec<f32> = gauss(o * n, 5, 1.0).iter().map(|&v| v as f32).collect();
    let (_, _, wd) = loop_words(&write("d.param", [o as u64, n as u64, 0, 0], &[&d]));
    let want: Vec<i64> = d
        .iter()
        .map(|&v| (f64::from(v) * 4_294_967_296.0).round() as i64)
        .collect();
    assert_eq!(wd, want);
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("o")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("n")) as usize;
    let w = b[16..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect();
    (o, n, w)
}

fn write_words(path: &str, o: usize, n: usize, w: &[i64]) {
    let mut b = Vec::with_capacity(16 + 8 * w.len());
    b.extend_from_slice(&(o as u64).to_le_bytes());
    b.extend_from_slice(&(n as u64).to_le_bytes());
    for x in w {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// `tr(E H Eᵀ)` over rows, threaded: `O(o·n²)` multiply-adds.
fn energy(h: &[f64], e: &[f64], o: usize, n: usize) -> f64 {
    let th = 32usize;
    let per = o.div_ceil(th);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..th)
            .map(|t| {
                s.spawn(move || {
                    let mut acc = 0.0;
                    for u in (t * per)..((t + 1) * per).min(o) {
                        let row = &e[u * n..(u + 1) * n];
                        for i in 0..n {
                            let hi = &h[i * n..(i + 1) * n];
                            let mut d = 0.0;
                            for j in 0..n {
                                d += hi[j] * row[j];
                            }
                            acc += row[i] * d;
                        }
                    }
                    acc
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("thread")).sum()
    })
}

/// ` | {cmp} relH … nq/{cmp} x…` for each cached `{prefix}-{cmp}.i64` of `APEIRON_NQ_CMP`
/// (default `q4km`).
fn cmp_line(
    prefix: &str,
    h: &[f64],
    w: &[f64],
    (o, n): (usize, usize),
    base: f64,
    rel: f64,
) -> String {
    let mut line = String::new();
    for cmp in std::env::var("APEIRON_NQ_CMP")
        .unwrap_or_else(|_| String::from("q4km"))
        .split(',')
    {
        let p = format!("{prefix}-{cmp}.i64");
        if !std::path::Path::new(&p).exists() {
            continue;
        }
        let (_, _, qw) = read_words(&p);
        let e: Vec<f64> = qw.iter().zip(w).map(|(&q, x)| from_grid(q) - x).collect();
        let rc = energy(h, &e, o, n) / base;
        let _ = write!(line, " | {cmp} relH {rc:.6} nq/{cmp} x{:.2}", rel / rc);
    }
    line
}

/// Fit stage. `APEIRON_NQ_KIND` (comma list of `gate`, `up`, `down`, `o`), `APEIRON_NQ_LAYERS`
/// (default `0,3,7,11,15,19,23`, the cached Gram layers), `APEIRON_NQ_ITERS` (400),
/// `APEIRON_NQ_BITS_PPM` (1000000), `APEIRON_NQ_SCHEDULE` (`linear`, or `cubic`), `APEIRON_NQ_TAG`.
/// `APEIRON_NQ_GRAM_TAG` (default `c6x29`) names the cached Gram `g{l}.{tag}.{gate_up,down,o}.i64`;
/// `APEIRON_NQ_MODEL` the teacher and `APEIRON_NQ_CACHE` the directory both are read from. The
/// factor is saved to `l{l}.{kind}-nq{tag}.nqf`; `APEIRON_NQ_FROM=<tag>` loads that factor in
/// place of the ADMM. `APEIRON_NQ_TUNE_STEPS` (0) runs [`tune`] at `APEIRON_NQ_TUNE_LR` and
/// `APEIRON_NQ_TUNE_LR_SCALE` (both `1e-5`). `APEIRON_NQ_FLOOR` sets the preconditioner floor of
/// [`factorize_floor`] (unset: [`factorize`]'s own). `APEIRON_NQ_RELH=0` drops the two dense
/// `energy` passes and the `Q4_K_M` comparison from the printed line (and from the run's cost);
/// `i_norm = diag(H)` of the cached `g{l}.c6x29.*` Gram (the reference's unclipped `dbf` statistic up to a constant, to which the
/// ADMM is equivariant); `o_norm = 1`. Prints the ADMM loss and the relative `H`-energy
/// `tr(EHEᵀ)/tr(WHWᵀ)` of the fit and of the cached same-scope `Q4_K_M` part.
#[test]
#[ignore = "opens the teacher and runs the ADMM per tensor; run explicitly"]
fn nq_fit() {
    let (model, cache, gtag) = (model_path(), cache_dir(), syrk_tag());
    let engine = Engine::open(&model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    std::fs::create_dir_all(&cache).expect("cache dir");
    println!(
        "NQ model {model} cache {cache} gram {gtag} n_embd {} n_ff {} n_layer {} head {} kv {} head_dim {}",
        cfg.n_embd, cfg.n_ff, cfg.n_layer, cfg.n_head, cfg.n_head_kv, cfg.head_dim
    );
    let kinds = std::env::var("APEIRON_NQ_KIND").unwrap_or_else(|_| String::from("down"));
    let layers: Vec<usize> = std::env::var("APEIRON_NQ_LAYERS")
        .unwrap_or_else(|_| String::from("0,3,7,11,15,19,23"))
        .split(',')
        .map(|s| s.parse().expect("layer"))
        .collect();
    let iters = env_usize("APEIRON_NQ_ITERS", 400);
    let ppm = env_usize("APEIRON_NQ_BITS_PPM", 1_000_000) as u64;
    let seeds: Vec<u64> = std::env::var("APEIRON_NQ_SEEDS")
        .unwrap_or_else(|_| String::from("0"))
        .split(',')
        .map(|s| s.parse().expect("seed"))
        .collect();
    let tag = factor_tag();
    let from = std::env::var("APEIRON_NQ_FROM").ok();
    let schedule = match std::env::var("APEIRON_NQ_SCHEDULE").as_deref() {
        Ok("cubic") => Schedule::Cubic,
        Ok("linear") | Err(_) => Schedule::Linear,
        Ok(s) => panic!("schedule {s}"),
    };
    let tc = TuneConfig {
        lr_binary: env_f64("APEIRON_NQ_TUNE_LR", 1e-5),
        lr_scale: env_f64("APEIRON_NQ_TUNE_LR_SCALE", 1e-5),
        ..TuneConfig::reference(env_usize("APEIRON_NQ_TUNE_STEPS", 0))
    };
    for &l in &layers {
        let ll = layer_from_engine(&engine, &cfg, l).expect("layer linears");
        for kind in kinds.split(',') {
            let (names, gram): (&[&str], &str) = match kind {
                "gate" => (&["ffn_gate"], "gate_up"),
                "up" => (&["ffn_up"], "gate_up"),
                "down" => (&["ffn_down"], "down"),
                "o" => (&["attn_output", "ssm_out"], "o"),
                k => panic!("kind {k}"),
            };
            let lin = ll
                .linears
                .iter()
                .find(|x| names.contains(&x.name))
                .unwrap_or_else(|| panic!("blk.{l} {kind}"));
            let (o, n) = (lin.rows, lin.cols);
            let (ho, hn, hw) = read_words(&format!("{cache}/g{l}.{gtag}.{gram}.i64"));
            assert_eq!((ho, hn), (n, n), "layer {l} {kind}: Gram");
            let h: Vec<f64> = hw.iter().map(|&x| from_grid(x)).collect();
            let i_norm: Vec<f64> = (0..n).map(|i| h[i * n + i]).collect();
            let r = rank_for_bits(o, n, ppm, 32);
            for &seed in &seeds {
                let c = NqConfig {
                    outer_iters: iters,
                    schedule,
                    seed,
                    ..NqConfig::reference(r)
                };
                let t = Instant::now();
                let mut f = match &from {
                    Some(ft) => load_factor(&format!("{cache}/l{l}.{kind}-nq{ft}.nqf")),
                    None => match std::env::var("APEIRON_NQ_FLOOR") {
                        Ok(fl) => factorize_floor(
                            &lin.w,
                            o,
                            n,
                            &i_norm,
                            &vec![1.0; o],
                            &c,
                            fl.parse().expect("floor"),
                        )
                        .expect("factorize"),
                        Err(_) => {
                            factorize(&lin.w, o, n, &i_norm, &vec![1.0; o], &c).expect("factorize")
                        }
                    },
                };
                let fit_s = t.elapsed().as_secs_f64();
                let mut tune_line = String::new();
                if tc.steps > 0 {
                    let mut sk_s = 0.0;
                    let t = Instant::now();
                    let tr = match env_usize("APEIRON_NQ_TUNE_SKETCH", 0) {
                        0 => tune(&mut f, &lin.w, &hw, &tc).expect("tune"),
                        s => {
                            let sk = h_sketch(&hw, n, s, env_usize("APEIRON_NQ_PASSES", 3), 0)
                                .expect("sketch");
                            sk_s = t.elapsed().as_secs_f64();
                            tune_sketch(&mut f, &lin.w, &hw, &sk, &tc).expect("tune_sketch")
                        }
                    };
                    let _ = write!(
                    tune_line,
                    " sketch {sk_s:.2} s tune {} steps lr {:e}/{:e} {:.2} s relH {:.5}->{:.5} best {:.5}@{} flips {}",
                    tc.steps,
                    tc.lr_binary,
                    tc.lr_scale,
                    t.elapsed().as_secs_f64() - sk_s,
                    tr.rel_before,
                    tr.rel_after,
                    tr.rel_best.0,
                    tr.rel_best.1,
                    tr.flips
                );
                }
                // `wq` is written whatever the screen; the two dense `energy` passes and the
                // `Q4_K_M` comparison are the screen.
                save_factor(&format!("{cache}/l{l}.{kind}-nq{tag}.nqf"), &f);
                let r = f.rank;
                let wq = f.weight().expect("weight");
                let path = format!("{cache}/l{l}.{kind}-nq{tag}.i64");
                write_words(&path, o, n, &wq);
                let mut line = format!(
                "NQ layer {l:02} {kind} {o}x{n} seed {seed} r {r} b/w {:.4} iters {iters} {fit_s:.1} s admm_loss {:.5}{tune_line}",
                stored_bits(o, n, r) as f64 / (o * n) as f64,
                f.admm_loss
            );
                if relh_enabled() {
                    let w: Vec<f64> = lin.w.iter().map(|&x| from_grid(x)).collect();
                    let base = energy(&h, &w, o, n);
                    let e: Vec<f64> = wq.iter().zip(&w).map(|(&q, x)| from_grid(q) - x).collect();
                    let rel = energy(&h, &e, o, n) / base;
                    let _ = write!(line, " relH {rel:.5}");
                    line += &cmp_line(&format!("{cache}/l{l}.{kind}"), &h, &w, (o, n), base, rel);
                } else {
                    line += " relH n/a";
                }
                println!("{line}");
            }
        }
    }
}

/// Premise of the sketched tune: the share of `tr(H)`, `tr(WHWᵀ)` and `tr(EHEᵀ)` (each cached
/// `l{l}.down-{tag}.i64` of `APEIRON_NQ_ARMS`) carried by `PHP`, `P` the rank-`s` [`h_sketch`]
/// subspace, for each `s` of `APEIRON_NQ_SKETCH` (default `64,128,256,512`), layer
/// `APEIRON_NQ_LAYERS` (23), `down`.
#[test]
#[ignore = "opens the teacher and sketches the cached Gram; run explicitly"]
fn nq_sketch_share() {
    let (cache, gtag) = (cache_dir(), syrk_tag());
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let l = env_usize("APEIRON_NQ_LAYERS", 23);
    let ll = layer_from_engine(&engine, &cfg, l).expect("layer linears");
    let lin = ll
        .linears
        .iter()
        .find(|x| x.name == "ffn_down")
        .expect("down");
    let (o, n) = (lin.rows, lin.cols);
    let (_, _, hw) = read_words(&format!("{cache}/g{l}.{gtag}.down.i64"));
    let h: Vec<f64> = hw.iter().map(|&x| from_grid(x)).collect();
    let w: Vec<f64> = lin.w.iter().map(|&x| from_grid(x)).collect();
    let tr_h: f64 = (0..n).map(|i| h[i * n + i]).sum();
    let arms: Vec<(String, Vec<f64>, f64)> = std::env::var("APEIRON_NQ_ARMS")
        .unwrap_or_else(|_| String::from("nqw2a,nqw2b,q4km"))
        .split(',')
        .map(|t| {
            let (_, _, qw) = read_words(&format!("{cache}/l{l}.down-{t}.i64"));
            let e: Vec<f64> = qw.iter().zip(&w).map(|(&q, x)| from_grid(q) - x).collect();
            let full = energy(&h, &e, o, n);
            (t.to_string(), e, full)
        })
        .collect();
    let base = energy(&h, &w, o, n);
    for s in std::env::var("APEIRON_NQ_SKETCH")
        .unwrap_or_else(|_| String::from("64,128,256,512"))
        .split(',')
        .map(|x| x.parse::<usize>().expect("s"))
    {
        let t = Instant::now();
        let sk = h_sketch(&hw, n, s, env_usize("APEIRON_NQ_PASSES", 3), 0).expect("sketch");
        let sk_s = t.elapsed().as_secs_f64();
        let tr_s: f64 = (0..s).map(|i| sk.hs[i * s + i]).sum();
        let mut line = format!(
            "SKETCH l{l} down s {s} {sk_s:.1} s trH {:.4} W {:.4}",
            tr_s / tr_h,
            sk.energy(&w, o).expect("energy") / base
        );
        for (tag, e, full) in &arms {
            let _ = write!(
                line,
                " | {tag} E {:.4}",
                sk.energy(e, o).expect("energy") / full
            );
        }
        println!("{line}");
    }
}

/// The student image of the teacher: the F16 `GGUF` re-encoded from `Weights::from_engine`, with
/// the cache's fitted words spliced onto the four linears `NanoQuant` owns. Board
/// : this file is the first same-harness port row's model.
///
/// `APEIRON_NQ_OUT` (required) is the written path. `APEIRON_NQ_SPLICE=none` leaves every teacher
/// tensor in place, which must reproduce the teacher byte-for-byte; otherwise each layer `l` reads
/// `{cache}/l{l}.{kind}-nq{tag}.i64` for `down`, `o`, `up`, `gate` (`tag` = [`factor_tag`]), asserts
/// the shape `nq_fit` fitted against and fails naming a missing file. `Q`/`K`/`V`, the norms and
/// the embeddings stay the teacher's. `APEIRON_NQ_KINDS` (comma list of `down`, `o`, `up`, `gate`;
/// unset = all four) restricts the splice to the named kinds, skips the rest without reading their
/// files, and panics on a name outside that set. The kind restriction separates a splice defect
/// from compounding damage in .
///
/// # Complexity
/// Time and space `O(2·W)` for the `W`-weight teacher: one decode of `W` into `Q32.32` words, one
/// encode pass over the image and one whole-image buffer.
#[test]
#[ignore = "opens the teacher and re-encodes the whole model; run explicitly"]
fn nq_student() {
    const ALL: [&str; 4] = ["down", "o", "up", "gate"];
    let t = Instant::now();
    let (model, cache, tag) = (model_path(), cache_dir(), factor_tag());
    let out = std::env::var("APEIRON_NQ_OUT").expect("APEIRON_NQ_OUT");
    let splice = !matches!(std::env::var("APEIRON_NQ_SPLICE").as_deref(), Ok("none"));
    let wanted: Vec<String> = match std::env::var("APEIRON_NQ_KINDS") {
        Ok(list) => list
            .split(',')
            .map(|k| {
                let k = k.trim();
                assert!(ALL.contains(&k), "APEIRON_NQ_KINDS: unknown kind {k:?}");
                String::from(k)
            })
            .collect(),
        Err(_) => ALL.iter().map(|k| String::from(*k)).collect(),
    };
    let engine = Engine::open(&model).expect("open teacher");
    let (mut w, cfg) = Weights::from_engine(&engine).expect("student weights");
    let mut spliced = 0usize;
    if let Ok(dir) = std::env::var("APEIRON_NQ_LOOP") {
        // `block_loop`'s state: every layer `l` with `{dir}/l{l}/done` takes all seven slots from
        // its `.param` files; a layer without the marker stays the teacher, as in the reference's
        // per-block PPL (`utils/eval_utils.py:91-104`).
        for l in 0..cfg.n_layer {
            if !std::path::Path::new(&format!("{dir}/l{l}/done")).exists() {
                continue;
            }
            for key in LOOP_KEYS {
                let path = format!("{dir}/l{l}/{key}.param");
                let (ro, cn, words) = loop_words(&path);
                let lw = &mut w.layers[l];
                let slot = match key {
                    "self_attn.q_proj" => &mut lw.wq,
                    "self_attn.k_proj" => &mut lw.wk,
                    "self_attn.v_proj" => &mut lw.wv,
                    "self_attn.o_proj" => &mut lw.wo,
                    "mlp.gate_proj" => &mut lw.w_gate,
                    "mlp.up_proj" => &mut lw.w_up,
                    _ => &mut lw.w_down,
                };
                assert_eq!(ro * cn, slot.len(), "{path}: teacher {}", slot.len());
                *slot = words;
                spliced += 1;
            }
        }
    } else if splice {
        let kinds: [(&str, &[&str]); 4] = [
            ("down", &["ffn_down"]),
            ("o", &["attn_output", "ssm_out"]),
            ("up", &["ffn_up"]),
            ("gate", &["ffn_gate"]),
        ];
        for l in 0..cfg.n_layer {
            let ll = layer_from_engine(&engine, &cfg, l).expect("layer linears");
            for (kind, names) in kinds {
                if !wanted.iter().any(|k| k.as_str() == kind) {
                    continue;
                }
                let lin = ll
                    .linears
                    .iter()
                    .find(|x| names.contains(&x.name))
                    .unwrap_or_else(|| panic!("blk.{l} {kind}"));
                let path = format!("{cache}/l{l}.{kind}-nq{tag}.i64");
                let (ro, cn, words) = read_words(&path);
                assert_eq!((ro, cn), (lin.rows, lin.cols), "{path}: shape");
                let lw = &mut w.layers[l];
                let slot = match kind {
                    "down" => &mut lw.w_down,
                    "o" => &mut lw.wo,
                    "up" => &mut lw.w_up,
                    _ => &mut lw.w_gate,
                };
                assert_eq!(ro * cn, slot.len(), "{path}: teacher {}", slot.len());
                *slot = words;
                spliced += 1;
            }
        }
    }
    drop(engine);
    let bytes =
        apeiron::learn::fold::run::convert::student_gguf_bytes(&model, &w).expect("student image");
    let tmp = format!("{out}.tmp");
    std::fs::write(&tmp, &bytes).unwrap_or_else(|e| panic!("{tmp}: {e}"));
    std::fs::rename(&tmp, &out).unwrap_or_else(|e| panic!("{out}: {e}"));
    println!(
        "NQ student out {out} bytes {} spliced {spliced} wall {:.1}",
        bytes.len(),
        t.elapsed().as_secs_f64()
    );
}
