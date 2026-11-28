//! `tests/learn_fold_gen.rs` — board : the **per-tensor** generalisation
//! law of the one-shot fold on the crate's engine path.
//!
//! W17 split the held-out generalisation error `G = KL_out/KL_in` into `G = O·T·Q` and reported `Q`
//! (KL per unit `J`) as the largest factor: `J_out/J_in = 1.17` against `q2_out/q2_in = 2.98`. Its
//! stated next step, never run, is `J_out/J_in` **per tensor** against that tensor's own weight-error
//! propagation to the logits. This file runs it in Rust on the engine path.
//!
//! For one layer (0) and each of its seven linears, a *single-tensor* student is the teacher with
//! that one tensor replaced by its one-shot `fold_fit` result. For that student the file records
//!
//! | column | definition |
//! |---|---|
//! | `J_in`, `J_out` | `tr(E G_cal Eᵀ)`, `tr(E G_ho Eᵀ)` with `E = Ŵ − W` in `Q32.32` and `G` the mean activation Gram of the tensor's input |
//! | `KL_in`, `KL_out` | `held_out` KL of the student against the teacher on the calibration chunks and on four disjoint held-out chunks |
//! | `O_t`, `Q_t` | `J_out/J_in` and `(KL_out/KL_in)/O_t` |
//!
//! `E` is read back from the written student `GGUF` (`Weights::from_engine` on the reopened engine),
//! so it is the error the engine executes, including the `F16` re-encoding of the fitted words. No
//! `F16`-dequantised forward exists here: every KL runs through [`held_out`] on two opened engines.
//!
//! [`gen_fixture`] is the gate: on the one-layer synthetic `GGUF` of `tests/learn_fold_oneshot.rs` it
//! asserts `J_in > 0`, that `J_out` comes from the **disjoint** Gram (its value differs from the
//! calibration-Gram value), and that replacing a tensor by itself gives `KL 0` and `J 0`.
//! [`gen_qwen05b`] is the production measurement (ignored); `APEIRON_GEN_NCAL` selects the calibration
//! chunk count, `APEIRON_GEN_LW=1` the Ledoit–Wolf arm, `APEIRON_GEN_PROBE=1` the cost-equation probe.
//!
//! Grade: `tested` (the wiring and its control), `measured` (the production walls and the table).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::unreadable_literal,
    reason = "the fixture maps `f64` samples into `f32` payload words and the engine's `f64` eps \
              into its `Q32.32` word; the single-letter names are the geometry's symbols; the \
              fixture's power-of-two literals are binary16 fields"
)]

use std::fs;
use std::io::Write;
use std::time::Instant;

use apeiron::engine::{Engine, KernelMode};
use apeiron::format::quant;
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig};
use apeiron::learn::fold::layer::driver::DriverCfg;
use apeiron::learn::fold::run::npy::{chunks, read_npy_i4, split};
use apeiron::learn::fold::run::transfer::write_student_gguf;
use apeiron::learn::fold::stats::capture::{LayerSyrk, Normalisation, SyrkAcc};
use apeiron::learn::fold::stats::heldout::held_out;
use apeiron::learn::fold::stats::lw::lw_intensity;
use apeiron::learn::fold::stats::parts::LayerCfg;
use apeiron::learn::grad::model::{Config, LayerWeights, Weights};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The real `Qwen2.5-0.5B` F16 teacher.
const MODEL: &str = "~/models/gguf/qwen05b-f16.gguf";

/// The C4 calibration tokens (`kl_distill.train_ids`, disjoint from the `seal/calib-c4.txt` shard).
const TOKENS: &str = "~/lanes/w18/kd/c4data/train-c4.qwen.npy";

/// The fit's working directory: its checkpoints, its student `GGUF`s and its JSON receipt. A
/// separate `const` keeps every expression that names it short enough to lay out the same way.
const GEN_DIR: &str = "~/gen-w1-t01";

/// Tokens per chunk, as `oneshot_qwen05b` and `tests/learn_fold_heldout.rs` chunk the stream.
const CHUNK: usize = 512;

/// Held-out chunks scored: the four chunks after the 50 calibration chunks of the split.
const N_HO: usize = 4;

/// `SolveParams::default().n_ref`, the stage scale's reference width.
const N_REF: usize = 896;

/// Stage slots of the tap rows (`src/engine/step/tap.rs`, private to that module; the capture
/// module keeps its own copies of the same four at `src/learn/fold/stats/capture.rs:105-113`).
const STAGE_LN1_OUT: usize = 1;
const STAGE_ATTN_OUT: usize = 2;
const STAGE_LN2_OUT: usize = 3;
const STAGE_M: usize = 4;

// ---- the fit helpers, shared with `tests/learn_fold_oneshot.rs` --------------------------------

/// The production ridge/damp frame: `base · n / n_ref` as one `Q32.32` word, round-half-to-even
/// (`learn::fold::layer::driver::stage_scale`).
fn stage_scale(base: i64, n: usize, n_ref: usize) -> i64 {
    let num = i128::from(base) * i128::try_from(n).expect("width");
    let den = i128::try_from(n_ref).expect("n_ref");
    let (q, r) = (num / den, num % den);
    let up = 2 * r > den || (2 * r == den && q % 2 != 0);
    i64::try_from(q + i128::from(up)).expect("word")
}

/// The `fold_fit` configuration at input width `n`.
fn fit_cfg(base: &DriverCfg, n: usize) -> FoldFitConfig {
    FoldFitConfig {
        ridge: stage_scale(base.quant_ridge, n, N_REF),
        damp: stage_scale(base.quant_damp, n, N_REF),
        ..base.fit
    }
}

/// One linear fitted once against the real Gram `h` (`[n, n]`, mean-normalised): the student words,
/// the bit budget (`o·n` sign plane plus the pack's `r`/`c`/`d`), the weight count and the wall.
fn fold_one(ws: &[i64], h: &[f64], o: usize, n: usize, base: &DriverCfg) -> (Vec<i64>, u64, u64) {
    let hw: Vec<i64> = h
        .iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("gram word on grid"))
        .collect();
    let f = fold_fit(ws, &hw, o, n, None, &fit_cfg(base, n)).expect("one-shot fold_fit");
    (
        f.wp,
        u64::try_from(o * n).expect("weights") + f.bits_r + f.bits_c + f.bits_d,
        u64::try_from(o * n).expect("weights"),
    )
}

/// [`fold_one`] behind the checkpoint `{dir}/{tag}.bin` (`u64` bits, then the `o·n` words), so a
/// rerun resumes at the first absent fit. Returns `true` on a hit.
fn fold_saved(
    dir: &str,
    tag: &str,
    ws: &[i64],
    h: &[f64],
    o: usize,
    n: usize,
    base: &DriverCfg,
) -> (Vec<i64>, u64, bool) {
    let path = format!("{dir}/{tag}.bin");
    if let Ok(b) = fs::read(&path) {
        if b.len() == 8 + 8 * o * n {
            let bits = u64::from_le_bytes(b[..8].try_into().expect("bits word"));
            let mut wp = vec![0_i64; o * n];
            for (i, w) in wp.iter_mut().enumerate() {
                let at = 8 + 8 * i;
                *w = i64::from_le_bytes(b[at..at + 8].try_into().expect("weight word"));
            }
            return (wp, bits, true);
        }
    }
    let (wp, bits, _) = fold_one(ws, h, o, n, base);
    let mut b = Vec::with_capacity(8 + 8 * wp.len());
    b.extend_from_slice(&bits.to_le_bytes());
    for w in &wp {
        b.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(&path, &b).expect("write fit checkpoint");
    (wp, bits, false)
}

// ---- the geometry -----------------------------------------------------------------------------

/// The four input stages a layer's seven linears read: `q`/`k`/`v` share `ln1_out`, `o` reads
/// `attn_out`, `gate`/`up` share `ln2_out` and `down` reads the `SwiGLU` product `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Qkv,
    O,
    GateUp,
    Down,
}

impl Stage {
    /// The stage's mean activation Gram.
    fn gram(self, g: &LayerSyrk) -> &[f64] {
        match self {
            Stage::Qkv => &g.qkv,
            Stage::O => &g.o,
            Stage::GateUp => &g.gate_up,
            Stage::Down => &g.down,
        }
    }

    /// The stage's moments over the calibration samples.
    fn moments(self, m: &Moments) -> &StageMom {
        match self {
            Stage::Qkv => &m.qkv,
            Stage::O => &m.o,
            Stage::GateUp => &m.gate_up,
            Stage::Down => &m.down,
        }
    }
}

/// One layer-0 linear: its name, its stage, `o` and `n`.
struct Tensor {
    name: &'static str,
    stage: Stage,
    o: usize,
    n: usize,
}

/// The seven linears of a layer in the order [`one_layer_specs`] fits them.
fn layer_tensors(cfg: &Config) -> Vec<Tensor> {
    let hid = cfg.n_embd;
    let qo = cfg.q_out().expect("q_out");
    let kvo = cfg.kv_out().expect("kv_out");
    let inter = cfg.n_ff;
    vec![
        Tensor {
            name: "q",
            stage: Stage::Qkv,
            o: qo,
            n: hid,
        },
        Tensor {
            name: "k",
            stage: Stage::Qkv,
            o: kvo,
            n: hid,
        },
        Tensor {
            name: "v",
            stage: Stage::Qkv,
            o: kvo,
            n: hid,
        },
        Tensor {
            name: "o",
            stage: Stage::O,
            o: hid,
            n: qo,
        },
        Tensor {
            name: "gate",
            stage: Stage::GateUp,
            o: inter,
            n: hid,
        },
        Tensor {
            name: "up",
            stage: Stage::GateUp,
            o: inter,
            n: hid,
        },
        Tensor {
            name: "down",
            stage: Stage::Down,
            o: hid,
            n: inter,
        },
    ]
}

/// A layer's tensor words by name (`&[]` for a name the struct does not carry).
fn tensor<'a>(l: &'a LayerWeights, name: &str) -> &'a [i64] {
    match name {
        "q" => &l.wq,
        "k" => &l.wk,
        "v" => &l.wv,
        "o" => &l.wo,
        "gate" => &l.w_gate,
        "up" => &l.w_up,
        "down" => &l.w_down,
        _ => &[],
    }
}

/// Replace layer 0's `name` with `wp`.
fn set_tensor(l: &mut LayerWeights, name: &str, wp: &[i64]) {
    let slot = match name {
        "q" => &mut l.wq,
        "k" => &mut l.wk,
        "v" => &mut l.wv,
        "o" => &mut l.wo,
        "gate" => &mut l.w_gate,
        "up" => &mut l.w_up,
        "down" => &mut l.w_down,
        _ => panic!("unknown tensor {name}"),
    };
    *slot = wp.to_vec();
}

// ---- the SyrkAcc and the calibration moments ----------------------------------------------------

/// The Schäfer–Strimmer inputs the tap pass can accumulate beside the Gram: `s4 = Σ_t ‖s_t‖⁴` and
/// `q = Σ_t s_t⁴` per input coordinate, over the same samples whose scatter is the Gram.
///
/// Time `O(tokens·n)` per stage, space `O(n)`; the scatter itself is the [`SyrkAcc`] accumulator's.
#[derive(Clone, Debug, Default)]
struct StageMom {
    s4: f64,
    q: Vec<f64>,
}

impl StageMom {
    /// Fold one tap column into the moments.
    fn add(&mut self, col: &[i64]) {
        if self.q.is_empty() {
            self.q = vec![0.0; col.len()];
        }
        let mut norm2 = 0.0;
        for (i, &w) in col.iter().enumerate() {
            let x = from_grid(w);
            let x2 = x * x;
            norm2 += x2;
            self.q[i] += x2 * x2;
        }
        self.s4 += norm2 * norm2;
    }
}

/// The four stages' moments over one calibration pass.
#[derive(Clone, Debug, Default)]
struct Moments {
    qkv: StageMom,
    o: StageMom,
    gate_up: StageMom,
    down: StageMom,
}

impl Moments {
    /// Fold one layer's tap row into the four stage moments.
    fn add(&mut self, stages: &[Vec<Vec<i64>>], widths: [usize; 4]) {
        for (slot, mom, w) in [
            (STAGE_LN1_OUT, &mut self.qkv, widths[0]),
            (STAGE_ATTN_OUT, &mut self.o, widths[1]),
            (STAGE_LN2_OUT, &mut self.gate_up, widths[2]),
            (STAGE_M, &mut self.down, widths[3]),
        ] {
            for col in &stages[slot] {
                assert_eq!(col.len(), w, "tap column width");
                mom.add(col);
            }
        }
    }
}

/// Teacher-input SyrkAcc over `cal` for the first `only` layers, plus the same pass's calibration
/// moments of layer 0: one tapped prefill per chunk (a fresh `DecodeState`, so a chunk never attends
/// to its predecessor), folded per layer.
///
/// Temporal `O(Σ_chunks (prefill + tap + Σ_layer t·w²/2))`; spatial `O(only·Σ w²)` words for the
/// accumulator plus `O(Σ_stage w)` for the moments.
fn gen_syrks(
    engine: &Engine,
    cfg: &Config,
    cal: &[Vec<u32>],
    only: usize,
) -> (Vec<LayerSyrk>, Moments, f64) {
    let gcfg = Config {
        n_layer: only,
        ..cfg.clone()
    };
    let widths = [
        cfg.n_embd,
        cfg.q_out().expect("q_out"),
        cfg.n_embd,
        cfg.n_ff,
    ];
    let mut grams = SyrkAcc::new(&gcfg).expect("grams");
    let mut moments = Moments::default();
    let t = Instant::now();
    for (i, chunk) in cal.iter().enumerate() {
        let mut st = engine.state(chunk.len()).expect("decode state");
        let (_, rows) = engine
            .prefill_tap_cols(&mut st, chunk, KernelMode::Q8Act, true)
            .expect("tapped prefill");
        grams.accumulate_tap(&rows[..only]).expect("tap accumulate");
        moments.add(&rows[0], widths);
        eprintln!(
            "[gen] gram chunk {}/{}: {:.2}s",
            i + 1,
            cal.len(),
            t.elapsed().as_secs_f64()
        );
    }
    let wall = t.elapsed().as_secs_f64();
    (
        grams.finish(Normalisation::Mean).expect("finish"),
        moments,
        wall,
    )
}

// ---- the weight-error propagation -------------------------------------------------------------

/// `tr(E G Eᵀ)` for `E` row-major `o × n` (real units) and the mean Gram `G` (`n × n`).
///
/// `tr(E G Eᵀ) = ⟨G, EᵀE⟩`: the `n × n` inner product `C = EᵀE` is accumulated as `o` rank-one
/// updates and folded against `G`'s upper triangle, so the work is `o·n²/2 + n²` multiply-adds
/// rather than the `o²·n` of forming `E G Eᵀ` at `o > n`. Temporal `O(o·n²)`, spatial `O(n²)` for
/// `C`; the accumulation is `f64` (the task's carrier: `i128`/`f64`), whose 53-bit significand is
/// exact here because the operands are Q32.32-derived reals of magnitude below `2³`.
fn quad(g: &[f64], e: &[f64], o: usize, n: usize) -> f64 {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..o {
        let a = &e[i * n..(i + 1) * n];
        for j in 0..n {
            let aj = a[j];
            if aj == 0.0 {
                continue;
            }
            let row = &mut c[j * n + j..j * n + n];
            for (cc, &ak) in row.iter_mut().zip(&a[j..]) {
                *cc += aj * ak;
            }
        }
    }
    let mut j = 0.0_f64;
    for r in 0..n {
        for k in r..n {
            let w = if r == k { 1.0 } else { 2.0 };
            j += w * g[r * n + k] * c[r * n + k];
        }
    }
    j
}

/// `E = Ŵ − W` from the two word rows, in the grid's real units.
fn err_words(hat: &[i64], w: &[i64]) -> Vec<f64> {
    assert_eq!(hat.len(), w.len(), "error row length");
    hat.iter()
        .zip(w)
        .map(|(&a, &b)| from_grid(a) - from_grid(b))
        .collect()
}

/// The Ledoit–Wolf arm's per-stage intensity and shrunk Gram.
///
/// [`lw_intensity`] is homogeneous of degree 0 in the sample scale (its `S4 − ΣQ` numerator and
/// `off(M)` are both quartic in the samples), so the samples are rescaled by a power of two when
/// the `Q32.32` words of `s4`/`q` would leave `i64`; the intensity is unchanged and the rescale is
/// exact in `f64`. `λ` is the `(num, λ)` pair's `λ` word; the Gram is shrunk toward its diagonal,
/// `G ← (1−λ)·G + λ·diag(G)`, off-diagonals only.
fn lw_shrunk(g: &[f64], n: usize, tokens: u64, m: &StageMom) -> (f64, Vec<f64>) {
    let mut s4 = m.s4;
    let mut q = m.q.clone();
    let lim = 2f64.powi(61);
    let mut k = 0_u32;
    while (s4 * 4_294_967_296.0 >= lim
        || q.iter().copied().fold(0.0, f64::max) * 4_294_967_296.0 >= lim)
        && k < 64
    {
        s4 *= 2f64.powi(-4);
        for v in &mut q {
            *v *= 2f64.powi(-4);
        }
        k += 1;
    }
    let c2 = 2f64.powi(-2 * i32::try_from(k).expect("k"));
    let nt = f64::from(u32::try_from(tokens).expect("tokens"));
    let hw: Vec<i64> = g
        .iter()
        .map(|&v| to_grid(v * nt * c2, GRID_BOUND).expect("scatter word on grid"))
        .collect();
    let qw: Vec<i64> = q
        .iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("q word on grid"))
        .collect();
    let s4w = to_grid(s4, GRID_BOUND).expect("s4 word on grid");
    let (_, lambda) = lw_intensity(s4w, &qw, &hw, n, usize::try_from(tokens).expect("tokens"))
        .expect("lw intensity");
    let lam = from_grid(lambda);
    let mut out = g.to_vec();
    for r in 0..n {
        for c in 0..n {
            if r != c {
                out[r * n + c] *= 1.0 - lam;
            }
        }
    }
    (lam, out)
}

// ---- the student ------------------------------------------------------------------------------

/// The single-tensor student, built by **moving** the teacher's buffers into it: a clone would hold
/// two copies of the model's words (10.6 GB for the 0.5B teacher's `Q32.32` rows) beside the write
/// buffers and the two engines. `tw` is left empty; [`with_tensor`] rebuilds it from the read-back.
fn student_from(tw: &mut Weights, name: &str, wp: &[i64]) -> Weights {
    let mut sw = Weights {
        layers: core::mem::take(&mut tw.layers),
        output_norm: core::mem::take(&mut tw.output_norm),
        token_embd: core::mem::take(&mut tw.token_embd),
        lm_head: core::mem::take(&mut tw.lm_head),
    };
    set_tensor(&mut sw.layers[0], name, wp);
    sw
}

/// `w` with layer 0's `name` set to `words`: the next iteration's teacher, from the student's
/// read-back. The writer copies every tensor it carries through the `F16` encoding the teacher's own
/// payload came from, so a read-back equals the teacher word for word outside `name` — the property
/// [`gen_fixture`] asserts on the whole struct.
fn with_tensor(mut w: Weights, name: &str, words: &[i64]) -> Weights {
    set_tensor(&mut w.layers[0], name, words);
    w
}

/// Write `sw` under the teacher's directory, open it, and return the engine.
fn write_and_open(teacher_path: &str, sw: &Weights, out: &str) -> Engine {
    write_student_gguf(teacher_path, sw, out).expect("write student gguf");
    Engine::open(out).expect("open student")
}

// ---- the measurement ---------------------------------------------------------------------------

/// One row of the table.
struct Row {
    name: &'static str,
    o: usize,
    n: usize,
    bits: u64,
    jin: f64,
    jout: f64,
    klin: f64,
    klout: f64,
    top1: f64,
    gamma: f64,
    lambda: f64,
    fit_s: f64,
    eval_s: f64,
}

impl Row {
    /// `O_t = J_out/J_in`.
    fn o_ratio(&self) -> f64 {
        self.jout / self.jin
    }

    /// `Q_t = (KL_out/KL_in)/O_t`.
    fn q_ratio(&self) -> f64 {
        (self.klout / self.klin) / self.o_ratio()
    }

    /// `KL_in/J_in`.
    fn kl_per_j(&self) -> f64 {
        self.klin / self.jin
    }

    fn json(&self, n_cal: usize, lw: bool) -> String {
        format!(
            "{{\"ncal\":{n_cal},\"arm\":\"{}\",\"tensor\":\"{}\",\"o\":{},\"n\":{},\"o_x_n\":{},\
             \"bits\":{},\"J_in\":{:.12e},\"J_out\":{:.12e},\"O_t\":{:.6},\"KL_in\":{:.12e},\
             \"KL_out\":{:.12e},\"Q_t\":{:.6},\"top1_out\":{:.6},\"KL_in_per_J\":{:.6e},\
             \"gamma\":{:.6e},\"lambda\":{:.6},\"fit_s\":{:.3},\"eval_s\":{:.3}}}",
            if lw { "lw" } else { "plain" },
            self.name,
            self.o,
            self.n,
            self.o * self.n,
            self.bits,
            self.jin,
            self.jout,
            self.o_ratio(),
            self.klin,
            self.klout,
            self.q_ratio(),
            self.top1,
            self.kl_per_j(),
            self.gamma,
            self.lambda,
            self.fit_s,
            self.eval_s
        )
    }
}

/// The table's header row.
fn print_header() {
    println!("| t | o×n | J_in | J_out | O_t | KL_in | KL_out | Q_t | KL_in/J_in | top-1 | λ |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
}

/// Print one row.
fn print_row(r: &Row) {
    println!(
        "| {} | {}×{} | {:.4e} | {:.4e} | {:.4} | {:.4e} | {:.4e} | {:.4} | {:.4e} | {:.4} | {:.4} |",
        r.name,
        r.o,
        r.n,
        r.jin,
        r.jout,
        r.o_ratio(),
        r.klin,
        r.klout,
        r.q_ratio(),
        r.kl_per_j(),
        r.top1,
        r.lambda
    );
}

fn rss_gb() -> f64 {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1e6;
        }
    }
    0.0
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The `Config` of an opened engine.
fn engine_config(engine: &Engine) -> Config {
    let ec = &engine.config;
    Config {
        n_layer: ec.n_layer,
        n_head: ec.n_head,
        n_head_kv: ec.n_head_kv,
        n_embd: ec.n_embd,
        n_ff: ec.n_ff,
        vocab: ec.vocab_size,
        head_dim: ec.head_dim,
        eps: (ec.rms_eps.to_f64() * 4_294_967_296.0).round() as i64,
        rope_theta: ec.rope_theta.to_f64(),
    }
}

/// The layer geometry [`DriverCfg::production`] needs.
fn engine_layer_cfg(cfg: &Config) -> LayerCfg {
    LayerCfg {
        n_head: cfg.n_head,
        n_kv_head: cfg.n_head_kv,
        head_dim: cfg.head_dim,
        hidden: cfg.n_embd,
        inter: cfg.n_ff,
        eps: from_grid(cfg.eps),
        theta: cfg.rope_theta,
    }
}

/// The calibration/held-out chunks `oneshot_qwen05b` uses: 512-token chunks, the 50/20/40 split's
/// first `50` calibration chunks and the four chunks after them (disjoint by construction).
fn load_chunks() -> (Vec<Vec<u32>>, Vec<Vec<u32>>, f64) {
    let t = Instant::now();
    let ids_raw = fs::read(TOKENS).expect("read token npy");
    let ids = read_npy_i4(&ids_raw).expect("parse token npy");
    drop(ids_raw);
    let all = chunks(&ids, CHUNK).expect("token chunks");
    drop(ids);
    let sp = split(all.len(), 50, 20, 40).expect("50/20/40 split");
    let cal: Vec<Vec<u32>> = sp.cal.iter().map(|&c| all[c].clone()).collect();
    let ho: Vec<Vec<u32>> = sp.heldout[..N_HO].iter().map(|&c| all[c].clone()).collect();
    drop(all);
    (cal, ho, t.elapsed().as_secs_f64())
}

/// The cost-equation probe: the unit walls the projection needs, then the projection `T(N)` for
/// each `N ∈ {2, 4, 8}` — `T = N·t_syrk + Σ fits + 7·(N + 4)·t_eval_chunk + 7·(write + open +
/// readback + quad)` (`AGENTS.md` §10: the projection is written before launch).
fn probe(
    engine: &Engine,
    cfg: &Config,
    base: &DriverCfg,
    cal: &[Vec<u32>],
    ho: &[Vec<u32>],
    dir: &str,
    t_all: &Instant,
) {
    let (g, _, t_syrk2) = gen_syrks(engine, cfg, &cal[..2], 1);
    let (g_ho, _, t_syrk4) = gen_syrks(engine, cfg, ho, 1);
    let t = Instant::now();
    let (mut tw, _cfg) = Weights::from_engine(engine).expect("teacher weights");
    let t_weights = t.elapsed().as_secs_f64();
    eprintln!(
        "[gen] probe: teacher weights {t_weights:.2}s, rss {:.2} GB",
        rss_gb()
    );

    let wq = tw.layers[0].wq.clone();
    let sw = student_from(&mut tw, "q", &wq);
    let path = format!("{dir}/probe-identity.gguf");
    let t = Instant::now();
    let se = write_and_open(MODEL, &sw, &path);
    let t_write = t.elapsed().as_secs_f64();
    drop(sw);
    let t = Instant::now();
    let teacher = Engine::open(MODEL).expect("reopen teacher");
    let t_open = t.elapsed().as_secs_f64();
    let cal_refs: Vec<&[u32]> = cal[..2].iter().map(Vec::as_slice).collect();
    let ho_refs: Vec<&[u32]> = ho.iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let h2 = held_out(&teacher, &se, &cal_refs).expect("probe held-out 2");
    let t_2 = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let h4 = held_out(&teacher, &se, &ho_refs).expect("probe held-out 4");
    let t_4 = t.elapsed().as_secs_f64();
    let t_eval_chunk = t_4 / f64::from(u32::try_from(N_HO).expect("ho"));
    drop(teacher);
    let t = Instant::now();
    let rb = Weights::from_engine(&se).expect("read back").0;
    let t_read = t.elapsed().as_secs_f64();
    assert_eq!(
        tensor(&rb.layers[0], "q"),
        wq.as_slice(),
        "self-replacement reads back word for word"
    );
    tw = with_tensor(rb, "q", &wq);
    drop(wq);
    eprintln!(
        "[gen] probe: write + open {t_write:.2}s, teacher reopen {t_open:.2}s, read-back \
         {t_read:.2}s, rss {:.2} GB",
        rss_gb()
    );
    drop(se);
    let _ = fs::remove_file(&path);

    let e_q: Vec<f64> = tw.layers[0].wq.iter().map(|&w| from_grid(w)).collect();
    let e_down: Vec<f64> = tw.layers[0].w_down.iter().map(|&w| from_grid(w)).collect();
    let t = Instant::now();
    let _ = quad(g[0].qkv.as_slice(), &e_q, cfg.n_embd, cfg.n_embd);
    let t_q = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = quad(g_ho[0].down.as_slice(), &e_down, cfg.n_embd, cfg.n_ff);
    let t_d = t.elapsed().as_secs_f64();
    drop(e_q);
    drop(e_down);

    // The quoted fit walls of the same layer (`tests/learn_fold_oneshot.rs`, measured there):
    // q 7.19 s, gate 34.0 s, down 87.5 s; `k`/`v` at `kv_out = 128` and `o` at `q_out` are the `q`
    // shape's `n × n` solve, so the seven fits are `4·7.19 + 2·34.0 + 87.5`.
    let t_fits = 4.0 * 7.19 + 2.0 * 34.0 + 87.5;
    let _ = (base, t_syrk4);
    eprintln!(
        "[gen] probe: gram {:.2}s/chunk; held-out 2 chunks {t_2:.2}s, 4 chunks {t_4:.2}s          ({t_eval_chunk:.2}s/chunk, KL2 {:.4e}, top1 {:.4}); quad q {t_q:.2}s, quad down {t_d:.2}s; \
         rss {:.2} GB, wall {:.1}s",
        t_syrk2 / 2.0,
        from_grid(h2.kl),
        from_grid(h4.top1),
        rss_gb(),
        t_all.elapsed().as_secs_f64()
    );
    for n in [2_usize, 4, 8] {
        let grams = f64::from(u32::try_from(n).expect("n")) * t_syrk2 / 2.0;
        let evals = 7.0 * f64::from(u32::try_from(n + N_HO).expect("n")) * t_eval_chunk;
        let sides = 7.0 * (t_write + t_open + t_read) + 4.0 * (t_q + t_d);
        let total = grams + t_fits + evals + sides;
        eprintln!(
            "[gen] T({n}) = grams {grams:.0} + fits {t_fits:.0} + 7·({n}+4)·{t_eval_chunk:.2} = \
             {evals:.0} + student write/read {sides:.0} -> {total:.0}s"
        );
    }
}

/// `gen_qwen05b` (ignored): the per-tensor generalisation law on the real 0.5B teacher.
///
/// `APEIRON_GEN_NCAL` (default `2`) selects the calibration chunk count; `APEIRON_GEN_LW=1` runs the
/// Ledoit–Wolf arm instead of the plain one; `APEIRON_GEN_PROBE=1` prints the cost equation and
/// stops; `APEIRON_GEN_DIR` (default `~/gen-w1-t01`) holds the fit checkpoints, the
/// student `GGUF`s and the JSON receipt.
#[test]
#[ignore = "loads the 0.5B model and fits 7 linears; run --release --ignored --nocapture"]
fn gen_qwen05b() {
    let t_all = Instant::now();
    let dir = env("APEIRON_GEN_DIR").unwrap_or_else(|| GEN_DIR.to_string());
    for sub in ["", "/fit", "/students"] {
        fs::create_dir_all(format!("{dir}{sub}")).expect("gen dir");
    }
    let n_cal: usize = env("APEIRON_GEN_NCAL")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let lw = env("APEIRON_GEN_LW").as_deref() == Some("1");
    let probe_only = env("APEIRON_GEN_PROBE").as_deref() == Some("1");
    // `APEIRON_GEN_ONLY=q,down` resumes a run cut by the wall-clock bound: the named tensors only,
    // appending to the same receipt, the finished fits taken from their checkpoints.
    let only: Vec<String> = env("APEIRON_GEN_ONLY")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let (cal, ho, load_s) = load_chunks();
    assert!(
        n_cal >= 2 && n_cal <= cal.len(),
        "N_cal in [2, {}]",
        cal.len()
    );

    let engine = Engine::open(MODEL).expect("open teacher");
    let config = engine_config(&engine);
    let base = DriverCfg::production(engine_layer_cfg(&config), CHUNK, n_cal);
    let tensors = layer_tensors(&config);
    eprintln!(
        "[gen] teacher {MODEL}: n_layer {} hidden {} q_out {} inter {}; cal {n_cal} chunks of \
         {CHUNK}, {N_HO} held-out, loaded at {load_s:.1}s",
        config.n_layer,
        config.n_embd,
        config.q_out().expect("q_out"),
        config.n_ff
    );
    if probe_only {
        probe(&engine, &config, &base, &cal, &ho, &dir, &t_all);
        return;
    }

    let (g_cal, moments, t_gcal) = gen_syrks(&engine, &config, &cal[..n_cal], 1);
    let (g_ho, _, t_gho) = gen_syrks(&engine, &config, &ho, 1);
    eprintln!(
        "[gen] grams: cal {t_gcal:.2}s, held-out {t_gho:.2}s, rss {:.2} GB",
        rss_gb()
    );
    let g_cal0 = &g_cal[0];
    let g_ho0 = &g_ho[0];
    let tokens = g_cal0.tokens;
    eprintln!(
        "[gen] calibration tokens {tokens}, held-out tokens {}",
        g_ho0.tokens
    );

    let cal_refs: Vec<&[u32]> = cal[..n_cal].iter().map(Vec::as_slice).collect();
    let ho_refs: Vec<&[u32]> = ho.iter().map(Vec::as_slice).collect();
    let arm = if lw { "lw" } else { "plain" };
    let receipt = format!("{dir}/gen-n{n_cal}{}.json", if lw { "-lw" } else { "" });
    if only.is_empty() {
        let _ = fs::remove_file(&receipt);
    }
    let mut rows: Vec<Row> = Vec::with_capacity(tensors.len());
    let mut tw: Weights = Weights::from_engine(&engine).expect("teacher weights").0;
    drop(engine);

    for t in &tensors {
        if !only.is_empty() && !only.iter().any(|n| n == t.name) {
            continue;
        }
        let tf = Instant::now();
        let gram = t.stage.gram(g_cal0);
        let (lambda, shrunk) = if lw {
            lw_shrunk(gram, t.n, tokens, t.stage.moments(&moments))
        } else {
            (0.0, Vec::new())
        };
        let fit_syrk: &[f64] = if lw { &shrunk } else { gram };
        let ws = tensor(&tw.layers[0], t.name).to_vec();
        let tag = format!("{arm}-n{n_cal}-l0-{}", t.name);
        let (wp, bits, hit) =
            fold_saved(&format!("{dir}/fit"), &tag, &ws, fit_syrk, t.o, t.n, &base);
        let fit_s = tf.elapsed().as_secs_f64();
        eprintln!(
            "[gen] {arm} {} {}×{}: fit {fit_s:.2}s{}{}",
            t.name,
            t.o,
            t.n,
            if hit { " (checkpoint)" } else { "" },
            if lw {
                format!(" λ {lambda:.4}")
            } else {
                String::new()
            }
        );
        drop(ws);
        drop(shrunk);

        // `J` and the read-back come after the KL evals, and the teacher's `Weights` and `Engine`
        // are not resident across the student's write or the eval's two `t×vocab` logit blocks:
        // measured at 12.0 GB (the cap) with all four alive, 6.7 GB in this order.
        let w = tensor(&tw.layers[0], t.name).to_vec();
        let sw = student_from(&mut tw, t.name, &wp);
        let out = format!("{dir}/students/{arm}-n{n_cal}-{}.gguf", t.name);
        let te = Instant::now();
        let se = write_and_open(MODEL, &sw, &out);
        drop(sw);
        let t_engine = Instant::now();
        let teacher = Engine::open(MODEL).expect("reopen teacher");
        let t_open = t_engine.elapsed().as_secs_f64();
        let hin = held_out(&teacher, &se, &cal_refs).expect("KL_in");
        let hout = held_out(&teacher, &se, &ho_refs).expect("KL_out");
        drop(teacher);
        let rb = Weights::from_engine(&se).expect("student weights").0;
        let e = err_words(tensor(&rb.layers[0], t.name), &w);
        tw = with_tensor(rb, t.name, &w);
        let jin = quad(t.stage.gram(g_cal0), &e, t.o, t.n);
        let jout = quad(t.stage.gram(g_ho0), &e, t.o, t.n);
        drop(e);
        drop(w);
        let eval_s = te.elapsed().as_secs_f64();
        drop(se);
        let _ = fs::remove_file(&out);
        eprintln!("[gen] {arm} {}: teacher reopen {t_open:.2}s", t.name);

        let row = Row {
            name: t.name,
            o: t.o,
            n: t.n,
            bits,
            jin,
            jout,
            klin: from_grid(hin.kl),
            klout: from_grid(hout.kl),
            top1: from_grid(hout.top1),
            gamma: t.n as f64 / (f64::from(u32::try_from(tokens).expect("tokens"))),
            lambda,
            fit_s,
            eval_s,
        };
        print_row(&row);
        // Append the row at once, so a run cut by the wall-clock bound still leaves its receipt.
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&receipt)
            .expect("open receipt");
        f.write_all(row.json(n_cal, lw).as_bytes())
            .expect("append receipt");
        f.write_all(b"\n").expect("append receipt");
        rows.push(row);
        eprintln!(
            "[gen] rss {:.2} GB, wall {:.1}s",
            rss_gb(),
            t_all.elapsed().as_secs_f64()
        );
    }

    let out_path = receipt.clone();
    print_header();
    for r in &rows {
        print_row(r);
    }
    let sum_o: f64 = rows.iter().map(Row::o_ratio).sum();
    let mean_klin: f64 = rows.iter().map(|r| r.klin).sum::<f64>() / rows.len() as f64;
    let mean_klout: f64 = rows.iter().map(|r| r.klout).sum::<f64>() / rows.len() as f64;
    println!(
        "[gen] {arm} N_cal {n_cal}: Σ_t O_t {sum_o:.4}; Σ_t KL_out {:.4e}, Σ_t KL_in {:.4e}, \
         ratio {:.4} (the full-student reference is 3.675); receipt {out_path}; wall {:.1}s; \
         rss {:.2} GB",
        rows.iter().map(|r| r.klout).sum::<f64>(),
        rows.iter().map(|r| r.klin).sum::<f64>(),
        rows.iter().map(|r| r.klout).sum::<f64>() / rows.iter().map(|r| r.klin).sum::<f64>(),
        t_all.elapsed().as_secs_f64(),
        rss_gb()
    );
    let _ = (mean_klin, mean_klout, sum_o, &base, &moments);
}

// ---- the fixture wiring gate ------------------------------------------------------------------

/// Fixture residual width (a multiple of `GROUP = 128`).
const GEN_HID: usize = 128;
/// Fixture query heads (`head_dim = GEN_HID / GEN_NH = 64`).
const GEN_NH: usize = 2;
/// Fixture KV heads (`kv_out = GEN_HID`).
const GEN_NKV: usize = 2;
/// Fixture feed-forward width.
const GEN_FF: usize = 128;
/// Fixture vocabulary.
const GEN_V: usize = 32;
/// Fixture tokens per chunk.
const GEN_TOK: usize = 8;

fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// `n` `f32` words of `1.0`, the fixture's norm gains.
fn gen_gain(n: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(n * 4);
    for _ in 0..n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    b
}

/// Round-to-nearest-even `f16` bits for a finite normal `f32` in `(2^-14, 65504)`.
#[allow(
    clippy::cast_sign_loss,
    reason = "the significand and exponent are masked to their unsigned widths and OR-ed into a \
              sign bit; no negative value reaches the cast"
)]
fn f16_bits(x: f32) -> u16 {
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

/// A deterministic `F16` matrix of `rows · cols` entries in `{-2, …, 2}·2⁻⁵` (exact in `f16`).
fn gen_matrix(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_bits(code as f32 * SCALE).to_le_bytes());
        }
    }
    b
}

/// One dense `qwen2` trunk layer: matrices and the tied embedding `F16`, the three norm gains `F32`.
fn gen_gguf() -> Vec<u8> {
    let (e, q, kv, ff, v) = (GEN_HID, GEN_HID, GEN_HID, GEN_FF, GEN_V);
    let f32t = quant::GGML_TYPE_F32;
    let f16t = quant::GGML_TYPE_F16;
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("blk.0.attn_norm.weight", vec![u(e)], f32t, gen_gain(e)),
        (
            "blk.0.attn_q.weight",
            vec![u(e), u(q)],
            f16t,
            gen_matrix(q, e, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![u(e), u(kv)],
            f16t,
            gen_matrix(kv, e, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![u(e), u(kv)],
            f16t,
            gen_matrix(kv, e, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![u(q), u(e)],
            f16t,
            gen_matrix(e, q, 4),
        ),
        ("blk.0.ffn_norm.weight", vec![u(e)], f32t, gen_gain(e)),
        (
            "blk.0.ffn_gate.weight",
            vec![u(e), u(ff)],
            f16t,
            gen_matrix(ff, e, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![u(e), u(ff)],
            f16t,
            gen_matrix(ff, e, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![u(ff), u(e)],
            f16t,
            gen_matrix(e, ff, 7),
        ),
        ("output_norm.weight", vec![u(e)], f32t, gen_gain(e)),
        (
            "token_embd.weight",
            vec![u(e), u(v)],
            f16t,
            gen_matrix(v, e, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("tensors"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(
        &mut b,
        "qwen2.attention.head_count",
        u32::try_from(GEN_NH).expect("heads"),
    );
    kv_u32(
        &mut b,
        "qwen2.attention.head_count_kv",
        u32::try_from(GEN_NKV).expect("heads"),
    );
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(GEN_HID).expect("width"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(GEN_FF).expect("width"),
    );
    let mut off = 0_u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank"));
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += u64::try_from(payload.len()).expect("payload");
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, u64::try_from(s.len()).expect("len"));
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

/// A scratch directory under the lane's disk, never the tmpfs.
fn scratch(name: &str) -> String {
    let root = env("APEIRON_GEN_TEST_DIR").unwrap_or_else(|| format!("{GEN_DIR}/test"));
    let dir = format!("{root}/{name}-{}", std::process::id());
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The wiring gate: on one dense fixture layer, `teacher tap → SyrkAcc → fold_fit → student GGUF →
/// read-back error → J_in/J_out`, with the disjoint-Gram assertion and the self-replacement case.
///
/// `J_in > 0` requires a fitted `q` whose re-encoded words differ from the teacher's; `J_out` is
/// computed from the held-out Gram and must differ from the calibration value (the assertion the
/// control's `J_out ← cal` mutation reddens); the self-replacement student must read back exactly
/// (`J 0`) and score `KL 0`.
#[test]
fn gen_fixture() {
    let dir = scratch("gen-fixture");
    let teacher_path = format!("{dir}/teacher.gguf");
    fs::write(&teacher_path, gen_gguf()).expect("write fixture");
    let engine = Engine::open(&teacher_path).expect("open fixture");
    let (tw, config) = Weights::from_engine(&engine).expect("fixture weights");
    assert_eq!(config.n_layer, 1, "fixture layer count");
    let vocab = u32::try_from(GEN_V).expect("vocab");
    let cal: Vec<u32> = (0..GEN_TOK as u32).map(|i| (i * 5 + 3) % vocab).collect();
    let ho: Vec<u32> = (0..GEN_TOK as u32).map(|i| (i * 7 + 11) % vocab).collect();

    let (g_cal, mom, wall) = gen_syrks(&engine, &config, std::slice::from_ref(&cal), 1);
    let (g_ho, _, _) = gen_syrks(&engine, &config, std::slice::from_ref(&ho), 1);
    assert_eq!(g_cal[0].tokens, u64::try_from(GEN_TOK).expect("tokens"));
    assert_ne!(
        g_cal[0].qkv, g_ho[0].qkv,
        "calibration and held-out SyrkAcc"
    );
    let base = DriverCfg::production(engine_layer_cfg(&config), GEN_TOK, 1);

    // Fit `q` on the calibration Gram, then build the single-tensor student.
    let wq = tw.layers[0].wq.clone();
    let (wp, _bits, _linear) = fold_one(&wq, &g_cal[0].qkv, GEN_HID, GEN_HID, &base);
    let mut tw = tw;
    let sw = student_from(&mut tw, "q", &wp);
    let fitted = format!("{dir}/student-q.gguf");
    let se = write_and_open(&teacher_path, &sw, &fitted);
    drop(sw);
    let rb = Weights::from_engine(&se).expect("fitted read-back").0;
    let e = err_words(&rb.layers[0].wq, &wq);
    let (o, n) = (GEN_HID, GEN_HID);
    let jin = quad(&g_cal[0].qkv, &e, o, n);
    let jout = quad(&g_ho[0].qkv, &e, o, n);
    let hf = held_out(&engine, &se, std::slice::from_ref(&ho.as_slice())).expect("fitted KL");
    drop(se);

    // The teacher rebuilt from the fitted student's read-back: every other buffer went through the
    // writer unchanged, so restoring `q` must give the teacher's exact struct — the property the
    // measurement relies on between tensors.
    let mut tw = with_tensor(rb, "q", &wq);

    // The self-replacement student: layer 0's `q` is the teacher's own words.
    let sid = student_from(&mut tw, "q", &wq);
    let ident = format!("{dir}/student-identity.gguf");
    let ie = write_and_open(&teacher_path, &sid, &ident);
    drop(sid);
    let irb = Weights::from_engine(&ie).expect("identity read-back").0;
    assert_eq!(
        irb.layers[0].wq, wq,
        "self-replacement must read back word for word"
    );
    let e0 = err_words(&irb.layers[0].wq, &wq);
    let jin0 = quad(&g_cal[0].qkv, &e0, o, n);
    let jout0 = quad(&g_ho[0].qkv, &e0, o, n);
    let hi = held_out(&engine, &ie, std::slice::from_ref(&ho.as_slice())).expect("identity KL");
    drop(ie);

    println!(
        "gen fixture: gram {wall:.3}s; fitted q J_in {jin:.6e} J_out {jout:.6e} ratio {:.6}; \
         identity J_in {jin0:.1} J_out {jout0:.1} KL {:.6} top1 {:.6}; fitted KL {:.6e}",
        jout / jin,
        from_grid(hi.kl),
        from_grid(hi.top1),
        from_grid(hf.kl)
    );
    assert!(jin > 0.0, "J_in must be positive on a fitted tensor: {jin}");
    assert!(
        (jout - jin).abs() > 1e-6 * jin,
        "J_out must come from the disjoint held-out Gram: J_in {jin:.6e}, J_out {jout:.6e}"
    );
    assert_eq!(jin0, 0.0, "self-replacement J_in");
    assert_eq!(jout0, 0.0, "self-replacement J_out");
    assert_eq!(from_grid(hi.kl), 0.0, "self-replacement KL");
    assert_eq!(from_grid(hi.top1), 1.0, "self-replacement top-1");
    fs::remove_dir_all(&dir).expect("clean scratch");
    let _ = (&mom, hf);
}
