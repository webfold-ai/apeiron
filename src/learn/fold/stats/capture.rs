//! Calibration activation-Gram (input-side Hessian) capture on the crate's own forward.
//!
//! Two sources feed the same accumulator:
//!
//! | source | fn | activations |
//! |---|---|---|
//! | the executing engine's prefill tap | [`Grams::accumulate_tap`] | the rows [`crate::engine::Engine::prefill_tap_cols`] records |
//! | a Q32.32 [`Tape`] (the engine's `KernelMode::Exact` prefill) | [`Grams::accumulate`] | the exact reference rows |
//!
//! Both accumulate, per trunk layer and per linear input, the unweighted scatter
//! `H = Σ_t x_t x_tᵀ` over the tokens, `x_t` the linear's input row. The engine source is the
//! production path: it reads the activations the engine actually computes, so the Gram is of the
//! teacher's own words and not of a second forward's. The [`Tape`] source stays as the exact
//! reference the engine path is tested against (`tests/learn_fold_capture_engine.rs`).
//! [`Grams::finish`] mirrors the upper triangle to the lower one and applies the requested
//! normalisation. The port is the activation-Gram half of the reference `rstar_fast.py` `Gram`
//! (`~/lanes/port/spec/rstar_fast.py:225-254`), whose four `H:<key>` tensors are
//! exactly these four inputs (`rstar_fast.py:641-653`).
//!
//! | [`LayerGrams`] field | linear | stage ([`crate::engine::step`] tap slot) | width |
//! |---|---|---|---|
//! | `qkv` | Q/K/V projection | `ln1_out` (`attn_norm` output) | `n_embd` |
//! | `o` | output projection | `attn_out` | `q_out = n_head·head_dim` |
//! | `gate_up` | gate and up | `ln2_out` (`ffn_norm` output) | `n_embd` |
//! | `down` | down projection | `m` (`SwiGLU` product) | `n_ff` |
//!
//! The tap rows are `[layer][stage][column][width]`; a stage the schedule does not route through a
//! tap point is empty, so `accumulate_tap` refuses a single-column prefill, whose fused FFN records
//! no `m` row.
//!
//! # Sum or mean, weighting, accumulation precision
//!
//! The accumulator is the raw sum `Σ_t x_t x_tᵀ`; [`Normalisation`] selects what the consumer is
//! handed, and the doc comment on each variant names the reference site it reproduces. The
//! reference `Gram.add` adds one **unweighted** `aᵀa` per token batch (`rstar_fast.py:234-245`) —
//! there is no per-token weight — and `Gram.result(nt)` divides by the token count `nt = nc·CHUNK`
//! (`rstar_fast.py:247-254`, `:677`), so [`Normalisation::Mean`] is the reference's `H`; the
//! imatrix `diag(H) = (Σ_t x_t²)/nt` of `forward_all` (`rstar_fast.py:367`) confirms it.
//! [`Normalisation::MeanDiagonal`] reproduces the different normalisation `gq_capture` applies to
//! its output-gradient `HO` (mean diagonal 1, then `0.01` added to the diagonal,
//! `~/lanes/port/spec/fold_transfer.py:1322-1323`).
//!
//! Accumulation from the [`Tape`] is Q32.32 `i64` words, the tape's own carrier: each entry sums
//! the exact `i128` product of two words rounded once by [`round_q32_even`], so a term carries at
//! most half a grid quantum (`2⁻³³` real) and a `t`-token entry at most `t·2⁻³³`. Accumulation from
//! the engine tap goes through the one limb adapter, [`syrk_q32`] at the default three limbs
//! (AGENTS.md §0 K2/K3): the entry is the limb expansion's word word-sum, whose error is
//! [`syrk_q32`]'s stated bound `G(n, 3, max|x|, max|x|)` — the 3-limb truncation dominates, about
//! `2⁻¹⁵` relative — and the entries are then summed exactly in `i64`. The engine's own forward also
//! rounds where the reference does not, so the two sources agree to the measured bound on the
//! fixture, not bit for bit (`tests/learn_fold_capture_engine.rs`). The reference accumulates `fp32`
//! (`Gram.acc`, `rstar_fast.py:231`) and its `fp16s` mode fixes a per-column scale from the first
//! batch (`:238-241`); neither rounding is reproduced. [`Grams::finish`] converts each finished
//! word to `f64` once, at the `LayerGrams` boundary.
//!
//! # Measured difference on the fixture
//!
//! `tests/learn_fold_capture_engine.rs` runs one dense layer through both sources and prints these
//! numbers every run. In [`crate::engine::KernelMode::Exact`] the tap rows against the reference
//! activations (entry-wise relative): `ln1_out` `0`, `attn_out` `2.7e-6`, `ln2_out` `2.0e-7`. In
//! the production [`crate::engine::KernelMode::Q8Act`] the Grams against the reference Grams
//! (difference over the matrix's own scale): `qkv` `1.6e-6`, `o` `4.4e-3`, `gate_up` `1.5e-3`,
//! `down` `1.1e-2`. The `qkv` input is the residual norm before the first quantised projection, so
//! it carries only the two forwards' fixed-point difference; the other three carry the 8-bit
//! activation quantiser through the hidden state.
//!
//! The reference's mean-diagonal floor `1e-30` is sub-quantum here: the smallest positive mean a
//! nonzero diagonal can produce is `2⁻³²/d` real, above `1e-30` for every `d < 2⁶⁸`, so the only
//! reachable branch is a zero diagonal, whose quotient is zero.
//!
//! # Residual: what `gq_capture` captures that this module does not
//!
//! Only the activation Gram is ported. `gq_capture` (`fold_transfer.py:1163-1341`) additionally
//! captures `A`, `B`, `HO`, `S4`/`Q4`, `ho_lw`, `PB`, `ho_lb`, `max_dlogit` and the `K` sampled
//! targets; those need a backward pass this forward-only capture does not run, and none is stubbed.
//! (The output-gradient half is [`super::hocapture`].)
//!
//! # Evidence grade
//!
//! `tested`: `tests/learn_fold_capture.rs` for the reference source and
//! `tests/learn_fold_capture_engine.rs` for the engine source and the two sources' per-entry
//! difference on the fixture. No Lean refinement links this file.
//!
//! # Trust boundary
//!
//! The capture reads Q32.32 `i64` rows only and accumulates in the same carrier; the finished `f64`
//! Gram is the caller's representation of those words. `rmsnorm`, attention and the dense kernels
//! are [`crate::learn::grad::model`]'s and the engine's contracts. Not covered: the output-gradient
//! Grams above, the thread pool's scheduling, and any activation the source does not carry.

#![allow(
    clippy::many_single_char_names,
    reason = "the single-letter names are the matrix entry's row and column `i`/`j`, the token `r` \
              and the row width `w`; the accumulator `x` is the tape's row block"
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::engine::kern::limb::{syrk_q32, LimbError};
use crate::learn::grad::model::{Config, Tape};
use crate::learn::solve::q32::{from_grid, round_q32_even};

/// Limbs per operand in the tap Gram's limb product: the default three (AGENTS.md §0 K3), 24-bit
/// digits, the precision of the `fp32` reference.
const TAP_LIMBS: u32 = 3;

/// Stage slot of the attention-side `RMSNorm` output in the engine's tap rows
/// (`src/engine/step/tap.rs:STAGE_LN1_OUT`, private to the engine step module).
const TAP_LN1_OUT: usize = 1;
/// Stage slot of the output projection's input (`STAGE_ATTN_OUT`).
const TAP_ATTN_OUT: usize = 2;
/// Stage slot of the FFN-side `RMSNorm` output (`STAGE_LN2_OUT`).
const TAP_LN2_OUT: usize = 3;
/// Stage slot of the `SwiGLU` product (`STAGE_M`).
const TAP_M: usize = 4;
/// Stage slots in one layer's tap (`STAGE_COUNT`): the rows this capture reads plus the attention
/// rows, the `SwiGLU` gate/up pre-activations it does not read, and the backward pass's
/// post-attention residual and layer output.
const TAP_STAGES: usize = 12;

/// A refusal from the Gram capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The tape or configuration disagrees with the accumulator's geometry.
    Shape(String),
    /// A length product left `usize`, a word sum left `i64`, or a rounded product left the grid.
    Overflow,
}

/// What [`Grams::finish`] hands back for one matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalisation {
    /// The raw scatter `Σ_t x_t x_tᵀ`, unscaled.
    Sum,
    /// `(Σ_t x_t x_tᵀ) / tokens`, the reference `Gram.result(nt)` (`rstar_fast.py:247-254`).
    Mean,
    /// `(Σ_t x_t x_tᵀ) / mean(diag) + eps·I` — the mean-diagonal-1 normalisation `gq_capture` applies
    /// to its output-gradient `HO` (`fold_transfer.py:1322-1323`).
    MeanDiagonal {
        /// Added to every diagonal entry after the division, as a `Q32.32` word (`0.01` is
        /// `42_949_673`).
        eps: i64,
    },
}

/// One layer's four activation Grams, each row-major `w × w` (`[LayerGrams]`'s field docs name the
/// width), plus the token count summed into them.
///
/// The entries are the accumulator's words rendered once by `from_grid`; `tests/learn_fold_basis.rs`
/// converts them straight back with `to_grid`, so word-valued fields are what that consumer can take.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGrams {
    /// `Σ_t ln1_out_t ln1_out_tᵀ`, row-major `n_embd × n_embd`.
    pub qkv: Vec<f64>,
    /// `Σ_t attn_out_t attn_out_tᵀ`, row-major `q_out × q_out`.
    pub o: Vec<f64>,
    /// `Σ_t ln2_out_t ln2_out_tᵀ`, row-major `n_embd × n_embd`.
    pub gate_up: Vec<f64>,
    /// `Σ_t m_t m_tᵀ`, row-major `n_ff × n_ff`.
    pub down: Vec<f64>,
    /// Tokens accumulated into this layer's Grams.
    pub tokens: u64,
}

/// One layer's four Q32.32 accumulators, upper triangle only, plus the token count.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LayerGramQ {
    qkv: Vec<i64>,
    o: Vec<i64>,
    gate_up: Vec<i64>,
    down: Vec<i64>,
    tokens: u64,
}

/// The per-layer activation-Gram accumulator for one calibration pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grams {
    /// Per trunk layer, in forward order, as Q32.32 words.
    layers: Vec<LayerGramQ>,
    n_embd: usize,
    q_out: usize,
    n_ff: usize,
}

impl Grams {
    /// The zero accumulator for `cfg`'s geometry.
    ///
    /// # Errors
    /// [`CaptureError::Shape`] for a zero model dimension; [`CaptureError::Overflow`] when
    /// `n_head · head_dim` or a squared width leaves `usize`.
    ///
    /// # Complexity
    /// `O(n_layer)` time; space `O(n_layer·(n_embd² + q_out² + n_ff²))` `i64` words.
    pub fn new(cfg: &Config) -> Result<Grams, CaptureError> {
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        if cfg.n_layer == 0 || n_embd == 0 || n_ff == 0 || cfg.n_head == 0 || cfg.head_dim == 0 {
            return Err(CaptureError::Shape("zero model dimension".to_string()));
        }
        let q_out = cfg.q_out().map_err(|_| CaptureError::Overflow)?;
        let e2 = square(n_embd)?;
        let q2 = square(q_out)?;
        let f2 = square(n_ff)?;
        let mut layers: Vec<LayerGramQ> = Vec::with_capacity(cfg.n_layer);
        let mut l = 0usize;
        while l < cfg.n_layer {
            layers.push(LayerGramQ {
                qkv: vec![0; e2],
                o: vec![0; q2],
                gate_up: vec![0; e2],
                down: vec![0; f2],
                tokens: 0,
            });
            l += 1;
        }
        Ok(Grams {
            layers,
            n_embd,
            q_out,
            n_ff,
        })
    }

    /// Fold `tape`'s four per-layer linear inputs into the accumulator, upper triangle only.
    ///
    /// # Errors
    /// [`CaptureError::Shape`] when `cfg` disagrees with the geometry [`Grams::new`] fixed, when
    /// `tape.layers.len() != cfg.n_layer`, or when a stage buffer is not `tape.t · w`;
    /// [`CaptureError::Overflow`] when a length product, a rounded `i128` product or a word sum
    /// leaves its range.
    ///
    /// # Complexity
    /// Unit cost `t·w·(w+1)/2` grid-quantised multiply-adds per matrix, once per `(layer, matrix)`
    /// per call: `≥ 10⁶` word ops at any model width, so each matrix is split by upper-triangle row
    /// across [`width_global`] pool tasks. `O(1)` scratch.
    pub fn accumulate(&mut self, tape: &Tape, cfg: &Config) -> Result<(), CaptureError> {
        let q_out = cfg.q_out().map_err(|_| CaptureError::Overflow)?;
        if cfg.n_embd != self.n_embd
            || q_out != self.q_out
            || cfg.n_ff != self.n_ff
            || cfg.n_layer != self.layers.len()
        {
            return Err(CaptureError::Shape(format!(
                "config {}×{}×{} / {} layers != Grams {}×{}×{} / {} layers",
                cfg.n_embd,
                q_out,
                cfg.n_ff,
                cfg.n_layer,
                self.n_embd,
                self.q_out,
                self.n_ff,
                self.layers.len()
            )));
        }
        if tape.layers.len() != cfg.n_layer {
            return Err(CaptureError::Shape(format!(
                "tape has {} layers, config {}",
                tape.layers.len(),
                cfg.n_layer
            )));
        }
        let t = tape.t;
        let mut l = 0usize;
        while l < self.layers.len() {
            let acc = &mut self.layers[l];
            let lt = &tape.layers[l];
            expect(t, self.n_embd, lt.ln1_out.len(), "ln1_out")?;
            expect(t, self.q_out, lt.attn_out.len(), "attn_out")?;
            expect(t, self.n_embd, lt.ln2_out.len(), "ln2_out")?;
            expect(t, self.n_ff, lt.m.len(), "m")?;
            fold_rows(&mut acc.qkv, &lt.ln1_out, t, self.n_embd)?;
            fold_rows(&mut acc.o, &lt.attn_out, t, self.q_out)?;
            fold_rows(&mut acc.gate_up, &lt.ln2_out, t, self.n_embd)?;
            fold_rows(&mut acc.down, &lt.m, t, self.n_ff)?;
            let rows = u64::try_from(t).map_err(|_| CaptureError::Overflow)?;
            acc.tokens = acc.tokens.checked_add(rows).ok_or(CaptureError::Overflow)?;
            l += 1;
        }
        Ok(())
    }

    /// Fold the engine's `prefill_tap_cols` rows into the accumulator.
    ///
    /// `rows` is `[layer][stage][column][width]`, the four stage slots [`TAP_LN1_OUT`],
    /// [`TAP_ATTN_OUT`], [`TAP_LN2_OUT`] and [`TAP_M`] the linear inputs. Each stage's columns are
    /// stacked row-major and folded through [`add_gram_stage`], so every Gram executes on
    /// [`syrk_q32`] (AGENTS.md §0 K2).
    ///
    /// # Errors
    /// [`CaptureError::Shape`] when the layer count or a stage width disagrees with [`Grams::new`]'s
    /// geometry, when the four stage column counts differ, or when a stage is empty (a
    /// single-column prefill records no `m` row, so it cannot feed this path);
    /// [`CaptureError::Overflow`] when a limb product leaves its range or a word leaves `i64`.
    ///
    /// # Complexity
    /// `O(n_layer·t·(n_embd + q_out + n_ff))` to stack the rows plus the limb product's
    /// `O(t·w²·limbs/2)` per matrix; space `O(t·n_ff)` for the widest stage.
    pub fn accumulate_tap(&mut self, rows: &[Vec<Vec<Vec<i64>>>]) -> Result<(), CaptureError> {
        if rows.len() != self.layers.len() {
            return Err(CaptureError::Shape(format!(
                "tap has {} layers, Grams {}",
                rows.len(),
                self.layers.len()
            )));
        }
        let n_embd = self.n_embd;
        let q_out = self.q_out;
        let n_ff = self.n_ff;
        // One layer per step; the loop carries the first refusal out beside its index (no `?` or
        // early return inside a loop, the pinned Aeneas shape).
        let mut res: Result<(), CaptureError> = Ok(());
        let mut l = 0usize;
        while l < rows.len() && res.is_ok() {
            let stages = &rows[l];
            if stages.len() == TAP_STAGES {
                let t = stages[TAP_LN1_OUT].len();
                if t == 0 {
                    res = Err(CaptureError::Shape(format!("layer {l}: no tap columns")));
                } else if stages[TAP_ATTN_OUT].len() != t
                    || stages[TAP_LN2_OUT].len() != t
                    || stages[TAP_M].len() != t
                {
                    res = Err(CaptureError::Shape(format!(
                        "layer {l}: stage columns {} / {} / {} / {} differ",
                        t,
                        stages[TAP_ATTN_OUT].len(),
                        stages[TAP_LN2_OUT].len(),
                        stages[TAP_M].len()
                    )));
                } else {
                    let acc = &mut self.layers[l];
                    let mut r = add_gram_stage(&mut acc.qkv, &stages[TAP_LN1_OUT], n_embd, t);
                    if r.is_ok() {
                        r = add_gram_stage(&mut acc.o, &stages[TAP_ATTN_OUT], q_out, t);
                    }
                    if r.is_ok() {
                        r = add_gram_stage(&mut acc.gate_up, &stages[TAP_LN2_OUT], n_embd, t);
                    }
                    if r.is_ok() {
                        r = add_gram_stage(&mut acc.down, &stages[TAP_M], n_ff, t);
                    }
                    if r.is_ok() {
                        r = match u64::try_from(t) {
                            Ok(rows_t) => match acc.tokens.checked_add(rows_t) {
                                Some(n) => {
                                    acc.tokens = n;
                                    Ok(())
                                }
                                None => Err(CaptureError::Overflow),
                            },
                            Err(_) => Err(CaptureError::Overflow),
                        };
                    }
                    res = r;
                }
            } else {
                res = Err(CaptureError::Shape(format!(
                    "layer {l}: {} tap stages != {TAP_STAGES}",
                    stages.len()
                )));
            }
            l += 1;
        }
        res
    }

    /// The mirrored, normalised copy of the accumulator, each word converted to `f64` once here.
    ///
    /// # Errors
    /// [`CaptureError::Shape`] for [`Normalisation::Mean`] or [`Normalisation::MeanDiagonal`] on a
    /// zero token count; [`CaptureError::Overflow`] when a scaled word or `eps` leaves its range.
    ///
    /// # Complexity
    /// `O(n_layer·(n_embd² + q_out² + n_ff²))` time, one `i64` copy of the accumulator.
    pub fn finish(&self, normalisation: Normalisation) -> Result<Vec<LayerGrams>, CaptureError> {
        let mut out = Vec::with_capacity(self.layers.len());
        let mut l = 0usize;
        let mut res: Result<(), CaptureError> = Ok(());
        while l < self.layers.len() && res.is_ok() {
            match finish_layer(
                &self.layers[l],
                self.n_embd,
                self.q_out,
                self.n_ff,
                normalisation,
            ) {
                Ok(layer) => out.push(layer),
                Err(e) => res = Err(e),
            }
            l += 1;
        }
        res?;
        Ok(out)
    }

    /// Add another shard's accumulator into this one: the exact merge of two calibration shards.
    ///
    /// Every word is an integer sum, so the merge is associative and commutative, and
    /// `merge(accumulate(A), accumulate(B))` is bit-identical to accumulating `A` then `B` into one
    /// accumulator, whatever the shard count or arrival order. Nothing is rounded. `other` is
    /// untouched; `self` is untouched on error (the sums are checked before any word is written).
    ///
    /// # Errors
    /// [`CaptureError::Shape`] when the two geometries differ; [`CaptureError::Overflow`] when a
    /// word or the token count would leave its range.
    ///
    /// # Complexity
    /// Two passes over the accumulator words: time `O(n_layer·(n_embd² + q_out² + n_ff²))`, space `O(1)`.
    pub fn merge(&mut self, other: &Grams) -> Result<(), CaptureError> {
        if self.n_embd != other.n_embd
            || self.q_out != other.q_out
            || self.n_ff != other.n_ff
            || self.layers.len() != other.layers.len()
        {
            return Err(CaptureError::Shape("merge: geometries differ".to_string()));
        }
        let mut fits = true;
        let mut l = 0usize;
        while l < self.layers.len() && fits {
            let (a, b) = (&self.layers[l], &other.layers[l]);
            fits = a.tokens.checked_add(b.tokens).is_some()
                && words_fit(&a.qkv, &b.qkv)
                && words_fit(&a.o, &b.o)
                && words_fit(&a.gate_up, &b.gate_up)
                && words_fit(&a.down, &b.down);
            l += 1;
        }
        if !fits {
            return Err(CaptureError::Overflow);
        }
        let mut l = 0usize;
        while l < self.layers.len() {
            let b = &other.layers[l];
            let a = &mut self.layers[l];
            a.tokens = a.tokens.wrapping_add(b.tokens);
            words_add(&mut a.qkv, &b.qkv);
            words_add(&mut a.o, &b.o);
            words_add(&mut a.gate_up, &b.gate_up);
            words_add(&mut a.down, &b.down);
            l += 1;
        }
        Ok(())
    }

    /// The shard's wire form: a header (`APGR1`, then `n_layer`, `n_embd`, `q_out`, `n_ff` as
    /// little-endian `u64`), then per layer the token count and the four upper triangles
    /// (`i ≤ j`, row-major) as little-endian `i64` Q32.32 words. The lower triangle is never
    /// written by [`Grams::accumulate_tap`], so it is not sent.
    ///
    /// # Complexity
    /// Time and space `O(n_layer·(n_embd² + q_out² + n_ff²)/2)` words.
    #[must_use]
    pub fn to_wire(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"APGR1");
        for v in [self.layers.len(), self.n_embd, self.q_out, self.n_ff] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        let mut l = 0usize;
        while l < self.layers.len() {
            let a = &self.layers[l];
            out.extend_from_slice(&a.tokens.to_le_bytes());
            put_upper(&mut out, &a.qkv, self.n_embd);
            put_upper(&mut out, &a.o, self.q_out);
            put_upper(&mut out, &a.gate_up, self.n_embd);
            put_upper(&mut out, &a.down, self.n_ff);
            l += 1;
        }
        out
    }

    /// Read a shard written by [`Grams::to_wire`].
    ///
    /// # Errors
    /// [`CaptureError::Shape`] for a wrong magic, a truncated or over-long buffer, or a geometry
    /// whose size does not fit; [`CaptureError::Overflow`] when a length product leaves `usize`.
    ///
    /// # Complexity
    /// Time and space as [`Grams::to_wire`].
    pub fn from_wire(bytes: &[u8]) -> Result<Grams, CaptureError> {
        let bad = |m: &str| CaptureError::Shape(format!("wire: {m}"));
        if bytes.len() < 5 + 32 || &bytes[..5] != b"APGR1" {
            return Err(bad("magic or header"));
        }
        let rd = |at: usize| -> usize {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bytes[at..at + 8]);
            usize::try_from(u64::from_le_bytes(w)).unwrap_or(usize::MAX)
        };
        let (n_layer, n_embd, q_out, n_ff) = (rd(5), rd(13), rd(21), rd(29));
        let tri = |w: usize| -> Result<usize, CaptureError> {
            w.checked_mul(w.checked_add(1).ok_or(CaptureError::Overflow)?)
                .map(|x| x / 2)
                .ok_or(CaptureError::Overflow)
        };
        let per_layer = 8usize
            .checked_add(
                tri(n_embd)?
                    .checked_mul(2)
                    .and_then(|x| x.checked_add(tri(q_out).ok()?))
                    .and_then(|x| x.checked_add(tri(n_ff).ok()?))
                    .and_then(|x| x.checked_mul(8))
                    .ok_or(CaptureError::Overflow)?,
            )
            .ok_or(CaptureError::Overflow)?;
        let want = per_layer
            .checked_mul(n_layer)
            .and_then(|x| x.checked_add(37))
            .ok_or(CaptureError::Overflow)?;
        if bytes.len() != want {
            return Err(bad("length does not match the header geometry"));
        }
        let (e2, q2, f2) = (square(n_embd)?, square(q_out)?, square(n_ff)?);
        let mut at = 37usize;
        let mut layers = Vec::with_capacity(n_layer);
        let mut l = 0usize;
        while l < n_layer {
            let mut t = [0u8; 8];
            t.copy_from_slice(&bytes[at..at + 8]);
            at += 8;
            let mut lay = LayerGramQ {
                qkv: vec![0; e2],
                o: vec![0; q2],
                gate_up: vec![0; e2],
                down: vec![0; f2],
                tokens: u64::from_le_bytes(t),
            };
            get_upper(bytes, &mut at, &mut lay.qkv, n_embd);
            get_upper(bytes, &mut at, &mut lay.o, q_out);
            get_upper(bytes, &mut at, &mut lay.gate_up, n_embd);
            get_upper(bytes, &mut at, &mut lay.down, n_ff);
            layers.push(lay);
            l += 1;
        }
        Ok(Grams { layers, n_embd, q_out, n_ff })
    }
}

/// `true` when every `dst[i] + src[i]` stays in `i64`.
fn words_fit(dst: &[i64], src: &[i64]) -> bool {
    let mut i = 0usize;
    let mut ok = dst.len() == src.len();
    while ok && i < dst.len() {
        ok = dst[i].checked_add(src[i]).is_some();
        i += 1;
    }
    ok
}

/// `dst[i] += src[i]`; the caller has already proved no word overflows.
fn words_add(dst: &mut [i64], src: &[i64]) {
    let mut i = 0usize;
    while i < dst.len() {
        dst[i] = dst[i].wrapping_add(src[i]);
        i += 1;
    }
}

/// Append the `i ≤ j` triangle of a `w × w` row-major matrix as little-endian words.
fn put_upper(out: &mut Vec<u8>, m: &[i64], w: usize) {
    let mut i = 0usize;
    while i < w {
        let mut j = i;
        while j < w {
            out.extend_from_slice(&m[i * w + j].to_le_bytes());
            j += 1;
        }
        i += 1;
    }
}

/// Read the `i ≤ j` triangle written by [`put_upper`], advancing `at`.
fn get_upper(bytes: &[u8], at: &mut usize, m: &mut [i64], w: usize) {
    let mut i = 0usize;
    while i < w {
        let mut j = i;
        while j < w {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[*at..*at + 8]);
            m[i * w + j] = i64::from_le_bytes(b);
            *at += 8;
            j += 1;
        }
        i += 1;
    }
}

/// One layer's finished Grams: the zero-token refusal, then the four mirrored matrices.
///
/// # Errors
/// [`CaptureError::Shape`] for a normalisation over zero tokens; [`CaptureError::Overflow`] from
/// [`mirror`].
///
/// # Complexity
/// `O(n_embd² + q_out² + n_ff²)`; space one layer's `f64` Grams.
fn finish_layer(
    lg: &LayerGramQ,
    n_embd: usize,
    q_out: usize,
    n_ff: usize,
    normalisation: Normalisation,
) -> Result<LayerGrams, CaptureError> {
    if lg.tokens == 0 && normalisation != Normalisation::Sum {
        return Err(CaptureError::Shape(
            "normalisation over zero tokens".to_string(),
        ));
    }
    Ok(LayerGrams {
        qkv: mirror(&lg.qkv, n_embd, normalisation, lg.tokens)?,
        o: mirror(&lg.o, q_out, normalisation, lg.tokens)?,
        gate_up: mirror(&lg.gate_up, n_embd, normalisation, lg.tokens)?,
        down: mirror(&lg.down, n_ff, normalisation, lg.tokens)?,
        tokens: lg.tokens,
    })
}

/// `x · x`, or [`CaptureError::Overflow`].
fn square(x: usize) -> Result<usize, CaptureError> {
    x.checked_mul(x).ok_or(CaptureError::Overflow)
}

/// `.len() == t · w` or [`CaptureError::Shape`].
fn expect(t: usize, w: usize, len: usize, what: &str) -> Result<usize, CaptureError> {
    match t.checked_mul(w) {
        Some(n) if n == len => Ok(n),
        Some(n) => Err(CaptureError::Shape(format!(
            "{what}: {len} != {t}·{w} = {n}"
        ))),
        None => Err(CaptureError::Overflow),
    }
}

/// The pool task of [`fold_rows`]: one contiguous band of upper-triangle rows.
struct FoldRows<'a> {
    x: &'a [i64],
    t: usize,
    w: usize,
    first: &'a FirstErr<usize, CaptureError>,
}

impl ChunkTask<i64> for FoldRows<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let row0 = start / self.w;
        let band = piece.len() / self.w;
        if let Err(e) = fold_upper_rows(piece, self.x, row0, band, self.t, self.w) {
            self.first.record(row0, e);
        }
    }
}

/// Split the `w × w` accumulator into contiguous row bands and fold each through the pool.
///
/// Every entry `(i, j)`, `i ≤ j`, is written by the one task owning row `i` and summed over the
/// tokens in ascending order there, so the result is bit-identical to the serial body.
///
/// # Complexity
/// Time `t·w·(w+1)/2` multiply-adds, once per matrix; space `O(1)` beyond the in-place bands.
fn fold_rows(acc: &mut [i64], x: &[i64], t: usize, w: usize) -> Result<(), CaptureError> {
    let chunk_rows = w.div_ceil(width_global()).max(1);
    let first: FirstErr<usize, CaptureError> = FirstErr::new();
    chunks_mut_global(
        acc,
        chunk_rows * w,
        &FoldRows {
            x,
            t,
            w,
            first: &first,
        },
    );
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Add the engine stage `cols` (one Q32.32 row per token, each `w` wide) to the `w × w`
/// upper-triangle accumulator `acc`, through [`syrk_q32`] on the transposed operand.
///
/// # Errors
/// [`CaptureError::Shape`] when a column is not `w` wide or `acc` is not `w²`;
/// [`CaptureError::Overflow`] on a limb refusal or a word sum leaving `i64`.
///
/// # Complexity
/// `O(t·w)` words copied into the transpose plus [`syrk_q32`]'s `O(w²·t·limbs/2)`; space `O(t·w)`.
fn add_gram_stage(
    acc: &mut [i64],
    cols: &[Vec<i64>],
    w: usize,
    t: usize,
) -> Result<(), CaptureError> {
    let d = w.checked_mul(w).ok_or(CaptureError::Overflow)?;
    if acc.len() != d {
        return Err(CaptureError::Shape(format!("gram {d} != {}", acc.len())));
    }
    let need = t.checked_mul(w).ok_or(CaptureError::Overflow)?;
    // Every column must be the stage's width; the first bad index is reported after the loop.
    let mut bad: Option<usize> = None;
    let mut r = 0usize;
    while r < t && bad.is_none() {
        if cols[r].len() != w {
            bad = Some(r);
        }
        r += 1;
    }
    if let Some(r) = bad {
        return Err(CaptureError::Shape(format!(
            "tap column {r}: {} != {w}",
            cols[r].len()
        )));
    }
    // `syrk_q32(a, m, n)` is `a·aᵀ` of an `m × n` operand, so the width Gram `Σ_t x_t x_tᵀ` is the
    // transpose `w × t` (one row per coordinate) contracted against itself.
    let mut xt: Vec<i64> = vec![0; need];
    let mut c = 0usize;
    while c < w {
        let mut r = 0usize;
        while r < t {
            xt[c * t + r] = cols[r][c];
            r += 1;
        }
        c += 1;
    }
    let gram = match syrk_q32(&xt, w, t, TAP_LIMBS) {
        Ok(g) => g,
        Err(LimbError::Shape) => return Err(CaptureError::Shape(String::from("limb shape"))),
        Err(LimbError::Range | LimbError::NoKernel) => return Err(CaptureError::Overflow),
    };
    if gram.len() != d {
        return Err(CaptureError::Shape(format!(
            "limb gram {} != {d}",
            gram.len()
        )));
    }
    let mut i = 0usize;
    let mut res: Result<(), CaptureError> = Ok(());
    while i < w && res.is_ok() {
        let mut j = i;
        while j < w && res.is_ok() {
            let e = i * w + j;
            match acc[e].checked_add(gram[e]) {
                Some(s) => acc[e] = s,
                None => res = Err(CaptureError::Overflow),
            }
            j += 1;
        }
        i += 1;
    }
    res
}

/// Add `Σ_{r<t} x_r x_rᵀ` to the upper triangle of rows `row0 .. row0 + band`, `x` row-major
/// `t × w`.
///
/// # Errors
/// [`CaptureError::Overflow`] when a rounded product leaves the grid or a row sum leaves `i64`.
///
/// # Complexity
/// Time `O(band·t·w)`; space `O(1)`.
fn fold_upper_rows(
    acc: &mut [i64],
    x: &[i64],
    row0: usize,
    band: usize,
    t: usize,
    w: usize,
) -> Result<(), CaptureError> {
    for i in 0..band {
        let gi = row0 + i;
        for j in gi..w {
            let mut s: i64 = 0;
            for r in 0..t {
                let p = i128::from(x[r * w + gi]) * i128::from(x[r * w + j]);
                s = s
                    .checked_add(round_q32_even(p).map_err(|_| CaptureError::Overflow)?)
                    .ok_or(CaptureError::Overflow)?;
            }
            if let Some(slot) = acc.get_mut(i * w + j) {
                *slot = slot.checked_add(s).ok_or(CaptureError::Overflow)?;
            }
        }
    }
    Ok(())
}

/// The sum of the diagonal words of the row-major `d × d` accumulator `a`, in `i128`.
///
/// # Complexity
/// `O(d)`.
fn diagonal_sum(a: &[i64], d: usize) -> i128 {
    let mut s: i128 = 0;
    let mut i = 0usize;
    while i < d {
        let row = i * d;
        if let Some(&v) = a.get(row + i) {
            s += i128::from(v);
        }
        i += 1;
    }
    s
}

/// `n / d` rounded to nearest, ties away from zero; `d > 0`, `n ≥ 0`. `q32` has no general
/// divide-and-round, so this is the one such rule in the file (it is `grad::head`'s private one).
fn round_div(n: i128, d: i128) -> i128 {
    if d == 0 {
        return 0;
    }
    let (q, r) = (n / d, n % d);
    if 2 * r >= d {
        q + 1
    } else {
        q
    }
}

/// The mirrored, scaled copy of the upper-triangle accumulator `a` (row-major `d × d`): entry
/// `(i, j)` is `round_div(a[max·d+min]·num, den)` words, plus `diag_add` on the diagonal. `num`,
/// `den` and `diag_add` follow [`Normalisation`]; a zero denominator (a zero mean diagonal) has the
/// zero quotient.
///
/// # Errors
/// [`CaptureError::Shape`] when `a.len() != d²`; [`CaptureError::Overflow`] when a word leaves
/// `i64`.
///
/// # Complexity
/// `O(d²)` time and space: one division and one conversion per entry.
fn mirror(
    a: &[i64],
    d: usize,
    normalisation: Normalisation,
    tokens: u64,
) -> Result<Vec<f64>, CaptureError> {
    if a.len() != square(d)? {
        return Err(CaptureError::Shape(format!("gram {} != {d}²", a.len())));
    }
    // `H/mean(diag)` as a grid word is `round(word·d·2³²/Σ diag word)`.
    let (num, den, diag_add) = match normalisation {
        Normalisation::Sum => (1, 1, 0),
        Normalisation::Mean => (1, i128::from(tokens), 0),
        Normalisation::MeanDiagonal { eps } => {
            let num = match i128::try_from(d) {
                Ok(d) => match d.checked_shl(32) {
                    Some(n) => n,
                    None => return Err(CaptureError::Overflow),
                },
                Err(_) => return Err(CaptureError::Overflow),
            };
            (num, diagonal_sum(a, d), eps)
        }
    };
    let mut out = vec![0.0f64; a.len()];
    let mut i = 0usize;
    let mut res: Result<(), CaptureError> = Ok(());
    while i < d && res.is_ok() {
        let mut j = i;
        while j < d && res.is_ok() {
            let mut w = if den <= 0 {
                0
            } else {
                round_div(i128::from(a[i * d + j]) * num, den)
            };
            if i == j {
                w += i128::from(diag_add);
            }
            match i64::try_from(w) {
                Ok(v) => {
                    out[i * d + j] = from_grid(v);
                    out[j * d + i] = from_grid(v);
                }
                Err(_) => res = Err(CaptureError::Overflow),
            }
            j += 1;
        }
        i += 1;
    }
    res?;
    Ok(out)
}
