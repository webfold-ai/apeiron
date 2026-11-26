//! Whole-model Qwen2 backward in Q32.32 (W19 wave 2) over the engine's exact forward tape.
//!
//! The forward is the engine's: `Engine::prefill_tap_cols` under `KernelMode::Exact`, materialised
//! as a [`Tape`] by
//! [`engine_tape_mode`](crate::learn::fold::stats::hocapture::engine_tape_mode). Its per-layer
//! schedule — `attn_norm` → Q/K/V projection with the Qwen2 attention bias → `RoPE` → causal GQA
//! attention → O projection → residual; then `ffn_norm` → `SwiGLU` → down → residual; then
//! `output_norm` → head — is `Engine::compose_cols`'s (`src/engine/step/cols.rs`) and
//! `Engine::step_logits`'s exact path. This module holds the backward over that tape, the checked
//! Q32.32 stage helpers the extracted word bodies call ([`add_flat`], [`col_sum`],
//! [`rmsnorm_bwd_rows`], [`rope_rows_bwd`], [`embedding_grad`]), and [`dense`]'s and
//! [`attn`]'s per-op kernels.
//!
//! Every weight is an `i64` Q32.32 row-major buffer; the engine-reading load lives in [`super::load`],
//! which holds the row dequantizer and the resolved decode-vector reads. This module is the
//! extraction unit's subject and reads no `GGUF`.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path. The test module asserts the backward's
//! analytic gradient against a central difference of the KL loss whose logits and tape are the
//! engine's `KernelMode::Exact` prefill. The integer helpers are translated by Aeneas through
//! `proof/extract/grad_solve_port.rs`; `backward` is not (it builds a host `f64` `RopeTable`). No Lean
//! refinement links this file yet (`proved` is open).
//!
//! # Trust boundary
//!
//! The backward differentiates the Q32.32 forward the engine runs; the engine's `Exact` prefill is
//! the oracle. `i128` accumulation and the per-op rounding are the per-op modules' contracts. Not
//! covered here: a float carrier's attention, partial rotary (`rot_dims < head_dim`), MoE/routed
//! experts, the affine mixer's backward, the per-head QK `RMSNorm`, the attention output gate,
//! Hadamard-folded weights and the stochastic sampler. The residual and bias adds are checked `i64`
//! adds; every overflow returns [`ModelError::Overflow`]. No float and no `unsafe` appear after the
//! load boundary.
//!
//! The load path is wider than the backward: [`super::load::layer_from_engine`] names a layer's fitted
//! linears by `GGUF` tensor suffix and shapes them from the engine's own configuration, so an affine
//! layer's `attn_qkv`/`attn_gate`/`ssm_out` set is available to a writer that does not yet fit it.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the f64→Q32.32 boundary is range-checked immediately before the `as i64`; the \
              single-letter names are the schedule's symbols (`t`, `e`, `hd`, `r`, `o`, `i`) and \
              the gradient carries `dh`/`dh2`/`dln1`/`dln2`/`dh_attn` name the same stage on the \
              backward and the forward; the layer loop and the forward/backward are each one linear \
              composition"
)]

use super::attn::{self, AttnError};
use super::dense::{self, DenseError};
use crate::engine::num::fp::F64;
use crate::engine::{EngineError, RopeTable};

/// A refused whole-model forward or backward evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// A dimension, a buffer length or a token id disagrees with the model geometry.
    Shape(String),
    /// A checked add (the residual or bias accumulation) left `i64`.
    Overflow,
    /// A named tensor is absent from the engine directory.
    Missing(String),
    /// The engine refused a payload read or a row dequantization.
    Engine(EngineError),
    /// The configuration cannot form the model (a non-finite or out-of-range `rms_eps`).
    Config(String),
    /// A dense forward/backward kernel refused.
    Dense(DenseError),
    /// An attention forward/backward kernel refused.
    Attn(AttnError),
}

impl From<DenseError> for ModelError {
    fn from(e: DenseError) -> Self {
        ModelError::Dense(e)
    }
}

impl From<AttnError> for ModelError {
    fn from(e: AttnError) -> Self {
        ModelError::Attn(e)
    }
}

/// The resolved model geometry plus the Q32.32 `RMSNorm` epsilon the engine's step uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Trunk layer count.
    pub n_layer: usize,
    /// Query heads.
    pub n_head: usize,
    /// KV heads (GQA); `n_head % n_head_kv == 0`.
    pub n_head_kv: usize,
    /// Residual width.
    pub n_embd: usize,
    /// Feed-forward width.
    pub n_ff: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// `round(rms_eps · 2^32)`, the engine's `eps_fixed` value.
    pub eps: i64,
    /// `rope_theta`, the `RoPE` table's load input.
    pub rope_theta: f64,
}

impl Config {
    /// `q_out = n_head · head_dim`, or [`ModelError::Overflow`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Overflow`] when the product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn q_out(&self) -> Result<usize, ModelError> {
        self.n_head
            .checked_mul(self.head_dim)
            .ok_or(ModelError::Overflow)
    }

    /// `kv_out = n_head_kv · head_dim`, or [`ModelError::Overflow`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Overflow`] when the product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn kv_out(&self) -> Result<usize, ModelError> {
        self.n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(ModelError::Overflow)
    }

    /// The dimension sanity the forward and backward both assume.
    ///
    /// # Errors
    ///
    /// [`ModelError::Shape`] for a zero model dimension or GQA head counts that do not divide;
    /// [`ModelError::Overflow`] from [`Config::q_out`] or [`Config::kv_out`].
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn checked(&self) -> Result<(), ModelError> {
        if self.n_layer == 0 || self.n_embd == 0 || self.n_ff == 0 || self.head_dim == 0 {
            return Err(ModelError::Shape("zero model dimension".to_string()));
        }
        if self.n_head_kv == 0 || !self.n_head.is_multiple_of(self.n_head_kv) {
            return Err(ModelError::Shape("GQA head counts".to_string()));
        }
        let _ = self.q_out()?;
        let _ = self.kv_out()?;
        Ok(())
    }
}

/// One layer's Q32.32 parameters, row-major `o × i` with the input dimension fastest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerWeights {
    /// `attn_norm.weight`, `n_embd` wide.
    pub attn_norm: Vec<i64>,
    /// `attn_q.weight`, `(n_head·head_dim) × n_embd`.
    pub wq: Vec<i64>,
    /// `attn_q.bias`, `n_head·head_dim` wide, or `None`.
    pub bq: Option<Vec<i64>>,
    /// `attn_k.weight`, `(n_head_kv·head_dim) × n_embd`.
    pub wk: Vec<i64>,
    /// `attn_k.bias`, `n_head_kv·head_dim` wide, or `None`.
    pub bk: Option<Vec<i64>>,
    /// `attn_v.weight`, `(n_head_kv·head_dim) × n_embd`.
    pub wv: Vec<i64>,
    /// `attn_v.bias`, `n_head_kv·head_dim` wide, or `None`.
    pub bv: Option<Vec<i64>>,
    /// `attn_output.weight`, `n_embd × (n_head·head_dim)`.
    pub wo: Vec<i64>,
    /// `ffn_norm.weight`, `n_embd` wide.
    pub ffn_norm: Vec<i64>,
    /// `ffn_gate.weight`, `n_ff × n_embd`.
    pub w_gate: Vec<i64>,
    /// `ffn_up.weight`, `n_ff × n_embd`.
    pub w_up: Vec<i64>,
    /// `ffn_down.weight`, `n_embd × n_ff`.
    pub w_down: Vec<i64>,
}

/// The whole model's Q32.32 parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Weights {
    /// Per trunk layer.
    pub layers: Vec<LayerWeights>,
    /// `output_norm.weight`, `n_embd` wide.
    pub output_norm: Vec<i64>,
    /// `token_embd.weight`, `vocab × n_embd`.
    pub token_embd: Vec<i64>,
    /// The head (`output.weight` when present, else tied `token_embd.weight`), `vocab × n_embd`.
    pub lm_head: Vec<i64>,
}

/// `.len() == t · w` or [`ModelError::Shape`].
///
/// # Complexity
/// `O(1)`.
pub(crate) fn expect(t: usize, w: usize, len: usize, what: &str) -> Result<usize, ModelError> {
    match t.checked_mul(w) {
        Some(n) if n == len => Ok(n),
        Some(n) => Err(ModelError::Shape(format!("{what}: {len} != {t}·{w} = {n}"))),
        None => Err(ModelError::Overflow),
    }
}

/// Checked `a[i] += b[i]` over equal-width rows.
///
/// # Complexity
/// `O(len)`.
pub(crate) fn add_flat(a: &mut [i64], b: &[i64]) -> Result<(), ModelError> {
    if a.len() != b.len() {
        return Err(ModelError::Shape("add width".to_string()));
    }
    // Flag loop: a `?` inside a loop has no fixed point in the pinned Charon/Aeneas fragment, so
    // the first refusal is carried in `err` and one `return` follows the sweep. The element order
    // and the first-refusal order are unchanged.
    let mut err: Option<ModelError> = None;
    let mut i = 0usize;
    while i < a.len() && err.is_none() {
        match a[i].checked_add(b[i]) {
            Some(v) => a[i] = v,
            None => err = Some(ModelError::Overflow),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The column sums of a `t × w` row-major buffer: `out[c] = Σ_r x[r·w + c]`, checked.
///
/// # Complexity
/// `O(t·w)`.
pub(crate) fn col_sum(x: &[i64], t: usize, w: usize) -> Result<Vec<i64>, ModelError> {
    expect(t, w, x.len(), "col_sum")?;
    let mut out = vec![0i64; w];
    // Flag loops: a `?` inside these nested sweeps has no fixed point in the pinned Charon/Aeneas
    // fragment, so the first refusal is carried in `err`, both loop conditions are guarded by
    // `err.is_none()`, and one `return` follows the sweep. The column order and the first-refusal
    // order are unchanged.
    let mut err: Option<ModelError> = None;
    let mut r = 0usize;
    while r < t && err.is_none() {
        let mut c = 0usize;
        while c < w && err.is_none() {
            match out[c].checked_add(x[r * w + c]) {
                Some(v) => out[c] = v,
                None => err = Some(ModelError::Overflow),
            }
            c += 1;
        }
        r += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// `RMSNorm` backward over every row of a `t × w` buffer: `(dx, Σ_r dgamma_r)`.
///
/// # Complexity
/// `O(t·w)`; one [`dense::rmsnorm_bwd`] per row.
pub(crate) fn rmsnorm_bwd_rows(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
    t: usize,
    w: usize,
) -> Result<(Vec<i64>, Vec<i64>), ModelError> {
    expect(t, w, x.len(), "rmsnorm_bwd_rows x")?;
    expect(t, w, dy.len(), "rmsnorm_bwd_rows dy")?;
    let mut dx = vec![0i64; x.len()];
    let mut dg = vec![0i64; w];
    // Flag loops as in `col_sum`; the kernel's refusal maps through the same `ModelError::Dense`
    // arm its `From<DenseError>` impl names, so the error value and the row order are unchanged.
    let mut err: Option<ModelError> = None;
    let mut r = 0usize;
    while r < t && err.is_none() {
        match dense::rmsnorm_bwd(&x[r * w..r * w + w], gamma, eps, &dy[r * w..r * w + w]) {
            Err(e) => err = Some(ModelError::Dense(e)),
            Ok((dxr, dgr)) => {
                dx[r * w..r * w + w].copy_from_slice(&dxr);
                let mut c = 0usize;
                while c < w && err.is_none() {
                    match dg[c].checked_add(dgr[c]) {
                        Some(v) => dg[c] = v,
                        None => err = Some(ModelError::Overflow),
                    }
                    c += 1;
                }
            }
        }
        r += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok((dx, dg))
}

/// `RoPE` backward on every `row`-wide row of a `t × row` gradient buffer.
///
/// # Complexity
/// `O(t·row)`; one [`attn::rope_bwd`] per token.
pub(crate) fn rope_rows_bwd(
    dy: &[i64],
    t: usize,
    row: usize,
    rope: &RopeTable,
) -> Result<Vec<i64>, ModelError> {
    expect(t, row, dy.len(), "rope_rows_bwd")?;
    let mut out = Vec::with_capacity(dy.len());
    for r in 0..t {
        let pos = u32::try_from(r).map_err(|_| ModelError::Overflow)?;
        out.extend_from_slice(&attn::rope_bwd(&dy[r * row..r * row + row], pos, rope)?);
    }
    Ok(out)
}

/// Every stage input the backward reads for one layer, captured by the engine's exact prefill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTape {
    /// Hidden before `attn_norm`, `t × n_embd`.
    pub ln1_in: Vec<i64>,
    /// `attn_norm` output (the Q/K/V input), `t × n_embd`.
    pub ln1_out: Vec<i64>,
    /// Query after `RoPE`, `t × q_out`.
    pub q: Vec<i64>,
    /// Key after `RoPE`, `t × kv_out`.
    pub k: Vec<i64>,
    /// Value, `t × kv_out`.
    pub v: Vec<i64>,
    /// Attention probabilities, `n_head × t × t`.
    pub p: Vec<i64>,
    /// Attention output, `t × q_out`.
    pub attn_out: Vec<i64>,
    /// Hidden before `ffn_norm`, `t × n_embd`.
    pub h2: Vec<i64>,
    /// `ffn_norm` output, `t × n_embd`.
    pub ln2_out: Vec<i64>,
    /// `SwiGLU` gate projection, `t × n_ff`.
    pub gate: Vec<i64>,
    /// `SwiGLU` up projection, `t × n_ff`.
    pub up: Vec<i64>,
    /// `SwiGLU` product, `t × n_ff`.
    pub m: Vec<i64>,
}

/// The forward tape: the token sequence, the per-layer stage inputs and the head's inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tape {
    /// Token count.
    pub t: usize,
    /// The token ids the forward ran (the embedding backward's scatter index).
    pub tokens: Vec<u32>,
    /// Per layer, in forward order.
    pub layers: Vec<LayerTape>,
    /// Hidden after every layer (the `output_norm` input), `t × n_embd`.
    pub final_hidden: Vec<i64>,
    /// `output_norm` output (the head input), `t × n_embd`.
    pub final_normed: Vec<i64>,
}

/// The per-layer gradient buffers, mirroring [`LayerWeights`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGrads {
    /// `dL/d attn_norm.weight`.
    pub attn_norm: Vec<i64>,
    /// `dL/d attn_q.weight`.
    pub wq: Vec<i64>,
    /// `dL/d attn_q.bias`.
    pub bq: Vec<i64>,
    /// `dL/d attn_k.weight`.
    pub wk: Vec<i64>,
    /// `dL/d attn_k.bias`.
    pub bk: Vec<i64>,
    /// `dL/d attn_v.weight`.
    pub wv: Vec<i64>,
    /// `dL/d attn_v.bias`.
    pub bv: Vec<i64>,
    /// `dL/d attn_output.weight`.
    pub wo: Vec<i64>,
    /// `dL/d ffn_norm.weight`.
    pub ffn_norm: Vec<i64>,
    /// `dL/d ffn_gate.weight`.
    pub w_gate: Vec<i64>,
    /// `dL/d ffn_up.weight`.
    pub w_up: Vec<i64>,
    /// `dL/d ffn_down.weight`.
    pub w_down: Vec<i64>,
}

/// The whole-model gradient buffers, mirroring [`Weights`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grads {
    /// Per trunk layer, in forward order.
    pub layers: Vec<LayerGrads>,
    /// `dL/d output_norm.weight`.
    pub output_norm: Vec<i64>,
    /// `dL/d token_embd.weight` (a scatter-add over the token ids).
    pub token_embd: Vec<i64>,
    /// `dL/d lm_head`.
    pub lm_head: Vec<i64>,
}

impl Grads {
    /// Every gradient buffer in one fixed order: per layer the twelve buffers, then the output
    /// norm, the embedding and the head.
    fn buffers(&self) -> Vec<&Vec<i64>> {
        let mut v: Vec<&Vec<i64>> = Vec::new();
        for l in &self.layers {
            v.extend([
                &l.attn_norm, &l.wq, &l.bq, &l.wk, &l.bk, &l.wv, &l.bv, &l.wo, &l.ffn_norm,
                &l.w_gate, &l.w_up, &l.w_down,
            ]);
        }
        v.extend([&self.output_norm, &self.token_embd, &self.lm_head]);
        v
    }

    fn buffers_mut(&mut self) -> Vec<&mut Vec<i64>> {
        let mut v: Vec<&mut Vec<i64>> = Vec::new();
        for l in &mut self.layers {
            v.extend([
                &mut l.attn_norm, &mut l.wq, &mut l.bq, &mut l.wk, &mut l.bk, &mut l.wv,
                &mut l.bv, &mut l.wo, &mut l.ffn_norm, &mut l.w_gate, &mut l.w_up, &mut l.w_down,
            ]);
        }
        v.extend([&mut self.output_norm, &mut self.token_embd, &mut self.lm_head]);
        v
    }

    /// Add another data-parallel shard's gradient into this one: the exact all-reduce.
    ///
    /// Each word is an integer sum, so the merge is associative and commutative: summing the
    /// gradients of disjoint sequences is bit-identical in any order, shard count or tree shape.
    /// `self` is untouched on error (all sums are checked before any word is written).
    ///
    /// # Errors
    /// [`ModelError::Shape`] when the buffer layout differs; [`ModelError::Overflow`] when a word
    /// would leave `i64`.
    ///
    /// # Complexity
    /// Two passes over the gradient words, space `O(1)` beyond the buffer list.
    pub fn merge(&mut self, other: &Grads) -> Result<(), ModelError> {
        let (a, b) = (self.buffers(), other.buffers());
        if a.len() != b.len() || a.iter().zip(&b).any(|(x, y)| x.len() != y.len()) {
            return Err(ModelError::Shape("Grads::merge: layouts differ".to_string()));
        }
        if a.iter().zip(&b).any(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.checked_add(*q).is_none())) {
            return Err(ModelError::Overflow);
        }
        for (x, y) in self.buffers_mut().into_iter().zip(b) {
            for (p, q) in x.iter_mut().zip(y.iter()) {
                *p = p.wrapping_add(*q);
            }
        }
        Ok(())
    }

    /// The shard's wire form: `APGD1`, the buffer count, then per buffer its length and its words,
    /// all little-endian.
    ///
    /// # Complexity
    /// Time and space `O(words)`.
    #[must_use]
    pub fn to_wire(&self) -> Vec<u8> {
        let bufs = self.buffers();
        let mut out = Vec::new();
        out.extend_from_slice(b"APGD1");
        out.extend_from_slice(&(self.layers.len() as u64).to_le_bytes());
        for b in bufs {
            out.extend_from_slice(&(b.len() as u64).to_le_bytes());
            for w in b {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Read a shard written by [`Grads::to_wire`].
    ///
    /// # Errors
    /// [`ModelError::Shape`] for a wrong magic or a truncated, over-long or inconsistent buffer.
    ///
    /// # Complexity
    /// Time and space `O(words)`.
    pub fn from_wire(bytes: &[u8]) -> Result<Grads, ModelError> {
        let bad = |m: &str| ModelError::Shape(format!("Grads wire: {m}"));
        if bytes.len() < 13 || &bytes[..5] != b"APGD1" {
            return Err(bad("magic or header"));
        }
        let rd = |at: usize| -> Option<u64> {
            let s = bytes.get(at..at + 8)?;
            let mut w = [0u8; 8];
            w.copy_from_slice(s);
            Some(u64::from_le_bytes(w))
        };
        let n_layer = usize::try_from(rd(5).ok_or_else(|| bad("header"))?).map_err(|_| bad("layers"))?;
        let mut at = 13usize;
        let mut take = |at: &mut usize| -> Result<Vec<i64>, ModelError> {
            let len = usize::try_from(rd(*at).ok_or_else(|| bad("truncated length"))?).map_err(|_| bad("length"))?;
            *at += 8;
            let end = len.checked_mul(8).and_then(|n| at.checked_add(n)).ok_or_else(|| bad("length overflow"))?;
            let body = bytes.get(*at..end).ok_or_else(|| bad("truncated words"))?;
            *at = end;
            Ok(body.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let mut layers = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            layers.push(LayerGrads {
                attn_norm: take(&mut at)?, wq: take(&mut at)?, bq: take(&mut at)?,
                wk: take(&mut at)?, bk: take(&mut at)?, wv: take(&mut at)?, bv: take(&mut at)?,
                wo: take(&mut at)?, ffn_norm: take(&mut at)?, w_gate: take(&mut at)?,
                w_up: take(&mut at)?, w_down: take(&mut at)?,
            });
        }
        let g = Grads {
            layers,
            output_norm: take(&mut at)?,
            token_embd: take(&mut at)?,
            lm_head: take(&mut at)?,
        };
        if at != bytes.len() {
            return Err(bad("trailing bytes"));
        }
        Ok(g)
    }
}

/// `dembedding[token[r], :] += dh[r, :]`, checked, over a `t × w` hidden gradient.
///
/// # Complexity
/// `O(t·w)`.
pub(crate) fn embedding_grad(
    dh: &[i64],
    tokens: &[u32],
    t: usize,
    w: usize,
    vocab: usize,
) -> Result<Vec<i64>, ModelError> {
    expect(t, w, dh.len(), "embedding_grad")?;
    let mut out = vec![0i64; vocab * w];
    let mut err: Option<ModelError> = None;
    let mut r = 0usize;
    while r < t && err.is_none() {
        match usize::try_from(tokens[r]) {
            Err(_) => err = Some(ModelError::Overflow),
            Ok(id) => {
                if id >= vocab {
                    err = Some(ModelError::Shape(format!("token {} >= vocab", tokens[r])));
                } else {
                    let mut c = 0usize;
                    while c < w && err.is_none() {
                        match out[id * w + c].checked_add(dh[r * w + c]) {
                            Some(v) => out[id * w + c] = v,
                            None => err = Some(ModelError::Overflow),
                        }
                        c += 1;
                    }
                }
            }
        }
        r += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// The whole-model backward: `dL/dW` for every weight from the forward [`Tape`] and the logit
/// gradient `dlogits` (`t × vocab`).
///
/// # Errors
///
/// [`ModelError::Shape`] when the tape or `dlogits` disagrees with the geometry;
/// [`ModelError::Overflow`] on a checked add or a width mismatch's length product; and the
/// [`ModelError::Dense`] and [`ModelError::Attn`] refusals of the kernels, as in [`forward`].
///
/// # Complexity
/// Mirrors [`forward`]'s `O(t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim)` per layer plus the head.
pub fn backward(
    w: &Weights,
    cfg: &Config,
    tape: &Tape,
    dlogits: &[i64],
) -> Result<Grads, ModelError> {
    cfg.checked()?;
    let t = tape.t;
    let e = cfg.n_embd;
    let q_out = cfg.q_out()?;
    let kv_out = cfg.kv_out()?;
    let n_ff = cfg.n_ff;
    if tape.layers.len() != cfg.n_layer || tape.tokens.len() != t {
        return Err(ModelError::Shape("tape shape".to_string()));
    }
    expect(t, cfg.vocab, dlogits.len(), "dlogits")?;
    let rope = RopeTable::new(F64::from_f64(cfg.rope_theta), cfg.head_dim)
        .map_err(|err| ModelError::Attn(AttnError::Rope(err)))?;

    let d_normed = dense::linear_bwd_input(dlogits, &w.lm_head, t, e, cfg.vocab)?;
    let lm_head = dense::linear_bwd_weight(dlogits, &tape.final_normed, t, e, cfg.vocab)?;
    let (mut dh, output_norm) =
        rmsnorm_bwd_rows(&tape.final_hidden, &w.output_norm, cfg.eps, &d_normed, t, e)?;

    let mut grads: Vec<LayerGrads> = Vec::with_capacity(cfg.n_layer);
    for l in (0..cfg.n_layer).rev() {
        let lw = &w.layers[l];
        let lt = &tape.layers[l];
        // h3 = h2 + down(m): the down branch's gradient, and the residual identity that carries it
        // into h2.
        let dd = dense::linear_bwd_input(&dh, &lw.w_down, t, n_ff, e)?;
        let w_down = dense::linear_bwd_weight(&dh, &lt.m, t, n_ff, e)?;
        let (dgate, dup) = dense::swiglu_bwd(&lt.gate, &lt.up, &dd)?;
        let dln2_g = dense::linear_bwd_input(&dgate, &lw.w_gate, t, e, n_ff)?;
        let w_gate = dense::linear_bwd_weight(&dgate, &lt.ln2_out, t, e, n_ff)?;
        let dln2_u = dense::linear_bwd_input(&dup, &lw.w_up, t, e, n_ff)?;
        let w_up = dense::linear_bwd_weight(&dup, &lt.ln2_out, t, e, n_ff)?;
        let mut dln2 = dln2_g;
        add_flat(&mut dln2, &dln2_u)?;
        let (dh2_ffn, ffn_norm) = rmsnorm_bwd_rows(&lt.h2, &lw.ffn_norm, cfg.eps, &dln2, t, e)?;
        // The residual identity: h3's gradient flows unchanged into h2, plus the ffn-norm branch.
        let mut dh2 = dh;
        add_flat(&mut dh2, &dh2_ffn)?;
        let d_attn_out = dense::linear_bwd_input(&dh2, &lw.wo, t, q_out, e)?;
        let wo = dense::linear_bwd_weight(&dh2, &lt.attn_out, t, q_out, e)?;
        let (dq_rope, dk_rope, dv) = attn::attn_bwd(
            &lt.q,
            &lt.k,
            &lt.v,
            &lt.p,
            &d_attn_out,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
        )?;
        let dq = rope_rows_bwd(&dq_rope, t, q_out, &rope)?;
        let dk = rope_rows_bwd(&dk_rope, t, kv_out, &rope)?;
        let bq = col_sum(&dq, t, q_out)?;
        let bk = col_sum(&dk, t, kv_out)?;
        let bv = col_sum(&dv, t, kv_out)?;
        let dln1_q = dense::linear_bwd_input(&dq, &lw.wq, t, e, q_out)?;
        let dln1_k = dense::linear_bwd_input(&dk, &lw.wk, t, e, kv_out)?;
        let dln1_v = dense::linear_bwd_input(&dv, &lw.wv, t, e, kv_out)?;
        let wq = dense::linear_bwd_weight(&dq, &lt.ln1_out, t, e, q_out)?;
        let wk = dense::linear_bwd_weight(&dk, &lt.ln1_out, t, e, kv_out)?;
        let wv = dense::linear_bwd_weight(&dv, &lt.ln1_out, t, e, kv_out)?;
        let mut dln1 = dln1_q;
        add_flat(&mut dln1, &dln1_k)?;
        add_flat(&mut dln1, &dln1_v)?;
        let (dh_attn, attn_norm) =
            rmsnorm_bwd_rows(&lt.ln1_in, &lw.attn_norm, cfg.eps, &dln1, t, e)?;
        add_flat(&mut dh2, &dh_attn)?;
        dh = dh2;
        grads.push(LayerGrads {
            attn_norm,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        });
    }
    grads.reverse();

    let token_embd = embedding_grad(&dh, &tape.tokens, t, e, cfg.vocab)?;
    Ok(Grads {
        layers: grads,
        output_norm,
        token_embd,
        lm_head,
    })
}

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
