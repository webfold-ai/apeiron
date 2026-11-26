//! The decode step: the rows 0-8 composition and the `Engine` entry points.

#![allow(
    clippy::manual_let_else,
    clippy::single_match_else,
    clippy::ref_option,
    reason = "match/let forms and a by-ref Option parameter: the pinned Aeneas translates these and rejects the Option::map / let-else / as_deref forms (borrow abstractions)"
)]

use std::borrow::Cow;

use crate::format::quant;

use super::chart::ffn::{projections_q8act, FfnScratch, ProjSource, ProjSpec};
use super::chart::task::HeadTask;
use super::error::{
    TensorKey, N_BLK_PARTS, N_LAYER_PARTS, PART_OUTPUT, TENSOR_PART_BYTES, TENSOR_PART_LEN,
};
#[cfg(any(test, target_arch = "wasm32"))]
use super::kern::lut::q8act_projection;
use super::kern::lut::LUT_ROW_ALIGN;
use super::kern::nq::NqError;
use super::loader::{opt_slice, HadamardMeta, PayloadRef};
use super::mixer::AffineState;
use super::{
    argmax_fixed, fixed_vector_checksum, fwht_fixed, rms_norm_fixed, AttnError, Engine,
    EngineError, F32Error, FfnError, HadamardError, KvState, LmHeadError, MatvecError, MixerKind,
    NormError, RopeError, RopeTable, SamplerError, TensorEntry,
};
use crate::engine::num::dequant::dequant_row_fixed;
use crate::engine::num::value::round_shift_32;
use crate::engine::ops::lm_head;

// Unchunked kernel: the `Sched::Sequential` oracle (tests and the wasm32 build); executing
// `Sched::Resident` is `par::matvec_rows_chunked`.
#[cfg(any(test, target_arch = "wasm32"))]
use super::matvec_fixed;

/// One greedy decode step's result (rows 0–8 composed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepResult {
    /// The chosen token: [`argmax_fixed`] over the step's Q32.32 logits.
    pub token: u32,
    /// [`fixed_vector_checksum`] of the step's Q32.32 logits.
    pub logits_checksum: u64,
}

/// A refused decode step (rows 0–8 composed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// The token id left the vocabulary, or the embedding read refused.
    Embedding(EngineError),
    /// A tensor payload read refused.
    Read(EngineError),
    /// `RoPE` refused.
    Rope(RopeError),
    /// A norm refused.
    Norm(NormError),
    /// The KV cache refused (capacity, position, shape, softmax window).
    Kv(AttnError),
    /// A projection refused.
    Matvec(MatvecError),
    /// A `NanoQuant` sign-rank product refused.
    Nq(NqError),
    /// The FFN refused.
    Ffn(FfnError),
    /// The LM head refused.
    LmHead(LmHeadError),
    /// The sampler refused.
    Sampler(SamplerError),
    /// The folded-weight activation rotation refused ([`fwht_fixed`]).
    Hadamard(HadamardError),
    /// An `explicit` fold contract carries no sign vector for the activation width.
    MissingHadamardSigns {
        /// Tensor being projected.
        tensor: TensorKey,
        /// Activation width `x.len()` that needs a sign vector.
        width: usize,
    },
    /// `prism.hadamard.version = 2` (tied output) is not executed: the tied embedding path needs the
    /// inverse of the signed transform, which the forward [`fwht_fixed`] does not provide.
    TiedOutputUnsupported,
    /// An `F32` gain or bias refused.
    Gain(F32Error),
    /// A required tensor is absent from the directory.
    MissingTensor(TensorKey),
    /// A tensor's quant family has no step reader.
    UnsupportedQuant {
        /// ggml type id.
        ggml_type: u32,
    },
    /// A tensor's stored dims are not the dims this stage requires.
    BadShape {
        /// Tensor name.
        tensor: String,
        /// Stored dims, fastest-varying first.
        dims: Vec<u64>,
        /// Required dims, fastest-varying first.
        expected: Vec<u64>,
    },
    /// The supplied KV cache geometry is not the engine config's geometry.
    KvGeometry {
        /// Supplied layer count.
        kv_layer: usize,
        /// Supplied query-head count.
        kv_head: usize,
        /// Supplied KV-head count.
        kv_head_kv: usize,
        /// Supplied head width.
        kv_head_dim: usize,
    },
    /// A file-backed KV cache could not be created, allocated or mapped.
    KvFile(String),
    /// A config field cannot form a step quantity.
    BadConfig(String),
    /// A length did not fit the target integer width.
    LengthOverflow,
    /// A checked fixed-point addition or narrowing left its range.
    Overflow,
}

impl core::fmt::Display for StepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StepError::Embedding(e) => write!(f, "embedding: {e}"),
            StepError::Read(e) => write!(f, "read: {e}"),
            StepError::Rope(e) => write!(f, "rope: {e:?}"),
            StepError::Norm(e) => write!(f, "norm: {e:?}"),
            StepError::Kv(e) => write!(f, "kv: {e:?}"),
            StepError::Matvec(e) => write!(f, "matvec: {e:?}"),
            StepError::Nq(e) => write!(f, "nq: {e}"),
            StepError::Ffn(e) => write!(f, "ffn: {e:?}"),
            StepError::LmHead(e) => write!(f, "lm head: {e:?}"),
            StepError::Sampler(e) => write!(f, "sampler: {e:?}"),
            StepError::Hadamard(e) => write!(f, "hadamard: {e:?}"),
            StepError::MissingHadamardSigns { tensor, width } => {
                write!(f, "hadamard: {tensor} needs a sign vector of width {width}")
            }
            StepError::TiedOutputUnsupported => {
                write!(f, "hadamard: version 2 tied output is not executed")
            }
            StepError::Gain(e) => write!(f, "gain: {e}"),
            StepError::MissingTensor(n) => write!(f, "no tensor {n}"),
            StepError::UnsupportedQuant { ggml_type } => {
                write!(f, "no step reader for ggml type {ggml_type}")
            }
            StepError::BadShape {
                tensor,
                dims,
                expected,
            } => write!(f, "tensor {tensor} dims {dims:?} != {expected:?}"),
            StepError::KvGeometry {
                kv_layer,
                kv_head,
                kv_head_kv,
                kv_head_dim,
            } => write!(
                f,
                "kv geometry {kv_layer}x{kv_head}x{kv_head_kv}x{kv_head_dim} != config"
            ),
            StepError::KvFile(e) => write!(f, "kv file: {e}"),
            StepError::BadConfig(k) => write!(f, "bad config field {k}"),
            StepError::LengthOverflow => write!(f, "length overflow"),
            StepError::Overflow => write!(f, "fixed-point overflow"),
        }
    }
}

impl std::error::Error for StepError {}

impl From<RopeError> for StepError {
    fn from(e: RopeError) -> Self {
        StepError::Rope(e)
    }
}

impl From<NormError> for StepError {
    fn from(e: NormError) -> Self {
        StepError::Norm(e)
    }
}

impl From<AttnError> for StepError {
    fn from(e: AttnError) -> Self {
        StepError::Kv(e)
    }
}

impl From<MatvecError> for StepError {
    fn from(e: MatvecError) -> Self {
        StepError::Matvec(e)
    }
}

impl From<FfnError> for StepError {
    fn from(e: FfnError) -> Self {
        StepError::Ffn(e)
    }
}

impl From<LmHeadError> for StepError {
    fn from(e: LmHeadError) -> Self {
        StepError::LmHead(e)
    }
}

impl From<SamplerError> for StepError {
    fn from(e: SamplerError) -> Self {
        StepError::Sampler(e)
    }
}

impl From<F32Error> for StepError {
    fn from(e: F32Error) -> Self {
        StepError::Gain(e)
    }
}

/// Which projection kernel a decode step uses: [`KernelMode::Exact`] the pre-chart path,
/// [`KernelMode::Q8Act`] the int8-activation chart (`matvec ∘ quant_act`) or LUT kernel,
/// [`KernelMode::Ternary`] the ternary-activation reader over a `NanoQuant` pack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelMode {
    /// The pre-change path over the dequantized activation.
    #[default]
    Exact,
    /// The int8-activation chart kernel, or the `PQ2_0` lookup-table kernel.
    Q8Act,
    /// The ternary-activation reader over the fitted linears; stored tensors keep the `Q8Act`
    /// kernels, because the dial only changes the activation reader over the factors.
    Ternary,
}

/// How a step sources each tensor payload: [`Sched::Resident`] the executing borrowed path,
/// [`Sched::Sequential`] the unchunked oracle (owned `read_range` in tests, the resident slice on
/// wasm32, one [`matvec_fixed`] in both).
#[derive(Clone, Copy)]
pub(crate) enum Sched {
    Resident,
    #[cfg(any(test, target_arch = "wasm32"))]
    Sequential,
}

/// The schedule the step entry points select, fixed at compile time: native executes the resident
/// path; wasm32 has no scoped threads or worker pool, so it executes the sequential one. The host
/// carries no runtime test of this choice.
#[cfg(target_arch = "wasm32")]
const STEP_SCHED: Sched = Sched::Sequential;
#[cfg(not(target_arch = "wasm32"))]
const STEP_SCHED: Sched = Sched::Resident;
/// The wasm build cannot reach the resident path's scoped threads, so the selection is checked at
/// compile time rather than by a test that only runs on the host.
#[cfg(target_arch = "wasm32")]
const _: () = assert!(matches!(STEP_SCHED, Sched::Sequential));

/// Checked `x[i] += y[i]` over two equal-width Q32.32 rows (the residual adds and bias add).
fn add_assign_fixed(x: &mut [i64], y: &[i64]) -> Result<(), StepError> {
    if x.len() != y.len() {
        return Err(StepError::BadConfig(String::from("residual row width")));
    }
    let mut i = 0usize;
    while i < x.len() {
        x[i] = x[i].checked_add(y[i]).ok_or(StepError::Overflow)?;
        i += 1;
    }
    Ok(())
}

/// [`rms_norm_fixed`] followed by the per-channel Q32.32 gain, the fold `cpu_rms` order. A width
/// mismatch refuses with [`StepError::BadConfig`].
fn rms_with_gain_slice(gain: &[i64], x: &[i64], eps: i64) -> Result<Vec<i64>, StepError> {
    if gain.len() != x.len() {
        return Err(StepError::BadConfig(String::from("norm gain width")));
    }
    let base = rms_norm_fixed(x, eps).map_err(StepError::Norm)?;
    let mut out: Vec<i64> = Vec::with_capacity(base.len());
    let mut i = 0usize;
    let mut err: Option<StepError> = None;
    while i < base.len() && err.is_none() {
        match i128::from(base[i]).checked_mul(i128::from(gain[i])) {
            None => err = Some(StepError::Overflow),
            Some(prod) => match i64::try_from(round_shift_32(prod)) {
                Ok(v) => out.push(v),
                Err(_) => err = Some(StepError::Overflow),
            },
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// The folded-weight activation rotation: `None` when `key` is not in the fold contract, otherwise
/// a fresh copy of `x` rotated by [`fwht_fixed`] (`x' = round(c·H_{2^k}·(s⊙x)/2^32)`); `x` is never
/// modified. In `explicit` mode the sign vector is the contract entry whose width equals `x.len()`
/// (a missing width refuses with [`StepError::MissingHadamardSigns`]); in `identity` mode none is
/// applied. A kernel refusal maps to [`StepError::Hadamard`].
///
/// # Complexity
///
/// `O(n log block)` time and `O(n)` space; the contract scan is `O(weight_names + signs)` bytes.
fn folded_activation(
    meta: Option<&HadamardMeta>,
    key: TensorKey,
    x: &[i64],
) -> Result<Option<Vec<i64>>, StepError> {
    let Some(meta) = meta else {
        return Ok(None);
    };
    // Only the weights the contract lists are folded; every other tensor keeps `x` verbatim.
    let mut listed = false;
    let mut i = 0usize;
    while i < meta.weight_names.len() {
        if key_matches(meta.weight_names[i].as_bytes(), key) {
            listed = true;
        }
        i += 1;
    }
    if !listed {
        return Ok(None);
    }
    // The sign vector, when the mode is explicit, is keyed by the activation width.
    let mut sign_at: Option<usize> = None;
    if let Some(table) = meta.signs.as_ref() {
        let mut found = false;
        let mut j = 0usize;
        while j < table.len() {
            if table[j].0 as usize == x.len() {
                sign_at = Some(j);
                found = true;
            }
            j += 1;
        }
        if !found {
            return Err(StepError::MissingHadamardSigns {
                tensor: key,
                width: x.len(),
            });
        }
    }
    let mut signs: Option<&[i8]> = None;
    if let Some(j) = sign_at {
        if let Some(table) = meta.signs.as_ref() {
            signs = Some(table[j].1.as_slice());
        }
    }
    let mut folded: Vec<i64> = Vec::with_capacity(x.len());
    let mut k = 0usize;
    while k < x.len() {
        folded.push(x[k]);
        k += 1;
    }
    fwht_fixed(&mut folded, signs, meta.block_size as usize).map_err(StepError::Hadamard)?;
    Ok(Some(folded))
}

/// [`Engine::step_proj_sched`]'s and [`Engine::lm_head_fixed_sched`]'s resident `Q8Act` leaf.
impl Engine {
    /// One `Q8Act` projection over `x`: one pooled
    /// [`quant_act_pooled`](super::chart::ffn::quant_act_pooled) of `x`, then the open-time
    /// `PQ2_0` lookup-table row kernel of directory entry `idx` or the chart row kernel split by
    /// `chunks`. Same result and refusals as `kern::lut::q8act_projection`; taking the
    /// quantization here lets a caller that already holds `q` skip a second pass.
    #[allow(
        clippy::too_many_arguments,
        reason = "the engine, tensor identity, payload and shape, activation and dispatch width \
                  are the projection's full input"
    )]
    fn proj_q8act_quantified(
        &self,
        idx: usize,
        ggml_type: u32,
        at: PayloadRef,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<i64>, MatvecError> {
        let q = super::chart::ffn::quant_act_pooled(x)
            .ok_or(MatvecError::Dequant(EngineError::FixedOverflow))?;
        if ggml_type == quant::GGML_TYPE_PQ2_0 {
            self.pq2_lut(idx)?;
            let spec = ProjSpec {
                n_out,
                row_align: LUT_ROW_ALIGN,
                source: ProjSource::Rows(idx),
            };
            let mut out = super::chart::ffn::projections_q8act_prepared(
                self.resident(),
                self.pq2_luts(),
                n_in,
                &q,
                &[spec],
                chunks,
            )?;
            Ok(out.pop().expect("one projection"))
        } else {
            let raw = &self.resident()[at.off..at.off + at.len];
            super::chart::ffn::matvec_q8act_quant_chunked(ggml_type, raw, n_out, n_in, &q, chunks)
        }
    }
}

/// Prefill block width: the columns one pass carries, so the served weights are read once per
/// block and the column tile folds each read across the block's columns. `APEIRON_PREFILL_COLS`
/// overrides it (read once).
///
/// Default 1024 on a pool of four or more participants and 256 below: the pass pays a per-block
/// fixed cost (the serial residual add, attention's per-pass setup, the norm and the pool
/// dispatches), so the wider block pays it once per 1,024 columns instead of once per 256. On the
/// OVH box, 0.6B pack, 1,827 columns (`bench/cpu/1795573862-prefill-block-and-group.md`):
/// 1024/256 is 1.17× at eight participants and 1.15× at four; the phase split that locates the
/// effect was taken at width 8, and width 4 was 1.007 before the group hoist against 1.152 after
/// it, so the two changes are not separated there. At one participant the pass is instruction-bound
/// and 1024 is 0.97× of 256. The rows are one model on one host; the threshold is the lowest
/// measured winning width, not a located crossover, and widths 2–3 and the laptop are unmeasured.
pub(crate) fn prefill_block() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("APEIRON_PREFILL_COLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                if super::chart::pool::width_global() >= 4 {
                    1024
                } else {
                    256
                }
            })
    })
}

/// The bit-plane head on the greedy path unless `APEIRON_BITPLANE=0`; read once per process.
fn bitplane_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("APEIRON_BITPLANE").map_or(true, |v| v != "0"))
}

/// Number of layers of kind `kind`.
///
/// # Complexity
///
/// `O(mixers.len())` time and `O(1)` space, counted in comparisons: one per layer.
fn count_mixers(mixers: &[MixerKind], kind: MixerKind) -> usize {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < mixers.len() {
        if mixers[i] == kind {
            n += 1;
        }
        i += 1;
    }
    n
}
/// Each layer's index in the softmax list carrier (`KvState`), `None` for an affine layer. With an
/// all-softmax schedule this is the identity.
///
/// # Complexity
///
/// `O(n_layer)` time and space.
fn softmax_slots(mixers: &[MixerKind]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(mixers.len());
    let mut next = 0usize;
    for m in mixers {
        if *m == MixerKind::Softmax {
            out.push(Some(next));
            next += 1;
        } else {
            out.push(None);
        }
    }
    out
}

/// Each layer's index in the [`AffineState`], `None` for a softmax layer.
///
/// # Complexity
///
/// `O(n_layer)` time and space.
fn affine_slots(mixers: &[MixerKind]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(mixers.len());
    let mut next = 0usize;
    for m in mixers {
        if *m == MixerKind::Affine {
            out.push(Some(next));
            next += 1;
        } else {
            out.push(None);
        }
    }
    out
}

/// `RoPE` on the first `rope.head_dim()` components of every `hd`-wide head of `x` (partial
/// rotary: Qwen3.5 rotates 64 of 256). When the table width equals `hd` this is one
/// [`RopeTable::rotate_one`] per head, the same arithmetic as rotating the whole vector.
///
/// # Complexity
///
/// `O(x.len() / hd · rope.head_dim())` multiplications, no allocation.
fn rotate_head_prefixes(
    table: &RopeTable,
    x: &mut [i64],
    hd: usize,
    pos: u32,
) -> Result<(), StepError> {
    let rot = table.head_dim();
    if hd == 0 || rot > hd || !x.len().is_multiple_of(hd) {
        return Err(StepError::Rope(RopeError::VectorLength));
    }
    let mut base = 0usize;
    while base < x.len() {
        table
            .rotate_one(&mut x[base..base + rot], pos)
            .map_err(StepError::Rope)?;
        base += hd;
    }
    Ok(())
}

/// Split a gated `attn_q` output, laid out per head as `[query(hd) | gate(hd)]` (llama.cpp
/// `qwen35.cpp` `build_layer_attn` views), into the query and the gate, each `n_head·hd` wide.
///
/// # Complexity
///
/// `O(n_head·hd)` copies into two new vectors.
fn split_query_gate(
    full: &[i64],
    n_head: usize,
    hd: usize,
) -> Result<(Vec<i64>, Vec<i64>), StepError> {
    if full.len() != n_head * hd * 2 {
        return Err(StepError::BadConfig(String::from("gated query width")));
    }
    let mut q = Vec::with_capacity(n_head * hd);
    let mut g = Vec::with_capacity(n_head * hd);
    let mut h = 0usize;
    while h < n_head {
        let base = h * 2 * hd;
        q.extend_from_slice(&full[base..base + hd]);
        g.extend_from_slice(&full[base + hd..base + 2 * hd]);
        h += 1;
    }
    Ok((q, g))
}

/// `att[i] ← att[i] · sigmoid(gate[i])`, the attention output gate, through the one
/// [`mul_q32`](super::value::mul_q32).
///
/// # Complexity
///
/// `O(att.len())` sigmoid evaluations and products; the body is [`attn_block::sigmoid_gate`],
/// word-equal to the `fixed_sigmoid` + `mul_q32` loop.
fn apply_sigmoid_gate(att: &mut [i64], gate: &[i64]) -> Result<(), StepError> {
    attn_block::sigmoid_gate(att, gate)
}

/// One head of [`qk_norm_and_rope`]: its QK `RMSNorm`, then its rotation.
fn qk_norm_head(
    gain: &Option<Vec<i64>>,
    dst: &mut [i64],
    table: &RopeTable,
    eps: i64,
    pos: u32,
) -> Result<(), StepError> {
    super::loader::qk_norm_heads(opt_slice(gain), dst, dst.len(), eps)?;
    rotate_head_prefixes(table, dst, dst.len(), pos)
}

/// One head of the rotation-only path of [`qk_norm_and_rope`] (no QK-norm tensors).
fn rope_head(dst: &mut [i64], table: &RopeTable, pos: u32) -> Result<(), StepError> {
    rotate_head_prefixes(table, dst, dst.len(), pos)
}

/// The shared borrows of one QK-norm head dispatch; `run` is the single call to [`qk_norm_head`].
struct QkNormCtx<'a> {
    gain: &'a Option<Vec<i64>>,
    table: &'a RopeTable,
    eps: i64,
    pos: u32,
}

impl HeadTask<i64, StepError> for QkNormCtx<'_> {
    fn run(&self, _h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        qk_norm_head(self.gain, dst, self.table, self.eps, self.pos)
    }
}

/// The shared borrows of one rotation-only head dispatch; `run` is the single call to [`rope_head`].
struct RopeCtx<'a> {
    table: &'a RopeTable,
    pos: u32,
}

impl HeadTask<i64, StepError> for RopeCtx<'_> {
    fn run(&self, _h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        rope_head(dst, self.table, self.pos)
    }
}

/// `vectors.apply_qk_norms` plus the two per-vector `RoPE` rotations, over the pool when
/// `parallel`: every `hd`-wide head of `query` (then of `key`) gets its QK `RMSNorm` and its
/// rotation in one pool task. Each head's norm and rotation are head-local, so the split is
/// bit-identical to the serial norm-then-rotate pair; a refusal keeps the first head index.
#[allow(
    clippy::too_many_arguments,
    reason = "the two QK gains, the rotation table, both vectors, the head geometry, the norm \
              epsilon, the position and the schedule are the call's full input"
)]
fn qk_norm_and_rope(
    vectors: &super::loader::DecodeVectors,
    layer: usize,
    table: &RopeTable,
    query: &mut [i64],
    key: &mut [i64],
    hd: usize,
    eps: i64,
    pos: u32,
    parallel: bool,
) -> Result<(), StepError> {
    if !parallel || hd == 0 || !query.len().is_multiple_of(hd) || !key.len().is_multiple_of(hd) {
        // The composed geometry is a whole number of `hd`-wide heads; anything else keeps the
        // serial refusals unchanged.
        vectors.apply_qk_norms(layer, query, key, hd, eps)?;
        return attn_block::rotate_pair_prefixes(table, query, key, hd, pos);
    }
    match vectors.qk_norms_of(layer) {
        Some(n) => {
            let g_q = &n[0];
            let g_k = &n[1];
            if let Some((_, e)) = super::chart::ffn::attn_heads(
                query.len() / hd,
                hd,
                query,
                true,
                &QkNormCtx {
                    gain: g_q,
                    table,
                    eps,
                    pos,
                },
            ) {
                return Err(e);
            }
            if let Some((_, e)) = super::chart::ffn::attn_heads(
                key.len() / hd,
                hd,
                key,
                true,
                &QkNormCtx {
                    gain: g_k,
                    table,
                    eps,
                    pos,
                },
            ) {
                return Err(e);
            }
        }
        None => {
            if let Some((_, e)) = super::chart::ffn::attn_heads(
                query.len() / hd,
                hd,
                query,
                true,
                &RopeCtx { table, pos },
            ) {
                return Err(e);
            }
            if let Some((_, e)) = super::chart::ffn::attn_heads(
                key.len() / hd,
                hd,
                key,
                true,
                &RopeCtx { table, pos },
            ) {
                return Err(e);
            }
        }
    }
    Ok(())
}

/// The seven `blk.{layer}.*` Q8-active decode weights of one layer, resolved once per step.
///
/// The tensor directory's `get` is a linear scan; the composed step calls it once per projection
/// per layer, so a token pays `n_layer · 7` scans and seven `format!` allocations per layer. This
/// table is one pass over the directory; each accessor is then an index, and the tensor's own
/// `name` serves as the `pq2_lut` cache key without an allocation. Part order is
/// `[q, k, v, attn_output, ffn_gate, ffn_up, ffn_down]`, the order the composed step consumes them,
/// so a missing tensor reports the same [`StepError::MissingTensor`] the name-based path reports.
#[derive(Clone, Copy)]
struct LayerIndex {
    idx: [Option<usize>; N_PARTS],
}

/// Parts per layer in [`LayerIndex`]: the parts `0..N_PARTS` of [`TENSOR_PART_BYTES`], in
/// consumption order `[q, k, v, attn_output, ffn_gate, ffn_up, ffn_down]`, then the affine mixer's
/// projections (`[q | k | v]` into the convolution, the output gate `z`, the per-head `β` and decay
/// inputs, the output) and the multi-token-prediction block's input projection.
const N_PARTS: usize = N_LAYER_PARTS;

impl LayerIndex {
    /// The directory index of part `p`, or the [`StepError::MissingTensor`] the key-based
    /// `tensor_payload` returns.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "p < N_PARTS = 13, so the part number fits u8"
    )]
    fn at(&self, p: usize, layer: usize) -> Result<usize, StepError> {
        let slot: Option<usize> = self.idx[p];
        match slot {
            Some(i) => Ok(i),
            None => Err(StepError::MissingTensor(TensorKey::blk(layer, p as u8))),
        }
    }
}

/// Whether `needle` occurs in `hay` starting at `at`.
///
/// # Complexity
///
/// `O(needle.len())` byte comparisons, `O(1)` space: one loop, one comparison per iteration.
pub(crate) fn bytes_at(hay: &[u8], at: usize, needle: &[u8]) -> bool {
    if at > hay.len() || hay.len() - at < needle.len() {
        return false;
    }
    let mut k = 0usize;
    let mut same = true;
    while k < needle.len() {
        if hay[at + k] != needle[k] {
            same = false;
        }
        k += 1;
    }
    same
}

/// Whether some vector of `xs` has a length other than `n`.
///
/// # Complexity
///
/// `O(xs.len())` length comparisons, `O(1)` space: one scan, stopping at the first mismatch.
pub(crate) fn any_vec_len_ne<T>(xs: &[Vec<T>], n: usize) -> bool {
    let mut i = 0usize;
    while i < xs.len() {
        if xs[i].len() != n {
            return true;
        }
        i += 1;
    }
    false
}

/// Whether `a` and `b` are the same bytes.
///
/// # Complexity
///
/// `O(a.len())` byte comparisons, `O(1)` space: the length test, then [`bytes_at`].
pub(crate) fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && bytes_at(a, 0, b)
}

/// Whether `nm[at..]` starts with the text of tensor part `part` ([`TENSOR_PART_BYTES`]).
///
/// # Complexity
///
/// `O(len)` byte comparisons, `O(1)` space, `len = TENSOR_PART_LEN[part]`: one loop.
fn part_at(nm: &[u8], at: usize, part: usize) -> bool {
    let len = TENSOR_PART_LEN[part];
    if at > nm.len() || nm.len() - at < len {
        return false;
    }
    let mut k = 0usize;
    let mut same = true;
    while k < len {
        if nm[at + k] != TENSOR_PART_BYTES[part][k] {
            same = false;
        }
        k += 1;
    }
    same
}

/// Whether the directory name `nm` is the name of `key`: `blk.{layer}.{part text}` with `layer` in
/// plain decimal (no sign, no leading zero) for a layer part, the part text for a top-level part.
/// Equal to comparing `nm` with the text [`TensorKey`]'s `Display` renders.
///
/// # Complexity
///
/// `O(nm.len())` byte comparisons, `O(1)` space: the digit scan and one [`part_at`] loop.
pub(crate) fn key_matches(nm: &[u8], key: TensorKey) -> bool {
    let p = key.part as usize;
    if p >= TENSOR_PART_LEN.len() {
        return false;
    }
    if p >= N_BLK_PARTS {
        return nm.len() == TENSOR_PART_LEN[p] && part_at(nm, 0, p);
    }
    if !bytes_at(nm, 0, b"blk.") {
        return false;
    }
    let mut pos = 4usize;
    let mut layer_no = 0usize;
    while pos < nm.len() && nm[pos] >= b'0' && nm[pos] <= b'9' {
        layer_no = layer_no
            .saturating_mul(10)
            .saturating_add((nm[pos] - b'0') as usize);
        pos += 1;
    }
    let plain_decimal = pos > 4 && (nm[4] != b'0' || pos == 5);
    plain_decimal
        && layer_no == key.layer as usize
        && pos < nm.len()
        && nm[pos] == b'.'
        && nm.len() == pos + 1 + TENSOR_PART_LEN[p]
        && part_at(nm, pos + 1, p)
}

impl Engine {
    /// The per-layer index table of the seven Q8-active weights, from one pass over the directory.
    ///
    /// # Complexity
    ///
    /// `O(directory bytes)` once per step, replacing `O(n_layer · 7 · directory bytes)` scans.
    fn resolve_layers(&self) -> Vec<LayerIndex> {
        let entries = self.table().entries();
        let n_layers = self.config.n_layer + self.config.n_nextn;
        let mut out: Vec<LayerIndex> = vec![
            LayerIndex {
                idx: [None; N_PARTS]
            };
            n_layers
        ];
        let mut ei = 0usize;
        while ei < entries.len() {
            let nm = entries[ei].name.as_bytes();
            if bytes_at(nm, 0, b"blk.") {
                let mut pos = 4usize;
                let mut layer_no = 0usize;
                while pos < nm.len() && nm[pos] >= b'0' && nm[pos] <= b'9' {
                    layer_no = layer_no
                        .saturating_mul(10)
                        .saturating_add((nm[pos] - b'0') as usize);
                    pos += 1;
                }
                if pos > 4 && pos < nm.len() && nm[pos] == b'.' && layer_no < n_layers {
                    pos += 1;
                    let mut pi = 0usize;
                    while pi < N_PARTS {
                        if nm.len() == pos + TENSOR_PART_LEN[pi] && part_at(nm, pos, pi) {
                            out[layer_no].idx[pi] = Some(ei);
                        }
                        pi += 1;
                    }
                }
            }
            ei += 1;
        }
        out
    }

    /// The checked payload extent of directory entry `idx` for a `[n_in, n_out]` tensor: the same
    /// refusals as `tensor_payload`, without the name scan or the up-front `expected` allocation
    /// (built only on the refusal path), and no borrowed bytes: the caller slices
    /// [`Engine::resident`] by the returned [`PayloadRef`] at the use. The test-only
    /// [`Sched::Sequential`] arm still performs its owned `read_range` (same refusals) and asserts
    /// it equals that slice.
    fn payload_at(
        &self,
        idx: usize,
        n_out: usize,
        n_in: usize,
        sched: Sched,
    ) -> Result<(u32, PayloadRef), StepError> {
        let t = &self.table().entries()[idx];
        if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
            return Err(StepError::BadShape {
                tensor: t.name.clone(),
                dims: t.dims.clone(),
                expected: vec![n_in as u64, n_out as u64],
            });
        }
        let n_bytes = match t.n_bytes {
            Some(n) => n,
            None => {
                return Err(StepError::UnsupportedQuant {
                    ggml_type: t.ggml_type,
                })
            }
        };
        #[cfg(test)]
        let owned = match sched {
            Sched::Resident => None,
            Sched::Sequential => match self.read_range(t.byte_range.start, n_bytes) {
                Ok(p) => Some(p),
                Err(e) => return Err(StepError::Read(e)),
            },
        };
        #[cfg(not(test))]
        match sched {
            Sched::Resident => {}
            // The wasm build has no file to read: the sequential schedule borrows the resident
            // bytes, and only the row kernel differs from `Resident`.
            #[cfg(target_arch = "wasm32")]
            Sched::Sequential => {}
        }
        let at = match self.payload_ref(t.byte_range.start, n_bytes) {
            Ok(p) => p,
            Err(e) => return Err(StepError::Read(e)),
        };
        #[cfg(test)]
        if let Some(p) = owned {
            assert_eq!(p.as_slice(), &self.resident()[at.off..at.off + at.len]);
        }
        Ok((t.ggml_type, at))
    }

    /// `step_proj_sched`'s resident `Q8Act` arm over a resolved directory index.
    fn proj_q8act_at(
        &self,
        idx: usize,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<i64>, StepError> {
        let (ggml_type, at) = self.payload_at(idx, n_out, n_in, Sched::Resident)?;
        let folded = folded_activation(self.hadamard(), key, x)?;
        let activation: &[i64] = opt_slice(&folded).unwrap_or(x);
        self.proj_q8act_quantified(idx, ggml_type, at, n_out, n_in, activation, chunks)
            .map_err(StepError::Matvec)
    }

    /// One `Q8Act` projection group: the layer parts `parts` (indices into [`TENSOR_PART_BYTES`])
    /// all read the same input `x`, so one pooled quantization of `x` feeds every row kernel. Used
    /// by the attention `q/k/v` group and the affine mixer's four input projections.
    ///
    /// A `Hadamard`-folded member keeps one `proj_q8act_at` per part, as the single-projection path
    /// keeps one `step_proj_sched` per part.
    ///
    /// # Complexity
    ///
    /// `O(Σ outs · n_in)` multiply-adds over the pool, plus one `O(n_in)` quantization; the
    /// payloads are `PayloadRef` extents of the resident file (no weight byte is copied).
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        reason = "the layer index, parts, shapes, activation and dispatch width are the group's input; the loop counters are single-character"
    )]
    fn group_q8act_at(
        &self,
        li: &LayerIndex,
        layer: usize,
        parts: &[usize],
        outs: &[usize],
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        if parts.len() != outs.len() {
            return Err(StepError::BadConfig(String::from("projection group shape")));
        }
        let n = parts.len();
        let mut idxs = Vec::with_capacity(n);
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < n && err.is_none() {
            match li.at(parts[i], layer) {
                Ok(ix) => idxs.push(ix),
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut types = Vec::with_capacity(n);
        let mut ats: Vec<PayloadRef> = Vec::with_capacity(n);
        let mut folded_out: Option<Vec<Vec<i64>>> = None;
        let mut i = 0usize;
        while i < n && err.is_none() && folded_out.is_none() {
            match folded_activation(self.hadamard(), part_key(layer, parts[i]), x) {
                Err(e) => err = Some(e),
                Ok(Some(_)) => {
                    let mut out = Vec::with_capacity(n);
                    let mut j = 0usize;
                    while j < n && err.is_none() {
                        match self.proj_q8act_at(
                            idxs[j],
                            part_key(layer, parts[j]),
                            outs[j],
                            n_in,
                            x,
                            chunks,
                        ) {
                            Ok(v) => out.push(v),
                            Err(e) => err = Some(e),
                        }
                        j += 1;
                    }
                    if err.is_none() {
                        folded_out = Some(out);
                    }
                }
                Ok(None) => match self.payload_at(idxs[i], outs[i], n_in, Sched::Resident) {
                    Ok((t, at)) => {
                        types.push(t);
                        ats.push(at);
                    }
                    Err(e) => err = Some(e),
                },
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        if let Some(out) = folded_out {
            return Ok(out);
        }
        let mut i = 0usize;
        while i < n && err.is_none() {
            if types[i] == quant::GGML_TYPE_PQ2_0 {
                if let Err(e) = self.pq2_lut(idxs[i]) {
                    err = Some(e.into());
                }
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut specs: Vec<ProjSpec> = Vec::with_capacity(n);
        for i in 0..n {
            specs.push(ffn_spec(types[i], ats[i], idxs[i], outs[i]));
        }
        projections_q8act(self.resident(), self.pq2_luts(), n_in, x, &specs, chunks)
            .map_err(StepError::Matvec)
    }

    /// `ffn_q8act_step` over resolved directory indices: the same three projections and pooled
    /// quantizations, with the `PQ2_0` table taken from the open-time slot of the directory index.
    fn ffn_q8act_at(
        &self,
        li: &LayerIndex,
        layer: usize,
        n_ff: usize,
        n_embd: usize,
        x: &[i64],
        scratch: &mut FfnScratch,
    ) -> Result<Vec<i64>, StepError> {
        let idxs = [li.at(4, layer)?, li.at(5, layer)?, li.at(6, layer)?];
        let (t0, at0) = self.payload_at(idxs[0], n_ff, n_embd, Sched::Resident)?;
        let (t1, at1) = self.payload_at(idxs[1], n_ff, n_embd, Sched::Resident)?;
        let (t2, at2) = self.payload_at(idxs[2], n_embd, n_ff, Sched::Resident)?;
        self.check_pq2_lut(t0, idxs[0])?;
        self.check_pq2_lut(t1, idxs[1])?;
        self.check_pq2_lut(t2, idxs[2])?;
        let gate = ffn_spec(t0, at0, idxs[0], n_ff);
        let up = ffn_spec(t1, at1, idxs[1], n_ff);
        let down = ffn_spec(t2, at2, idxs[2], n_embd);
        super::chart::ffn::ffn_q8act_prepared_into(
            self.resident(),
            self.pq2_luts(),
            n_embd,
            x,
            gate,
            up,
            down,
            n_ff,
            crate::engine::ops::par::row_chunk_budget(n_ff),
            crate::engine::ops::par::row_chunk_budget(n_embd),
            scratch,
        )
        .map_err(StepError::Ffn)
    }

    /// Refuses with the stored table error when the `PQ2_0` tensor at directory index `idx` has
    /// none; any other quant family passes.
    fn check_pq2_lut(&self, ggml_type: u32, idx: usize) -> Result<(), MatvecError> {
        if ggml_type == quant::GGML_TYPE_PQ2_0 {
            self.pq2_lut(idx)?;
        }
        Ok(())
    }
}

/// One projection's row kernel: its open-time `PQ2_0` table (`Rows(idx)`, the directory index),
/// else the chart row core over the payload extent `at`.
fn ffn_spec(ggml_type: u32, at: PayloadRef, idx: usize, n_out: usize) -> ProjSpec {
    if ggml_type == quant::GGML_TYPE_PQ2_0 {
        ProjSpec {
            n_out,
            row_align: LUT_ROW_ALIGN,
            source: ProjSource::Rows(idx),
        }
    } else {
        ProjSpec {
            n_out,
            row_align: 1,
            source: ProjSource::Chart { ggml_type, at },
        }
    }
}

/// The key of layer part `part` (an index into [`TENSOR_PART_BYTES`]) of `layer`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "part < N_LAYER_PARTS = 13, so the part number fits u8"
)]
pub(crate) fn part_key(layer: usize, part: usize) -> TensorKey {
    TensorKey::blk(layer, part as u8)
}

impl Engine {
    /// One `Q8Act` FFN block (`SwiGLU`) for the directory entries `idxs` (gate, up, down) with the
    /// payload extents and quant families `ws`: the `PQ2_0` tensors use their open-time lookup
    /// tables, the others the chart row core, and
    /// [`ffn_q8act_prepared`](super::chart::ffn::ffn_q8act_prepared) takes one pooled `quant_act`
    /// of `x` and one of the hidden.
    ///
    /// # Errors
    ///
    /// As [`FfnError`]: the geometry refusal, a projection's payload/quant/accumulator refusal and
    /// the elementwise overflow.
    fn ffn_q8act_step(
        &self,
        idxs: [usize; 3],
        ws: [(u32, PayloadRef); 3],
        n_ff: usize,
        x: &[i64],
    ) -> Result<Vec<i64>, FfnError> {
        let n_embd = x.len();
        self.check_pq2_lut(ws[0].0, idxs[0])?;
        self.check_pq2_lut(ws[1].0, idxs[1])?;
        self.check_pq2_lut(ws[2].0, idxs[2])?;
        super::chart::ffn::ffn_q8act_prepared(
            self.resident(),
            self.pq2_luts(),
            n_embd,
            x,
            ffn_spec(ws[0].0, ws[0].1, idxs[0], n_ff),
            ffn_spec(ws[1].0, ws[1].1, idxs[1], n_ff),
            ffn_spec(ws[2].0, ws[2].1, idxs[2], n_embd),
            n_ff,
            crate::engine::ops::par::row_chunk_budget(n_ff),
            crate::engine::ops::par::row_chunk_budget(n_embd),
        )
    }

    /// The Q32.32 embedding vector of token `id`.
    ///
    /// Reads one row of `token_embd.weight` (row stride from the quant geometry) and dequantizes it
    /// with no float; the row range is checked against the tensor extent and the read against the
    /// file length.
    ///
    /// # Errors
    ///
    /// [`EngineError::TokenOutOfRange`], [`EngineError::UnsupportedEmbeddingQuant`],
    /// [`EngineError::RowOutOfBounds`], [`EngineError::NonFiniteScale`] and the read/quant errors.
    pub fn token_embedding(&self, id: u32) -> Result<Vec<i64>, EngineError> {
        self.embedding_sched(id, Sched::Resident)
    }

    /// [`Engine::token_embedding`]'s body with the payload source chosen by `sched`: a borrowed
    /// resident slice under [`Sched::Resident`], an owned `read_range` copy under the test-only
    /// [`Sched::Sequential`].
    fn embedding_sched(&self, id: u32, sched: Sched) -> Result<Vec<i64>, EngineError> {
        let t = self.token_embd();
        let vocab = self.config.vocab_size;
        let id_usize =
            usize::try_from(id).map_err(|_| EngineError::TokenOutOfRange { id, vocab })?;
        if id_usize >= vocab {
            return Err(EngineError::TokenOutOfRange { id, vocab });
        }
        let (qk, bb) =
            quant::block_geom(t.ggml_type).ok_or(EngineError::UnsupportedEmbeddingQuant {
                ggml_type: t.ggml_type,
            })?;
        let qk = u64::try_from(qk).map_err(|_| EngineError::LengthOverflow)?;
        let row_elems =
            u64::try_from(self.config.n_embd).map_err(|_| EngineError::LengthOverflow)?;
        let blocks = row_elems.div_ceil(qk);
        let row_bytes = blocks
            .checked_mul(u64::try_from(bb).map_err(|_| EngineError::LengthOverflow)?)
            .ok_or(EngineError::LengthOverflow)?;
        let at = t
            .byte_range
            .start
            .checked_add(
                u64::from(id)
                    .checked_mul(row_bytes)
                    .ok_or(EngineError::LengthOverflow)?,
            )
            .ok_or(EngineError::LengthOverflow)?;
        let end = at
            .checked_add(row_bytes)
            .ok_or(EngineError::LengthOverflow)?;
        if end > t.byte_range.end {
            return Err(EngineError::RowOutOfBounds {
                at,
                n: row_bytes,
                end: t.byte_range.end,
            });
        }
        let raw = match sched {
            Sched::Resident => Cow::Borrowed(self.payload(at, row_bytes)?),
            #[cfg(all(test, not(target_arch = "wasm32")))]
            Sched::Sequential => Cow::Owned(self.read_range(at, row_bytes)?),
            #[cfg(target_arch = "wasm32")]
            Sched::Sequential => Cow::Borrowed(self.payload(at, row_bytes)?),
        };
        let mut out: Vec<i64> = Vec::with_capacity(self.config.n_embd);
        dequant_row_fixed(t.ggml_type, raw.as_ref(), self.config.n_embd, &mut out)?;
        if out.len() != self.config.n_embd {
            return Err(EngineError::ConfigMismatch(format!(
                "embedding row produced {} of {} weights",
                out.len(),
                self.config.n_embd
            )));
        }
        Ok(out)
    }

    /// The load-time `RoPE` frequency table for this model (port-map row 3).
    ///
    /// Builds [`RopeTable`] once from the loader's `f64` `rope_theta`/`head_dim` metadata; a decode
    /// loop holds it and calls [`RopeTable::rotate`] per token.
    ///
    /// # Errors
    ///
    /// [`RopeError`] for a malformed head geometry, a non-positive or non-finite `rope_theta`, or a
    /// frequency that does not fit Q32.32.
    pub fn rope_table(&self) -> Result<RopeTable, RopeError> {
        let mut table = RopeTable::new(self.config.rope_theta, self.config.rot_dims)?;
        // A `llama` GGUF stores Q and K in the permuted layout ggml rotates as adjacent pairs.
        if self.config.architecture == "llama" {
            table = table.with_adjacent_pairs();
        }
        // Llama-3.1 ships `rope_freqs.weight` (one `F32` factor per rotary pair): the llama3 RoPE
        // scaling. `ggml_rope_ext` divides each pair's angle by its factor, so without it the
        // low-frequency pairs rotate too fast and the model degrades as position grows.
        match self.rope_freq_factors() {
            Ok(Some(factors)) => table.divided_by_q32(&factors[..]),
            Ok(None) => Ok(table),
            Err(_) => Err(RopeError::TableLength),
        }
    }

    /// The output (LM-head) tensor entry (port-map row 7): `output.weight` when the directory
    /// carries it, otherwise the tied `token_embd.weight`.
    ///
    /// Both are `[n_embd, vocab]` row-major, so the head projection is the row-4 kernel with
    /// `n_out = vocab` and `n_in = n_embd`. The choice is read from the loaded tensor table, never
    /// assumed.
    #[must_use]
    pub fn lm_head(&self) -> &TensorEntry {
        self.table()
            .get("output.weight")
            .unwrap_or_else(|| self.token_embd())
    }

    /// The directory index of [`Engine::lm_head`]: `output.weight` when present, else the
    /// `token_embd.weight` entry.
    fn lm_head_index(&self) -> usize {
        match self.index_of(TensorKey::top(PART_OUTPUT)) {
            Some(i) => i,
            None => self.token_embd_index(),
        }
    }

    /// The Q32.32 LM-head logits of a residual activation `x` (port-map row 7).
    ///
    /// The head tensor is [`Engine::lm_head`] (`output.weight` when present, else the tied
    /// `token_embd.weight`), read over its exact checked extent and projected by [`matvec_fixed`]
    /// with `n_out = vocab`, `n_in = n_embd`; output `o` is `round(Σ_i W[o,i]·x_i / 2^32)`. No
    /// float appears on the executing path.
    ///
    /// # Errors
    ///
    /// [`LmHeadError`]'s shape, quant, payload, length, read and matvec refusals; no clamping.
    ///
    /// # Complexity
    ///
    /// `O(vocab·n_embd)` checked multiply-adds plus one dequantization per weight element;
    /// `O(vocab·n_embd)` payload bytes and `O(n_embd)` row scratch.
    pub fn lm_head_fixed(&self, x: &[i64]) -> Result<Vec<i64>, LmHeadError> {
        self.lm_head_fixed_sched(x, Sched::Resident, KernelMode::Exact)
    }

    /// The Q32.32 logits of every residual column in `xs`: one head projection per column through
    /// the engine's own path, with `mode` choosing the arithmetic (port-map row 7).
    ///
    /// [`KernelMode::Exact`] is the oracle: column `c` is
    /// [`Engine::lm_head_fixed_sched`]`(&xs[c], Resident, Exact)`, the checked `i128` per-column
    /// projection. [`KernelMode::Q8Act`] is the shipped head: for more than one column the whole
    /// block runs through the engine's multi-column route ([`Engine::head_cols_kernel`]:
    /// `matmat_q8act`/`gemm_q8_0` on a `Q8_0` head, the cached limb tile on an `F16` head), reading
    /// the head weights once for all columns; a head of any other quant family, a folded
    /// (`Hadamard`) model or any kernel refusal falls back to one
    /// [`Engine::lm_head_fixed_sched`]`(&xs[c], Resident, Q8Act)` per column. The multi-column
    /// `F16` limb route is not bit-identical to the per-column `Q8Act` path; its deviation is
    /// bounded in `tests/engine_f16_limb.rs` (K4), not claimed equal.
    ///
    /// # Errors
    ///
    /// [`LmHeadError`]'s shape, quant, payload, length, read and matvec refusals; no clamping. A
    /// column whose width is not `n_embd` refuses with [`LmHeadError::ShapeMismatch`].
    ///
    /// # Complexity
    ///
    /// `O(k·vocab·n_embd)` checked multiply-adds and one dequantization per weight element; the
    /// multi-column `Q8Act` route reads the `O(vocab·n_embd)` payload once, the per-column path
    /// once per column; space `O(k·vocab)` logit words.
    pub fn lm_head_cols(
        &self,
        xs: &[Vec<i64>],
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, LmHeadError> {
        let n_embd = self.config.n_embd;
        let vocab = self.config.vocab_size;
        if xs.len() > 1 && matches!(mode, KernelMode::Q8Act | KernelMode::Ternary) {
            let t = self.lm_head();
            let need = lm_head::head_payload_len(t.ggml_type, &t.dims, n_embd, vocab)?;
            let extent = t.byte_range.end - t.byte_range.start;
            if need > extent {
                return Err(LmHeadError::ShortPayload { need, have: extent });
            }
            let mut i = 0usize;
            while i < xs.len() {
                if xs[i].len() != n_embd {
                    return Err(LmHeadError::ShapeMismatch {
                        x: xs[i].len(),
                        n_embd,
                    });
                }
                i += 1;
            }
            let at = self
                .payload_ref(t.byte_range.start, need)
                .map_err(LmHeadError::Read)?;
            let idx = self.lm_head_index();
            if let Some(cols) = self.head_cols_kernel(idx, t.ggml_type, at, vocab, n_embd, xs) {
                return Ok(cols);
            }
        }
        let mut out = Vec::with_capacity(xs.len());
        let mut i = 0usize;
        while i < xs.len() {
            out.push(self.lm_head_fixed_sched(&xs[i], Sched::Resident, mode)?);
            i += 1;
        }
        Ok(out)
    }

    /// [`Engine::lm_head_fixed`]'s body with the payload source and row schedule chosen by
    /// `sched`: a borrowed resident slice projected by [`super::par::matvec_rows_chunked`] under
    /// [`Sched::Resident`], an owned `read_range` copy projected by the unchunked
    /// [`lm_head::lm_head_fixed`] under the test-only [`Sched::Sequential`]; `mode` selects the
    /// `Q8Act` chart kernel.
    fn lm_head_fixed_sched(
        &self,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<i64>, LmHeadError> {
        let n_embd = self.config.n_embd;
        let vocab = self.config.vocab_size;
        if x.len() != n_embd {
            return Err(LmHeadError::ShapeMismatch { x: x.len(), n_embd });
        }
        let t = self.lm_head();
        let need = lm_head::head_payload_len(t.ggml_type, &t.dims, n_embd, vocab)?;
        let extent = t.byte_range.end - t.byte_range.start;
        if need > extent {
            return Err(LmHeadError::ShortPayload { need, have: extent });
        }
        #[cfg(test)]
        let owned = match sched {
            Sched::Resident => None,
            Sched::Sequential => match self.read_range(t.byte_range.start, need) {
                Ok(p) => Some(p),
                Err(e) => return Err(LmHeadError::Read(e)),
            },
        };
        #[cfg(not(test))]
        match sched {
            Sched::Resident => {}
            #[cfg(target_arch = "wasm32")]
            Sched::Sequential => {}
        }
        let at = match self.payload_ref(t.byte_range.start, need) {
            Ok(p) => p,
            Err(e) => return Err(LmHeadError::Read(e)),
        };
        let raw: &[u8] = &self.resident()[at.off..at.off + at.len];
        #[cfg(test)]
        if let Some(p) = owned {
            assert_eq!(p.as_slice(), raw);
        }
        match (sched, mode) {
            (Sched::Resident, KernelMode::Exact) => {
                // One row chunk is exactly `lm_head_fixed`'s unchunked body; keep that leaf on
                // the executing path for the single-chunk case and chunk only when it pays.
                let budget = crate::engine::ops::par::row_chunk_budget(vocab);
                if budget <= 1 {
                    lm_head::lm_head_fixed(t.ggml_type, raw, vocab, n_embd, x)
                } else {
                    match crate::engine::ops::par::matvec_rows_chunked(
                        t.ggml_type,
                        raw,
                        vocab,
                        n_embd,
                        x,
                        budget,
                    ) {
                        Ok(v) => Ok(v),
                        Err(e) => Err(LmHeadError::Matvec(e)),
                    }
                }
            }
            (Sched::Resident, KernelMode::Q8Act | KernelMode::Ternary) => match self
                .proj_q8act_quantified(
                    self.lm_head_index(),
                    t.ggml_type,
                    at,
                    vocab,
                    n_embd,
                    x,
                    crate::engine::ops::par::row_chunk_budget(vocab),
                ) {
                Ok(v) => Ok(v),
                Err(e) => Err(LmHeadError::Matvec(e)),
            },
            #[cfg(any(test, target_arch = "wasm32"))]
            (Sched::Sequential, KernelMode::Exact) => {
                lm_head::lm_head_fixed(t.ggml_type, raw, vocab, n_embd, x)
            }
            #[cfg(any(test, target_arch = "wasm32"))]
            (Sched::Sequential, KernelMode::Q8Act | KernelMode::Ternary) => match q8act_projection(
                self,
                self.lm_head_index(),
                t.ggml_type,
                raw,
                vocab,
                n_embd,
                x,
                1,
            ) {
                Ok(v) => Ok(v),
                Err(e) => Err(LmHeadError::Matvec(e)),
            },
        }
    }

    /// `rms_eps` rounded once to Q32.32: `round(rms_eps · 2^32)`.
    ///
    /// This is the row-2 load-time float boundary for the norm addend. A negative, non-finite or
    /// out-of-`i64` epsilon refuses with [`StepError::BadConfig`].
    fn eps_fixed(&self) -> Result<i64, StepError> {
        match crate::engine::num::fp::prim::eps_q32(self.config.rms_eps) {
            Some(q) => Ok(q),
            None => Err(StepError::BadConfig(String::from("rms_eps"))),
        }
    }

    /// The directory index of the first entry named by `key`, or `None`.
    ///
    /// # Complexity
    ///
    /// `O(Σ name lengths)` byte comparisons, `O(1)` space: one [`key_matches`] per entry until the
    /// first match. Equal to the first-match scan of `TensorDirectory::get` over the rendered name.
    pub(crate) fn index_of(&self, key: TensorKey) -> Option<usize> {
        let entries = self.table().entries();
        let mut i = 0usize;
        while i < entries.len() {
            if key_matches(entries[i].name.as_bytes(), key) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// The checked payload of the `[n_in, n_out]` tensor `key` for `sched`: its directory index,
    /// quant family and payload extent.
    ///
    /// Refuses a missing tensor, a stored shape that is not `[n_in, n_out]`, and a quant family
    /// with no geometry (so no byte count). The extent is a [`PayloadRef`] into the resident file
    /// under [`Sched::Resident`]; the test-only [`Sched::Sequential`] also reads an owned
    /// `read_range` copy and asserts it equals that extent, so the dequantizers see identical
    /// bytes.
    fn tensor_payload(
        &self,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        sched: Sched,
    ) -> Result<(usize, u32, PayloadRef), StepError> {
        let idx = match self.index_of(key) {
            Some(i) => i,
            None => return Err(StepError::MissingTensor(key)),
        };
        let (ggml_type, at) = self.payload_at(idx, n_out, n_in, sched)?;
        Ok((idx, ggml_type, at))
    }

    /// One projection stage with the payload source and row schedule chosen by `sched`.
    ///
    /// A tensor named in `prism.hadamard.weight_names` is projected with its activation rotated by
    /// [`folded_activation`] first; every other tensor projects `x` verbatim. [`Sched::Resident`]
    /// projects the resident payload with [`super::par::matvec_rows_chunked`] over [`super::par::row_chunk_budget`] rows;
    /// the test-only [`Sched::Sequential`] projects it with one unchunked [`matvec_fixed`].
    fn step_proj_sched(
        &self,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        // A packed NanoQuant linear replaces the payload: the factors carry the weight, so no GGUF
        // row is read. The kernel mode chooses the activation reader over those factors: the exact
        // Q32.32 words, the Q8 block codes or the ternary trit masks.
        if let Some(out) = self.nq_proj(key, n_out, n_in, x, mode)? {
            return Ok(out);
        }
        let (idx, ggml_type, at) = self.tensor_payload(key, n_out, n_in, sched)?;
        let folded = folded_activation(self.hadamard(), key, x)?;
        let activation: &[i64] = opt_slice(&folded).unwrap_or(x);
        let out = match (sched, mode) {
            (Sched::Resident, KernelMode::Exact) => crate::engine::ops::par::matvec_rows_chunked(
                ggml_type,
                &self.resident()[at.off..at.off + at.len],
                n_out,
                n_in,
                activation,
                crate::engine::ops::par::row_chunk_budget(n_out),
            ),
            (Sched::Resident, KernelMode::Q8Act | KernelMode::Ternary) => self
                .proj_q8act_quantified(
                    idx,
                    ggml_type,
                    at,
                    n_out,
                    n_in,
                    activation,
                    crate::engine::ops::par::row_chunk_budget(n_out),
                ),
            #[cfg(any(test, target_arch = "wasm32"))]
            (Sched::Sequential, KernelMode::Exact) => matvec_fixed(
                ggml_type,
                &self.resident()[at.off..at.off + at.len],
                n_out,
                n_in,
                activation,
            ),
            #[cfg(any(test, target_arch = "wasm32"))]
            (Sched::Sequential, KernelMode::Q8Act | KernelMode::Ternary) => q8act_projection(
                self,
                idx,
                ggml_type,
                &self.resident()[at.off..at.off + at.len],
                n_out,
                n_in,
                activation,
                1,
            ),
        };
        out.map_err(StepError::Matvec)
    }

    /// Q/K/V projections for `layer` over `x` under the schedule and kernel `sched`/`mode` choose:
    /// under the resident `Q8Act` path, when none of the three is `Hadamard`-folded, they share one
    /// `quant_act(x)` and one [`projections_q8act`] dispatch; every other case keeps the three
    /// [`Engine::step_proj_sched`] calls. Errors as [`Engine::step_proj_sched`], Q/K/V in order.
    fn step_qkv_sched(
        &self,
        layer: usize,
        outs: [usize; 3],
        n_in: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<[Vec<i64>; 3], StepError> {
        let keys = [part_key(layer, 0), part_key(layer, 1), part_key(layer, 2)];
        // A packed linear has no GGUF payload for the fused group to read; the three separate
        // calls take the sign-rank product.
        if keys.iter().any(|&key| self.nq_linear(key).is_some())
            || !matches!(sched, Sched::Resident)
            || !matches!(mode, KernelMode::Q8Act | KernelMode::Ternary)
        {
            return self.step_qkv_separate(&keys, outs, n_in, x, sched, mode);
        }
        // A folded tensor rotates its own activation copy; the three then share no `quant_act`.
        match folded_activation(self.hadamard(), keys[0], x) {
            Err(e) => return Err(e),
            Ok(Some(_)) => return self.step_qkv_separate(&keys, outs, n_in, x, sched, mode),
            Ok(None) => {}
        }
        let (i0, t0, at0) = self.tensor_payload(keys[0], outs[0], n_in, Sched::Resident)?;
        match folded_activation(self.hadamard(), keys[1], x) {
            Err(e) => return Err(e),
            Ok(Some(_)) => return self.step_qkv_separate(&keys, outs, n_in, x, sched, mode),
            Ok(None) => {}
        }
        let (i1, t1, at1) = self.tensor_payload(keys[1], outs[1], n_in, Sched::Resident)?;
        match folded_activation(self.hadamard(), keys[2], x) {
            Err(e) => return Err(e),
            Ok(Some(_)) => return self.step_qkv_separate(&keys, outs, n_in, x, sched, mode),
            Ok(None) => {}
        }
        let (i2, t2, at2) = self.tensor_payload(keys[2], outs[2], n_in, Sched::Resident)?;
        self.check_pq2_lut(t0, i0)?;
        self.check_pq2_lut(t1, i1)?;
        self.check_pq2_lut(t2, i2)?;
        let specs = [
            ffn_spec(t0, at0, i0, outs[0]),
            ffn_spec(t1, at1, i1, outs[1]),
            ffn_spec(t2, at2, i2, outs[2]),
        ];
        let out = projections_q8act(
            self.resident(),
            self.pq2_luts(),
            n_in,
            x,
            &specs,
            crate::engine::ops::par::row_chunk_budget(outs[0]),
        )?;
        let mut out = out;
        assert!(out.len() == 3, "three projections");
        let value = out.pop().unwrap_or_default();
        let key = out.pop().unwrap_or_default();
        let query = out.pop().unwrap_or_default();
        Ok([query, key, value])
    }

    /// The three separate `step_proj_sched` calls, Q/K/V in order: the non-fused arm of
    /// [`Engine::step_qkv_sched`].
    fn step_qkv_separate(
        &self,
        keys: &[TensorKey; 3],
        outs: [usize; 3],
        n_in: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<[Vec<i64>; 3], StepError> {
        Ok([
            self.step_proj_sched(keys[0], outs[0], n_in, x, sched, mode)?,
            self.step_proj_sched(keys[1], outs[1], n_in, x, sched, mode)?,
            self.step_proj_sched(keys[2], outs[2], n_in, x, sched, mode)?,
        ])
    }

    /// One projection stage over the resident payload (see [`Engine::step_proj_sched`]).
    ///
    /// Test-only: the executing [`Engine::compose_logits`] calls [`Engine::step_proj_sched`]
    /// directly; the existing projection tests keep this resident-path entry point.
    #[cfg(test)]
    fn step_proj(
        &self,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        x: &[i64],
    ) -> Result<Vec<i64>, StepError> {
        self.step_proj_sched(key, n_out, n_in, x, Sched::Resident, KernelMode::Exact)
    }

    /// One greedy decode step with a caller-owned KV cache (the stateful rows 0–8 composition).
    ///
    /// The position is the cache's `KvState::occupied(0)`, and the layer order and residual
    /// placement are the fold order named in the module documentation.
    ///
    /// # Errors
    ///
    /// Each stage's typed refusal through [`StepError`] — the token, cache, tensor, geometry and
    /// kernel refusals; no value is clamped.
    pub fn step_with_kv(&self, kv: &mut KvState, token_id: u32) -> Result<StepResult, StepError> {
        let logits = self.step_logits(kv, token_id)?;
        let token = argmax_fixed(&logits).map_err(StepError::Sampler)?;
        Ok(StepResult {
            token,
            logits_checksum: fixed_vector_checksum(&logits),
        })
    }

    /// The Q32.32 logits of one decode step with a caller-owned KV cache (rows 0–8 composed).
    ///
    /// [`Engine::step_with_kv`]'s body separated from its sampler: it advances `kv` by one position
    /// for `token_id` and returns the logit vector [`Engine::lm_head_fixed`] produces from the final
    /// normed residual, at the cache's `KvState::occupied(0)`.
    ///
    /// # Errors
    ///
    /// Each stage's typed refusal through [`StepError`]; no value is clamped.
    ///
    /// # Complexity
    ///
    /// `O(n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd)` checked multiply-adds plus the LM-head
    /// dequantization; `O(vocab)` for the returned logits.
    pub fn step_logits(&self, kv: &mut KvState, token_id: u32) -> Result<Vec<i64>, StepError> {
        self.compose_logits(kv, None, token_id, STEP_SCHED, KernelMode::Exact, false)
    }

    /// The Q32.32 logits of one decode step under the projection kernel `mode`.
    ///
    /// `KernelMode::Exact` is [`Engine::step_logits`] byte-for-byte; `KernelMode::Q8Act` routes every
    /// projection through the pooled int8-activation kernels.
    ///
    /// # Errors
    /// As [`Engine::step_logits`].
    pub fn step_logits_with_mode(
        &self,
        kv: &mut KvState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            kv,
            None,
            token_id,
            STEP_SCHED,
            mode,
            // The Q8Act head dispatch is the pool's; the wasm build keeps the serial head even
            // when a caller asks for that kernel.
            matches!(mode, KernelMode::Q8Act | KernelMode::Ternary)
                && !cfg!(target_arch = "wasm32"),
        )
    }

    /// [`Engine::compose_final`] followed by the head: one step's logits.
    ///
    /// # Errors
    ///
    /// As [`Engine::compose_final`], plus the head's [`LmHeadError`].
    fn compose_logits(
        &self,
        kv: &mut KvState,
        affine: Option<&mut AffineState>,
        token_id: u32,
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let xf = self.compose_final(kv, affine, token_id, sched, mode, parallel_heads)?;
        self.lm_head_fixed_sched(&xf, sched, mode)
            .map_err(StepError::LmHead)
    }

    /// The trunk of one step: embedding, every block, and the output norm; returns the normed final
    /// state the head reads. `parallel_heads` selects the [`super::chart::ffn::attn_heads`] pool
    /// dispatch, the serial loop being the test oracle.
    #[allow(clippy::too_many_lines, reason = "one linear rows 0-8 composition")]
    pub(crate) fn compose_final(
        &self,
        kv: &mut KvState,
        affine: Option<&mut AffineState>,
        token_id: u32,
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let mut out =
            self.compose_cols(kv, affine, &[token_id], sched, mode, parallel_heads, None)?;
        out.pop()
            .ok_or(StepError::BadConfig(String::from("empty column block")))
    }

    /// The output-normed trunk state of one decode step: embedding, the first `config.n_layer`
    /// blocks and the output norm, before the head.
    ///
    /// [`Engine::step_logits_state`]'s trunk without its head; the check surface for a device
    /// that executes the trunk (a caller that lowers `config.n_layer` reads a partial trunk).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    pub fn step_trunk_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_final(
            &mut st.kv,
            st.affine.as_mut(),
            token_id,
            Sched::Resident,
            mode,
            matches!(mode, KernelMode::Q8Act | KernelMode::Ternary),
        )
    }

    /// The pre-change sequential reference for the bit-identity test: [`Engine::step_logits`]'s
    /// composition with owned `read_range` payloads and one unchunked [`matvec_fixed`] per
    /// projection. Test-only; the executing path is [`Engine::step_logits`].
    #[cfg(test)]
    fn step_logits_sequential(
        &self,
        kv: &mut KvState,
        token_id: u32,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            kv,
            None,
            token_id,
            Sched::Sequential,
            KernelMode::Exact,
            false,
        )
    }

    /// One greedy decode step with a fresh one-position [`KvState`].
    ///
    /// The stateless single-token forward at position 0; a decode loop that accumulates context
    /// calls [`Engine::step_with_kv`].
    ///
    /// # Errors
    /// As [`Engine::step_with_kv`].
    pub fn step(&mut self, token_id: u32) -> Result<StepResult, StepError> {
        let mut kv = self.decode_state(1)?;
        self.step_with_kv(&mut kv, token_id)
    }

    /// A decode [`KvState`] with this engine's geometry and capacity `cap`.
    ///
    /// The loader-side wiring for [`Engine::step_with_kv`]: the caller asks for the capacity its
    /// decode loop needs, bounded by the config's `context_length` when the file states one.
    ///
    /// # Errors
    ///
    /// [`StepError::BadConfig`] for a zero or above-context capacity, and [`StepError::Kv`] with
    /// [`AttnError`] for a geometry [`KvState::new`] refuses.
    pub fn decode_state(&self, cap: usize) -> Result<KvState, StepError> {
        if cap == 0 {
            return Err(StepError::BadConfig(String::from("kv capacity")));
        }
        let ctx = self.config.context_length;
        if ctx != 0 && cap > ctx {
            return Err(StepError::BadConfig(String::from(
                "kv capacity above context_length",
            )));
        }
        KvState::new(
            count_mixers(&self.config.mixers, MixerKind::Softmax),
            self.config.n_head,
            self.config.n_head_kv,
            self.config.head_dim,
            cap,
        )
        .map_err(StepError::Kv)
    }

    /// A decode state for this engine's schedule: the softmax list carrier sized `cap`, and the
    /// affine carrier when any layer is affine.
    ///
    /// # Errors
    ///
    /// As [`Engine::decode_state`].
    ///
    /// # Complexity
    ///
    /// The [`KvState`] planes plus `O(n_affine · state words)`, zeroed once.
    pub fn state(&self, cap: usize) -> Result<DecodeState, StepError> {
        let kv = self.decode_state(cap)?;
        Ok(self.state_with_kv(kv))
    }

    /// A decode state whose KV planes live in one writable cache file at `path`.
    ///
    /// The file is created (or truncated), allocated to its full capacity and mapped
    /// `MAP_SHARED`; see [`KvState::with_file`]. The capacity check is the allocation, so a cache
    /// the filesystem cannot hold is refused here.
    ///
    /// # Errors
    ///
    /// [`StepError::BadConfig`] for a zero or above-context capacity, [`StepError::Kv`] for a
    /// geometry refusal, and [`StepError::KvFile`] for a refused file.
    pub fn state_with_file(&self, cap: usize, path: &str) -> Result<DecodeState, StepError> {
        if cap == 0 {
            return Err(StepError::BadConfig(String::from("kv capacity")));
        }
        let ctx = self.config.context_length;
        if ctx != 0 && cap > ctx {
            return Err(StepError::BadConfig(String::from(
                "kv capacity above context_length",
            )));
        }
        let kv = KvState::with_file(
            path,
            count_mixers(&self.config.mixers, MixerKind::Softmax),
            self.config.n_head,
            self.config.n_head_kv,
            self.config.head_dim,
            cap,
        )
        .map_err(|e| StepError::KvFile(e.to_string()))?;
        Ok(self.state_with_kv(kv))
    }

    /// Wrap a resolved [`KvState`] with the affine carrier this model's schedule needs.
    /// A decode state around a caller-supplied [`KvState`]: the affine state, when the model has
    /// affine mixers, is allocated fresh; the KV planes are the caller's.
    pub fn state_with_kv(&self, kv: KvState) -> DecodeState {
        let n_affine = count_mixers(&self.config.mixers, MixerKind::Affine);
        let affine = match self.config.affine {
            Some(g) if n_affine > 0 => Some(AffineState::new(&g, n_affine)),
            _ => None,
        };
        DecodeState { kv, affine }
    }

    /// One greedy decode step over any schedule; `mode` selects the projection kernel.
    ///
    /// # Errors
    ///
    /// As [`Engine::step_with_kv`], plus [`StepError::BadConfig`] when an affine layer has no
    /// affine state.
    pub fn step_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<StepResult, StepError> {
        let logits = self.step_logits_state(st, token_id, mode)?;
        let token = argmax_fixed(&logits).map_err(StepError::Sampler)?;
        Ok(StepResult {
            token,
            logits_checksum: fixed_vector_checksum(&logits),
        })
    }

    /// The logits of one decode step over any schedule (the one executor).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    pub fn step_logits_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            &mut st.kv,
            st.affine.as_mut(),
            token_id,
            Sched::Resident,
            mode,
            matches!(mode, KernelMode::Q8Act | KernelMode::Ternary),
        )
    }

    /// Prefill `tokens` in blocks of near-equal width bounded by [`prefill_block`],
    /// returning every position's output-normed final state; exactly the states of one-token
    /// steps (`qwen35_two_columns_equal_two_single_steps`).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    ///
    /// # Complexity
    ///
    /// `⌈n / prefill_block()⌉` block passes; each reads the `Q8_0` weights once.
    pub(crate) fn prefill_cols(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        self.prefill_cols_width(st, tokens, mode, prefill_block())
    }

    pub(crate) fn prefill_cols_width(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
        w: usize,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let mut out = Vec::with_capacity(tokens.len());
        let num_blocks = tokens.len().div_ceil(w.max(1));
        let base = tokens.len().checked_div(num_blocks).unwrap_or(0);
        let rem = tokens.len().checked_rem(num_blocks).unwrap_or(0);
        let mut start = 0;
        for i in 0..num_blocks {
            let size = if i < rem { base + 1 } else { base };
            let block = &tokens[start..start + size];
            start += size;
            out.extend(self.compose_cols(
                &mut st.kv,
                st.affine.as_mut(),
                block,
                Sched::Resident,
                mode,
                matches!(mode, KernelMode::Q8Act | KernelMode::Ternary),
                None,
            )?);
        }
        Ok(out)
    }

    /// Prefill `tokens` in blocks and return the greedy next token (the prompt's continuation).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`]; [`StepError::BadConfig`] for an empty prompt.
    pub fn prefill_state(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
    ) -> Result<u32, StepError> {
        self.prefill_state_with_width(st, tokens, mode, prefill_block())
    }

    /// Prefill `tokens` and return every position's output-normed final state, before the head.
    ///
    /// Column `c` is exactly the state of one single-token step at position `c`, at the
    /// prefill block width the engine chooses. The caller reads the head itself when it needs the
    /// logits of every position, as the perplexity harness does.
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`]; [`StepError::BadConfig`] for an empty prompt.
    ///
    /// # Complexity
    ///
    /// `⌈n / prefill_block()⌉` block passes, each reading the served weights once where the kernel
    /// supports a multi-column pass; `O(n · n_embd)` returned words.
    pub fn prefill_states(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        if tokens.is_empty() {
            return Err(StepError::BadConfig(String::from("empty prompt")));
        }
        self.prefill_cols(st, tokens, mode)
    }

    pub(crate) fn prefill_state_with_width(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
        w: usize,
    ) -> Result<u32, StepError> {
        if tokens.is_empty() {
            return Err(StepError::BadConfig(String::from("empty prompt")));
        }
        let num_blocks = tokens.len().div_ceil(w.max(1));
        let base = tokens.len() / num_blocks;
        let rem = tokens.len() % num_blocks;
        let mut start = 0;
        let mut last_col = None;
        for i in 0..num_blocks {
            let size = if i < rem { base + 1 } else { base };
            let block = &tokens[start..start + size];
            start += size;
            let mut cols = self.compose_cols(
                &mut st.kv,
                st.affine.as_mut(),
                block,
                Sched::Resident,
                mode,
                matches!(mode, KernelMode::Q8Act | KernelMode::Ternary),
                None,
            )?;
            if i + 1 == num_blocks {
                last_col = cols.pop();
            }
        }
        let xf = last_col.ok_or(StepError::BadConfig(String::from("empty prompt")))?;
        self.head_greedy_state(&xf, mode)
    }

    /// Greedily select the next token from an output-normed final state.
    fn head_greedy_state(&self, xf: &[i64], mode: KernelMode) -> Result<u32, StepError> {
        let (vocab, n_embd) = (self.config.vocab_size, self.config.n_embd);
        let _head = super::trace::span(
            super::trace::Phase::Head,
            1,
            super::trace::q8_bytes(vocab, n_embd),
            super::trace::macs(vocab, n_embd, 1),
        );
        if matches!(mode, KernelMode::Q8Act | KernelMode::Ternary) && bitplane_enabled() {
            if let Some(bp) = self.bitplane_head() {
                if let Some(q) = super::chart::ffn::quant_act_pooled(xf) {
                    if super::chart::q8act_fast_bound_ok(xf.len(), 128, &q, true) {
                        if let Some((token, _)) = bp.argmax(&q) {
                            return Ok(token);
                        }
                    }
                }
            }
        }
        let logits = self
            .lm_head_fixed_sched(xf, Sched::Resident, mode)
            .map_err(StepError::LmHead)?;
        argmax_fixed(&logits).map_err(StepError::Sampler)
    }

    /// The greedy next token of one decode step over any schedule. In `Q8Act` with a `Q8_0` head it
    /// reads the head through the bit-plane argmax ([`super::kern::bitplane`], licensed by
    /// `BitplaneArgmax.bitplane_argmax_exact`); it returns exactly `argmax_fixed` of
    /// [`Engine::step_logits_state`]'s logits, without forming them.
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    ///
    /// # Complexity
    ///
    /// The trunk plus, on the bit-plane path, `18/34` of the head's bytes and the survivors' low
    /// planes; otherwise the dense head.
    pub fn next_token_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<u32, StepError> {
        let xf = self.compose_final(
            &mut st.kv,
            st.affine.as_mut(),
            token_id,
            Sched::Resident,
            mode,
            matches!(mode, KernelMode::Q8Act | KernelMode::Ternary),
        )?;
        self.head_greedy_state(&xf, mode)
    }
}

/// The decode state of any schedule: the softmax list carrier and, when the schedule has affine
/// layers, the affine carrier.
#[derive(Debug, Clone)]
pub struct DecodeState {
    /// Keys and values of the softmax layers.
    pub kv: KvState,
    /// The affine layers' fixed-size states.
    pub affine: Option<AffineState>,
}

#[path = "step/attn_block.rs"]
pub(crate) mod attn_block;
#[path = "step/cols.rs"]
mod cols;
#[path = "step/softmax.rs"]
mod softmax;
#[path = "step/spec.rs"]
mod spec;
pub use spec::{MtpState, SpecStats};
#[path = "step/cold_store.rs"]
pub mod cold_store;

#[cfg(test)]
#[path = "step/prefill_tests.rs"]
mod prefill_tests;
#[cfg(test)]
#[path = "step/tests.rs"]
mod tests;
