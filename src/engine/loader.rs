//! GGUF engine loader: Qwen2 config resolution, a named tensor table over the file bytes and the
//! bounds arithmetic the embedding reader relies on.
//!
//! The loader reads the header through [`crate::format::gguf`] and resolves the `qwen2.*` metadata
//! into an [`EngineConfig`]. It then walks the tensor directory once, records every entry as
//! `(name, ggml type, dims, quant kind, byte range)` and refuses the load if any entry's range
//! leaves the file. Quant unpacking lives in `crate::format::quant`; this module only records the
//! directory extent and the exact geometry where that module has it.
//!
//! Payloads stay in the file image: on Linux [`Engine::open`] maps the GGUF read-only and
//! [`Engine::payload`] serves a borrowed slice of the mapping, so the pages are file-backed and
//! reclaimable instead of a second anonymous copy of the skeleton; other platforms read the file
//! into a `Box<[u8]>` ([`Engine::from_bytes`]). No directory value is used before it is checked,
//! and no failure is a panic. [`Engine::read_range`] is retained for callers that need owned bytes.

use core::ops::Range;
use std::sync::Arc;

#[cfg(not(target_os = "linux"))]
use crate::format::gguf::host_read_all;
use crate::format::gguf::{host_read_at, parse_header, GVal, GgufHeader};
use crate::format::quant;

pub use super::error::EngineError;
use super::error::{
    TensorKey, PART_SSM_A, PART_SSM_CONV1D, PART_SSM_DT_BIAS, PART_SSM_NORM, PART_TOKEN_EMBD,
};
use super::kern::lut::Pq2Lut;
use crate::engine::num::f32_convert::dequant_f32_row;
use crate::engine::num::fp::F64;
use crate::engine::num::value::round_shift_32;

#[path = "loader/hadamard.rs"]
mod hadamard;
pub use hadamard::{hadamard_meta, HadamardMeta};

#[path = "loader/nq.rs"]
mod nq;
use crate::format::nq::NqPack;
use nq::{build_nq_table, NqTable};

/// Bits of the default `RoPE` base (`10_000.0_f64`) when `*.rope.freq_base` is absent. A bit pattern,
/// not a float constant: the extraction cannot translate float constants.
pub const DEFAULT_ROPE_THETA_BITS: u64 = 0x40C3_8800_0000_0000;
/// Bits of the default `RMSNorm` epsilon (`1e-6_f64`) when `*.attention.layer_norm_rms_epsilon` is absent.
pub const DEFAULT_RMS_EPS_BITS: u64 = 0x3EB0_C6F7_A0B5_ED8D;

/// The ggml quant families this crate names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// ggml `F32`.
    F32,
    /// ggml `F16`.
    F16,
    /// ggml `Q4_K`, a 256-weight affine super-block.
    Q4K,
    /// ggml `Q6_K`, a 256-weight affine super-block.
    Q6K,
    /// ggml `Q8_0`, a 32-weight block with one `f16` scale.
    Q8_0,
    /// ggml `Q5_0`, a 32-weight block with one `f16` scale, a high-bit plane and nibbles.
    Q5_0,
    /// A ggml type `crate::format::quant` has no geometry for; the raw id is kept.
    Other(u32),
}

impl QuantKind {
    /// Classify a ggml type id.
    #[must_use]
    pub fn from_ggml(t: u32) -> QuantKind {
        match t {
            quant::GGML_TYPE_F32 => QuantKind::F32,
            quant::GGML_TYPE_F16 => QuantKind::F16,
            quant::GGML_TYPE_Q4_K => QuantKind::Q4K,
            quant::GGML_TYPE_Q6_K => QuantKind::Q6K,
            quant::GGML_TYPE_Q8_0 => QuantKind::Q8_0,
            quant::GGML_TYPE_Q5_0 => QuantKind::Q5_0,
            other => QuantKind::Other(other),
        }
    }

    /// The ggml name, for diagnostics.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            QuantKind::F32 => String::from("F32"),
            QuantKind::F16 => String::from("F16"),
            QuantKind::Q4K => String::from("Q4_K"),
            QuantKind::Q6K => String::from("Q6_K"),
            QuantKind::Q8_0 => String::from("Q8_0"),
            QuantKind::Q5_0 => String::from("Q5_0"),
            QuantKind::Other(_) => String::from("OTHER"),
        }
    }
}

/// The sequence mixer a layer runs: one of the two carriers of the mixer primitive.
///
/// `Softmax` keeps the append-only key/value list and reads it with softmax attention; `Affine`
/// keeps a fixed-size state updated by an affine map `S ← A_t·S + B_t` and read by `Sᵀq` (Gated
/// `DeltaNet` in `qwen35`). A model is a per-layer schedule of these, not a separate code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    /// Softmax attention over the append-only key/value list.
    Softmax,
    /// The fixed-size affine-state mixer.
    Affine,
}

/// Geometry of the affine-state mixer (`*.ssm.*` metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineGeom {
    /// `*.ssm.conv_kernel`: taps of the causal depthwise convolution.
    pub conv_kernel: usize,
    /// `*.ssm.group_count`: key heads.
    pub n_k_heads: usize,
    /// `*.ssm.time_step_rank`: value heads (one decay and one β each).
    pub n_v_heads: usize,
    /// `*.ssm.state_size`: key width per head.
    pub head_k: usize,
    /// `*.ssm.inner_size / time_step_rank`: value width per head.
    pub head_v: usize,
}

/// The resolved model configuration: dimensions, the per-layer mixer schedule and the attention
/// flags that distinguish one decoder from another.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    /// `general.architecture`.
    pub architecture: String,
    /// `*.block_count`.
    pub n_layer: usize,
    /// `*.attention.head_count`.
    pub n_head: usize,
    /// `*.attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `*.embedding_length`.
    pub n_embd: usize,
    /// `*.feed_forward_length`.
    pub n_ff: usize,
    /// Vocabulary size, from `token_embd.weight` dims and cross-checked against the token array.
    pub vocab_size: usize,
    /// `*.context_length`, or `0` when unstated.
    pub context_length: usize,
    /// `*.rope.freq_base`.
    pub rope_theta: F64,
    /// `*.attention.layer_norm_rms_epsilon`.
    pub rms_eps: F64,
    /// `*.attention.key_length` when stated, else `n_embd / n_head` (checked integral).
    pub head_dim: usize,
    /// Per trunk layer, the mixer it runs. `*.full_attention_interval` = `f` makes layer `l` softmax
    /// iff `(l + 1) % f == 0` and affine otherwise; without the key every layer is softmax.
    pub mixers: Vec<MixerKind>,
    /// Leading dimensions of each head that `RoPE` rotates: `*.rope.dimension_count`, else `head_dim`.
    pub rot_dims: usize,
    /// Whether `attn_q` also emits a per-head sigmoid output gate: its rows are `2·n_head·head_dim`.
    pub attn_gate: bool,
    /// The affine mixer's geometry, present iff some layer is [`MixerKind::Affine`].
    pub affine: Option<AffineGeom>,
    /// `*.nextn_predict_layers`: multi-token-prediction blocks after the trunk (`blk.{n_layer}..`),
    /// softmax-attention blocks the speculative decoder drafts with; `0` when absent.
    pub n_nextn: usize,
}

/// One tensor-directory entry with its resolved quant kind and checked file extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// GGUF name, e.g. `token_embd.weight`.
    pub name: String,
    /// ggml type id as stored.
    pub ggml_type: u32,
    /// The named quant family.
    pub quant: QuantKind,
    /// Dimensions as stored, fastest-varying first.
    pub dims: Vec<u64>,
    /// Total element count, checked.
    pub n_elements: u64,
    /// Exact storage bytes when `crate::format::quant` has the geometry; `None` otherwise.
    pub n_bytes: Option<u64>,
    /// Absolute half-open byte range: from this entry's data offset to the next entry's data
    /// offset in directory-offset order (the file end for the last entry).
    pub byte_range: Range<u64>,
}

/// The named tensor directory.
#[derive(Debug, Clone)]
pub struct TensorTable {
    entries: Vec<TensorEntry>,
}

impl TensorTable {
    /// The entry with `name`, compared by bytes.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        let mut found: Option<&TensorEntry> = None;
        let mut i = 0usize;
        while i < self.entries.len() && found.is_none() {
            if self.entries[i].name.as_bytes() == name.as_bytes() {
                found = Some(&self.entries[i]);
            }
            i += 1;
        }
        found
    }

    /// Every entry, in file directory order.
    #[must_use]
    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The per-layer `F32` gain and attention-bias vectors the decode step needs, resolved to Q32.32
/// once at load.
///
/// The step composes the fold `cpu_rms` gain (`attn_norm.weight`, `ffn_norm.weight`,
/// `output_norm.weight`) and the Qwen2 attention biases (`attn_{q,k,v}.bias`). Those are the only
/// `F32` tensors on the decode path; reading them per layer per token would be I/O-bound, so the
/// loader resolves them once. An absent optional bias stays `None` (the fold `if let Some(b)`
/// arms); a present tensor with the wrong shape or type refuses at load with
/// [`EngineError::BadVector`], and a non-finite or out-of-range value with [`EngineError::F32`].
#[derive(Debug, Clone, Default)]
pub struct DecodeVectors {
    /// Per layer: `[attn_norm.weight, ffn_norm.weight]`, each `n_embd` wide.
    pub norms: Vec<[Option<Vec<i64>>; 2]>,
    /// Per layer: `[attn_q.bias, attn_k.bias, attn_v.bias]`; Q is `n_head·head_dim` wide, K and V
    /// are `n_head_kv·head_dim`.
    pub biases: Vec<[Option<Vec<i64>>; 3]>,
    /// Per layer: `[attn_q_norm.weight, attn_k_norm.weight]`, the Qwen3 per-head QK `RMSNorm`
    /// gains, each `head_dim` wide. Both are `None` on a Qwen2 model, which has no such tensor.
    pub qk_norms: Vec<[Option<Vec<i64>>; 2]>,
    /// `output_norm.weight`, `n_embd` wide.
    pub output_norm: Option<Vec<i64>>,
    /// Per layer: the affine mixer's `F32` vectors, `Some` exactly on [`MixerKind::Affine`] layers.
    pub affine: Vec<Option<AffineVectors>>,
    /// Per multi-token-prediction block: `[enorm, hnorm, shared_head_norm]`, each `n_embd` wide.
    pub nextn: Vec<[Option<Vec<i64>>; 3]>,
}

/// One affine-mixer layer's `F32` parameters in Q32.32.
#[derive(Debug, Clone, Default)]
pub struct AffineVectors {
    /// `ssm_conv1d.weight`, channel-major: tap `t` of channel `c` at `c·conv_kernel + t`.
    pub conv: Vec<i64>,
    /// `ssm_dt.bias`, one per value head: added to the decay input before `softplus`.
    pub dt: Vec<i64>,
    /// `ssm_a`, one per value head: `−exp(A_log)`, the decay coefficient.
    pub a: Vec<i64>,
    /// `ssm_norm.weight`, `head_v` wide: the gated output norm's gain.
    pub norm: Vec<i64>,
}

impl DecodeVectors {
    /// `[attn_norm, ffn_norm]` for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn norms_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 2]> {
        self.norms.get(layer)
    }

    /// `[q, k, v]` biases for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn biases_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 3]> {
        self.biases.get(layer)
    }

    /// `[attn_q_norm, attn_k_norm]` for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn qk_norms_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 2]> {
        self.qk_norms.get(layer)
    }

    /// Apply `layer`'s per-head QK `RMSNorm` to `query` and `key`, before `RoPE`.
    ///
    /// A model with no QK-norm tensors (Qwen2) gets `None` gains and both vectors are unchanged;
    /// Qwen3 gets the `head_dim`-wide gains applied per `hd`-wide head by [`qk_norm_heads`].
    ///
    /// # Errors
    ///
    /// As [`qk_norm_heads`].
    pub(crate) fn apply_qk_norms(
        &self,
        layer: usize,
        query: &mut [i64],
        key: &mut [i64],
        hd: usize,
        eps: i64,
    ) -> Result<(), super::StepError> {
        let present = self.qk_norms_of(layer).is_some();
        let q_gain = if present {
            opt_slice(&self.qk_norms[layer][0])
        } else {
            None
        };
        let k_gain = if present {
            opt_slice(&self.qk_norms[layer][1])
        } else {
            None
        };
        qk_norm_heads(q_gain, query, hd, eps)?;
        qk_norm_heads(k_gain, key, hd, eps)
    }
}

/// `o` as an optional slice. `Option::as_deref` extracts to a `DerefVec` instance that the Lean model
/// declares with an implicit type argument, so the extracted call does not elaborate.
#[allow(
    clippy::ref_option,
    reason = "callers hold a borrowed slot of a bias array; Option::as_ref does not change the extracted call shape"
)]
pub(crate) fn opt_slice<T>(o: &Option<Vec<T>>) -> Option<&[T]> {
    match o {
        Some(v) => Some(v),
        None => None,
    }
}

/// The Qwen3 per-head QK `RMSNorm` gain: apply [`rms_norm_fixed`](super::norm::rms_norm_fixed) to
/// each `hd`-wide head of `x` in place, then the `head_dim`-wide per-channel gain. `gain` is `None`
/// on a model with no QK-norm tensor, where `x` is unchanged.
///
/// A `None` gain is a no-op; a gain whose width is not `hd`, a non-multiple-of-`hd` vector and the
/// norm's own range refusals propagate through [`super::StepError`].
///
/// # Complexity
///
/// `O(x.len())` time for the norm and the gain, `O(hd)` scratch per head.
#[allow(
    clippy::many_single_char_names,
    reason = "x, i, r, v are the head vector, index, product and narrowed value"
)]
pub(crate) fn qk_norm_heads(
    gain: Option<&[i64]>,
    x: &mut [i64],
    hd: usize,
    eps: i64,
) -> Result<(), super::StepError> {
    let Some(gain) = gain else {
        return Ok(());
    };
    if hd == 0 || gain.len() != hd || !x.len().is_multiple_of(hd) {
        return Err(super::StepError::BadConfig(String::from(
            "qk norm geometry",
        )));
    }
    let heads = x.len() / hd;
    let mut h = 0usize;
    let mut err: Option<super::StepError> = None;
    while h < heads && err.is_none() {
        let base = h * hd;
        match crate::engine::ops::norm::rms_norm_fixed(&x[base..base + hd], eps) {
            Err(e) => err = Some(super::StepError::from(e)),
            Ok(normed) => {
                let mut i = 0usize;
                while i < hd && err.is_none() {
                    // |normed[i]·gain[i]| <= 2^126 < 2^127, so the i128 product cannot overflow.
                    let r = round_shift_32(i128::from(normed[i]) * i128::from(gain[i]));
                    if (i128::from(i64::MIN)..=i128::from(i64::MAX)).contains(&r) {
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "r is in i64 range by the check above"
                        )]
                        let v = r as i64;
                        x[base + i] = v;
                    } else {
                        x[base + i] = 0;
                        err = Some(super::StepError::Overflow);
                    }
                    i += 1;
                }
            }
        }
        h += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The resident GGUF image: an owned read for an in-memory image, or the read-only file mapping
/// [`Engine::open`] uses on Linux.
///
/// Mapping matters at 8B: the skeleton is 1.33 GB, but a decode step reads one embedding row and
/// the whole head. Keeping the file-backed pages reclaimable lets the kernel keep the head hot and
/// evict embedding rows the step never touches, instead of pinning the whole skeleton in anonymous
/// memory.
#[derive(Debug)]
enum Image {
    /// The whole file, copied into the engine.
    Owned(Box<[u8]>),
    /// The whole file, file-backed.
    #[cfg(target_os = "linux")]
    Mapped(crate::format::nq::MappedFile),
}

impl Image {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(b) => b,
            #[cfg(target_os = "linux")]
            Self::Mapped(m) => m.bytes(),
        }
    }
}

/// A loaded GGUF: resolved config, checked tensor directory and the resident file bytes.
#[derive(Debug)]
pub struct Engine {
    /// Path the file was opened from.
    pub path: String,
    /// Total file size in bytes.
    pub file_len: u64,
    /// The resolved Qwen2.5 config.
    pub config: EngineConfig,
    /// The validated `prism.hadamard.*` fold contract, or `None` for an unrotated model.
    pub hadamard: Option<HadamardMeta>,
    /// The whole file, opened once; [`Engine::payload`] serves checked slices of it.
    data: Image,
    table: TensorTable,
    token_embd: usize,
    vectors: DecodeVectors,
    /// The `PQ2_0` lookup table of each rank-2 `PQ2_0` directory entry, indexed by directory index and
    /// repacked once at [`Engine::open`]; a decode step reads each projection tensor's table every token.
    pq2_luts: Vec<Option<Result<Pq2Lut, super::MatvecError>>>,
    /// The head split into bit planes, built on first greedy `Q8Act` step; `None` inside when the
    /// head is not `Q8_0` or the model carries a Hadamard fold.
    bitplane: std::sync::OnceLock<Option<Arc<super::kern::bitplane::BitplaneHead>>>,
    /// The `F16` limb tile of each projection tensor, keyed by directory index and packed on the
    /// first multi-column `Q8Act` projection of that tensor; owned by this engine, dropped with it.
    f16_tiles: std::sync::Mutex<std::collections::HashMap<usize, Arc<super::kern::limb::F16Tile>>>,
    /// The packed `NanoQuant` factors that replace the served linears; empty for a plain GGUF.
    pub(crate) nq: NqTable,
}

/// A checked byte extent of the resident file: the payload `resident()[off..off + len]`, resolved
/// once by [`Engine::payload_ref`] and carried by value (two words) instead of a borrowed slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadRef {
    /// Start offset in the resident file.
    pub off: usize,
    /// Byte count.
    pub len: usize,
}

impl PayloadRef {
    /// The extent's bytes of `file` (the resident file this reference was resolved against).
    #[inline]
    pub(crate) fn bytes(self, file: &[u8]) -> &[u8] {
        &file[self.off..self.off + self.len]
    }
}

/// `header.meta_i64("<arch>.<suffix>")`, refusing when absent or non-integral.
fn require_i64(h: &GgufHeader, arch: &str, suffix: &str) -> Result<i64, EngineError> {
    let key = format!("{arch}.{suffix}");
    h.meta_i64(&key).ok_or(EngineError::MissingMetadata(key))
}

/// `header.meta("<arch>.<suffix>")`, interpreted as an `f64` at this boundary, or `default` when
/// absent.
///
/// [`crate::format::gguf`] keeps `FLOAT32`/`FLOAT64` metadata as raw bits so its parse path stays
/// integer-only; the widening to `f64` happens here, where the engine consumes the value. An
/// integer limb is still accepted, matching the previous accessor's domain.
#[allow(
    clippy::cast_precision_loss,
    reason = "the integer-limb arm reproduces the format accessor's prior widening; GGUF stores \
              these keys as FLOAT32/FLOAT64, so the lossy arm is a nonconforming-file fallback"
)]
fn optional_f64(h: &GgufHeader, arch: &str, suffix: &str, default_bits: u64) -> F64 {
    let key = format!("{arch}.{suffix}");
    match h.meta(&key) {
        #[allow(
            clippy::cast_lossless,
            reason = "f64::from(f32) is not extractable; the cast is exact"
        )]
        #[allow(
            clippy::cast_lossless,
            reason = "f64::from(f32) is not extractable; the cast is exact"
        )]
        Some(GVal::F32Bits(bits, _)) => F64::from_f64(f32::from_bits(*bits) as f64),
        Some(GVal::F64Bits(bits, _)) => F64(*bits),
        Some(GVal::Int(v, _)) => F64::from_f64(*v as f64),
        _ => F64(default_bits),
    }
}

/// An `i64` metadata field as a positive `usize`.
fn positive_usize(v: i64, key: &str) -> Result<usize, EngineError> {
    if v <= 0 {
        return Err(EngineError::BadMetadata(key.to_string()));
    }
    usize::try_from(v).map_err(|_| EngineError::BadMetadata(key.to_string()))
}

/// Resolve the `qwen2.*` metadata and the `token_embd.weight` shape into an [`EngineConfig`].
///
/// Each stage below is its own function so no single extracted body carries the whole resolver's
/// live set; the stages run in the order the checks were written, so the first refusal is unchanged.
fn resolve_config(h: &GgufHeader) -> Result<EngineConfig, EngineError> {
    let architecture = h.arch();
    let (n_layer, n_nextn) = resolve_layer_count(h, &architecture)?;
    let [n_head, n_head_kv, n_embd, n_ff] = resolve_core_dims(h, &architecture)?;
    let meta = resolve_attention_meta(h, &architecture, [n_embd, n_head], n_layer)?;
    let (context_length, rope_theta, rms_eps) = resolve_numeric_meta(h, &architecture)?;
    let vocab_size = resolve_vocab(h, n_embd)?;

    Ok(EngineConfig {
        architecture,
        n_layer,
        n_head,
        n_head_kv,
        n_embd,
        n_ff,
        vocab_size,
        context_length,
        rope_theta,
        rms_eps,
        head_dim: meta.head_dim,
        mixers: meta.mixers,
        rot_dims: meta.rot_dims,
        attn_gate: meta.attn_gate,
        affine: meta.affine,
        n_nextn,
    })
}

/// The trunk layer count and the trailing multi-token-prediction block count, as
/// `(n_layer, n_nextn)`. Every trunk layer owns at least one `blk.<l>.*` tensor, so a count above
/// the directory is unrunnable; refusing it here bounds [`resolve_mixers`] by the parsed
/// directory, which `parse_header` bounds by the file length (fuzz `gguf_open`: `block_count =
/// u32::MAX` in a 9600-byte file asked for a 4 GiB schedule).
fn resolve_layer_count(h: &GgufHeader, arch: &str) -> Result<(usize, usize), EngineError> {
    let n_block = positive_usize(require_i64(h, arch, "block_count")?, "block_count")?;
    // Multi-token-prediction blocks follow the trunk and are not part of the decode step.
    let n_nextn = match h.meta_i64(&format!("{arch}.nextn_predict_layers")) {
        Some(v) if v >= 0 => usize::try_from(v).map_err(|_| EngineError::LengthOverflow)?,
        Some(_) => return Err(EngineError::BadMetadata("nextn_predict_layers".to_string())),
        None => 0,
    };
    let n_layer = n_block
        .checked_sub(n_nextn)
        .filter(|&n| n > 0)
        .ok_or_else(|| EngineError::BadMetadata("block_count".to_string()))?;
    if n_layer > h.tensors.len() {
        return Err(EngineError::ConfigMismatch(format!(
            "block_count {n_layer} exceeds {} tensors",
            h.tensors.len()
        )));
    }
    Ok((n_layer, n_nextn))
}

/// `[n_head, n_head_kv, n_embd, n_ff]` from the four required positive metadata counts.
fn resolve_core_dims(h: &GgufHeader, arch: &str) -> Result<[usize; 4], EngineError> {
    let n_head = positive_usize(
        require_i64(h, arch, "attention.head_count")?,
        "attention.head_count",
    )?;
    let n_head_kv = positive_usize(
        require_i64(h, arch, "attention.head_count_kv")?,
        "attention.head_count_kv",
    )?;
    let n_embd = positive_usize(
        require_i64(h, arch, "embedding_length")?,
        "embedding_length",
    )?;
    let n_ff = positive_usize(
        require_i64(h, arch, "feed_forward_length")?,
        "feed_forward_length",
    )?;
    Ok([n_head, n_head_kv, n_embd, n_ff])
}

/// The head geometry, mixer schedule and gate geometry of [`EngineConfig`]. `dims` is
/// `[n_embd, n_head]`.
fn resolve_attention_meta(
    h: &GgufHeader,
    arch: &str,
    dims: [usize; 2],
    n_layer: usize,
) -> Result<AttentionMeta, EngineError> {
    let [n_embd, n_head] = dims;
    let (head_dim, rot_dims) = resolve_head_geometry(h, arch, n_embd, n_head)?;
    let mixers = resolve_mixers(h, arch, n_layer)?;
    let affine = if mixers.contains(&MixerKind::Affine) {
        Some(resolve_affine(h, arch)?)
    } else {
        None
    };
    let attn_gate = resolve_attn_gate(h, &mixers, n_head, head_dim)?;
    Ok(AttentionMeta {
        head_dim,
        rot_dims,
        mixers,
        affine,
        attn_gate,
    })
}

/// `(context_length, rope_theta, rms_eps)`; an unstated context length is `0`.
fn resolve_numeric_meta(h: &GgufHeader, arch: &str) -> Result<(usize, F64, F64), EngineError> {
    let context_length = match h.meta_i64(&format!("{arch}.context_length")) {
        Some(v) if v > 0 => usize::try_from(v).map_err(|_| EngineError::LengthOverflow)?,
        _ => 0,
    };
    let rope_theta = optional_f64(h, arch, "rope.freq_base", DEFAULT_ROPE_THETA_BITS);
    let rms_eps = optional_f64(
        h,
        arch,
        "attention.layer_norm_rms_epsilon",
        DEFAULT_RMS_EPS_BITS,
    );
    Ok((context_length, rope_theta, rms_eps))
}

/// The vocabulary size from the `token_embd.weight` shape, cross-checked against the token array.
fn resolve_vocab(h: &GgufHeader, n_embd: usize) -> Result<usize, EngineError> {
    let td = h
        .tensor("token_embd.weight")
        .ok_or_else(|| EngineError::MissingTensor(TensorKey::top(PART_TOKEN_EMBD)))?;
    if td.dims.len() != 2 {
        return Err(EngineError::ConfigMismatch(format!(
            "token_embd.weight rank {} is not 2",
            td.dims.len()
        )));
    }
    let (row_len, vocab_dim) = (td.dims[0], td.dims[1]);
    if row_len != n_embd as u64 {
        return Err(EngineError::ConfigMismatch(format!(
            "token_embd.weight row {row_len} != embedding_length {n_embd}"
        )));
    }
    let vocab_size = usize::try_from(vocab_dim).map_err(|_| EngineError::LengthOverflow)?;
    if let Some(GVal::Arr(_, toks)) = h.meta("tokenizer.ggml.tokens") {
        if toks.len() != vocab_size {
            return Err(EngineError::ConfigMismatch(format!(
                "token array {} != token_embd rows {vocab_size}",
                toks.len()
            )));
        }
    }
    Ok(vocab_size)
}

/// The resolved head, mixer and gate geometry of [`EngineConfig`].
struct AttentionMeta {
    head_dim: usize,
    rot_dims: usize,
    mixers: Vec<MixerKind>,
    affine: Option<AffineGeom>,
    attn_gate: bool,
}

/// `(head_dim, rot_dims)`: `*.attention.key_length` (checked equal to `value_length` when stated)
/// or `n_embd / n_head`, and `*.rope.dimension_count` or the whole head, even and at most `head_dim`.
fn resolve_head_geometry(
    h: &GgufHeader,
    arch: &str,
    n_embd: usize,
    n_head: usize,
) -> Result<(usize, usize), EngineError> {
    let head_dim = if let Some(k) = h.meta_i64(&format!("{arch}.attention.key_length")) {
        let k = positive_usize(k, "attention.key_length")?;
        if let Some(v) = h.meta_i64(&format!("{arch}.attention.value_length")) {
            if positive_usize(v, "attention.value_length")? != k {
                return Err(EngineError::ConfigMismatch(format!(
                    "attention.value_length {v} != key_length {k}"
                )));
            }
        }
        k
    } else {
        if !n_embd.is_multiple_of(n_head) {
            return Err(EngineError::ConfigMismatch(format!(
                "embedding_length {n_embd} not divisible by head_count {n_head}"
            )));
        }
        n_embd / n_head
    };
    // Only a stated partial width is checked here; the whole-head default keeps its refusals in
    // `RopeTable::new`, as before the schedule existed.
    let rot_dims = match h.meta_i64(&format!("{arch}.rope.dimension_count")) {
        Some(v) => {
            let r = positive_usize(v, "rope.dimension_count")?;
            if r > head_dim || !r.is_multiple_of(2) {
                return Err(EngineError::ConfigMismatch(format!(
                    "rope.dimension_count {r} is odd or exceeds head_dim {head_dim}"
                )));
            }
            r
        }
        None => head_dim,
    };
    Ok((head_dim, rot_dims))
}

/// The per-layer mixer schedule from `*.full_attention_interval` (llama.cpp `qwen35.cpp`
/// `load_arch_hparams`: layer `l` is recurrent iff `(l + 1) % interval != 0`).
///
/// # Complexity
///
/// `O(n_layer)` time and space: one entry per trunk layer.
fn resolve_mixers(
    h: &GgufHeader,
    arch: &str,
    n_layer: usize,
) -> Result<Vec<MixerKind>, EngineError> {
    let interval = match h.meta_i64(&format!("{arch}.full_attention_interval")) {
        Some(v) => Some(positive_usize(v, "full_attention_interval")?),
        None => None,
    };
    let mut out = Vec::with_capacity(n_layer);
    let mut l = 0usize;
    while l < n_layer {
        out.push(match interval {
            Some(f) if !(l + 1).is_multiple_of(f) => MixerKind::Affine,
            _ => MixerKind::Softmax,
        });
        l += 1;
    }
    Ok(out)
}

/// The affine mixer's geometry from `*.ssm.*`, refusing a missing key or an inner size that is not
/// a whole number of value heads.
fn resolve_affine(h: &GgufHeader, arch: &str) -> Result<AffineGeom, EngineError> {
    let get =
        |k: &str| -> Result<usize, EngineError> { positive_usize(require_i64(h, arch, k)?, k) };
    let conv_kernel = get("ssm.conv_kernel")?;
    let key_heads = get("ssm.group_count")?;
    let value_heads = get("ssm.time_step_rank")?;
    let head_k = get("ssm.state_size")?;
    let inner = get("ssm.inner_size")?;
    if !inner.is_multiple_of(value_heads) || !value_heads.is_multiple_of(key_heads) {
        return Err(EngineError::ConfigMismatch(format!(
            "ssm.inner_size {inner} / time_step_rank {value_heads} / group_count {key_heads}"
        )));
    }
    Ok(AffineGeom {
        conv_kernel,
        n_k_heads: key_heads,
        n_v_heads: value_heads,
        head_k,
        head_v: inner / value_heads,
    })
}

/// Whether the softmax layers' `attn_q` carries the per-head output gate, read from its row count:
/// `n_head·head_dim` (no gate) or `2·n_head·head_dim` (query and gate interleaved per head).
fn resolve_attn_gate(
    h: &GgufHeader,
    mixers: &[MixerKind],
    n_head: usize,
    head_dim: usize,
) -> Result<bool, EngineError> {
    let mut first = 0usize;
    while first < mixers.len() && mixers[first] != MixerKind::Softmax {
        first += 1;
    }
    if first == mixers.len() {
        return Ok(false);
    }
    let l = first;
    let name = format!("blk.{l}.attn_q.weight");
    let Some(t) = h.tensor(&name) else {
        return Ok(false);
    };
    let rows = t.dims.get(1).copied().unwrap_or(0);
    let q_out = (n_head as u64).saturating_mul(head_dim as u64);
    if rows == q_out {
        Ok(false)
    } else if rows == q_out.saturating_mul(2) {
        Ok(true)
    } else {
        Err(EngineError::ConfigMismatch(format!(
            "{name} rows {rows} are neither {q_out} nor {}",
            q_out.saturating_mul(2)
        )))
    }
}

/// One layer's decode vectors: norm gains, attention biases, QK-norm gains.
type LayerDecodeVectors = (
    [Option<Vec<i64>>; 2],
    [Option<Vec<i64>>; 3],
    [Option<Vec<i64>>; 2],
);

/// One directory entry as a [`TensorEntry`], with its extent checks; `Err` at the first failing check.
fn tensor_entry(
    desc: &crate::format::gguf::TensorDesc,
    start: u64,
    end: u64,
    file_len: u64,
) -> Result<TensorEntry, EngineError> {
    let name = String::from_utf8_lossy(&desc.name).into_owned();
    let n_elements = desc.n_elements()?;
    let n_bytes = desc.n_bytes()?;
    if end < start {
        return Err(EngineError::ConfigMismatch(format!(
            "directory offsets out of order at {name}"
        )));
    }
    if let Some(bytes) = n_bytes {
        let exact = start
            .checked_add(bytes)
            .ok_or(EngineError::LengthOverflow)?;
        if exact > file_len {
            return Err(EngineError::TensorOutOfBounds {
                name,
                at: start,
                n: bytes,
                file_len,
            });
        }
        if exact > end {
            return Err(EngineError::ConfigMismatch(format!(
                "tensor {name} storage {bytes} overlaps the next directory entry"
            )));
        }
    }
    Ok(TensorEntry {
        name,
        ggml_type: desc.ggml_type,
        quant: QuantKind::from_ggml(desc.ggml_type),
        dims: desc.dims.clone(),
        n_elements,
        n_bytes,
        byte_range: start..end,
    })
}

/// Walk the directory into a [`TensorTable`], checking every extent against the file.
///
/// A tensor whose type has no geometry in `crate::format::quant` still gets a range: its extent
/// runs from its data offset to the next entry's data offset in directory-offset order (the file
/// end for the last). That keeps every entry bounds-checked without inventing geometry; the exact
/// byte count is recorded only when the quant layer supplies it.
fn build_table(header: &GgufHeader, file_len: u64) -> Result<TensorTable, EngineError> {
    let count = header.tensors.len();
    let mut starts: Vec<u64> = Vec::with_capacity(count);
    let mut idx = 0usize;
    let mut err: Option<EngineError> = None;
    while idx < count && err.is_none() {
        let desc = &header.tensors[idx];
        match header.data0.checked_add(desc.offset) {
            None => err = Some(EngineError::LengthOverflow),
            Some(at) => {
                if at > file_len {
                    err = Some(EngineError::TensorOutOfBounds {
                        name: String::from_utf8_lossy(&desc.name).into_owned(),
                        at,
                        n: 0,
                        file_len,
                    });
                } else {
                    starts.push(at);
                }
            }
        }
        idx += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut order: Vec<usize> = Vec::with_capacity(count);
    let mut ord_fill = 0usize;
    while ord_fill < count {
        order.push(ord_fill);
        ord_fill += 1;
    }
    order.sort_by_key(|&ord| starts[ord]);
    let mut ends: Vec<u64> = vec![file_len; count];
    let mut rank = 0usize;
    while rank + 1 < count {
        ends[order[rank]] = starts[order[rank + 1]];
        rank += 1;
    }

    let mut entries: Vec<TensorEntry> = Vec::with_capacity(count);
    let mut entry_idx = 0usize;
    while entry_idx < count && err.is_none() {
        let desc = &header.tensors[entry_idx];
        let (start, end) = (starts[entry_idx], ends[entry_idx]);
        match tensor_entry(desc, start, end, file_len) {
            Ok(entry) => entries.push(entry),
            Err(e) => err = Some(e),
        }
        entry_idx += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(TensorTable { entries })
}

/// Keep freed per-call buffers (columns, packs, conv caches; each under 64 MiB) in the glibc heap
/// instead of returning them to the kernel. Under the default dynamic thresholds a freed buffer is
/// trimmed or unmapped and the next layer's call faults its pages in again (~77K minor
/// faults per pp200 run, ~3.7 faults per page; with the thresholds held, ~9K). Process-wide and
/// set once; a refused `mallopt` leaves the default and changes no result.
///
/// Release-only: the thresholds tune the allocator and change no computed value, while the
/// Charon/Aeneas extraction unit builds the debug profile (`debug_assertions` is on). Keeping the
/// call out of that profile keeps `libc::mallopt` and its `M_*` arguments out of the extracted
/// closure, where they arrive as unmodeled external axioms.
fn hold_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu", not(debug_assertions)))]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // SAFETY: `mallopt` only sets allocator parameters; it is called before any engine
            // buffer exists and takes plain integers.
            unsafe {
                libc::mallopt(libc::M_MMAP_THRESHOLD, 64 << 20);
                libc::mallopt(libc::M_TRIM_THRESHOLD, 1 << 30);
            }
        });
    }
}

impl Engine {
    /// The cached `F16` limb tile of directory entry `idx`, built by `build` on the first call;
    /// `None` when `build` refuses or the cache lock is poisoned (the caller keeps the scalar path).
    ///
    /// # Complexity
    ///
    /// `O(1)` on a hit; on a miss the cost of `build`.
    pub(crate) fn f16_tile(
        &self,
        idx: usize,
        build: impl FnOnce() -> Option<super::kern::limb::F16Tile>,
    ) -> Option<Arc<super::kern::limb::F16Tile>> {
        let mut guard = self.f16_tiles.lock().ok()?;
        if let Some(t) = guard.get(&idx) {
            return Some(Arc::clone(t));
        }
        let t = Arc::new(build()?);
        guard.insert(idx, Arc::clone(&t));
        Some(t)
    }

    /// Open a GGUF file, resolve the config and build the checked tensor table.
    ///
    /// The file is read once and handed to [`Engine::from_bytes`], so the disk path and an
    /// in-memory model share one loader. There is no second read: the header and the payload are
    /// the same bytes, so a size that changed between them cannot be observed or served.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Io`] on a read failure; every refusal of [`Engine::from_bytes`]
    /// otherwise.
    pub fn open(path: &str) -> Result<Engine, EngineError> {
        #[cfg(target_os = "linux")]
        {
            let map = crate::format::nq::MappedFile::open(path)
                .map_err(|e| EngineError::Io(e.to_string()))?;
            Engine::from_image(path, Image::Mapped(map))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let data = host_read_all(path).map_err(EngineError::Io)?;
            Engine::from_bytes(path, data)
        }
    }

    /// Build an engine from an in-memory GGUF image, labelled `label`.
    ///
    /// `data` is the whole file, header and tensor payloads. The header is parsed from these bytes
    /// ([`parse_header`]) with the byte count as the file length, so every directory extent is
    /// checked against exactly the buffer that backs the payloads; the sizes cannot disagree and
    /// there is no second read. The config, the `prism.hadamard.*` fold contract, the tensor table,
    /// the decode vectors and the `PQ2_0` lookup tables are resolved exactly as [`Engine::open`]
    /// resolves them, from this parse alone.
    ///
    /// `label` is stored in [`Engine::path`] verbatim; it is a diagnostic name, not a path that is
    /// read, so an in-memory model names itself.
    ///
    /// # Complexity
    ///
    /// `O(file bytes)` for the header parse's directory sweep plus `O(W)` for the decode-vector and
    /// `PQ2_0` table construction; the buffer moves into the engine and is not copied.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on a GGUF parse refusal, a missing or malformed config key, a
    /// refusal of the `prism.hadamard.*` fold contract ([`hadamard_meta`]), a vocabulary
    /// cross-check failure, a missing `token_embd.weight`, or a tensor extent that leaves the
    /// buffer.
    pub fn from_bytes(label: &str, data: Vec<u8>) -> Result<Engine, EngineError> {
        Engine::from_image(label, Image::Owned(data.into_boxed_slice()))
    }

    /// The shared open path: an owned image from [`Engine::from_bytes`] or the file mapping
    /// [`Engine::open`] makes.
    fn from_image(label: &str, data: Image) -> Result<Engine, EngineError> {
        hold_heap();
        let bytes = data.bytes();
        let file_len = u64::try_from(bytes.len()).map_err(|_| EngineError::LengthOverflow)?;
        let header = parse_header(bytes, file_len)?;
        let config = resolve_config(&header)?;
        let hadamard = hadamard_meta(&header)?;
        let table = build_table(&header, file_len)?;
        let entries = table.entries();
        let mut token_embd = 0usize;
        while token_embd < entries.len()
            && !super::step::bytes_eq(entries[token_embd].name.as_bytes(), b"token_embd.weight")
        {
            token_embd += 1;
        }
        if token_embd == entries.len() {
            return Err(EngineError::MissingTensor(TensorKey::top(PART_TOKEN_EMBD)));
        }
        let mut engine = Engine {
            path: label.to_string(),
            file_len,
            config,
            hadamard,
            data,
            table,
            token_embd,
            vectors: DecodeVectors::default(),
            pq2_luts: Vec::new(),
            bitplane: std::sync::OnceLock::new(),
            f16_tiles: std::sync::Mutex::new(std::collections::HashMap::new()),
            nq: NqTable::default(),
        };
        let vectors = engine.load_decode_vectors()?;
        engine.vectors = vectors;
        engine.pq2_luts = engine.build_pq2_luts();
        Ok(engine)
    }

    /// Open a GGUF skeleton and install a `NanoQuant` pack: [`Engine::open`] then
    /// [`Engine::from_bytes_with_nq`]'s table build.
    ///
    /// The GGUF may still carry the replaced linears (a full quantised image) or omit them (a
    /// skeleton). Either way the projection path reads the pack's factors, not the payload; the
    /// skeleton form is what keeps the resident model at the pack's rate.
    ///
    /// # Errors
    ///
    /// [`Engine::open`]'s refusals, then every [`Engine::from_bytes_with_nq`] refusal.
    pub fn open_with_nq(gguf: &str, pack: NqPack) -> Result<Engine, EngineError> {
        let mut engine = Engine::open(gguf)?;
        engine.install_nq(pack)?;
        Ok(engine)
    }

    /// Build an engine from an in-memory GGUF image and install a `NanoQuant` pack.
    ///
    /// The pack must cover exactly the config's trunk layers and every kind; each section's
    /// `(out, in)` is checked against the config's linear geometry. An empty pack leaves the engine
    /// a plain GGUF engine.
    ///
    /// # Errors
    ///
    /// [`Engine::from_bytes`]'s refusals, and [`EngineError::ConfigMismatch`] for a pack whose layer
    /// count, linear geometry or factor scales do not fit the model.
    pub fn from_bytes_with_nq(
        label: &str,
        data: Vec<u8>,
        pack: NqPack,
    ) -> Result<Engine, EngineError> {
        let mut engine = Engine::from_bytes(label, data)?;
        engine.install_nq(pack)?;
        Ok(engine)
    }

    fn install_nq(&mut self, pack: NqPack) -> Result<(), EngineError> {
        if !pack.linears.is_empty() {
            self.nq = build_nq_table(&self.config, self.table(), pack)?;
        }
        Ok(())
    }

    /// The resident weight bytes with the `NanoQuant` factors in place of the linears they replace:
    /// every directory tensor the pack replaces is counted as its factor's stored bytes, every
    /// other tensor as its stored GGUF bytes.
    ///
    /// # Trust boundary
    ///
    /// The loader still holds the whole GGUF buffer ([`Engine::from_bytes`]), so a replaced
    /// linear's GGUF payload is resident beside its factors until the skeleton image that omits
    /// those payloads is the one opened; this value is the budget accounting and not the process's
    /// resident set.
    ///
    /// # Complexity
    ///
    /// `O(entries · log fitted)`.
    #[must_use]
    pub fn resident_weight_bytes(&self) -> u64 {
        let mut total = 0u64;
        for (i, t) in self.table.entries().iter().enumerate() {
            if self.nq.entries.binary_search(&i).is_err() {
                total = total.saturating_add(t.n_bytes.unwrap_or(0));
            }
        }
        total.saturating_add(self.nq.factor_bytes)
    }

    /// The decode `F32` gain/bias vectors resolved once at load.
    #[must_use]
    pub fn decode_vectors(&self) -> &DecodeVectors {
        &self.vectors
    }

    /// The validated `prism.hadamard.*` fold contract, or `None` for an unrotated model.
    #[must_use]
    pub fn hadamard(&self) -> Option<&HadamardMeta> {
        self.hadamard.as_ref()
    }

    /// The llama3 `rope_freqs.weight` factors (one Q32.32 word per rotary pair), or `None`.
    pub(crate) fn rope_freq_factors(&self) -> Result<Option<Vec<i64>>, EngineError> {
        self.read_f32_vector("rope_freqs.weight", self.config.rot_dims / 2)
    }

    /// Read the `F32` vector `name` of length `len` into Q32.32, or `None` when absent.
    ///
    /// A present tensor must be rank-1 with `dims == [len]` and ggml type `F32`; otherwise it
    /// refuses with [`EngineError::BadVector`]. The payload is converted by
    /// [`super::f32_convert::dequant_f32_row`], which refuses a non-finite or out-of-`i64` value.
    fn read_f32_vector(&self, name: &str, len: usize) -> Result<Option<Vec<i64>>, EngineError> {
        let Some(t) = self.table.get(name) else {
            return Ok(None);
        };
        let expected = u64::try_from(len).map_err(|_| EngineError::LengthOverflow)?;
        if t.dims.len() != 1 || t.dims[0] != expected {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: String::from("dims"),
            });
        }
        if t.ggml_type != quant::GGML_TYPE_F32 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: String::from("type"),
            });
        }
        let n_bytes = t.n_bytes.ok_or(EngineError::BadVector {
            name: name.to_string(),
            reason: String::from("geometry"),
        })?;
        // The resident bytes, not a second file read: an engine over in-memory bytes has no path
        // to read. [`Engine::payload`] has `read_range`'s bounds and refusals, and the resident
        // bytes are the file's bytes.
        let raw = self.payload(t.byte_range.start, n_bytes)?;
        let mut out: Vec<i64> = Vec::with_capacity(len);
        dequant_f32_row(raw, len, &mut out)?;
        Ok(Some(out))
    }

    /// Each multi-token-prediction block's input norms and head norm.
    fn load_nextn_vectors(&self) -> Result<Vec<[Option<Vec<i64>>; 3]>, EngineError> {
        let cfg = &self.config;
        let mut out = Vec::with_capacity(cfg.n_nextn);
        let mut i = 0usize;
        while i < cfg.n_nextn {
            let l = cfg.n_layer + i;
            out.push([
                self.read_f32_vector(&format!("blk.{l}.nextn.enorm.weight"), cfg.n_embd)?,
                self.read_f32_vector(&format!("blk.{l}.nextn.hnorm.weight"), cfg.n_embd)?,
                self.read_f32_vector(
                    &format!("blk.{l}.nextn.shared_head_norm.weight"),
                    cfg.n_embd,
                )?,
            ]);
            i += 1;
        }
        Ok(out)
    }

    /// Every affine layer's `F32` parameters (`ssm_conv1d`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`),
    /// refusing an absent one; `None` for softmax layers.
    ///
    /// # Complexity
    ///
    /// `O(n_affine · conv_kernel · channels)` reads and conversions, once at load.
    fn load_affine_vectors(&self) -> Result<Vec<Option<AffineVectors>>, EngineError> {
        let cfg = &self.config;
        let mut out = Vec::with_capacity(cfg.n_layer);
        let mut l = 0usize;
        while l < cfg.n_layer {
            if cfg.mixers[l] != MixerKind::Affine {
                out.push(None);
                l += 1;
                continue;
            }
            let g = cfg
                .affine
                .ok_or_else(|| EngineError::ConfigMismatch("affine geometry".to_string()))?;
            let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
            let need = |part: u8, v: Option<Vec<i64>>| {
                v.ok_or(EngineError::MissingTensor(TensorKey::blk(l, part)))
            };
            let conv_name = format!("blk.{l}.ssm_conv1d.weight");
            let conv = self.read_f32_matrix(&conv_name, g.conv_kernel, channels)?;
            let dt_name = format!("blk.{l}.ssm_dt.bias");
            let a_name = format!("blk.{l}.ssm_a");
            let norm_name = format!("blk.{l}.ssm_norm.weight");
            out.push(Some(AffineVectors {
                conv: need(PART_SSM_CONV1D, conv)?,
                dt: need(
                    PART_SSM_DT_BIAS,
                    self.read_f32_vector(&dt_name, g.n_v_heads)?,
                )?,
                a: need(PART_SSM_A, self.read_f32_vector(&a_name, g.n_v_heads)?)?,
                norm: need(PART_SSM_NORM, self.read_f32_vector(&norm_name, g.head_v)?)?,
            }));
            l += 1;
        }
        Ok(out)
    }

    /// A two-dimensional `F32` tensor `[d0, d1]` (ggml order: `d0` contiguous) as one Q32.32 vector
    /// of `d0·d1` values, or `None` when absent; the same type and range refusals as
    /// `read_f32_vector`.
    fn read_f32_matrix(
        &self,
        name: &str,
        d0: usize,
        d1: usize,
    ) -> Result<Option<Vec<i64>>, EngineError> {
        let Some(t) = self.table.get(name) else {
            return Ok(None);
        };
        if t.dims.len() != 2 || t.dims[0] != d0 as u64 || t.dims[1] != d1 as u64 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: String::from("dims"),
            });
        }
        if t.ggml_type != quant::GGML_TYPE_F32 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: String::from("type"),
            });
        }
        let len = d0.checked_mul(d1).ok_or(EngineError::LengthOverflow)?;
        let n_bytes = t.n_bytes.ok_or(EngineError::BadVector {
            name: name.to_string(),
            reason: String::from("geometry"),
        })?;
        // The resident bytes, not a second file read: an engine over in-memory bytes has no path
        // to read. [`Engine::payload`] has `read_range`'s bounds and refusals, and the resident
        // bytes are the file's bytes.
        let raw = self.payload(t.byte_range.start, n_bytes)?;
        let mut out: Vec<i64> = Vec::with_capacity(len);
        dequant_f32_row(raw, len, &mut out)?;
        Ok(Some(out))
    }

    /// One layer's decode vectors: the two norm gains, the three attention biases and the two QK-norm gains.
    fn layer_decode_vectors(
        &self,
        layer: usize,
        q_out: usize,
        kv_out: usize,
    ) -> Result<LayerDecodeVectors, EngineError> {
        let cfg = &self.config;
        let norms = [
            self.read_f32_vector(&format!("blk.{layer}.attn_norm.weight"), cfg.n_embd)?,
            // The pre-FFN norm: `ffn_norm` (Qwen2), `post_attention_norm` (Qwen3.5).
            match self.read_f32_vector(&format!("blk.{layer}.ffn_norm.weight"), cfg.n_embd)? {
                Some(v) => Some(v),
                None => self.read_f32_vector(
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    cfg.n_embd,
                )?,
            },
        ];
        let biases = [
            self.read_f32_vector(&format!("blk.{layer}.attn_q.bias"), q_out)?,
            self.read_f32_vector(&format!("blk.{layer}.attn_k.bias"), kv_out)?,
            self.read_f32_vector(&format!("blk.{layer}.attn_v.bias"), kv_out)?,
        ];
        // Qwen3 only: the per-head QK `RMSNorm` gains, absent (`None`) on Qwen2.
        let qk_norms = [
            self.read_f32_vector(&format!("blk.{layer}.attn_q_norm.weight"), cfg.head_dim)?,
            self.read_f32_vector(&format!("blk.{layer}.attn_k_norm.weight"), cfg.head_dim)?,
        ];
        Ok((norms, biases, qk_norms))
    }

    /// Resolve every decode gain and attention bias the step reads, once.
    ///
    /// The names and widths are exactly [`Engine::step_with_kv`]'s reads: `blk.{l}.attn_norm`,
    /// `blk.{l}.ffn_norm` (`n_embd`), `blk.{l}.attn_{q,k,v}.bias` (`n_head·head_dim`,
    /// `n_head_kv·head_dim`), and `output_norm.weight` (`n_embd`).
    fn load_decode_vectors(&self) -> Result<DecodeVectors, EngineError> {
        let cfg = &self.config;
        let q_out = cfg
            .n_head
            .checked_mul(cfg.head_dim)
            .ok_or(EngineError::LengthOverflow)?;
        let kv_out = cfg
            .n_head_kv
            .checked_mul(cfg.head_dim)
            .ok_or(EngineError::LengthOverflow)?;
        let mut norms: Vec<[Option<Vec<i64>>; 2]> = Vec::with_capacity(cfg.n_layer);
        let mut biases: Vec<[Option<Vec<i64>>; 3]> = Vec::with_capacity(cfg.n_layer);
        let mut qk_norms: Vec<[Option<Vec<i64>>; 2]> = Vec::with_capacity(cfg.n_layer);
        let mut layer = 0usize;
        let mut err: Option<EngineError> = None;
        while layer < cfg.n_layer + cfg.n_nextn && err.is_none() {
            match self.layer_decode_vectors(layer, q_out, kv_out) {
                Ok((n, b, q)) => {
                    norms.push(n);
                    biases.push(b);
                    qk_norms.push(q);
                }
                Err(e) => err = Some(e),
            }
            layer += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let output_norm = self.read_f32_vector("output_norm.weight", cfg.n_embd)?;
        let affine = self.load_affine_vectors()?;
        let nextn = self.load_nextn_vectors()?;
        Ok(DecodeVectors {
            norms,
            biases,
            qk_norms,
            output_norm,
            affine,
            nextn,
        })
    }

    /// The head's bit planes, built once on first use (`O(vocab·n_embd)` byte operations and as
    /// many bytes of planes); `None` when the head is not `Q8_0` or the model is Hadamard-folded.
    pub(crate) fn bitplane_head(&self) -> Option<Arc<super::kern::bitplane::BitplaneHead>> {
        self.bitplane
            .get_or_init(|| {
                if self.hadamard.is_some() {
                    return None;
                }
                let t = self.lm_head();
                let raw = self.payload(t.byte_range.start, t.n_bytes?).ok()?;
                super::kern::bitplane::BitplaneHead::from_q8_0(
                    t.ggml_type,
                    raw,
                    self.config.vocab_size,
                    self.config.n_embd,
                )
            })
            .clone()
    }

    /// The `PQ2_0` lookup table of directory entry `idx`, built at [`Engine::open`].
    ///
    /// Returns the stored refusal of [`Pq2Lut::from_pq2_0`] (non-finite block scale, short payload,
    /// row width not a whole number of 128-weight blocks) at the use, as the former lazy cache
    /// did on the first projection; an entry that is not a rank-2 `PQ2_0` tensor with a readable
    /// payload has no table and refuses as an unsupported quant (every caller has already checked
    /// the entry's type and shape, so that arm is not reached).
    ///
    /// # Errors
    ///
    /// The [`MatvecError`](super::MatvecError) stored for `idx`.
    pub(crate) fn pq2_lut(&self, idx: usize) -> Result<&Pq2Lut, super::MatvecError> {
        match self.pq2_luts.get(idx) {
            Some(Some(Ok(lut))) => Ok(lut),
            Some(Some(Err(e))) => Err(e.clone()),
            _ => Err(super::MatvecError::UnsupportedQuant {
                ggml_type: quant::GGML_TYPE_PQ2_0,
            }),
        }
    }

    /// Every directory entry's `PQ2_0` table, for the dispatch (`ProjSource::Rows(idx)`).
    pub(crate) fn pq2_luts(&self) -> &[Option<Result<Pq2Lut, super::MatvecError>>] {
        &self.pq2_luts
    }

    /// The `PQ2_0` table of each rank-2 `PQ2_0` directory entry, repacked once at open.
    ///
    /// # Complexity
    ///
    /// Time `O(Σ n_out·n_in)` over the rank-2 `PQ2_0` entries (one [`Pq2Lut::from_pq2_0`] repack
    /// each, the work the lazy cache did on first use); space the tables, as the lazy cache
    /// held after the first token; `O(directory)` for the slots.
    fn build_pq2_luts(&self) -> Vec<Option<Result<Pq2Lut, super::MatvecError>>> {
        let entries = self.table.entries();
        let mut out = Vec::with_capacity(entries.len());
        for t in entries {
            let mut slot = None;
            if t.ggml_type == quant::GGML_TYPE_PQ2_0 && t.dims.len() == 2 {
                if let (Some(nb), Ok(n_in), Ok(n_out)) = (
                    t.n_bytes,
                    usize::try_from(t.dims[0]),
                    usize::try_from(t.dims[1]),
                ) {
                    if let Ok(raw) = self.payload(t.byte_range.start, nb) {
                        slot = Some(Pq2Lut::from_pq2_0(raw, n_out, n_in));
                    }
                }
            }
            out.push(slot);
        }
        out
    }

    /// The named tensor directory.
    #[must_use]
    pub fn table(&self) -> &TensorTable {
        &self.table
    }

    /// The `token_embd.weight` directory entry.
    #[must_use]
    pub fn token_embd(&self) -> &TensorEntry {
        &self.table.entries()[self.token_embd]
    }

    /// The directory index of [`Engine::token_embd`].
    pub(crate) fn token_embd_index(&self) -> usize {
        self.token_embd
    }

    /// Read `n` bytes of the file at absolute offset `at`, refusing past EOF.
    ///
    /// Opens the file and returns owned bytes; retained for callers that need a buffer they can
    /// hold independently of the [`Engine`]. The step path uses [`Engine::payload`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RowOutOfBounds`] when the range leaves the file and
    /// [`EngineError::Io`] when the read itself fails.
    pub fn read_range(&self, at: u64, n: u64) -> Result<Vec<u8>, EngineError> {
        let end = at.checked_add(n).ok_or(EngineError::LengthOverflow)?;
        if end > self.file_len {
            return Err(EngineError::RowOutOfBounds {
                at,
                n,
                end: self.file_len,
            });
        }
        let len = usize::try_from(n).map_err(|_| EngineError::LengthOverflow)?;
        host_read_at(&self.path, at, len).map_err(EngineError::Io)
    }

    /// Borrow `n` bytes of the resident file at absolute offset `at`, refusing past EOF.
    ///
    /// This is the step path's read: the bytes were read once at [`Engine::open`], so the borrow is
    /// O(1) and copies nothing. The bounds checks and refusals are
    /// [`Engine::read_range`]'s: [`EngineError::RowOutOfBounds`] when `at + n` leaves the file and
    /// [`EngineError::LengthOverflow`] when the range does not fit `usize`. The resident bytes are
    /// exactly the file's bytes, so a payload slice equals the range `read_range` returns.
    ///
    /// # Complexity
    ///
    /// `O(1)` time and no allocation; the returned slice borrows `self`.
    ///
    /// # Errors
    ///
    /// [`EngineError::RowOutOfBounds`], [`EngineError::LengthOverflow`].
    pub fn payload(&self, at: u64, n: u64) -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or(EngineError::LengthOverflow)?;
        if end > self.file_len {
            return Err(EngineError::RowOutOfBounds {
                at,
                n,
                end: self.file_len,
            });
        }
        let start = usize::try_from(at).map_err(|_| EngineError::LengthOverflow)?;
        let stop = usize::try_from(end).map_err(|_| EngineError::LengthOverflow)?;
        self.data
            .bytes()
            .get(start..stop)
            .ok_or(EngineError::LengthOverflow)
    }

    /// The checked extent `[at, at+n)` of the resident file as a [`PayloadRef`]: the refusals of
    /// [`Engine::payload`], without borrowing the bytes. The bytes are `resident()[off..off+len]`.
    ///
    /// # Errors
    ///
    /// [`EngineError::RowOutOfBounds`], [`EngineError::LengthOverflow`].
    pub fn payload_ref(&self, at: u64, n: u64) -> Result<PayloadRef, EngineError> {
        let end = at.checked_add(n).ok_or(EngineError::LengthOverflow)?;
        if end > self.file_len {
            return Err(EngineError::RowOutOfBounds {
                at,
                n,
                end: self.file_len,
            });
        }
        let start = usize::try_from(at).map_err(|_| EngineError::LengthOverflow)?;
        let stop = usize::try_from(end).map_err(|_| EngineError::LengthOverflow)?;
        if self.data.bytes().get(start..stop).is_none() {
            return Err(EngineError::LengthOverflow);
        }
        Ok(PayloadRef {
            off: start,
            len: stop - start,
        })
    }

    /// The whole resident file; a [`PayloadRef`] indexes it.
    #[must_use]
    pub fn resident(&self) -> &[u8] {
        self.data.bytes()
    }
}

#[cfg(test)]
mod tests;
