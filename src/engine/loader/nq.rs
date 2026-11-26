//! The engine's `NanoQuant` linear table: the packed factors each served projection runs, and the
//! map from a layer's GGUF tensor to its sign-rank kernel.
//!
//! The pack carries a section for a subset of the trunk layers and kinds (`q`, `k`, `v`, `o`,
//! `gate`, `up`, `down`). [`build_nq_table`] checks each section's `(out, in)` against the config's
//! linear geometry and decodes its binary16 scales once; the projection path then reads
//! [`NqLinear::project`] and never touches the GGUF payload of a replaced linear. A linear the pack
//! omits keeps its dense skeleton payload and is projected by the ordinary path.
//!
//! The table also records the directory index of each replaced tensor when the skeleton still
//! carries it, so [`Engine::resident_weight_bytes`](crate::engine::loader::Engine::resident_weight_bytes)
//! counts the factors instead of the payload. The loader maps the whole GGUF image either way (or
//! reads it, on a platform without the mapping); that value is the budget, not the process's
//! resident set.

use crate::engine::kern::nq::NqLinear;
use crate::engine::loader::{EngineConfig, EngineError, TensorTable};
use crate::engine::TensorKey;
use crate::format::nq::{NqPack, NQ_KINDS};

/// The fitted linears of one loaded model, indexed `layer * NQ_KINDS + kind`, and the accounting.
#[derive(Debug, Default)]
pub(crate) struct NqTable {
    pub(super) linears: Vec<Option<NqLinear>>,
    /// Directory indices of the entries a fitted linear replaces, ascending.
    pub(super) entries: Vec<usize>,
    /// Σ of every factor's stored bits, in bytes.
    pub(super) factor_bytes: u64,
    /// Σ of `out·in` over the fitted linears.
    pub(super) weights: u64,
}

/// `(out, in)` of linear `kind` of a layer: `q` is `q_out × n_embd`, `k`/`v` are
/// `kv_out × n_embd`, `o` is `n_embd × q_out`, `gate`/`up` are `n_ff × n_embd` and `down` is
/// `n_embd × n_ff`.
///
/// # Errors
///
/// [`EngineError::ConfigMismatch`] for a kind past [`NQ_KINDS`] or a head-count product that leaves
/// `usize`.
pub(super) fn nq_kind_dims(
    config: &EngineConfig,
    kind: usize,
) -> Result<(usize, usize), EngineError> {
    let q_out = config
        .n_head
        .checked_mul(config.head_dim)
        .ok_or_else(|| EngineError::ConfigMismatch(String::from("nq pack: q_out")))?;
    let kv_out = config
        .n_head_kv
        .checked_mul(config.head_dim)
        .ok_or_else(|| EngineError::ConfigMismatch(String::from("nq pack: kv_out")))?;
    match kind {
        0 => Ok((q_out, config.n_embd)),
        1 | 2 => Ok((kv_out, config.n_embd)),
        3 => Ok((config.n_embd, q_out)),
        4 | 5 => Ok((config.n_ff, config.n_embd)),
        6 => Ok((config.n_embd, config.n_ff)),
        _ => Err(EngineError::ConfigMismatch(format!(
            "nq pack: linear kind {kind} >= {NQ_KINDS}"
        ))),
    }
}

/// Build the fitted-linear table from the pack's sections.
///
/// # Errors
///
/// [`EngineError::ConfigMismatch`] when the pack's layer count is not the config's, a section's
/// `(out, in)` is not its linear's geometry, a kind is past [`NQ_KINDS`], or a factor's scales leave
/// Q32.32.
///
/// # Complexity
///
/// `O(linears · (out·in / 32 + out + in))` to decode and size the factors, plus one directory scan
/// per linear.
pub(super) fn build_nq_table(
    config: &EngineConfig,
    table: &TensorTable,
    pack: NqPack,
) -> Result<NqTable, EngineError> {
    if pack.layers != config.n_layer {
        return Err(EngineError::ConfigMismatch(format!(
            "nq pack covers {} layers, model has {}",
            pack.layers, config.n_layer
        )));
    }
    let mut out = NqTable {
        linears: (0..config.n_layer.saturating_mul(NQ_KINDS))
            .map(|_| None)
            .collect(),
        ..NqTable::default()
    };
    let entries = table.entries();
    for packed in pack.linears {
        let layer = packed.layer;
        let kind_at = packed.kind;
        let (rows, cols) = nq_kind_dims(config, kind_at)?;
        if packed.out != rows || packed.inn != cols {
            return Err(EngineError::ConfigMismatch(format!(
                "nq pack: blk.{layer} kind {kind_at} is {}x{}, model linear is {rows}x{cols}",
                packed.out, packed.inn
            )));
        }
        let linear = NqLinear::from_packed(packed).map_err(|e| {
            EngineError::ConfigMismatch(format!("nq pack: blk.{layer} kind {kind_at}: {e}"))
        })?;
        let at = layer
            .checked_mul(NQ_KINDS)
            .and_then(|x| x.checked_add(kind_at))
            .ok_or(EngineError::LengthOverflow)?;
        let slot = out
            .linears
            .get_mut(at)
            .ok_or_else(|| EngineError::ConfigMismatch(String::from("nq pack: linear index")))?;
        out.factor_bytes = out
            .factor_bytes
            .saturating_add(linear.stored_bits().saturating_add(7) / 8);
        out.weights = out.weights.saturating_add(linear.weights());
        *slot = Some(linear);
        let kind = u8::try_from(kind_at)
            .map_err(|_| EngineError::ConfigMismatch(String::from("nq pack: kind width")))?;
        let name = TensorKey::blk(layer, kind).to_string();
        if let Some((idx, _)) = entries
            .iter()
            .enumerate()
            .find(|(_, t)| t.name.as_bytes() == name.as_bytes())
        {
            out.entries.push(idx);
        }
    }
    out.entries.sort_unstable();
    out.entries.dedup();
    Ok(out)
}

impl crate::engine::loader::Engine {
    /// The fitted linear that replaces `key`, or `None` when the tensor is not fitted.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub(crate) fn nq_linear(&self, key: TensorKey) -> Option<&NqLinear> {
        let kind = usize::from(key.part);
        if kind >= NQ_KINDS {
            return None;
        }
        let at = usize::try_from(key.layer)
            .ok()?
            .checked_mul(NQ_KINDS)?
            .checked_add(kind)?;
        self.nq.linears.get(at)?.as_ref()
    }

    /// The sign-rank product of `x` through the linear that replaces `key`, or `None` when the
    /// tensor is not fitted. [`KernelMode::Exact`] reads the exact Q32.32 words
    /// ([`NqLinear::project`]); [`KernelMode::Q8Act`] reads the Q8 block codes
    /// ([`NqLinear::project_q8`]); [`KernelMode::Ternary`] reads the trit masks
    /// ([`NqLinear::project_ternary`]). Both block readers fall back to the exact reader when
    /// their bounds refuse.
    ///
    /// # Errors
    ///
    /// [`crate::engine::StepError::BadShape`] when the requested shape is not the factor's, and
    /// [`crate::engine::StepError::Nq`] for every [`crate::engine::NqError`] the product raises.
    pub(crate) fn nq_proj(
        &self,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        mode: crate::engine::KernelMode,
    ) -> Result<Option<Vec<i64>>, crate::engine::StepError> {
        let Some(linear) = self.nq_linear(key) else {
            return Ok(None);
        };
        if linear.out() != n_out || linear.inn() != n_in {
            return Err(crate::engine::StepError::BadShape {
                tensor: key.to_string(),
                dims: vec![
                    u64::try_from(linear.out()).unwrap_or(u64::MAX),
                    u64::try_from(linear.inn()).unwrap_or(u64::MAX),
                ],
                expected: vec![
                    u64::try_from(n_out).unwrap_or(u64::MAX),
                    u64::try_from(n_in).unwrap_or(u64::MAX),
                ],
            });
        }
        let out = match mode {
            crate::engine::KernelMode::Exact => linear.project(x),
            crate::engine::KernelMode::Q8Act => linear.project_q8(x).or_else(|e| {
                if e == crate::engine::NqError::Overflow {
                    linear.project(x)
                } else {
                    Err(e)
                }
            }),
            crate::engine::KernelMode::Ternary => linear.project_ternary(x).or_else(|e| {
                if e == crate::engine::NqError::Overflow {
                    linear.project(x)
                } else {
                    Err(e)
                }
            }),
        };
        out.map(Some).map_err(crate::engine::StepError::Nq)
    }

    /// The pooled form of [`EngineLoader::nq_proj`](Self::nq_proj) over `xs`: one `V` dispatch
    /// and one `U` dispatch for the whole column group, with each column's own reader,
    /// quantization and accumulation order. Every output column equals the per-column call's.
    ///
    /// Unlike [`Self::nq_proj`], a block reader that refuses is returned, not replaced by the
    /// exact reader: a whole-group fallback would change the columns that did not refuse, so the
    /// caller reruns the group through the per-column entry, which keeps that entry's per-column
    /// fallback.
    ///
    /// # Errors
    ///
    /// [`crate::engine::StepError::BadShape`] when the requested shape is not the factor's, and
    /// [`crate::engine::StepError::Nq`] for every [`crate::engine::NqError`] the product raises.
    pub(crate) fn nq_proj_pooled(
        &self,
        key: TensorKey,
        n_out: usize,
        n_in: usize,
        xs: &[Vec<i64>],
        mode: crate::engine::KernelMode,
    ) -> Result<Option<Vec<Vec<i64>>>, crate::engine::StepError> {
        let Some(linear) = self.nq_linear(key) else {
            return Ok(None);
        };
        if linear.out() != n_out || linear.inn() != n_in {
            return Err(crate::engine::StepError::BadShape {
                tensor: key.to_string(),
                dims: vec![
                    u64::try_from(linear.out()).unwrap_or(u64::MAX),
                    u64::try_from(linear.inn()).unwrap_or(u64::MAX),
                ],
                expected: vec![
                    u64::try_from(n_out).unwrap_or(u64::MAX),
                    u64::try_from(n_in).unwrap_or(u64::MAX),
                ],
            });
        }
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        let out = match mode {
            crate::engine::KernelMode::Exact => linear.project_pooled(&refs),
            crate::engine::KernelMode::Q8Act => linear.project_q8_pooled(&refs),
            crate::engine::KernelMode::Ternary => linear.project_ternary_pooled(&refs),
        };
        out.map(Some).map_err(crate::engine::StepError::Nq)
    }

    /// The fitted linears' stored bytes and the dense weights they carry: the two terms of the
    /// pack's bits per weight.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn nq_storage(&self) -> (u64, u64) {
        (self.nq.factor_bytes, self.nq.weights)
    }
}
