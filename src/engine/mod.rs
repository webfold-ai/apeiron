//! Integer inference engine: the fixed-point value path, including token-embedding lookup, on the
//! GGUF executing path.
//!
//! # Fixed-point convention
//!
//! The executing path carries no float. Every dequantized weight is an `i64` in **Q32.32** — an
//! `i64` `v` denotes the real number `v / 2^32` — matching
//! [`crate::numeric::fixed_exp`], [`crate::numeric::fixed_rsqrt`] and
//! [`crate::numeric::fixed_trig`], so one scale carries from the loaded weight through every
//! kernel.
//!
//! **Scale conversion is exact.** Every finite `f16` is an integer multiple of `2^-24` (the
//! smallest subnormal), so it is an exact multiple of the `2^-32` unit. [`f16_to_fixed`] decodes
//! the `f16` bits with shifts and adds only, and returns the exact value times `2^32`; `Inf` and
//! `NaN` refuse. The affine `Q4_K` combination, the `Q8_0` scale-times-code product and the `F16`
//! weight are therefore integer arithmetic with no rounding at load.
//!
//! **Representable ranges.** A finite `f16` maps into `[-2^48, 2^48]`; a `Q8_0` or `Q5_0` product
//! with a code in `[-128, 127]` into `[-2^55, 2^55]`; a `Q4_K` product with `sc, min <= 63` and
//! `code <= 15` into `[-2^59, 2^59]`; a `Q6_K` product with `sc <= 128` and `code <= 32` into
//! `[-2^60, 2^60]`; a `Q2_K` or `Q3_K` product into `[-2^58, 2^58]`. Every dequantized weight is an
//! exact `i64` multiple of the `2^-32` unit with no rounding; every product is nonetheless checked
//! and a failure is [`EngineError::FixedOverflow`], never a wrap.
//!
//! # Token embedding
//!
//! [`Engine::token_embedding`] reads one row of `token_embd.weight` and returns its `n_embd`
//! Q32.32 values. Supported ggml types: `F16`, `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, `Q2_K` and `Q3_K`.
//! `F32` refuses with [`EngineError::UnsupportedEmbeddingQuant`]: its domain needs a rounding
//! contract (an `f32` exponent can fall below `2^-32`). The K-quant scales come from the exact
//! `f16` to Q32.32 decode and the integer scale/code accessors in `crate::format::quant`.
//!
//! # `RoPE`
//!
//! [`RopeTable`] is the load-time frequency table of port-map row 3: `freq_j = theta^(-2j/head_dim)`
//! for `j < head_dim / 2`, each an `i64` Q32.32. The table is built once from
//! [`EngineConfig::rope_theta`] and reused across every layer and token.
//!
//! **Load-time float boundary.** [The table builder](RopeTable::new) computes the frequencies with
//! the host `f64` `powf` and rounds once to Q32.32. That is the only float in row 3 and it is the
//! load boundary; every per-token operation is integer. [`RopeTable::from_freqs`] accepts an
//! explicit Q32.32 table, so the boundary can be replaced without touching the rotation.
//!
//! **Per-token path.** [`RopeTable::rotate`] (and [`rope_rotate_fixed`]) forms the angle
//! `pos * freq_j` by a checked `i64` multiply, calls [`crate::numeric::fixed_trig::fixed_sin_cos`]
//! for the `sin`/`cos` (its quadrant reduction is the mod-`2pi` reduction), and writes each rotated
//! component through the value-path epilogue [`round_shift_32`], one round-to-nearest `>> 32` per
//! component. The admitted angle domain is the trig window `[-4096, 4096]` radians; a product or a
//! rotated component outside its fixed-point range refuses with [`RopeError`] and is never clamped.
//!
//! # Projections (port-map row 4)
//!
//! [`matvec_fixed`] is the single projection kernel for every served shape. It takes the raw
//! row-major weight payload, the ggml type, the output-row count `n_out` and the row width `n_in`,
//! dequantizes one row at a time to Q32.32 with the [`dequant_row_fixed`] family, and reduces each
//! row against the Q32.32 activation `x` with [`row_dot_fixed`]: products accumulate exactly in
//! `i128`, the sum takes one round-to-nearest `>> 32` epilogue per output (ties upward), and no
//! float appears on the executing path. Q/K/V use the GQA head counts (`n_head·head_dim` and
//! `n_head_kv·head_dim` rows), the FFN uses `n_ff`, and the LM head uses `vocab` rows; the shape is
//! the caller's, so there is no second kernel. A dimension mismatch, a short payload and an
//! accumulator or output outside `i128`/`i64` refuse with [`MatvecError`]; nothing is clamped.
//!
//! # Attention (port-map row 5)
//!
//! [`KvState`] is the per-layer, per-KV-head key/value cache (`n_head_kv` heads, GQA); keys and
//! values are Q32.32 and [`KvState::append`] writes the next position of a layer under a checked
//! capacity. [`KvState::attn_head`] reads one query head at position `pos` over the causal prefix
//! `0..=pos`: the score is `round(round(Σ q·k_j / 2^32) · scale / 2^32)` with
//! `scale = fixed_rsqrt(head_dim)` — the fold `attn_scale` — using [`row_dot_fixed`] and one
//! value-path epilogue. The softmax subtracts the exact maximum and evaluates [`fixed_exp`] over
//! `[-16·ONE, 0]`, saturating to `0` for an argument below `-16·ONE` (the dropped mass is
//! `≤ exp(−16) ≈ 1.1e-7` against a total `≥ 1`); the total is the checked `i128` sum, and each
//! weight is one round-to-nearest division `p_j = round(e_j·2^32 / Σe)`; the output is the checked
//! `i128` sum `round(Σ p_j·v_j / 2^32)` under the same epilogue. GQA maps query head `h` to KV head
//! `h / (n_head / n_head_kv)`. A shape, position, capacity, head or `i128`/`i64` range violation
//! refuses with [`AttnError`]; nothing is clamped. No float appears on the executing path;
//! `fixed_rsqrt` supplies the one load-time score constant.
//!
//! # FFN / `SwiGLU` (port-map row 6)
//!
//! [`silu_fixed`] is the scalar `SiLU` of the fold `moe.rs::swiglu` on Q32.32:
//! `silu(v) = v / (1 + exp(-v))`, evaluated as one checked negation, one [`fixed_exp`] call and one
//! round-to-nearest division `round(|v|·2^32 / (2^32 + exp(-v)))` with the sign reapplied (ties away
//! from zero). The exact path is the open interval `-21·ONE < v < 16·ONE`, which keeps `-v` strictly
//! inside the `fixed_exp` window `[-16·ONE, 21·ONE]`; outside it saturates, returning `v` for
//! `v ≥ 16·ONE` (bound `|silu(v) − v| ≤ |v|·exp(−v) ≤ 1.8e-6`) and `0` for `v ≤ −21·ONE` (bound
//! `|silu(v)| ≤ |v|·exp(v) ≤ 1.6e-8`). `silu(0) = 0` exactly.
//!
//! [`ffn_fixed`] is one token of the dense FFN block in the fold's order: `gate =
//! matvec_fixed(W_gate, x)`, `up = matvec_fixed(W_up, x)`, `h_i = round(silu(gate_i)·up_i / 2^32)`
//! (one value-path [`round_shift_32`] epilogue per element, the gate inside the `SiLU`), then
//! `y = matvec_fixed(W_down, h)`. `W_gate`/`W_up` are `n_ff × n_embd`, `W_down` is `n_embd × n_ff`,
//! so the block preserves the residual width. The gate/`up` ordering matches the fold's
//! concatenated `gateup` (`gate_proj` rows then `up_proj` rows) with the gate in the `SiLU`. Shape,
//! payload, quant, zero-denominator and `i128`/`i64` range violations refuse through [`FfnError`]
//! (the in-window `fixed_exp` refusal is unreachable and retained for that path); no float appears on
//! the executing path.
//!
//! # LM head (port-map row 7)
//!
//! [`Engine::lm_head_fixed`] projects the residual activation through the output projection to one
//! Q32.32 logit per vocabulary entry. The head weight is `output.weight` when the tensor directory
//! carries it, otherwise the tied `token_embd.weight`; both are `vocab × n_embd` row-major
//! (`dims = [n_embd, vocab]`), so the call is [`matvec_fixed`] with `n_out = vocab`, `n_in =
//! n_embd` over the exact tensor payload. Shape, stored-head-shape, extent, quant and projection
//! violations refuse through [`LmHeadError`]; nothing is clamped.
//!
//! # Sampler (port-map row 8)
//!
//! [`argmax_fixed`] is the v0 greedy path: the index of the maximal Q32.32 logit, ties resolved to
//! the lowest index, with an empty-logit refusal; it performs no division and no float.
//! [`top_p_fixed`] is the integer nucleus filter behind temperature/top-p: it shifts by the exact
//! maximum, evaluates [`fixed_exp`] over `[-16·ONE, 0]` (a logit below the window refuses), forms
//! each Q32.32 probability by one round-to-nearest division against the checked `i128` total,
//! orders the entries by descending logit (ties to the lowest index) and admits the shortest
//! prefix whose cumulative probability reaches the Q32.32 threshold `p`. It returns the candidate
//! set only; the stochastic draw from the nucleus and temperature scaling are not on this path.
//!
//! # Decode-loop port map (v0)
//!
//! The reference decode loop is the served CPU path in the sibling `fold` tree. Each row names the
//! fold function, its apeiron target and the fixed-point kernel or refusal it needs. Order is
//! dependency order: a row may use the rows above it.
//!
//! | # | stage | fold reference (`~/Workspace/fold/`) | apeiron target | v0 fixed-point kernel |
//! |---|---|---|---|---|
//! | 0 | config, tensor table | `bedrock/src/serve/model/loaders.rs` (GGUF load), `types.rs::Config` | `src/engine/loader.rs` | none (metadata; integer/`f64` config only) |
//! | 1 | token embedding | `forward.rs::cpu_step_f32` embed read `m.embed.cpu[t*hidden..]` | [`Engine::token_embedding`] | [`f16_to_fixed`] + the per-kind row dequantizer; **landed** |
//! | 2 | `RMSNorm` | `forward.rs::cpu_rms`, `forward.rs::rms_span`, `attn.rs::cpu_rms_into` | [`rms_norm_fixed`] | `numeric::fixed_rsqrt::fixed_rsqrt`; mean square is Q32.32 `round(sum x_i^2 / (n * 2^32))`, `eps_fixed` is a caller Q32.32 addend, and `y_i = round(x_i * r / 2^32)`; window `[2^-20, 2^20]`; **landed** |
//! | 3 | `RoPE` | `forward.rs::rope_table`, `forward.rs::cpu_rope_cached`, `forward.rs::cpu_rope` | `src/engine/mod.rs` | `numeric::fixed_trig::fixed_sin_cos`; the `1/theta^(2j/hd)` table is built once at load from `EngineConfig::rope_theta`; fold rotates the pair `(j, j + hd/2)`; **landed** |
//! | 4 | QKV / O / FFN projections | `matvec.rs::cpu_matvec_f32`, `matvec.rs::cpu_matvec_with`, `matvec.rs::cpu_matmat_f32_into`, `rowdot.rs::quantize_act_sb`, `rowdot.rs::row_dot_q2sb` | `src/engine/mod.rs` | [`matvec_fixed`] (with the row-dot helper [`row_dot_fixed`]): weight row (Q32.32) x activation (Q32.32) accumulate in `i128`, one round-to-nearest `>> 32` per output; **landed** |
//! | 5 | attention | `attn.rs::attn_head_into` (scores, softmax, value read), `state.rs::CpuState::kv_write` (int8 KV in the fold; Q32.32 here) | `src/engine/mod.rs` | scores are Q32.32 dots scaled by `fixed_rsqrt(hd)`; softmax weights via `numeric::fixed_exp::fixed_exp` (window `[-16, 21]`); GQA group `n_head / n_head_kv`; **landed** |
//! | 6 | FFN (dense) | `moe.rs::cpu_ffn`, `moe.rs::swiglu` | [`silu_fixed`], [`ffn_fixed`] | the fold's order `silu(gate) ⊙ up`: `silu_fixed(v) = round(\|v\|·2^32/(2^32 + exp(-v)))` on `-21·ONE < v < 16·ONE`, saturating to `v`/`0` outside, `h_i = round(silu(gate_i)·up_i / 2^32)`, then the down projection; **landed** |
//! | 7 | LM head | `forward.rs::apply_lm_f32`, `forward.rs::apply_lm_rows` | [`Engine::lm_head_fixed`] | the row-4 projection kernel over `vocab × n_embd`; `output.weight` when present, else the tied `token_embd.weight`; **landed** |
//! | 8 | sampler | `forward.rs::cpu_step_f32` (logits); `runtime.rs` argmax/softmax sampling | [`argmax_fixed`], [`top_p_fixed`] | v0 greedy is [`argmax_fixed`] (ties lowest index, no division); [`top_p_fixed`] is the integer nucleus filter over the `fixed_exp` window with one round-to-nearest division per probability; **landed** |
//! | 9 | decode loop (rows 0–8 composed) | `forward.rs::cpu_step_f32`, `attn.rs::cpu_hidden_with`, `moe.rs::swiglu` | [`Engine::step`], [`Engine::step_with_kv`] | the layer order, residual placement and once-per-step `RoPE` table above; projection/bias/gain reads from the tensor table; **landed** |
//!
//! **v0 epilogue.** Every activation, weight, norm output, score and logit is an `i64` Q32.32. Any
//! step that would introduce a fractional scale (projection accumulation, the `RMSNorm` mean, the
//! softmax normalisation, the `SiLU` division) applies exactly one documented `>> 32`
//! round-to-nearest; no step is allowed an undisclosed shift. A value outside a kernel's declared
//! window (`fixed_rsqrt`, `fixed_exp`, `fixed_sin_cos`) refuses rather than clamping. The full row
//! dequantizer and rows 2 (`RMSNorm`), 3 (`RoPE`), 4 (projections), 5 (attention), 6 (FFN), 7 (`LM
//! head`) and 8 (`sampler`) are landed, and rows 0–8 are composed into [`Engine::step`] /
//! [`Engine::step_with_kv`] (the decode-loop integration). Open on row 8: the stochastic draw from
//! the nucleus and temperature scaling, which need a deterministic RNG contract that is not on this
//! path.
//!
//! # Decode loop (rows 0–8 composed)
//!
//! [`Engine::step`] is the v0 greedy decode step: one token in, one sampled token out, the kernels
//! of rows 1–8 in the fold `forward.rs::cpu_step_f32` / `attn.rs::cpu_hidden_with` order. Per
//! layer: `attn_norm` `RMSNorm` → Q/K/V projections with the Qwen2 attention bias → `RoPE` at the
//! cache position → `KvState::append` → `attn_head` per query head → O projection → residual add;
//! then `ffn_norm` → `ffn_fixed` → residual add. After the layers: `output_norm` →
//! `lm_head_fixed` → `argmax_fixed`. The O residual is added left of the second norm and the FFN
//! residual right of it, exactly the `cpu_hidden_with` `x += o` / `h2 = cpu_rms(x)` / `x += d`
//! placement; the `RoPE` table is built once per step and reused across layers, as
//! `cpu_hidden_with` builds it before the layer loop.
//!
//! **KV state is the caller's.** The loader's `Engine` struct is not extended, so the two entry
//! points differ only in where the cache lives. [`Engine::step`] allocates a one-position
//! [`KvState`] and is a stateless single-token forward at position 0; [`Engine::step_with_kv`]
//! takes a caller-owned [`KvState`] and advances it, the stateful primitive a decode loop calls
//! with a capacity-sized cache. Both derive the position from `KvState::occupied`, so a position
//! cannot be passed out of sync with the cache.
//!
//! **Gain and bias are F32 at the load boundary.** The fold `cpu_rms` multiplies the normed row by
//! the per-channel `*_norm.weight`, and Qwen2 attention carries `attn_{q,k,v}.bias`; both are
//! stored `F32`. [`f32_to_fixed`] is the one rounding contract for those vectors (round-to-nearest
//! at Q32.32; `NaN`/`Inf` and out-of-`i64` refuse), used only for the norm gain and the bias.
//! Projection weights keep the exact row dequantizers, and an `F32` weight row still refuses
//! through [`dequant_row_fixed`]. The `rms_eps` metadata is rounded once to Q32.32 by
//! [`Engine::eps_fixed`].
//!
//! The per-head QK-norm (`attn_q_norm`/`attn_k_norm`, present in qwen3 files) runs when the layer
//! carries it, through `Engine::apply_qk_norms` after the Q/K projections and before `RoPE`.
//!
//! **Open.** The dense FFN only: MoE/routed experts, `YaRN` `RoPE` scaling, the stochastic draw and
//! temperature scaling are not on this path.
//!
//! [`prefix`] provides KV prefix caching on Apeiron's growable table primitive. [`trace`] is the
//! opt-in per-phase wall-clock attribution the benchmark harness reads.

pub mod loader;
pub mod prefix;
/// Fixture-path resolution shared by the unit and integration tests; not part of the public API.
#[doc(hidden)]
pub mod testdata;
pub mod trace;

/// Fractional bits of the executing fixed-point representation (Q32.32).
pub const FIXED_FRAC_BITS: u32 = 32;

/// The Q32.32 unit: `ONE` denotes `1.0`.
pub const ONE: i64 = 1 << FIXED_FRAC_BITS;

// One module per port-map row plus the shared value-path helpers and the decode step. Every item
// stays reachable at its previous `crate::engine::*` path through the re-exports below.
mod attn;
pub(crate) mod chart;
mod error;
pub(crate) mod kern;
mod mixer;
pub(crate) mod num;
mod ops;
mod step;
#[cfg(test)]
mod tests;

pub use self::attn::f32_core::attn_core_f32_strided;
pub use self::attn::{AttnError, KvState};
pub use self::error::{EngineError, TensorKey};
pub use self::kern::nq::NqError;
pub use self::loader::{AffineGeom, AffineVectors, Engine, MixerKind, TensorEntry};
pub use self::mixer::AffineState;
pub use self::num::dequant::{dequant_row_fixed, f16_to_fixed};
pub use self::num::f32_convert::{f32_to_fixed, F32Error};
pub use self::num::value::fixed_vector_checksum;
pub use self::ops::ffn::{
    ffn_fixed, ffn_fixed_tapped, silu_fixed, FfnError, FfnStages, ProjWeights,
};
pub use self::ops::hadamard::{fwht_fixed, HadamardError};
pub use self::ops::lm_head::LmHeadError;
pub use self::ops::matvec::{matvec_fixed, row_dot_fixed, MatvecError};
pub use self::ops::moe::{gate_weights, moe_ffn_fixed, route_topk, ExpertWeights, MoeError};
pub use self::ops::norm::{rms_norm_fixed, NormError};
pub use self::ops::rope::{rope_rotate_fixed, RopeError, RopeTable};
pub use self::ops::sampler::{argmax_fixed, top_p_fixed, SamplerError};
pub use self::step::cold_store::{ColdStore, ColdStoreError};
pub use self::step::{DecodeState, KernelMode, MtpState, SpecStats, StepError, StepResult};
