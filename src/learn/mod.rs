//! Learning track: training, finetuning and quantization ported from `../spinor-ttl` and `../fold`
//! One submodule per mechanism; each carries its focused
//! test `tests/learn_<x>.rs`.

/// GGUF v3 container writer: [`crate::format::gguf`]'s header, metadata and tensor types serialized
/// back to disk.
pub mod gguf;

/// llama.cpp imatrix `GGUF` reader: per-column activation importance vectors.
pub mod imatrix;

/// Activation-weighted `Q4_K` block quantization: the seal core emitting bytes
/// [`crate::format::quant::dequant_q4_k_block`] reads unchanged.
pub mod seal;

/// Bit budget of the one-bit `-nb` arm, split by component (sign, r, c, d, corrector, row index).
pub mod onebit_bits;

/// Exact-integer first-order sigma-delta: the executed body of the `sd1` arms, refined to
/// `Numeric.SigmaDelta`.
pub mod sigma_delta;

/// The NanoQuant binary low-rank factorisation (`SamsungLabs/NanoQuant` @ `a9e0a43`): ADMM to
/// `diag(s_post)·U·V·diag(s_pre)` with sign factors.
pub mod nanoquant;

/// Exact-integer Gram fold: `g += XᵀX` over a batch of `i8` rows, checked `i64`.
pub mod gram;

/// `IQ1_S` weighted-codebook quantizer and decoder: the port of llama.cpp `ece963f41`.
pub mod iq1s;

/// The 2048-entry `IQ1_S` grid extracted from llama.cpp `ece963f41`.
pub mod iq1s_grid;

/// Fixed codebooks: the 256-entry E8 ∪ O8 code of the one-bit fold transfer.
pub mod code;

/// Backward passes of the ported model: dense adjoints, attention with `RoPE`, KL/TV heads.
pub mod grad;

/// The fold-transfer pipeline: SyrkAcc, corrector, held-out evaluation and byte budget.
pub mod fold;

/// Q32.32 SPD solve and weighted Gram fold shared by the weight-fit steps.
pub mod solve;
