//! Fold-transfer pipeline on the crate build path: the weight-fit steps (`fit`) and the activation,
//! evaluation and accounting statistics they consume (`stats`).

/// Finishing stage: the frozen-code coded weight plane carrying the trainable log-multiplicative
/// latents.
pub mod finish;
/// Weight-fit steps: Jacobi eigendecomposition, scale quantisers, closed-form steps, ICM passes and
/// the low-rank corrector.
pub mod fit;
/// Per-layer production scaling steps these fits feed into (the `-resq` plane row scale).
pub mod layer;
/// Run glue: the `Q32.32` engine weights below the `f64` fold driver, in both directions.
pub mod run;
/// Statistics: activation SyrkAcc, held-out KL, layer-wise intensity and the exact byte budget.
pub mod stats;
