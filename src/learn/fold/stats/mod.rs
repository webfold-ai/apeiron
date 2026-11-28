//! Statistics the fold transfer consumes: activation SyrkAcc, held-out evaluation, shrinkage
//! intensity and byte accounting.

/// Input-basis concentration arms (`H`, `I`, `J`, `V`) and reverse water-filling at 1.18 b/w.
pub mod basis;
/// Exact 27B byte accounting at a target bits-per-weight, over the reference class table.
pub mod budget;
/// Per-layer activation SyrkAcc of the four linears, accumulated from the forward tape.
pub mod capture;
/// Softmax, sampled-label cross entropy and target draw of the held-out capture.
pub mod ce;
/// Held-out KL, TV, top-1 and NLL of a student against a teacher on the ported model.
pub mod heldout;
/// Output-side Hessian sketch `HO = Σ w_t g_t g_tᵀ` from sampled-label gradients on the ported backward pass.
pub mod hocapture;
/// Reader of llama.cpp's `--kl-divergence-base` log-probability file, one chunk at a time.
pub mod kldbase;
/// Analytic shrinkage intensities (Schäfer–Strimmer, Ledoit–Wolf) of the output-side Hessian sketch.
pub mod lw;
/// `f64` CPU forward of one `Qwen2` decoder layer, returning every intermediate the fold transfer
/// reads.
pub mod parts;
