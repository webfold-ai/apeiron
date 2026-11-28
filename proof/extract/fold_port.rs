//! Extraction unit for the fold statistics (`src/learn/fold/stats/`) and the closed-form weight-fit
//! steps (`src/learn/fold/fit/`). Each module is reached through its real `mod.rs` by `#[path]`, so
//! the extracted bodies are the real sources; each `word_*` wrapper fixes the monomorphic root and
//! calls the real function, restating no formula.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! The unit is declared at the crate root, as `learn_port`/`solve_port` are:
//!
//! ```text
//! #[path = "fold_port.rs"]
//! pub mod fold_port;
//! ```
//!
//! and the `learn` mirror carries the paths these sources name absolutely (a `#[path]` inside an
//! inline module cannot traverse `..`, so every module is declared at the crate root and only
//! re-exported here):
//!
//! ```text
//! pub mod learn {
//!     pub mod solve { pub use crate::solve_port::{chol, eig, gemm, inv, q32, syrk}; }
//!     pub mod fold {
//!         pub use crate::fold_port::{fit, stats};
//!     }
//!     pub mod grad {
//!         pub use crate::grad_solve_port::model;
//!         pub use crate::learn_port::{attn, dense, head};
//!     }
//!     pub mod code { pub use crate::learn_port::e8; }
//!     pub use crate::grad_solve_port::gguf;
//! }
//! ```
//!
//! `src/learn/fold/fit/steps.rs` also reads `crate::engine::{fwht_fixed, HadamardError}`, the real
//! `src/engine/mod.rs` re-exports; the engine mirror adds those two names.
//!
//! # Boundary
//!
//! The layer/run/finish surface (`fold/layer/`, `fold/run/`, `fold/finish/`) is not declared here:
//! those bodies use `Iterator` adapters and function items the pinned Aeneas does not translate
//! (the signature translation aborts on `Iterator`), so they are a rewrite lane, not a port.
//!
//! Five fit entry points stay unrooted for the same class of reason, one body each: `fnorm_fit`,
//! `fold_fit`, `gptq_e8`, `feedback_u` and `nb_refine` carry a `?`/early return inside a loop or a
//! `break` out of an outer loop. Their callees that are clean are still reached through the roots
//! below.

/// The real `src/learn/fold/fit/`: the corrector and its eigendecomposition, the closed-form
/// `c`/`r`/`d` steps, the hierarchical quantisers and the `-nb` post-pass.
#[path = "../../src/learn/fold/fit/mod.rs"]
pub mod fit;

/// The real `src/learn/fold/stats/`: the activation SyrkAcc, the shrinkage intensities and the
/// `f64` forward the transfer consumes.
#[path = "../../src/learn/fold/stats/mod.rs"]
pub mod stats;

/// Mirror of `crate::learn` for the extraction unit: the real `learn` sources name
/// `crate::learn::…`, which resolves here to the `solve_port`/`fold_port`/`learn_port` copies.
pub mod learn {
    pub mod solve {
        pub use crate::solve_port::{chol, eig, gemm, inv, q32, syrk};
    }
    pub mod fold {
        pub use crate::fold_port::{fit, stats};
        pub use crate::fold_finish_port::{convert, finish};
        /// The per-layer step (`src/learn/fold/layer/`), declared in `fold_layer_port.rs`.
        pub use crate::fold_layer_port::layer;
        /// The `run` sources name `crate::learn::fold::run::convert`; only `convert` is declared.
        pub mod run {
            pub use crate::fold_finish_port::convert;
        }
    }
    pub mod grad {
        pub use crate::grad_solve_port::model;
        pub use crate::learn_port::{attn, dense, head};
    }
    pub mod code {
        pub use crate::learn_port::e8;
    }
    pub use crate::grad_solve_port::gguf;
}

// ── `learn::fold::stats` ─────────────────────────────────────────────────────────────────────

/// The real `ceil(bits / 8)`, rooted at the concrete `u128`.
pub fn word_stats_budget_bytes_ceil(bits: u128) -> Option<u128> {
    stats::budget::bytes_ceil(bits)
}

/// The real largest-fitting uniform body rate, rooted at the concrete `u64` rationals.
pub fn word_stats_budget_max_body_bpw(
    total_weights: u64,
    embd_weights: u64,
    head_weights: u64,
    embd_bpw: (u64, u64),
    head_bpw: (u64, u64),
    budget_bytes: u64,
    reserve_bytes: u64,
) -> Option<(u64, u64)> {
    stats::budget::max_body_bpw(
        total_weights,
        embd_weights,
        head_weights,
        embd_bpw,
        head_bpw,
        budget_bytes,
        reserve_bytes,
    )
}

/// The real Schäfer–Strimmer intensity, rooted at the concrete Q32.32 words.
pub fn word_stats_lw_intensity(
    s4: i64,
    q: &[i64],
    h: &[i64],
    o: usize,
    n: usize,
) -> Result<(i64, i64), stats::lw::LwError> {
    stats::lw::lw_intensity(s4, q, h, o, n)
}

/// The real batch-means intensity, rooted at the concrete Q32.32 words.
pub fn word_stats_lw_block_intensity(
    p: i64,
    b: usize,
    h: &[i64],
    o: usize,
    n: usize,
) -> Result<(i64, i64), stats::lw::LwError> {
    stats::lw::lw_block_intensity(p, b, h, o, n)
}

/// The real `a·b` Q32.32 product, rooted at the concrete words.
pub fn word_stats_parts_mul(a: i64, b: i64) -> Option<i64> {
    stats::parts::mul(a, b)
}

/// The real floored Q32.32 square root, rooted at the concrete word.
pub fn word_stats_parts_sqrt(x: i64) -> Option<i64> {
    stats::parts::sqrt(x)
}

/// The real `(sin t, cos t)` turn reduction, rooted at the concrete word.
pub fn word_stats_parts_sin_cos(t: i64) -> Option<(i64, i64)> {
    stats::parts::sin_cos(t)
}

/// The real power-of-two grid shift, rooted at the concrete word slice.
pub fn word_stats_parts_shift(a: &mut [i64], k: i32) -> Option<()> {
    stats::parts::shift(a, k)
}

/// The real panel normalisation, rooted at the concrete word slice.
pub fn word_stats_parts_panel(v: &[i64]) -> Option<(Vec<i64>, i32)> {
    stats::parts::panel(v)
}

/// The real `RMSNorm` row pass, rooted at the concrete Q32.32 words.
pub fn word_stats_parts_rms(
    x: &[i64],
    w: &[i64],
    eps: i64,
    t: usize,
    width: usize,
) -> Result<Vec<i64>, stats::parts::PartsError> {
    stats::parts::rms(x, w, eps, t, width)
}

/// The real `RoPE` half-rotation, rooted at the concrete Q32.32 words and rope tables.
pub fn word_stats_parts_rope_apply(
    dst: &mut [i64],
    src: &[i64],
    t: usize,
    n_head: usize,
    hd: usize,
    rope: &stats::parts::RopeTables,
) -> Result<(), stats::parts::PartsError> {
    stats::parts::rope_apply(dst, src, t, n_head, hd, rope)
}

/// The real `silu` on one Q32.32 word, rooted at the concrete word.
pub fn word_stats_parts_silu(g: i64) -> Option<i64> {
    stats::parts::silu(g)
}

/// The real `silu(g)·u` on words, rooted at the concrete words.
pub fn word_stats_parts_silu_mul(g: i64, u: i64) -> Option<i64> {
    stats::parts::silu_mul(g, u)
}

/// The real [`stats::parts::RopeTables::head_dim`] accessor, rooted at the concrete table.
pub fn word_stats_parts_rope_head_dim(rope: &stats::parts::RopeTables) -> usize {
    rope.head_dim()
}

/// The real [`stats::parts::RopeTables::t_max`] accessor, rooted at the concrete table.
pub fn word_stats_parts_rope_t_max(rope: &stats::parts::RopeTables) -> usize {
    rope.t_max()
}

// ── `learn::fold::fit` ─────────────────────────────────────────────────────────────────────

/// The rounded low-rank corrector of `ΔW` in the metric `Hl`.
pub fn word_fit_lowrank_corrector(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
    mode: fit::corrector::Mode,
    factors: bool,
) -> Result<fit::corrector::LowrankCorrector, fit::corrector::CorrectorError> {
    fit::corrector::lowrank_corrector(delta_w, hl, o, n, rank, mode, factors)
}

/// The exact eigenstage `M = (ΔW·Hl)·ΔWᵀ` and its top-`rank` eigenvalue sum.
pub fn word_fit_eigen_parts(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<(Vec<i64>, i64), fit::corrector::CorrectorError> {
    fit::corrector::eigen_parts(delta_w, hl, o, n, rank)
}

/// The symmetric eigendecomposition.
pub fn word_fit_eigh(
    a: &[i64],
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>), fit::eigh::EighError> {
    fit::eigh::eigh(a, n)
}

/// The `o × o` Gram `(ΔW·Hl)·ΔWᵀ` in the word carrier.
pub fn word_fit_eigen_gram(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, i32), fit::eigh::EighError> {
    fit::eigh::eigen_gram(delta_w, hl, o, n)
}

/// The exact rank-`rank` corrector.
pub fn word_fit_lowrank_corrector_exact(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<fit::eigh::Corrector, fit::eigh::EighError> {
    fit::eigh::lowrank_corrector_exact(delta_w, hl, o, n, rank)
}

/// The hierarchical group-scale quantiser.
pub fn word_fit_quant_r_hier(
    r: &[i64],
    o: usize,
    g: usize,
) -> Result<fit::quant::RHier, fit::quant::QuantError> {
    fit::quant::quant_r_hier(r, o, g)
}

/// Affine column-scale quantization.
pub fn word_fit_quant_c_u8(
    c: &[i64],
    bits: u32,
) -> Result<(Vec<i64>, u64), fit::quant::QuantError> {
    fit::quant::quant_c_u8(c, bits)
}

/// Bit-exact binary16 rounding of a word.
pub fn word_fit_f16r(w: i64) -> Result<i64, fit::quant::QuantError> {
    fit::quant::f16r(w)
}

/// The stored-grid level of code `k` on `(lo, sp)`: `None` when it leaves the word.
pub fn word_fit_r_level(lo: i64, sp: i64, k: u8) -> Option<i64> {
    fit::quant::r_level(lo, sp, k)
}

/// [`RHier::reconstruct`]: decode `rq` from the stored per-row grid and 4-bit codes.
pub fn word_fit_r_hier_reconstruct(r: &fit::quant::RHier) -> Vec<i64> {
    r.reconstruct()
}

/// The exact power-of-two exponent [`eigen_parts`] applies to the raw Gram words.
pub fn word_fit_gram_scale(m: &[i64]) -> i32 {
    fit::eigh::gram_scale(m)
}

// ── `learn::fold::stats::basis` ─────────────────────────────────────────────────────────────

/// The real damped input Gram `H + DAMP_Q·mean(diag H)·I`, rooted at the concrete words.
pub fn word_stats_basis_damped_gram(
    h: &[i64],
    n: usize,
) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::damped_gram(h, n)
}

/// The real `n × n` identity basis, rooted at the concrete Q32.32 words.
pub fn word_stats_basis_identity_basis(n: usize) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::identity_basis(n)
}

/// The real normalised 128-block Hadamard basis, rooted at the concrete Q32.32 words.
pub fn word_stats_basis_hadamard_basis(n: usize) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::hadamard_basis(n)
}

/// The real row energies `r_i = Σ_j B_ij²`, rooted at the concrete words.
pub fn word_stats_basis_row_energies(
    b: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::row_energies(b, o, n)
}

/// The real column energies `c_j = Σ_i B_ij²`, rooted at the concrete words.
pub fn word_stats_basis_column_energies(
    b: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::column_energies(b, o, n)
}

/// The real `Σ_j ln c_j` over the positive column energies, rooted at the concrete words.
pub fn word_stats_basis_sum_log_column_energies(c: &[i64]) -> i64 {
    stats::basis::sum_log_column_energies(c)
}

/// The real reverse water-filling distortion `D(θ) = Σ min(λ, θ)`, rooted at the concrete words.
pub fn word_stats_basis_distortion_at(
    lambda: &[i64],
    theta: i64,
) -> Result<i64, stats::basis::BasisError> {
    stats::basis::distortion_at(lambda, theta)
}

/// The real reverse water-filling rate `R(θ)`, rooted at the concrete words.
pub fn word_stats_basis_rate_at(
    lambda: &[i64],
    theta: i64,
) -> Result<i64, stats::basis::BasisError> {
    stats::basis::rate_at(lambda, theta)
}

/// The real fixed-rate distortion `D_u`, rooted at the concrete words.
pub fn word_stats_basis_uniform_distortion(
    lambda: &[i64],
    budget_bits: i64,
    side_bits: i64,
) -> Result<i64, stats::basis::BasisError> {
    stats::basis::uniform_distortion(lambda, budget_bits, side_bits)
}

/// The real reverse water-filling solution at the target rate, rooted at the concrete words.
pub fn word_stats_basis_solve_water_fill(
    lambda: &[i64],
    budget_bits: i64,
    side_bits: i64,
) -> stats::basis::WaterFill {
    stats::basis::solve_water_fill(lambda, budget_bits, side_bits)
}

/// The real [`stats::basis::Density::side_bits`] charge at the concrete separable density.
pub fn word_stats_basis_density_side_bits(o: usize, n: usize) -> i64 {
    stats::basis::Density::Separable.side_bits(o, n)
}

/// The real [`stats::basis::Density::sigma`] coefficient-variance map at the separable density.
pub fn word_stats_basis_density_sigma(
    b: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, stats::basis::BasisError> {
    stats::basis::Density::Separable.sigma(b, o, n)
}

/// The real adapted-butterfly angle side charge, rooted at the concrete bit count.
pub fn word_stats_basis_butterfly_side_bits(n: usize) -> i64 {
    stats::basis::butterfly_side_bits(n)
}

// ── `learn::fold::stats::hocapture` ────────────────────────────────────────────────────────

/// The real Q32.32 softmax row, rooted at the concrete words.
pub fn word_stats_hocapture_softmax_row(
    z: &[i64],
) -> Result<Vec<i64>, stats::hocapture::HocaptureError> {
    stats::hocapture::softmax_row(z)
}

/// The real sampled-label cross-entropy sum, rooted at the concrete logits and targets.
pub fn word_stats_hocapture_cross_entropy_sum(
    logits: &[i64],
    t: usize,
    vocab: usize,
    targets: &[u32],
) -> Result<i64, stats::hocapture::HocaptureError> {
    stats::hocapture::cross_entropy_sum(logits, t, vocab, targets)
}

/// The real sampled-label logit gradient `p − onehot`, rooted at the concrete words.
pub fn word_stats_hocapture_logits_grad_ce(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    out: &mut [i64],
) -> Result<(), stats::hocapture::HocaptureError> {
    stats::hocapture::logits_grad_ce(logits, targets, vocab, out)
}

/// The real `K` sampled targets per token, rooted at the concrete words and seed.
pub fn word_stats_hocapture_sample_targets(
    logits: &[i64],
    t: usize,
    vocab: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<u32>, stats::hocapture::HocaptureError> {
    stats::hocapture::sample_targets(logits, t, vocab, k, seed)
}

/// The real zero accumulator of width `O`, rooted at the concrete width.
pub fn word_stats_hocapture_kind_stats_zeros(
    o: usize,
) -> Result<stats::hocapture::KindStats, stats::hocapture::HocaptureError> {
    stats::hocapture::KindStats::zeros(o)
}

// ── `learn::fold::stats::heldout` ──────────────────────────────────────────────────────────

/// The real per-row base scoring, rooted at the concrete Q32.32 words.
pub fn word_stats_heldout_score_base_row(
    z: &[i64],
    lp: &[i64],
    target: usize,
    base_argmax: Option<usize>,
) -> Result<stats::heldout::RowSums, stats::heldout::HeldOutError> {
    stats::heldout::score_base_row(z, lp, target, base_argmax)
}
