//! Input-basis concentration for the 1.18 b/w fold: does a basis that concentrates the weight's
//! column energy beat the Hadamard basis that flattens it, once the coder may vary its rate per
//! coefficient?
//!
//! # Model (grade `model`)
//!
//! For `W` (row-major `O × n`, Q32.32 words), a per-input orthogonal `Q` (block
//! diagonal in [`GROUP`] columns, like the 128-block Hadamard transform), the damped Gram
//! `H + 0.01·mean(diag H)·I` and
//! `B = W·Q`:
//!
//! - `M = Qᵀ·H·Q`; the feedback weights `d_j` are the `LDLᵀ` pivots of `M` (`d_j = L_jj²` with
//!   `M = L·Lᵀ`), and `Π_j d_j = det H` for every orthogonal `Q`;
//! - the coefficient variances are [`Density::Separable`] `σ²_ij = r_i·c_j/m` (`r_i = Σ_j B_ij²`,
//!   `c_j`, `m = Σ_ij B_ij²`; 16 side bits per row and column), [`Density::Tile`] the 16×16 tile mean
//!   of `B²` (8 bits per tile), or [`Density::Oracle`] `σ²_ij = B_ij²` (no density bits);
//! - reverse water-filling at `λ_ij = d_j·σ²_ij` has `D(θ) = Σ min(λ_ij, θ)` and
//!   `R(θ) = Σ max(0, ½·log₂(λ_ij/θ))`, with `θ` bisected in the log domain so that
//!   `R + side_bits = 1.18·O·n`; the fixed-rate baseline is `D_u = Σ λ_ij·2^{−2·(1.18 − side/(O·n))}`;
//! - the bases are the normalised 128-block Hadamard ([`hadamard_basis`]), the identity
//!   ([`identity_basis`]), the adapted butterfly ([`butterfly_basis`]: seven-stage radix-2 butterfly,
//!   each node a Givens rotation by the Jacobi angle `θ = ½·atan2(2·G_ab, G_aa − G_bb)` of the
//!   current `G_c = BᵀB`, then up to three sweeps while `Σ log c_j` falls by more than `1e-6`
//!   relative; 8 angle bits per node) and the per-block eigenbasis of `WᵀW` ([`eigen_basis`], side
//!   bits not counted: an upper reference). The output side is the identity in every arm.
//!
//! # Carrier and the pool
//!
//! Every quantity is a `Q32.32` word: one `i128` accumulation and one rounding per entry, with the
//! transcendentals from [`crate::numeric`] — `fixed_ln`, `fixed_exp`, `fixed_sin_cos` and
//! `fixed_atan2`. `B = W·Q` and the block Gram run on [`gemm_nn`] and
//! [`wsyrk`], each operand brought inside the `2³⁶` bound by an exact `2⁻ᵏ` panel shift that cancels
//! on the read-back. `M = Qᵀ·H·Q` is the one pooled task ([`MetricRows`]): `H` is unbounded, so its
//! exact accumulation is the model's premise. Every vector-sized loop (energies, pivots,
//! water-filling, rates, rotations, the density map) is serial.
//!
//! # Subsumption
//!
//! The empirical best for "rotation before low-bit quantisation" is a learned or folded rotation:
//! QuaRot (arXiv `2404.00456`), SpinQuant (arXiv `2405.16406`), DuQuant (arXiv `2406.01721`).
//! `rg -l -i "spinquant|quarot|duquant" ~/Workspace` returns only citations and no implementation, so
//! this lane is `not-subsumable` here: those methods FLATTEN the weight for a fixed-rate coder, while
//! this lane CONCENTRATES the coefficient energy and lets reverse water-filling spend 3–4 bits on the
//! occupied modes and 0 on the vacuum.
//!
//! # Evidence grade
//!
//! `model`: the densities model the coefficient magnitudes, the side-information counts are the
//! stated bit charges, and the distortion is the Gaussian rate-distortion model. No coder, transfer
//! or held-out KL is measured here; the synthetic tests are `tested`.
//!
//! # Trust boundary
//!
//! The activation Gram is the crate forward's Q32.32 tape and the weights are the engine's Q32.32
//! parameters; every entry point takes and returns Q32.32 words. Not covered: the quantiser's own
//! accumulation, any output-side rotation, any real coder, and any quality effect.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the module's arithmetic is over Q32.32 words and bit budgets, so `usize → i64/i128` \
              enters the exact accumulator domain; the single-letter names are the matrix symbols \
              of the model (O, n, Q, H, B, M, d, c, r, λ, θ); the flat row-major loops keep the \
              multiply-add count visible"
)]

use crate::engine::chart::pool::{chunks_mut_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::engine::kern::limb::{rowdot_q32, LimbError};
use crate::learn::fold::fit::eigh::{eigh, EighError};
use crate::learn::fold::fit::steps::fwht_words;
use crate::learn::solve::chol::{equilibrate, factor_scaled, SolveError};
use crate::learn::solve::gemm::{gemm_nn, GemmError};
use crate::learn::solve::q32::{div_rne, narrow, round_q32_even, rshift_rne};
use crate::learn::solve::syrk::{wsyrk, SyrkError};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_ln::{fixed_ln, LN2_Q32, ONE};
use crate::numeric::fixed_trig::{fixed_atan2, fixed_sin_cos};

use super::parts::{mul, panel, shift};

/// `log₂ x` for `x > 0`: the numeric `ln` kernel divided by `ln 2`, one rounding. `None` for
/// `x <= 0`. `O(1)`.
fn log2(x: i64) -> Option<i64> {
    let l = fixed_ln(x).ok()?;
    narrow(div_rne(i128::from(l) << 32, i128::from(LN2_Q32))?)
}

/// `2^t` for a `Q32.32` exponent `t`: `k = floor t` and `frac = t - k`, then the numeric `exp`
/// kernel at `frac·ln 2` scaled by the exact `2^k` (a checked left shift, a rounded right one).
/// `None` outside the kernel's window or off the grid. `O(1)`.
fn exp2(t: i64) -> Option<i64> {
    let k = t >> 32;
    let y = mul(t - (k << 32), LN2_Q32)?;
    let e = fixed_exp(y).ok()?;
    // The shift amounts are range-checked here rather than through `I128::checked_shl`, which the
    // model does not carry; `<<` discards the value bits exactly as the checked shift did.
    if k >= 0 {
        let s = u32::try_from(k).ok()?;
        if s >= 128 {
            return None;
        }
        narrow(i128::from(e) << s)
    } else {
        let mag = k.unsigned_abs();
        if mag > 126 {
            return None;
        }
        narrow(rshift_rne(i128::from(e), mag as u32)?)
    }
}

/// Side of one transform block. Every linear of the `Qwen2.5 0.5B` model has an input width that is
/// a multiple of this, matching the `fwht_words` block width.
pub const GROUP: usize = 128;

/// Diagonal damping of the input Gram: `H + DAMP·mean(diag H)·I`.
pub const DAMP_Q: i64 = 42_949_673;

/// Side of one coefficient-variance tile in [`Density::Tile`].
pub const TILE: usize = 16;

/// Reverse water-filling iterations; the rate-convergence threshold is `RATE_TOL_Q·N`.
const BISECTION_ITERS: usize = 200;

/// The rate-convergence threshold per coefficient, `|ΔR| < RATE_TOL_Q·N`, as a `Q32.32` word.
const RATE_TOL_Q: i64 = 4;

/// The relative fall of `Σ log c_j` that ends the butterfly's extra sweeps, as a `Q32.32` word.
const SWEEP_TOL_Q: i64 = 4_295;

/// A refused basis construction or distortion evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasisError {
    /// A slice length or dimension disagrees with the declared geometry.
    Shape(String),
    /// A length product left `usize`.
    Overflow,
    /// A Cholesky pivot of `M` was not positive, so the damped metric is not positive definite.
    NotPositiveDefinite,
    /// The symmetric eigendecomposition did not converge.
    Eigh(String),
    /// A value is outside the `Q32.32` grid, a primitive bound was violated, or a rounded result
    /// left the grid. The entry points refuse instead of saturating.
    OutOfRange,
}

impl core::fmt::Display for BasisError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BasisError::Shape(m) => write!(f, "shape: {m}"),
            BasisError::Overflow => write!(f, "length product left usize"),
            BasisError::NotPositiveDefinite => write!(f, "metric is not positive definite"),
            BasisError::Eigh(m) => write!(f, "eigendecomposition: {m}"),
            BasisError::OutOfRange => write!(f, "value outside the Q32.32 grid"),
        }
    }
}

impl std::error::Error for BasisError {}

fn square(rows: usize, cols: usize) -> Result<usize, BasisError> {
    rows.checked_mul(cols).ok_or(BasisError::Overflow)
}

fn expect(len: usize, rows: usize, cols: usize, what: &str) -> Result<usize, BasisError> {
    match square(rows, cols)? {
        n if n == len => Ok(n),
        n => Err(BasisError::Shape(format!(
            "{what}: {len} != {rows}·{cols} = {n}"
        ))),
    }
}

fn expect_blocks(n: usize) -> Result<usize, BasisError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(BasisError::Shape(format!(
            "n = {n} is not a positive multiple of {GROUP}"
        )));
    }
    Ok(n / GROUP)
}

/// The exact `i128` `Q64` accumulator rounded once to a `Q32.32` word. `O(1)`.
fn round64(acc: i128) -> Result<i64, BasisError> {
    round_q32_even(acc).map_err(|_| BasisError::OutOfRange)
}

/// The buffer a pool pass filled, or the first error it recorded.
fn pooled<T>(buf: T, first: FirstErr<usize, BasisError>) -> Result<T, BasisError> {
    first.take().map_or(Ok(buf), |(_, e)| Err(e))
}

/// [`GemmError`] as a [`BasisError`].
fn map_gemm(e: GemmError) -> BasisError {
    match e {
        GemmError::ZeroDim | GemmError::ShapeMismatch => BasisError::Shape("gemm".to_string()),
        GemmError::InputBound | GemmError::Overflow => BasisError::OutOfRange,
    }
}

/// [`SyrkError`] as a [`BasisError`].
fn map_syrk(e: SyrkError) -> BasisError {
    match e {
        SyrkError::ShapeMismatch | SyrkError::ZeroN | SyrkError::ZeroO => {
            BasisError::Shape("syrk".to_string())
        }
        SyrkError::InputBound | SyrkError::Overflow => BasisError::OutOfRange,
    }
}

/// [`SolveError`] as a [`BasisError`]; a non-positive diagonal or pivot is a definite-metric
/// refusal.
fn map_solve(e: SolveError) -> BasisError {
    match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            BasisError::NotPositiveDefinite
        }
        SolveError::Empty | SolveError::ShapeMismatch => BasisError::Shape("M".to_string()),
        SolveError::Overflow => BasisError::Overflow,
    }
}

/// `rows × cols` contiguous words copied out of `src` (stride `stride`, origin `(r0, c0)`).
/// `O(rows·cols)` time and space.
fn copy_block(
    src: &[i64],
    stride: usize,
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
) -> Vec<i64> {
    let mut out = vec![0_i64; rows * cols];
    for r in 0..rows {
        out[r * cols..][..cols].copy_from_slice(&src[(r0 + r) * stride + c0..][..cols]);
    }
    out
}

/// The damped input Gram `H + DAMP_Q·mean(diag H)·I`, a fresh row-major `n × n` matrix of
/// words; time and space `O(n²)`.
/// # Errors
/// [`BasisError::Shape`] when `h.len() != n·n`, `n == 0`, or the mean diagonal is not positive;
/// [`BasisError::OutOfRange`] when a value or the damped entry leaves the grid.
pub fn damped_gram(h: &[i64], n: usize) -> Result<Vec<i64>, BasisError> {
    expect(h.len(), n, n, "H")?;
    let mut out = h.to_vec();
    let mut sum: i128 = 0;
    for i in 0..n {
        sum += i128::from(out[i * n + i]);
    }
    let mean = div_rne(
        sum,
        i128::from(u64::try_from(n).map_err(|_| BasisError::Overflow)?),
    )
    .and_then(narrow)
    .ok_or(BasisError::Overflow)?;
    if mean <= 0 {
        return Err(BasisError::Shape(format!(
            "mean(diag H) word {mean} is not positive"
        )));
    }
    let delta = mul(DAMP_Q, mean).ok_or(BasisError::OutOfRange)?;
    for i in 0..n {
        out[i * n + i] = out[i * n + i]
            .checked_add(delta)
            .ok_or(BasisError::OutOfRange)?;
    }
    Ok(out)
}

/// The `n × n` identity basis, row-major.
/// # Errors
/// [`BasisError::Overflow`] when `n·n` leaves `usize`; time and space `O(n²)`.
pub fn identity_basis(n: usize) -> Result<Vec<i64>, BasisError> {
    let mut q = vec![0_i64; square(n, n)?];
    for i in 0..n {
        q[i * n + i] = ONE;
    }
    Ok(q)
}

/// The normalised 128-block Hadamard basis `H₁₂₈/√128`, block diagonal, row-major `n × n`: column
/// `j` is `fwht_words(e_j)` for the block-local unit vector.
/// # Errors
/// [`BasisError::Shape`] when `n` is not a positive multiple of [`GROUP`] or the transform refuses;
/// [`BasisError::Overflow`] when `n·n` leaves `usize`.
/// Costs `O(n·GROUP)` time (one 128-point transform per column) and `O(n²)` space.
pub fn hadamard_basis(n: usize) -> Result<Vec<i64>, BasisError> {
    let blocks = expect_blocks(n)?;
    let mut q = vec![0_i64; square(n, n)?];
    let mut e = vec![0_i64; GROUP];
    // Flag form: the fragment carries no early return inside a nested loop and no `Slice::fill`.
    // The first refusing transform is kept and returned after the loop, in the original order.
    let mut failed: Option<BasisError> = None;
    for b in 0..blocks {
        let off = b * GROUP;
        for j in 0..GROUP {
            for i in 0..GROUP {
                e[i] = 0;
            }
            e[j] = ONE;
            match fwht_words(&mut e) {
                Ok(()) => {
                    for i in 0..GROUP {
                        q[(off + i) * n + off + j] = e[i];
                    }
                }
                Err(err) => {
                    if failed.is_none() {
                        failed = Some(BasisError::Shape(format!("fwht: {err:?}")));
                    }
                }
            }
        }
    }
    match failed {
        Some(err) => Err(err),
        None => Ok(q),
    }
}

/// Every entry of `q` outside a [`GROUP`]×[`GROUP`] diagonal block is zero.
/// # Errors
/// [`BasisError::Shape`] on a non-block-diagonal basis; time and space `O(n²)`.
fn expect_block_diagonal(q: &[i64], n: usize) -> Result<(), BasisError> {
    expect(q.len(), n, n, "Q")?;
    for i in 0..n {
        for j in 0..n {
            if j / GROUP != i / GROUP && q[i * n + j] != 0 {
                return Err(BasisError::Shape(format!(
                    "Q[{i},{j}] lies outside a {GROUP}-block"
                )));
            }
        }
    }
    Ok(())
}

/// The coefficients `B = W·Q`, row-major `O × n`, for a block-diagonal `Q`: per block a `O × 128`
/// by `128 × 128` [`gemm_nn`] on the exact panel-shift form.
/// # Complexity
/// Time `O(O·n·GROUP)` pooled inside [`gemm_nn`]; space `O(O·n + (O + GROUP)·GROUP)`.
/// # Errors
/// [`BasisError::Shape`] on a length or block-structure disagreement; [`BasisError::OutOfRange`] on
/// a value or a rounded entry off the grid, or a bound refusal.
pub fn apply_basis(w: &[i64], o: usize, n: usize, q: &[i64]) -> Result<Vec<i64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    expect_block_diagonal(q, n)?;
    let mut b = vec![0_i64; square(o, n)?];
    for blk in 0..blocks {
        let off = blk * GROUP;
        let (ws, kp) = panel(&copy_block(w, n, 0, off, o, GROUP)).ok_or(BasisError::OutOfRange)?;
        let (qs, kq) =
            panel(&copy_block(q, n, off, off, GROUP, GROUP)).ok_or(BasisError::OutOfRange)?;
        let mut c = gemm_nn(&ws, &qs, o, GROUP, GROUP).map_err(map_gemm)?;
        shift(&mut c, -(kp + kq)).ok_or(BasisError::OutOfRange)?;
        for r in 0..o {
            b[r * n + off..][..GROUP].copy_from_slice(&c[r * GROUP..][..GROUP]);
        }
    }
    Ok(b)
}

/// One block-row of `M = Qᵀ·H·Q`: rows `[b1·GROUP, b1·GROUP+GROUP)` against every block of `H`.
struct MetricRows<'a> {
    h: &'a [i64],
    q: &'a [i64],
    n: usize,
    blocks: usize,
    first: &'a FirstErr<usize, BasisError>,
}

impl MetricRows<'_> {
    /// The pool body: `(Qᵀ·(H·Q))` for the block-row starting at `start`, written into `piece`.
    fn rows(&self, start: usize, piece: &mut [i64]) -> Result<(), BasisError> {
        let (n, g) = (self.n, GROUP);
        let b1 = start / (g * n);
        let mut t = vec![0_i64; g * g];
        for b2 in 0..self.blocks {
            for i in 0..g {
                for cc in 0..g {
                    let mut acc: i128 = 0;
                    for k in 0..g {
                        acc += i128::from(self.h[(b1 * g + i) * n + b2 * g + k])
                            * i128::from(self.q[(b2 * g + k) * n + b2 * g + cc]);
                    }
                    t[i * g + cc] = round64(acc)?;
                }
            }
            for c1 in 0..g {
                for c2 in 0..g {
                    let mut acc: i128 = 0;
                    for i in 0..g {
                        acc += i128::from(self.q[(b1 * g + i) * n + b1 * g + c1])
                            * i128::from(t[i * g + c2]);
                    }
                    piece[c1 * n + b2 * g + c2] = round64(acc)?;
                }
            }
        }
        Ok(())
    }
}

impl ChunkTask<i64> for MetricRows<'_> {
    fn run(&self, c: usize, start: usize, piece: &mut [i64]) {
        if let Err(e) = self.rows(start, piece) {
            self.first.record(c, e);
        }
    }
}

/// The metric `M = Qᵀ·H·Q`, row-major `n × n`, for a block-diagonal `Q`; `H` is the damped Gram.
/// Each block-row is one pool task writing its own rows, so an entry never depends on another
/// entry's rounding; time `O(n²·GROUP)` over `n/GROUP` block rows, space `O(n²)` plus `O(GROUP²)`
/// per task. `H` is unbounded, so this product keeps the exact `i128` accumulation, not the
/// bounded `gemm` primitive.
/// # Errors
/// [`BasisError::Shape`] on a length or block-structure disagreement; [`BasisError::OutOfRange`]
/// when a value or a rounded entry leaves the grid.
pub fn block_metric(q: &[i64], h: &[i64], n: usize) -> Result<Vec<i64>, BasisError> {
    expect(h.len(), n, n, "H")?;
    let blocks = expect_blocks(n)?;
    expect_block_diagonal(q, n)?;
    let mut m = vec![0_i64; square(n, n)?];
    let first = FirstErr::new();
    chunks_mut_global(
        &mut m,
        GROUP * n,
        &MetricRows {
            h,
            q,
            n,
            blocks,
            first: &first,
        },
    );
    pooled(m, first)
}

/// The successive-coding feedback weights `d_j = L_jj²` of the Cholesky factor of `m`:
/// [`equilibrate`] gives `M = D·Â·D` with `D = diag(d)` and [`factor_scaled`] gives `Â = L·Lᵀ`, so
/// the factor of `M` is `D·L` and the pivot is `d_j = (d_j·L_jj)²`; time `O(n³/3)` in the shared
/// blocked factor, space `O(n²)`.
/// # Errors
/// [`BasisError::Shape`] when `m.len() != n·n`; [`BasisError::NotPositiveDefinite`] when `M` is not
/// SPD; [`BasisError::OutOfRange`] when a value or a rounded result leaves the grid.
pub fn cholesky_pivots(m: &[i64], n: usize) -> Result<Vec<i64>, BasisError> {
    expect(m.len(), n, n, "M")?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let (hat, d) = equilibrate(m, n).map_err(map_solve)?;
    let l = factor_scaled(&hat, n).map_err(map_solve)?;
    let mut out = vec![0_i64; n];
    for j in 0..n {
        let dl = mul(d[j], l[j * n + j]).ok_or(BasisError::OutOfRange)?;
        out[j] = mul(dl, dl).ok_or(BasisError::OutOfRange)?;
    }
    Ok(out)
}

/// The limb count [`energy_words`] states to the kernel: [`rowdot_q32`]'s maximum, `8`. Eight
/// 7-bit limbs carry 56 bits of each word and drop a residual below `2^{−57}` of the block scale,
/// so the dot error is below `2^{−56}·Σv²`. One rounded word is `2^{32}` accumulator units, so the
/// computed word equals the exact `round64(Σv²)` whenever `Σv² < 2^{87}` (a whole `2^{32}`-word
/// row at `|v| = 2^{27.5}`); `tests/learn_fold_basis.rs` measures bit equality at `|v| ≤ 2^{20}`
/// over five shapes and the recorded digest fixture reproduces at `|v| ≲ 2^{35}`. A larger word can
/// differ in its low bits — the kernel returns a neighbour, not a refusal; the refusal case is the
/// block exponent leaving the scale-placement window (`|v|` near `i64::MAX`), which both the
/// kernel and the reference report as [`BasisError::OutOfRange`].
///
/// AGENTS §0 K3: the call site states its limb count; this is the largest the kernel supports and
/// the one with the widest exact range.
const ENERGY_LIMBS: u32 = 8;

/// [`LimbError`] as a [`BasisError`].
fn map_rowdot(e: LimbError) -> BasisError {
    match e {
        LimbError::Shape => BasisError::Shape("rowdot".to_string()),
        LimbError::Range | LimbError::NoKernel => BasisError::OutOfRange,
    }
}

/// Row (`axis = 0`) or column (`axis = 1`) energies `Σ B²`, length `O` or `n`, as words, on
/// [`rowdot_q32`]: axis 0 squares the rows in place, axis 1 copies the `O × n` transpose
/// (`bt[j·O + i] = b[i·n + j]`) and squares its `n` rows.
///
/// # Complexity
/// Time `O(O·n·limbs)` limb packing plus `O(O·n)` kernel work (and `O(O·n)` for the transpose
/// copy on axis 1); space `O(O)` or `O(n)` for the result, plus `O(O·n)` for the transpose.
///
/// # Errors
/// [`BasisError::Shape`] when `b.len() != O·n` or the kernel refuses the geometry;
/// [`BasisError::OutOfRange`] when a word's block exponents the scale placement cannot carry, or a
/// rounded entry leaves the grid.
fn energy_words(b: &[i64], o: usize, n: usize, axis: usize) -> Result<Vec<i64>, BasisError> {
    expect(b.len(), o, n, "B")?;
    // Flag form throughout: the fragment carries no `?` inside a loop body.
    let dots = if axis == 0 {
        rowdot_q32(b, b, o, n, ENERGY_LIMBS)
    } else {
        let mut bt = vec![0_i64; square(n, o)?];
        let mut i = 0usize;
        while i < o {
            let mut j = 0usize;
            while j < n {
                bt[j * o + i] = b[i * n + j];
                j += 1;
            }
            i += 1;
        }
        rowdot_q32(&bt, &bt, n, o, ENERGY_LIMBS)
    };
    match dots {
        Ok(e) => Ok(e),
        Err(e) => Err(map_rowdot(e)),
    }
}

/// Row energies `r_i = Σ_j B_ij²`, length `O`; `O(O·n)` time, `O(O)` space.
/// # Errors
/// [`BasisError::Shape`] when `b.len() != O·n`; [`BasisError::OutOfRange`] on a rounded entry.
pub fn row_energies(b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
    energy_words(b, o, n, 0)
}

/// Column energies `c_j = Σ_i B_ij²`, length `n`; `O(O·n)` time, `O(n)` space.
/// # Errors
/// [`BasisError::Shape`] when `b.len() != O·n`; [`BasisError::OutOfRange`] on a rounded entry.
pub fn column_energies(b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
    energy_words(b, o, n, 1)
}

/// `Σ_j ln c_j` over the positive column energies, as a `Q32.32` word; a zero or negative energy is
/// floored at the grid's least positive word `2⁻³²` so the sum stays finite and deterministic.
/// # Complexity
/// Time `O(n)`; space `O(1)`.
pub fn sum_log_column_energies(c: &[i64]) -> i64 {
    let mut acc: i128 = 0;
    for &word in c {
        let word = if word > 0 { word } else { 1 };
        // The `Result` is matched here: `Result::unwrap_or` is not in the model's boundary.
        let l = match fixed_ln(word) {
            Ok(v) => v,
            Err(_) => LN2_Q32 * -32,
        };
        acc += i128::from(l);
    }
    narrow(acc).unwrap_or(i64::MAX)
}

/// The `128 × 128` Gram `W_blockᵀ·W_block` of one column block: [`wsyrk`] on the block panel with
/// the exact `2⁻ᵏ` shift that fits the primitive's bound, and `2²ᵏ` undone on the Gram; time
/// `O(O·GROUP²)` pooled inside [`wsyrk`], space `O(GROUP² + O·GROUP)`.
/// # Errors
/// [`BasisError::OutOfRange`] on a bound refusal or a rounded entry; [`BasisError::Shape`] when `O
/// == 0`.
fn block_syrk(w: &[i64], o: usize, n: usize, blk: usize) -> Result<Vec<i64>, BasisError> {
    let (ws, k) =
        panel(&copy_block(w, n, 0, blk * GROUP, o, GROUP)).ok_or(BasisError::OutOfRange)?;
    let mut g = wsyrk(&ws, None, o, GROUP).map_err(map_syrk)?;
    shift(&mut g, -2 * k).ok_or(BasisError::OutOfRange)?;
    Ok(g)
}

/// Apply `Jᵀ·G·J` to the 128×128 Gram `g` for the Givens rotation `J = [[c, −s], [s, c]]` on
/// columns `(a, b)`. `O(GROUP)`.
fn rotate_syrk(g: &mut [i64], a: usize, b: usize, c: i64, s: i64) -> Option<()> {
    let j = GROUP;
    for r in 0..j {
        if r == a || r == b {
            continue;
        }
        let (gia, gib) = (g[r * j + a], g[r * j + b]);
        let nia = mul(c, gia)?.checked_add(mul(s, gib)?)?;
        let nib = mul(-s, gia)?.checked_add(mul(c, gib)?)?;
        g[r * j + a] = nia;
        g[a * j + r] = nia;
        g[r * j + b] = nib;
        g[b * j + r] = nib;
    }
    let (gaa, gbb, gab) = (g[a * j + a], g[b * j + b], g[a * j + b]);
    let (cc, ss, cs) = (mul(c, c)?, mul(s, s)?, mul(c, s)?);
    g[a * j + a] = mul(cc, gaa)?
        .checked_add(mul(2 * cs, gab)?)?
        .checked_add(mul(ss, gbb)?)?;
    g[b * j + b] = mul(ss, gaa)?
        .checked_sub(mul(2 * cs, gab)?)?
        .checked_add(mul(cc, gbb)?)?;
    let off = mul(-cs, gaa)?
        .checked_add(mul(cc - ss, gab)?)?
        .checked_add(mul(cs, gbb)?)?;
    g[a * j + b] = off;
    g[b * j + a] = off;
    Some(())
}

/// Right-multiply the `GROUP` rows of block `off` of the basis by the same Givens rotation; rows
/// outside the block hold zeros in both columns.
/// # Complexity
/// Time `O(GROUP)`; space `O(1)`.
fn rotate_block(
    q: &mut [i64],
    n: usize,
    off: usize,
    ca: usize,
    cb: usize,
    c: i64,
    s: i64,
) -> Option<()> {
    for r in off..off + GROUP {
        let base = r * n;
        let (qa, qb) = (q[base + ca], q[base + cb]);
        let na = mul(c, qa)?.checked_add(mul(s, qb)?)?;
        let nb = mul(-s, qa)?.checked_add(mul(c, qb)?)?;
        q[base + ca] = na;
        q[base + cb] = nb;
    }
    Some(())
}

/// Which angle a butterfly node uses.
#[derive(Clone, Copy, Debug, PartialEq)]
enum AngleRule {
    /// The Jacobi angle that diagonalises the node's current 2×2 Gram block.
    Jacobi,
    /// A fixed angle; `π/4` reproduces `fwht_words/√128` up to column signs.
    Fixed(i64),
}

/// One pass over the seven-stage butterfly graph of one block: `7·64` nodes in WHT order, each
/// rotating `GROUP` Gram entries and `GROUP` basis rows. The node order is a sequential recurrence
/// (each angle reads the Gram the previous node wrote), so the sweep is serial and `O(1)` in space.
fn sweep(
    q: &mut [i64],
    g: &mut [i64],
    n: usize,
    off: usize,
    rule: AngleRule,
) -> Result<(), BasisError> {
    let mut h = 1usize;
    while h < GROUP {
        let mut i = 0usize;
        while i < GROUP {
            for j in i..i + h {
                let (a, b) = (j, j + h);
                let theta = match rule {
                    AngleRule::Jacobi => {
                        // ½·atan2(2·G_ab, G_aa − G_bb): the angle that zeroes the node's off-diagonal.
                        let y = g[a * GROUP + b]
                            .checked_mul(2)
                            .ok_or(BasisError::OutOfRange)?;
                        let x = g[a * GROUP + a]
                            .checked_sub(g[b * GROUP + b])
                            .ok_or(BasisError::OutOfRange)?;
                        let ang = fixed_atan2(y, x).map_err(|_| BasisError::OutOfRange)?;
                        narrow(i128::from(ang) >> 1).ok_or(BasisError::OutOfRange)?
                    }
                    AngleRule::Fixed(t) => t,
                };
                let turn = fixed_sin_cos(theta).map_err(|_| BasisError::OutOfRange)?;
                let (s, c) = (turn.sin, turn.cos);
                rotate_syrk(g, a, b, c, s).ok_or(BasisError::OutOfRange)?;
                rotate_block(q, n, off, off + a, off + b, c, s).ok_or(BasisError::OutOfRange)?;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    Ok(())
}

/// `Σ log(diag g)` with the [`sum_log_column_energies`] floor; `O(GROUP)` time, `O(1)` space.
fn sum_log_diag(g: &[i64]) -> Option<i64> {
    let mut acc: i128 = 0;
    for i in 0..GROUP {
        let v = g[i * GROUP + i];
        acc += i128::from(if v > 0 {
            fixed_ln(v).ok()?
        } else {
            LN2_Q32 * -32
        });
    }
    narrow(acc)
}

/// Shared body of [`butterfly_basis`] and [`butterfly_basis_fixed`], on words; `Q` starts at the
/// identity, then each block's nodes rotate it against the block Gram.
/// # Complexity
/// See [`butterfly_basis`]; the adaptive rule performs at most three extra sweeps.
fn butterfly_words(w: &[i64], o: usize, n: usize, rule: AngleRule) -> Result<Vec<i64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    let mut q = vec![0_i64; square(n, n)?];
    for i in 0..n {
        q[i * n + i] = ONE;
    }
    for blk in 0..blocks {
        let mut g = block_syrk(w, o, n, blk)?;
        sweep(&mut q, &mut g, n, blk * GROUP, rule)?;
        if rule == AngleRule::Jacobi {
            let mut prev = sum_log_diag(&g).ok_or(BasisError::OutOfRange)?;
            for _ in 0..3 {
                sweep(&mut q, &mut g, n, blk * GROUP, rule)?;
                let cur = sum_log_diag(&g).ok_or(BasisError::OutOfRange)?;
                let drop = prev.checked_sub(cur).ok_or(BasisError::OutOfRange)?;
                let scale = mul(
                    SWEEP_TOL_Q,
                    prev.checked_abs().ok_or(BasisError::OutOfRange)?,
                )
                .ok_or(BasisError::OutOfRange)?;
                if drop <= scale {
                    break;
                }
                prev = cur;
            }
        }
    }
    Ok(q)
}

/// The adapted butterfly basis: each WHT node is a Givens rotation by the Jacobi angle of the
/// current column Gram `G_c = BᵀB`, one mandatory pass plus up to three sweeps while `Σ log c_j`
/// falls by more than `1e-6` relative. `w` and the result are row-major Q32.32 words.
/// # Errors
/// [`BasisError::Shape`] when `w.len() != O·n` or `n` is not a positive multiple of [`GROUP`];
/// [`BasisError::OutOfRange`] on a value or a `wsyrk` bound refusal.
/// # Complexity
/// Time `O(O·n·GROUP + n·GROUP²·sweeps)`; space `O(n² + O·n)`.
pub fn butterfly_basis(w: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
    butterfly_words(w, o, n, AngleRule::Jacobi)
}

/// The butterfly basis with every node fixed at `angle` radians (a Q32.32 word). With
/// `angle = π/4` the result is `fwht_words/√128` up to column signs; the seam exists so the test can
/// assert that identity.
/// # Errors
/// As [`butterfly_basis`].
/// # Complexity
/// Time `O(O·n·GROUP + n·GROUP²)` (one pass, no extra sweeps); space `O(n²)`.
pub fn butterfly_basis_fixed(
    w: &[i64],
    o: usize,
    n: usize,
    angle: i64,
) -> Result<Vec<i64>, BasisError> {
    butterfly_words(w, o, n, AngleRule::Fixed(angle))
}

/// The angle side information of the adapted butterfly: 8 bits per node, `7·64` nodes per block,
/// shared across the block's rows, `28·n` bits for an `n`-column tensor, as a word. `O(1)`.
pub fn butterfly_side_bits(n: usize) -> i64 {
    let nodes = (7 * (GROUP / 2)).saturating_mul(n.div_ceil(GROUP));
    i64::try_from(nodes)
        .ok()
        .and_then(|v| v.checked_shl(35))
        .unwrap_or(i64::MAX)
}

/// The dense per-block eigenbasis of the `128 × 128` diagonal blocks of `WᵀW`, using
/// [`eigh`](crate::learn::fold::fit::eigh::eigh) (column order ascending eigenvalue), on words in
/// and out. Side bits are not counted: this arm is an upper reference.
/// # Errors
/// [`BasisError::Shape`] when `w.len() != O·n` or `n` is not a positive multiple of [`GROUP`];
/// [`BasisError::Eigh`] when a block does not converge; [`BasisError::OutOfRange`] on an overflow.
/// # Complexity
/// Time `O(O·n·GROUP + (n/GROUP)·GROUP³·sweeps)`; space `O(n²)`.
pub fn eigen_basis(w: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    let mut q = vec![0_i64; square(n, n)?];
    for blk in 0..blocks {
        let (_, vecs) = eigh(&block_syrk(w, o, n, blk)?, GROUP).map_err(|err| match err {
            EighError::NotConverged => BasisError::Eigh("block did not converge".to_string()),
            other => BasisError::Shape(format!("eigh: {other:?}")),
        })?;
        let off = blk * GROUP;
        for i in 0..GROUP {
            for j in 0..GROUP {
                q[(off + i) * n + off + j] = vecs[i * GROUP + j];
            }
        }
    }
    Ok(q)
}

/// One coefficient-variance density model and its measured side-information charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Density {
    /// Separable `σ²_ij = r_i·c_j/m`, 16 side bits per row and per column.
    Separable,
    /// Mean of `B²` over the 16×16 tile containing `(i,j)`, 8 side bits per tile.
    Tile,
    /// `σ²_ij = B_ij²`, no density side information: the attainable reference.
    Oracle,
}

/// `σ²_ij = r_i·c_j/m` as a word, zero when the total energy `m` is zero; `O(1)`.
fn separable_word(r: i64, c: i64, m: i64) -> Result<i64, BasisError> {
    if m == 0 {
        return Ok(0);
    }
    div_rne(i128::from(r) * i128::from(c), i128::from(m))
        .and_then(narrow)
        .ok_or(BasisError::OutOfRange)
}

/// One `TILE × TILE` tile mean of `Σ B²`: `narrow(div_rne(sum, count·2³²))`, zero for an empty
/// tile; `O(1)`.
fn tile_mean(sum: i128, count: i128) -> Result<i64, BasisError> {
    if count <= 0 {
        return Ok(0);
    }
    narrow(div_rne(sum, count << 32).ok_or(BasisError::OutOfRange)?).ok_or(BasisError::OutOfRange)
}

impl Density {
    /// The model's own side-information charge in bits, as a `Q32.32` word, counting rows/columns
    /// or tiles for the whole `O × n` coefficient matrix. `O(1)`.
    pub fn side_bits(self, o: usize, n: usize) -> i64 {
        let bits: i128 = match self {
            Density::Separable => 16 * (o as i128 + n as i128),
            Density::Tile => 8 * o.div_ceil(TILE) as i128 * n.div_ceil(TILE) as i128,
            Density::Oracle => 0,
        };
        narrow(bits << 32).unwrap_or(i64::MAX)
    }

    /// The `O × n` coefficient-variance matrix of `b = W·Q`, as words: serial passes for the
    /// separable or tile statistics and one rounded map.
    /// # Complexity
    /// Time and space `O(O·n)`.
    /// # Errors
    /// [`BasisError::Shape`] when `b.len() != O·n`; [`BasisError::OutOfRange`] on a rounded entry.
    fn sigma_words(self, b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
        let len = expect(b.len(), o, n, "B")?;
        let mut out = vec![0_i64; len];
        // One flat index pass per arm: the fragment carries no `?` inside a nested loop and no
        // `Iterator::zip`, so the separable and tile maps call a helper at the loop's own depth.
        match self {
            Density::Oracle => {
                for k in 0..len {
                    out[k] = mul(b[k], b[k]).ok_or(BasisError::OutOfRange)?;
                }
            }
            Density::Separable => {
                let (r, c) = (energy_words(b, o, n, 0)?, energy_words(b, o, n, 1)?);
                let mut total: i128 = 0;
                for k in 0..r.len() {
                    total += i128::from(r[k]);
                }
                let m = narrow(total).ok_or(BasisError::OutOfRange)?;
                for k in 0..len {
                    out[k] = separable_word(r[k / n], c[k % n], m)?;
                }
            }
            Density::Tile => {
                let ntj = n.div_ceil(TILE);
                let mut sums = vec![0_i128; o.div_ceil(TILE) * ntj];
                let mut counts = vec![0_i128; sums.len()];
                for i in 0..o {
                    for j in 0..n {
                        let k = (i / TILE) * ntj + j / TILE;
                        let v = i128::from(b[i * n + j]);
                        sums[k] += v * v;
                        counts[k] += 1;
                    }
                }
                let mut means = vec![0_i64; sums.len()];
                for k in 0..sums.len() {
                    means[k] = tile_mean(sums[k], counts[k])?;
                }
                for i in 0..o {
                    for j in 0..n {
                        out[i * n + j] = means[(i / TILE) * ntj + j / TILE];
                    }
                }
            }
        }
        Ok(out)
    }

    /// The `O × n` coefficient-variance matrix of `b = W·Q`.
    /// # Errors
    /// As [`Density::sigma_words`].
    /// # Complexity
    /// As [`Density::sigma_words`].
    pub fn sigma(self, b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, BasisError> {
        self.sigma_words(b, o, n)
    }
}

/// A reverse water-filling solution at the target rate. Every field is a `Q32.32` word: `theta` is
/// in the `λ` units, `distortion` in the `D` units, `rate` in bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFill {
    /// The water level `θ`.
    pub theta: i64,
    /// `D(θ) = Σ min(λ_ij, θ)`.
    pub distortion: i64,
    /// `R(θ) = Σ max(0, ½·log₂(λ_ij/θ))` in bits.
    pub rate: i64,
}

/// `D(θ) = Σ min(λ, θ)` over words; `θ ≤ 0` gives zero. `O(N)` time, `O(1)` space.
fn distortion_words(lambda: &[i64], theta: i64) -> Result<i64, BasisError> {
    if theta <= 0 {
        return Ok(0);
    }
    let mut acc: i128 = 0;
    for &l in lambda {
        acc += i128::from(if l < theta { l } else { theta });
    }
    narrow(acc).ok_or(BasisError::OutOfRange)
}

/// `R(θ) = Σ max(0, ½·log₂(λ/θ))` over words; `θ ≤ 0` gives an infinite rate when any `λ > 0`.
/// Costs `O(N·S)` time with `S` the fixed `log₂` series, and `O(1)` space.
fn rate_words(lambda: &[i64], theta: i64) -> Result<i64, BasisError> {
    if theta <= 0 {
        // Flag form: `Iterator::any` is not in the model and a `?`/`continue` inside the loop is
        // outside the fragment.
        let mut any = false;
        for &l in lambda {
            if l > 0 {
                any = true;
            }
        }
        return Ok(if any { i64::MAX } else { 0 });
    }
    let t = log2(theta).ok_or(BasisError::OutOfRange)?;
    let mut acc: i128 = 0;
    let mut failed = false;
    for &l in lambda {
        if !failed && l > theta {
            match log2(l).and_then(|li| li.checked_sub(t)) {
                Some(d) => acc += i128::from(d),
                None => failed = true,
            }
        }
    }
    if failed {
        return Err(BasisError::OutOfRange);
    }
    narrow(acc >> 1).ok_or(BasisError::OutOfRange)
}

/// The reverse water-filling solution with `R(θ) + side_bits = budget_bits`, on words. The
/// bisection runs in the log domain (`t = log₂ θ`), where `R` is piecewise linear with slope
/// `−N/2`; each `log₂ λ_i` is computed once and the bracket walks down until `R > target`.
/// # Complexity
/// Time `O(N·S + BISECTION_ITERS·N)`; space `O(N)` for the logs.
fn water_fill_words(lambda: &[i64], budget: i64, side: i64) -> Result<WaterFill, BasisError> {
    let target = budget.checked_sub(side).ok_or(BasisError::OutOfRange)?;
    let ncoef = i64::try_from(lambda.len()).map_err(|_| BasisError::Overflow)?;
    let mut max = 0_i64;
    let mut min_pos = i64::MAX;
    let mut logs = vec![0_i64; lambda.len()];
    // Flag form: the fragment carries no `?` inside a loop body, and `Ord::max` is written out as
    // the comparison it is. The first refused log stops the pass, as the early return did.
    let mut err: Option<BasisError> = None;
    for k in 0..lambda.len() {
        let l = lambda[k];
        if l > max {
            max = l;
        }
        if l > 0 && l < min_pos {
            min_pos = l;
        }
        if err.is_none() {
            match water_log(l) {
                Ok(t) => logs[k] = t,
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    if lambda.is_empty() || max <= 0 {
        return Ok(WaterFill {
            theta: 0,
            distortion: 0,
            rate: 0,
        });
    }
    let mut sum_lambda: i128 = 0;
    for k in 0..lambda.len() {
        sum_lambda += i128::from(lambda[k]);
    }
    if target <= 0 {
        return Ok(WaterFill {
            theta: max,
            distortion: narrow(sum_lambda).unwrap_or(i64::MAX),
            rate: 0,
        });
    }
    let mut t_hi = log2(max)
        .and_then(|v| v.checked_add(ONE))
        .ok_or(BasisError::OutOfRange)?;
    let mut t_lo = log2(min_pos)
        .and_then(|v| v.checked_sub(ONE))
        .ok_or(BasisError::OutOfRange)?;
    // Flag form for both bisections: a `break`/`?` inside the body is outside the fragment, so each
    // loop carries its exit and the first refusal out and returns them once after the loop.
    let mut gap = ONE;
    let mut err: Option<BasisError> = None;
    let mut walked = false;
    for _ in 0..BISECTION_ITERS {
        if !walked && err.is_none() {
            match rate_at_logs(&logs, t_lo) {
                Ok(r) => {
                    if r >= target {
                        walked = true;
                    } else {
                        match t_lo.checked_sub(gap) {
                            Some(lo) => t_lo = lo,
                            None => err = Some(BasisError::OutOfRange),
                        }
                        match gap.checked_mul(2) {
                            Some(g) => gap = g,
                            None => err = Some(BasisError::OutOfRange),
                        }
                    }
                }
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    let tol = RATE_TOL_Q.checked_mul(ncoef).ok_or(BasisError::Overflow)?;
    let mut t = t_hi;
    let mut err = None;
    let mut done = false;
    for _ in 0..BISECTION_ITERS {
        if !done && err.is_none() {
            // `i128::midpoint` truncates toward zero and is not in the model; the floor shift plus
            // the odd negative correction is that same average, exact for either sign.
            let sum = i128::from(t_lo) + i128::from(t_hi);
            let mid = if sum < 0 && (sum & 1) != 0 {
                (sum >> 1) + 1
            } else {
                sum >> 1
            };
            match narrow(mid) {
                Some(mid) => {
                    t = mid;
                    match rate_at_logs(&logs, t) {
                        Ok(r) => match converged(r, target, tol) {
                            Ok(true) => done = true,
                            Ok(false) => {
                                if r > target {
                                    t_lo = t;
                                } else {
                                    t_hi = t;
                                }
                            }
                            Err(e) => err = Some(e),
                        },
                        Err(e) => err = Some(e),
                    }
                }
                None => err = Some(BasisError::OutOfRange),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    let theta = exp2(t).ok_or(BasisError::OutOfRange)?;
    Ok(WaterFill {
        theta,
        distortion: distortion_words(lambda, theta)?,
        rate: rate_words(lambda, theta)?,
    })
}

/// `log₂ λ` in the water-filling log domain; a non-positive `λ` carries `i64::MIN`, contributing to
/// neither bracket. `O(1)`.
fn water_log(l: i64) -> Result<i64, BasisError> {
    if l > 0 {
        log2(l).ok_or(BasisError::OutOfRange)
    } else {
        Ok(i64::MIN)
    }
}

/// `R(t) = ½·Σ max(0, log₂ λ_i − t)` from the precomputed logs: ½ of the exact sum of the positive
/// differences, rounded once. `O(N)`; space `O(1)`.
fn rate_at_logs(logs: &[i64], t: i64) -> Result<i64, BasisError> {
    let mut acc: i128 = 0;
    let mut failed = false;
    for k in 0..logs.len() {
        let l = logs[k];
        if !failed && l != i64::MIN && l > t {
            match l.checked_sub(t) {
                Some(d) => acc += i128::from(d),
                None => failed = true,
            }
        }
    }
    if failed {
        return Err(BasisError::OutOfRange);
    }
    narrow(acc >> 1).ok_or(BasisError::OutOfRange)
}

/// `|r − target| ≤ tol`. A difference of `i64::MIN` is refused rather than reaching `abs`'s panic.
fn converged(r: i64, target: i64, tol: i64) -> Result<bool, BasisError> {
    let d = r.checked_sub(target).ok_or(BasisError::OutOfRange)?;
    let mag = if d < 0 {
        // `0 - d` rather than `checked_neg`, which the model does not carry.
        0i64.checked_sub(d).ok_or(BasisError::OutOfRange)?
    } else {
        d
    };
    Ok(mag <= tol)
}

/// The reverse water-filling solution with `R(θ) + side_bits = budget_bits`, all in Q32.32 words; a
/// refused scalar returns the zero [`WaterFill`], as the entry point is infallible.
/// # Complexity
/// As [`water_fill_words`].
pub fn solve_water_fill(lambda: &[i64], budget_bits: i64, side_bits: i64) -> WaterFill {
    let fallback = WaterFill {
        theta: 0,
        distortion: 0,
        rate: 0,
    };
    match water_fill_words(lambda, budget_bits, side_bits) {
        Ok(w) => w,
        Err(_) => fallback,
    }
}

/// `D(θ) = Σ min(λ, θ)`; `θ ≤ 0` gives zero. `O(N)` time, `O(1)` space.
/// # Errors
/// [`BasisError::OutOfRange`] when the sum leaves `i64`.
pub fn distortion_at(lambda: &[i64], theta: i64) -> Result<i64, BasisError> {
    distortion_words(lambda, theta)
}

/// `R(θ) = Σ max(0, ½·log₂(λ/θ))`; `θ ≤ 0` gives an infinite rate when any `λ > 0`.
/// # Complexity
/// Time `O(N·S)`; space `O(1)`.
/// # Errors
/// [`BasisError::OutOfRange`] when the sum leaves `i64`.
pub fn rate_at(lambda: &[i64], theta: i64) -> Result<i64, BasisError> {
    rate_words(lambda, theta)
}

/// The fixed-rate distortion `D_u = Σ λ·2^{−2·(budget/N − side/N)}` over words. `O(N)` time, `O(1)`
/// space.
fn uniform_words(lambda: &[i64], budget: i64, side: i64) -> Result<i64, BasisError> {
    if lambda.is_empty() {
        return Ok(0);
    }
    let ncoef = i64::try_from(lambda.len()).map_err(|_| BasisError::Overflow)?;
    let bits = narrow(
        div_rne(
            i128::from(budget.checked_sub(side).ok_or(BasisError::OutOfRange)?),
            i128::from(ncoef),
        )
        .ok_or(BasisError::OutOfRange)?,
    )
    .ok_or(BasisError::OutOfRange)?;
    let factor =
        exp2(bits.checked_mul(-2).ok_or(BasisError::OutOfRange)?).ok_or(BasisError::OutOfRange)?;
    let mut acc: i128 = 0;
    for &l in lambda {
        acc += i128::from(l) * i128::from(factor);
    }
    round64(acc)
}

/// The fixed-rate distortion `D_u = Σ λ·2^{−2·(budget/N − side/N)}`, a word. `O(N)` time, `O(1)`
/// space.
/// # Errors
/// [`BasisError::OutOfRange`] when an intermediate leaves the grid.
pub fn uniform_distortion(
    lambda: &[i64],
    budget_bits: i64,
    side_bits: i64,
) -> Result<i64, BasisError> {
    uniform_words(lambda, budget_bits, side_bits)
}

/// The basis-independent part of one arm: coefficients `B = W·Q` and feedback weights `d`, words.
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients {
    /// `O`.
    pub o: usize,
    /// `n`.
    pub n: usize,
    /// `B = W·Q`, row-major `O × n`.
    pub b: Vec<i64>,
    /// The Cholesky pivots `d_j` of `QᵀHQ`, length `n`.
    pub d: Vec<i64>,
}

/// One arm's distortion at the registered rate: uniform (`D_u`) and reverse water-filling (`D_wf`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arm {
    /// `D_u = Σ λ·2^{−2·(1.18 − side/(O·n))}`, the fixed-rate coder, a word.
    pub uniform: i64,
    /// `D_wf`, the reverse water-filling distortion, a word.
    pub water_filled: i64,
    /// The achieved `R + side`, in bits, for the water-filling solution.
    pub rate: i64,
}

impl Coefficients {
    /// Form `B = W·Q`, `M = Qᵀ·H·Q` and the pivots `d` of `M` for one basis; every operand and
    /// result is a Q32.32 word.
    /// # Errors
    /// The [`apply_basis`], [`block_metric`] and [`cholesky_pivots`] refusals.
    /// # Complexity
    /// Dominated by the metric `O(n²·GROUP)` and the Cholesky `O(n³/3)`; space `O(O·n + n²)`.
    pub fn from_basis(
        w: &[i64],
        o: usize,
        n: usize,
        h: &[i64],
        q: &[i64],
    ) -> Result<Coefficients, BasisError> {
        let b = apply_basis(w, o, n, q)?;
        let d = cholesky_pivots(&block_metric(q, h, n)?, n)?;
        Ok(Coefficients { o, n, b, d })
    }

    /// Evaluate one density model at `budget_bits` (the pipeline's `1.18·O·n`) with the density's
    /// side information plus `basis_side_bits` charged against it; the two scalars are Q32.32
    /// words.
    /// # Errors
    /// [`BasisError::Shape`] when the stored geometry is inconsistent; [`BasisError::OutOfRange`]
    /// when a value leaves the grid.
    /// # Complexity
    /// Time `O(O·n·S + O·n·BISECTION_ITERS)`; space `O(O·n)` for `σ²` and `λ`.
    pub fn evaluate(
        &self,
        density: Density,
        basis_side_bits: i64,
        budget_bits: i64,
    ) -> Result<Arm, BasisError> {
        let sigma = density.sigma_words(&self.b, self.o, self.n)?;
        let mut lambda = vec![0_i64; square(self.o, self.n)?];
        for i in 0..self.o {
            for j in 0..self.n {
                lambda[i * self.n + j] =
                    mul(self.d[j], sigma[i * self.n + j]).ok_or(BasisError::OutOfRange)?;
            }
        }
        let side = density
            .side_bits(self.o, self.n)
            .checked_add(basis_side_bits)
            .ok_or(BasisError::OutOfRange)?;
        let wf = water_fill_words(&lambda, budget_bits, side)?;
        Ok(Arm {
            uniform: uniform_words(&lambda, budget_bits, side)?,
            water_filled: wf.distortion,
            rate: wf.rate,
        })
    }
}
