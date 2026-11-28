//! The NanoQuant binary low-rank factorisation: Samsung Labs, arXiv 2602.06694, ported from
//! `SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/admm_nq.py` (`factorize_admm_nanoquant`)
//! and `src/nanoquant/utils/utils.py:128` (`calculate_ranks`).
//!
//! `Ŵ = diag(s_post)·U·V·diag(s_pre)`, `U ∈ {±1}^{out×r}`, `V ∈ {±1}^{r×in}`; the stored bits are
//! `r·(out + in) + 16·(out + in)` (two binary16 scale vectors). The factors come from the ADMM of
//! `admm_nq.py`: a ridge least-squares `X`-update, a sign-value-independent rank-one `Z`-update
//! (`sign(M) ⊙ u vᵀ`, `u σ vᵀ` the top singular triplet of `|M|` by power iteration) and the scaled
//! dual `U`-update, under the penalty schedule [`Schedule`] (the reference CLI default is linear,
//! `ρ = t/T`, `main.py:81`; `cubic`, `ρ = (t/T)³`, is the function signature's default).
//!
//! Every matrix product runs on the kernel through [`crate::engine::kern::limb`] (AGENTS.md §0 K2):
//! the SyrkAcc and the right-hand sides by prepared products, the solves by `gemm_q32`, the power
//! iteration by prepared one-column products and the norms by `rowdot_q32`. The `r × r` ridge
//! system is solved as `F = H⁻¹R` with `H⁻¹` from the block Schur-complement recursion
//! [`spd_inverse`], so the `O(r²·k)` substitution is one kernel product instead of a scalar
//! triangular solve.
//!
//! Deviations from the reference, each a stated input, not a silent change: the Gaussian draws
//! come from [`Rng`] (splitmix64 and Box–Muller), not `torch.randn`; the `B` power iteration runs
//! on the transposed frame with its start vector on the same (input) side as the reference; the
//! norms take [`FLOOR_REL`] relative to their mean, where the reference's `eps` clamp is float-only
//! and a dead Gram coordinate would otherwise scale a column past the limb kernel's exponent range.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "single letters are the reference's symbols (`o`, `n`, `r`, `k`, `w`); the casts are \
              dimensions and 53-bit draws far below 2^52; every loop is one row-major index map"
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::engine::chart::pool::{width_global, Pool};
use crate::engine::kern::limb::{gemm_q32, prepare, rowdot_q32, syrk_q32, LimbError};
use crate::learn::solve::chol::SolveError;
use crate::learn::solve::inv::inv_factor_upper;
use crate::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// Limb count of every kernel call here: 3, the fp32-reference precision (AGENTS.md §0 K3).
pub(crate) const LIMBS: u32 = 3;

/// The reference's `eps` clamp on norms (`admm_nq.py`, `eps=1e-12`).
const EPS: f64 = 1e-12;

/// The preconditioner floor as a fraction of the mean `sqrt(diag)`: a coordinate dead in the
/// calibration Gram keeps a column the limb kernel can carry. Deviation from the reference, whose
/// `eps` clamp is float-only; a dead coordinate carries no weight in `tr(EHEᵀ)`, and the diagonal
/// rescaling cancels in the export, so the floor moves the fit's conditioning, not its objective.
/// `1e-6` is a 1e5 margin over the smallest floor that factorizes `Qwen3-0.6B-Base` layer 0 `gate`
/// (between `5e-12` and `1e-11`); every value to `1.0` factorizes it, `relH` 0.0606–0.0615.
/// Board unmeasured against held-out KL.
const FLOOR_REL: f64 = 1e-6;

/// A refusal of [`factorize`] or [`NqFactor::weight`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NqError {
    /// A slice length disagrees with the declared shape, or a dimension or the rank is zero.
    Shape,
    /// A value left the Q32.32 grid or is not finite.
    Range,
    /// The limb kernel refused a product (shape or scale placement).
    Kernel,
    /// The ridge system's inverse factor was refused.
    Solve(SolveError),
}

impl From<LimbError> for NqError {
    fn from(_: LimbError) -> Self {
        Self::Kernel
    }
}

impl From<SolveError> for NqError {
    fn from(e: SolveError) -> Self {
        Self::Solve(e)
    }
}

/// The two-scale rank of an `a × b` matrix at `bits_ppm` millionths of a bit per weight
/// (`calculate_ranks`, `num_scales = 2`): `r = ⌊a·b·bits/(a + b)⌋ − 16`, floored to a multiple of
/// 32, and `min_rank` when that is zero or below.
#[must_use]
pub fn rank_for_bits(a: usize, b: usize, bits_ppm: u64, min_rank: usize) -> usize {
    let ab = (a as u128) * (b as u128) * u128::from(bits_ppm);
    let per = ab / (1_000_000u128 * (a as u128 + b as u128));
    let r = (per as usize).saturating_sub(16) / 32 * 32;
    r.max(min_rank)
}

/// Stored bits of a two-scale rank-`r` factor of an `a × b` matrix: `r·(a + b) + 16·(a + b)`.
#[must_use]
pub fn stored_bits(a: usize, b: usize, r: usize) -> u64 {
    ((r + 16) * (a + b)) as u64
}

/// ADMM settings; [`NqConfig::reference`] is the README's `--admm_outer_iters 400` run.
#[derive(Debug, Clone, Copy)]
pub struct NqConfig {
    /// The rank `r`.
    pub rank: usize,
    /// Outer ADMM iterations `T` (reference 400).
    pub outer_iters: usize,
    /// Power-iteration steps of each rank-one `Z`-update (reference 5).
    pub inner_iters: usize,
    /// The ridge term added to the normalised system's diagonal (reference `3e-2`).
    pub reg: f64,
    /// Seed of the Gaussian initialisation and the power-iteration start vectors.
    pub seed: u64,
    /// The penalty schedule `ρ(t/T)`.
    pub schedule: Schedule,
}

/// The ADMM penalty schedule `ρ(x)`, `x = t/T` (`admm_nq.py:255-277`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// `ρ = x`: the reference CLI and `NanoQuantConfig` default.
    Linear,
    /// `ρ = min(1, x)³`.
    Cubic,
}

impl Schedule {
    fn rho(self, x: f64) -> f64 {
        match self {
            Self::Linear => x,
            Self::Cubic => x.min(1.0).powi(3),
        }
    }
}

impl NqConfig {
    /// The reference settings at rank `rank`.
    #[must_use]
    pub fn reference(rank: usize) -> Self {
        Self {
            rank,
            outer_iters: 400,
            inner_iters: 5,
            reg: 3e-2,
            seed: 0,
            schedule: Schedule::Linear,
        }
    }
}

/// The deployed factor: `Ŵ[i][j] = s_post[i]·Σ_k U[i][k]·V[k][j]·s_pre[j]`.
#[derive(Debug, Clone)]
pub struct NqFactor {
    /// Output rows `out`.
    pub out: usize,
    /// Input columns `in`.
    pub inn: usize,
    /// Rank `r`.
    pub rank: usize,
    /// `U`, row-major `out × r`, entries `±1`.
    pub u: Vec<i8>,
    /// `V`, row-major `r × in`, entries `±1`.
    pub v: Vec<i8>,
    /// `s_pre`, length `in`.
    pub s_pre: Vec<f64>,
    /// `s_post`, length `out`.
    pub s_post: Vec<f64>,
    /// The ADMM's own normalised loss `‖W_n − (A_z/m)·B_z‖²/‖W_n‖²` at exit (`admm_nq.py`'s
    /// printed `Loss`, `m` the row norms of `B_z`).
    pub admm_loss: f64,
    /// `U`'s latent, row-major `out × r`: `(A_ls + A_u)/norm_o·bf` (`admm_nq.py:215-232`);
    /// its sign is `u` wherever the rank-one magnitude is nonzero. The reference's `tune_fact`
    /// trains it through the sign.
    pub u_latent: Vec<f64>,
    /// `V`'s latent, row-major `r × in`: `(B_ls + B_u)/norm_i/bf`, the same way.
    pub v_latent: Vec<f64>,
}

/// Deterministic Gaussian source: splitmix64 words through Box–Muller.
struct Rng(u64);

/// The `SplitMix64` increment `γ`: [`Rng::word`] adds it to the state once per word.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

impl Rng {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn normal(&mut self) -> f64 {
        let u1 = ((self.word() >> 11) as f64 + 1.0) / 9_007_199_254_740_993.0;
        let u2 = (self.word() >> 11) as f64 / 9_007_199_254_740_992.0;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// The next `len` draws of [`Rng::normal`], `out[c]` holding draw `slot(c)`, with the state
    /// advanced past all of them. The state after `w` words is `s₀ + w·γ`, so draw `i` (words
    /// `2i`, `2i + 1`) is computed alone from `s₀ + 2i·γ` on the pool; each entry is the sequential
    /// one in every bit. `slot` is a bijection on `0..len`.
    ///
    /// # Complexity
    ///
    /// `O(len)` normals (one `ln`, `sqrt`, `cos` each) over the pool's width.
    fn normals(&mut self, len: usize, slot: impl Fn(usize) -> usize + Sync) -> Vec<f64> {
        let s0 = self.0;
        let at = |i: usize| Rng(s0.wrapping_add((2 * i as u64).wrapping_mul(GOLDEN))).normal();
        let mut out = vec![0.0f64; len];
        par_fill(&mut out, &|c, o| *o = at(slot(c)));
        self.0 = s0.wrapping_add((2 * len as u64).wrapping_mul(GOLDEN));
        out
    }
}

/// Below this many entries an elementwise pass runs on the caller: a pool dispatch costs more than
/// the pass.
const PAR_MIN: usize = 1 << 14;

/// `f(c, &mut out[c])` for every `c`, one contiguous chunk per pool worker (under [`PAR_MIN`]
/// entries on the caller). Each entry is computed alone, so the result is the sequential one.
/// `O(len)` calls of `f`.
fn par_fill<T: Send>(out: &mut [T], f: &(impl Fn(usize, &mut T) + Sync)) {
    if out.len() < PAR_MIN {
        for (c, o) in out.iter_mut().enumerate() {
            f(c, o);
        }
        return;
    }
    let chunk = out.len().div_ceil(width_global());
    Pool::global().chunks_mut(out, chunk, &|_, start, piece: &mut [T]| {
        for (k, o) in piece.iter_mut().enumerate() {
            f(start + k, o);
        }
    });
}

/// `to_grid(f(c))` for `c < len`, through [`par_fill`]; [`NqError::Range`] if any entry leaves the
/// grid. `O(len)` calls of `f`.
fn grid_map(len: usize, f: &(impl Fn(usize) -> f64 + Sync)) -> Result<Vec<i64>, NqError> {
    let bad = AtomicBool::new(false);
    let mut out = vec![0i64; len];
    par_fill(&mut out, &|c, o| match to_grid(f(c), GRID_BOUND) {
        Some(w) => *o = w,
        None => bad.store(true, Ordering::Relaxed),
    });
    if bad.into_inner() {
        Err(NqError::Range)
    } else {
        Ok(out)
    }
}

/// Every entry of `x` on the Q32.32 grid. `O(len)`.
fn grid(x: &[f64]) -> Result<Vec<i64>, NqError> {
    grid_map(x.len(), &|c| x[c])
}

/// Row-major `rows × cols` transposed to `cols × rows`. `O(rows·cols)` moves.
fn transpose<T: Copy + Default + Send + Sync>(m: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::default(); m.len()];
    par_fill(&mut t, &|c, o| *o = m[(c % rows) * cols + c / rows]);
    t
}

/// `‖m‖_F²` of a row-major `rows × cols` matrix: one `rowdot_q32` of `m` with itself, summed.
/// `O(rows·cols·limbs²)` kernel byte products.
fn fro2(m: &[f64], rows: usize, cols: usize) -> Result<f64, NqError> {
    let q = grid(m)?;
    let d = rowdot_q32(&q, &q, rows, cols, LIMBS)?;
    Ok(d.iter().map(|&w| from_grid(w)).sum())
}

/// The reference's `rank1_approx`: `sign(M) ⊙ (σu)vᵀ` with `(u, σ, v)` the top singular triplet
/// of `|M|` after `iters` power steps from the Gaussian `start`. `start_rows` puts the start vector
/// (`rows` long) on the row side (the reference's `B`, run here on `Bᵀ`), else (`cols` long) on the
/// column side.
///
/// `O(iters·rows·cols·limbs²)` kernel byte products through two prepared planes (`|M|` and
/// `|M|ᵀ`, `O(rows·cols·limbs)` bytes each), plus `O(rows·cols)` for the sign product.
pub(crate) fn rank1(
    m: &[f64],
    rows: usize,
    cols: usize,
    iters: usize,
    start_rows: bool,
    start: Vec<f64>,
) -> Result<Vec<f64>, NqError> {
    let abs: Vec<f64> = m.iter().map(|x| x.abs()).collect();
    let pa = prepare(&grid(&abs)?, rows, cols, LIMBS)?;
    let pt = prepare(&grid(&transpose(&abs, rows, cols))?, cols, rows, LIMBS)?;
    // `|M|·x` (x on the column side) and `|M|ᵀ·y` (y on the row side), each one prepared product.
    let mul = |x: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(pa
            .gemm(&grid(x)?, 1)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    let mul_t = |y: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(pt
            .gemm(&grid(y)?, 1)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    let unit = |x: Vec<f64>| -> Result<Vec<f64>, NqError> {
        let n = fro2(&x, 1, x.len())?.sqrt().max(EPS);
        Ok(x.into_iter().map(|e| e / n).collect())
    };
    let (u, v) = if start_rows {
        // Reference frame `Mᵀ` (`cols × rows`): its column-side start is our row side.
        let mut a = unit(start)?;
        for _ in 0..iters {
            let b = unit(mul_t(&a)?)?;
            a = unit(mul(&b)?)?;
        }
        let b_un = mul_t(&a)?;
        // The reference's `u` (σ folded in) is our column vector, its unit `v` our row vector.
        (a, b_un)
    } else {
        let mut v = unit(start)?;
        for _ in 0..iters {
            let u = unit(mul(&v)?)?;
            v = unit(mul_t(&u)?)?;
        }
        (mul(&v)?, v)
    };
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let s = if m[i * cols + j] < 0.0 { -1.0 } else { 1.0 };
            out[i * cols + j] = s * u[i] * v[j];
        }
    }
    Ok(out)
}

/// Order at or below which [`spd_inverse`] stops recursing and takes `UᵀU` from the exact inverse
/// factor. Measured at order 768 (`spd_inverse_base_wall`): 12.2 ms at 32 against 17.0 ms at 192;
/// residual at order 300 (`spd_inverse_inverts`) 7.7e-6 against 3.4e-5 without recursion.
const INV_BASE: usize = 32;

/// `H⁻¹` of an SPD `h` (row-major `n × n`, Q32.32), symmetric, by the block Schur-complement
/// recursion on `H = [A B; Bᵀ C]` (`A` of order `p = ⌊n/2⌋`, `C` of order `q = n − p`):
///
/// ```text
/// X = A⁻¹B,  S = C − BᵀX,  Y = X·S⁻¹,  H⁻¹ = [A⁻¹ + Y·Xᵀ, −Y; −Yᵀ, S⁻¹],
/// ```
///
/// the four products on the kernel at [`LIMBS`] (the fp32-reference precision of the products
/// that consume `H⁻¹`), `A⁻¹` and `S⁻¹` by recursion; at order `≤ INV_BASE` the base is `UᵀU` with
/// `U` from [`inv_factor_upper`]. `S` is SPD whenever `H` is (Schur complement of an SPD matrix).
///
/// # Errors
///
/// [`NqError::Solve`] as [`inv_factor_upper`] refuses a base block (a computed `S` that is not SPD
/// lands here); [`NqError::Kernel`] as a product refuses; [`NqError::Range`] if a sum leaves `i64`.
///
/// # Complexity
///
/// Four products of half order per level, `p·p·q + q·q·p + p·q·q + p·p·q = n³/2` multiply-adds at
/// `p = q = n/2`, so `T(n) = n³/2 + 2·T(n/2)` closes at `2n³/3` kernel multiply-adds, plus `n/INV_BASE`
/// bases of `INV_BASE³/2` checked `i128` multiply-adds and an `INV_BASE³/2` kernel Gram each.
/// `O(n²)` words live per level; the block reads, `S` and the assembly write `p² + q·p + q² + n²`
/// words per level through [`par_fill`], each entry alone, so the pooled result is the serial one.
pub(crate) fn spd_inverse(h: &[i64], n: usize) -> Result<Vec<i64>, NqError> {
    spd_inverse_at(h, n, INV_BASE)
}

/// [`spd_inverse`] with the recursion base order `base` as an argument.
fn spd_inverse_at(h: &[i64], n: usize, base: usize) -> Result<Vec<i64>, NqError> {
    if n <= base {
        let u = inv_factor_upper(h, n)?;
        return Ok(syrk_q32(&transpose(&u, n, n), n, n, LIMBS)?);
    }
    let (p, q) = (n / 2, n - n / 2);
    let mut a = vec![0i64; p * p];
    par_fill(&mut a, &|c, o| *o = h[(c / p) * n + c % p]);
    let ai = spd_inverse_at(&a, p, base)?;
    // `Bᵀ` (`q × p`) read out of `h` directly: entry `(i, j)` is `h[j, p + i]`.
    let mut bt = vec![0i64; q * p];
    par_fill(&mut bt, &|c, o| *o = h[(c % p) * n + p + c / p]);
    let x = gemm_q32(&ai, p, &bt, q, p, LIMBS)?;
    let btx = gemm_q32(&bt, q, &transpose(&x, p, q), q, p, LIMBS)?;
    let bad = AtomicBool::new(false);
    let mut s = vec![0i64; q * q];
    par_fill(&mut s, &|c, o| {
        let (i, j) = (c / q, c % q);
        // `(BᵀX)ᵢⱼ` and `(BᵀX)ⱼᵢ` agree in exact arithmetic; the mean keeps `S` symmetric.
        let m = (btx[c] >> 1) + (btx[j * q + i] >> 1);
        match h[(p + i) * n + p + j].checked_sub(m) {
            Some(v) => *o = v,
            None => bad.store(true, Ordering::Relaxed),
        }
    });
    if bad.load(Ordering::Relaxed) {
        return Err(NqError::Range);
    }
    let si = spd_inverse_at(&s, q, base)?;
    let y = gemm_q32(&x, p, &si, q, q, LIMBS)?;
    let yx = gemm_q32(&y, p, &x, p, q, LIMBS)?;
    let mut out = vec![0i64; n * n];
    par_fill(&mut out, &|c, o| {
        let (i, j) = (c / n, c % n);
        let v = match (i < p, j < p) {
            (true, true) => {
                let m = (yx[i * p + j] >> 1) + (yx[j * p + i] >> 1);
                ai[i * p + j].checked_add(m)
            }
            (true, false) => 0i64.checked_sub(y[i * q + j - p]),
            (false, true) => 0i64.checked_sub(y[j * q + i - p]),
            (false, false) => Some(si[(i - p) * q + j - p]),
        };
        match v {
            Some(v) => *o = v,
            None => bad.store(true, Ordering::Relaxed),
        }
    });
    if bad.into_inner() {
        return Err(NqError::Range);
    }
    Ok(out)
}

/// One ridge `X`-update in the transposed frame: returns `Fᵀ` (`k × r`) with
/// `F = (XᵀX + (ρ·mean diag + reg)·I)⁻¹ (XᵀY + ρ(Z − U))`, `X = Z_x/‖·‖` column-normalised.
///
/// `zx` is the `n × r` normalising factor (`Bᵀ_z` or `A_z`), `wp` the prepared `k × n` target
/// (`W_n` for `A`, `W_nᵀ` for `B`), `z`/`du` the `k × r` consensus and dual.
///
/// `O(r²·n + r·n·k + 2r³/3 + r²·k)` multiply-adds: the full-square Gram, the right-hand side, the
/// inverse ([`spd_inverse`]) and `Rᵀ·H⁻¹`, all on the kernel but the inverse's order-32 bases.
#[allow(
    clippy::too_many_arguments,
    reason = "the ADMM step's operands are distinct matrices"
)]
pub(crate) fn x_update(
    zx: &[f64],
    n: usize,
    r: usize,
    wp: &crate::engine::kern::limb::Prepared,
    k: usize,
    z: &[f64],
    du: &[f64],
    rho: f64,
    reg: f64,
) -> Result<Vec<f64>, NqError> {
    // `Z_xᵀ` on the grid in one pass: entry `(i, j)` of the `r × n` result is `zx[j·r + i]`.
    let zt = grid_map(r * n, &|c| zx[(c % n) * r + c / n])?;
    // The full square in one product: `syrk_q32` tiles `SYRK_COLS = 16` columns per kernel call.
    let g = prepare(&zt, r, n, LIMBS)?.gemm(&zt, r)?;
    let d: Vec<f64> = (0..r)
        .map(|i| from_grid(g[i * r + i]).sqrt().max(EPS))
        .collect();
    let hn = |i: usize, j: usize| from_grid(g[i * r + j]) / (d[i] * d[j]);
    let mean_diag = (0..r).map(|i| hn(i, i)).sum::<f64>() / r as f64;
    let stab = (rho * mean_diag.abs() + reg).max(EPS);
    // The reference's symmetrisation `(H + Hᵀ)/2` off the diagonal (IEEE addition commutes, so
    // both triangles take the same word), then the stabiliser on the diagonal.
    let hw = grid_map(r * r, &|c| {
        let (i, j) = (c / r, c % r);
        if i == j {
            hn(i, i) + stab
        } else {
            f64::midpoint(hn(i, j), hn(j, i))
        }
    })?;
    let hi = spd_inverse(&hw, r)?;
    // Rᵀ = Y_x·Z_x·D⁻¹ + ρ(Z − U), `k × r`.
    let wz = wp.gemm(&zt, r)?;
    let rt = grid_map(k * r, &|c| {
        from_grid(wz[c]) / d[c % r] + rho * (z[c] - du[c])
    })?;
    // Fᵀ = Rᵀ·H⁻¹: `gemm_q32(a, b) = a·bᵀ`, and `H⁻¹` is symmetric, so its rows serve.
    let f = gemm_q32(&rt, k, &hi, r, r, LIMBS)?;
    let mut out = vec![0.0f64; f.len()];
    par_fill(&mut out, &|c, o| *o = from_grid(f[c]));
    Ok(out)
}

/// The factor of `Wᵀ` (`inn × out`) read as the factor of `W`: `(U·diag·V)ᵀ = Vᵀ·diag·Uᵀ`, the
/// scales swapped. `O(rank·(out + inn))` copies.
fn transposed(f: NqFactor, out: usize, inn: usize) -> NqFactor {
    NqFactor {
        out,
        inn,
        rank: f.rank,
        u: transpose(&f.v, f.rank, out),
        v: transpose(&f.u, inn, f.rank),
        s_pre: f.s_post,
        s_post: f.s_pre,
        admm_loss: f.admm_loss,
        u_latent: transpose(&f.v_latent, f.rank, out),
        v_latent: transpose(&f.u_latent, inn, f.rank),
    }
}

/// `factorize_admm_nanoquant` on `w` (`out × inn`, Q32.32 row-major) under the input and output
/// statistics `i_norm` (length `inn`) and `o_norm` (length `out`), exported to the deployed
/// two-scale sign factor. A wide matrix (`out < inn`) runs on `wᵀ` with the statistics swapped
/// and is transposed back, as the reference's `is_transpose`.
///
/// # Errors
///
/// [`NqError::Shape`] on a mismatched length or a zero dimension or rank; [`NqError::Range`] on a
/// value off the Q32.32 grid; [`NqError::Kernel`] and [`NqError::Solve`] as the products and the
/// inverse factor refuse.
///
/// # Complexity
///
/// Per outer iteration (frame `o ≥ n`, rank `r`): two [`x_update`]s, `O(r²(o + n) + 2·r·o·n +
/// 4r³/3 + r²(o + n))` multiply-adds, and two [`rank1`]s, `O(inner·r·(o + n))`. Total
/// `T` times that; space `O(o·n + r·(o + n) + r²)` words.
pub fn factorize(
    w: &[i64],
    out: usize,
    inn: usize,
    i_norm: &[f64],
    o_norm: &[f64],
    cfg: &NqConfig,
) -> Result<NqFactor, NqError> {
    factorize_floor(w, out, inn, i_norm, o_norm, cfg, FLOOR_REL)
}

/// The factor's preprint scales `sqrt(diag)` floored at `floor_rel · mean(sqrt(diag))` (and at
/// [`EPS`]): a coordinate the calibration never excites would otherwise scale a column by `1e-12`,
/// which the limb kernel's block-exponent placement refuses. `O(n)` time and `O(n)` words over the
/// `n` coordinates, one pass for the mean and one for the clamp.
fn scales(diag: &[f64], floor_rel: f64) -> Vec<f64> {
    let raw: Vec<f64> = diag.iter().map(|x| x.sqrt()).collect();
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let floor = (floor_rel * mean).max(EPS);
    raw.iter().map(|&v| v.max(floor)).collect()
}

/// [`factorize`] with the preconditioner floor as a fraction of the mean `sqrt(diag)` ([`FLOOR_REL`]
/// in [`factorize`]). A fit stage reads it as one knob; the wide-matrix transpose path carries it
/// to the swapped call.
///
/// # Errors
///
/// As [`factorize`]; a floor small enough to leave a dead Gram coordinate's column outside the
/// limb kernel's exponent range still returns [`NqError::Kernel`] (`< 1e-11` on `Qwen3-0.6B-Base`
/// layer 0 `gate`).
///
/// # Complexity
///
/// As [`factorize`], plus `O(out + inn)` for [`scales`].
#[allow(
    clippy::too_many_lines,
    reason = "one ADMM loop over the two transposed frames; a split separates the sides' shared \
              shapes from their steps"
)]
pub fn factorize_floor(
    w: &[i64],
    out: usize,
    inn: usize,
    i_norm: &[f64],
    o_norm: &[f64],
    cfg: &NqConfig,
    floor_rel: f64,
) -> Result<NqFactor, NqError> {
    if out == 0 || inn == 0 || cfg.rank == 0 {
        return Err(NqError::Shape);
    }
    if w.len() != out * inn || i_norm.len() != inn || o_norm.len() != out {
        return Err(NqError::Shape);
    }
    if out < inn {
        let f = factorize_floor(
            &transpose(w, out, inn),
            inn,
            out,
            o_norm,
            i_norm,
            cfg,
            floor_rel,
        )?;
        return Ok(transposed(f, out, inn));
    }
    let (o, n, r) = (out, inn, cfg.rank);
    let ni: Vec<f64> = scales(i_norm, floor_rel);
    let no: Vec<f64> = scales(o_norm, floor_rel);
    let mut wn = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            wn[i * n + j] = from_grid(w[i * n + j]) * ni[j] * no[i];
        }
    }
    let (wg, wtg) = (grid(&wn)?, grid(&transpose(&wn, o, n))?);
    let (wp, wtp) = (prepare(&wg, o, n, LIMBS)?, prepare(&wtg, n, o, LIMBS)?);
    let (mut a_side, mut b_side) = (resident(&wg, o, n, r), resident(&wtg, n, o, r));
    let mut rng = Rng(cfg.seed);
    // A is `o × r`; B is held as Bᵀ, `n × r`.
    let mut a_ls = rng.normals(o * r, |c| c);
    // `B` is drawn row-major `r × n` and held transposed: slot `c = j·r + k` takes draw `k·n + j`.
    let mut b_ls = rng.normals(r * n, |c| (c % r) * n + c / r);
    let (mut a_z, mut b_z) = if cfg.outer_iters > 0 {
        (
            rank1_step(
                &mut a_side,
                &a_ls,
                (o, r),
                cfg.inner_iters,
                false,
                draw(&mut rng, r),
            )?,
            rank1_step(
                &mut b_side,
                &b_ls,
                (n, r),
                cfg.inner_iters,
                true,
                draw(&mut rng, n),
            )?,
        )
    } else {
        (a_ls.clone(), b_ls.clone())
    };
    let mut a_u: Vec<f64> = a_ls.iter().zip(&a_z).map(|(x, z)| x - z).collect();
    let mut b_u: Vec<f64> = b_ls.iter().zip(&b_z).map(|(x, z)| x - z).collect();
    // Device waits separate the host inverse's dispatches by more than the pool's park delay.
    let _hot = crate::engine::chart::pool::hot();
    let (t0, mut pending) = device_iters(
        [&mut a_side, &mut b_side],
        [
            [&mut a_ls, &mut b_ls],
            [&mut a_z, &mut b_z],
            [&mut a_u, &mut b_u],
        ],
        &mut rng,
        cfg,
        [r, n],
    );
    for t in t0..cfg.outer_iters {
        // Each side's step reads only the previous `Z`s and its own `U`, so the two run on two
        // threads: one side's `rank1` and host inverse proceed while the other waits on the
        // device. The starts are drawn in the serial order (`A` then `B`), so each result is the
        // serial one; refusals resolve in the serial order `X_A`, `X_B`, `Z_A`, `Z_B`.
        let (sa, sb) = pending
            .take()
            .unwrap_or_else(|| (draw(&mut rng, r), draw(&mut rng, n)));
        let rho = cfg.schedule.rho(t as f64 / cfg.outer_iters as f64);
        let p = (rho, cfg.reg, cfg.inner_iters);
        let (ra, rb) = std::thread::scope(|s| {
            let hb = s.spawn(|| {
                side_step(
                    &mut b_side,
                    (&a_z, o, r, &wtp, n),
                    &b_z,
                    &mut b_u,
                    p,
                    true,
                    sb,
                )
            });
            let ra = side_step(
                &mut a_side,
                (&b_z, n, r, &wp, o),
                &a_z,
                &mut a_u,
                p,
                false,
                sa,
            );
            (ra, hb.join())
        });
        let (xa, za) = ra?;
        let (xb, zb) = rb.unwrap_or_else(|p| std::panic::resume_unwind(p))?;
        (a_ls, b_ls) = (xa, xb);
        (a_z, b_z) = (za?, zb?);
    }
    let admm_loss = admm_loss(&wn, &a_z, &b_z, o, n, r)?;
    let a_lat: Vec<f64> = a_ls.iter().zip(&a_u).map(|(x, u)| x + u).collect();
    let b_lat: Vec<f64> = b_ls.iter().zip(&b_u).map(|(x, u)| x + u).collect();
    export(&[&a_z, &b_z, &a_lat, &b_lat], &ni, &no, o, n, r, admm_loss)
}

/// `‖W_n − (A_z/m)·B_z‖²/‖W_n‖²`, `m_k = ‖B_z row k‖` (the reference's printed loss).
/// `O(o·n·r)` on the kernel plus `O(o·n)`.
fn admm_loss(
    wn: &[f64],
    a_z: &[f64],
    b_z: &[f64],
    o: usize,
    n: usize,
    r: usize,
) -> Result<f64, NqError> {
    let bzr = transpose(b_z, n, r);
    let mid = rowdot_q32(&grid(&bzr)?, &grid(&bzr)?, r, n, LIMBS)?;
    let mut am = a_z.to_vec();
    for i in 0..o {
        for k in 0..r {
            am[i * r + k] /= from_grid(mid[k]).sqrt().max(EPS);
        }
    }
    let pred = gemm_q32(&grid(&am)?, o, &grid(b_z)?, n, r, LIMBS)?;
    let e: Vec<f64> = wn
        .iter()
        .zip(&pred)
        .map(|(w, &p)| w - from_grid(p))
        .collect();
    Ok(fro2(&e, o, n)? / fro2(wn, o, n)?.max(EPS))
}

/// The reference's final export: divide out the statistics, balance `A` and `B` to equal
/// Frobenius norm, scale `A`'s columns by `1/‖A_z col‖`, take `s_pre`/`s_post` as the mean
/// magnitudes and `U`/`V` as the signs (zero to `+1`). The latents `A_ls + A_u`, `B_ls + B_u`
/// take the same statistics and balance, without the column scale. `z = [A_z, B_zᵀ, A_lat,
/// B_latᵀ]`. `O(r·(o + n))` plus two `rowdot_q32`.
fn export(
    z: &[&[f64]; 4],
    ni: &[f64],
    no: &[f64],
    o: usize,
    n: usize,
    r: usize,
    admm_loss: f64,
) -> Result<NqFactor, NqError> {
    let [a_z, b_z, a_lat, b_lat] = *z;
    let mut a: Vec<f64> = a_z.to_vec();
    for i in 0..o {
        for k in 0..r {
            a[i * r + k] /= no[i];
        }
    }
    let mut b: Vec<f64> = b_z.to_vec();
    for j in 0..n {
        for k in 0..r {
            b[j * r + k] /= ni[j];
        }
    }
    let na = fro2(&a, o, r)?.sqrt().max(EPS);
    let nb = fro2(&b, n, r)?.sqrt().max(EPS);
    let bf = (nb / na).sqrt();
    let azt = transpose(a_z, o, r);
    let col = rowdot_q32(&grid(&azt)?, &grid(&azt)?, r, o, LIMBS)?;
    for i in 0..o {
        for k in 0..r {
            a[i * r + k] *= bf / from_grid(col[k]).sqrt().max(EPS);
        }
    }
    for x in &mut b {
        *x /= bf;
    }
    let mut ul = a_lat.to_vec();
    for i in 0..o {
        for k in 0..r {
            ul[i * r + k] *= bf / no[i];
        }
    }
    let mut vl = b_lat.to_vec();
    for j in 0..n {
        for k in 0..r {
            vl[j * r + k] /= ni[j] * bf;
        }
    }
    let sign = |x: f64| -> i8 {
        if x < 0.0 {
            -1
        } else {
            1
        }
    };
    let s_post: Vec<f64> = (0..o)
        .map(|i| a[i * r..(i + 1) * r].iter().map(|x| x.abs()).sum::<f64>() / r as f64)
        .collect();
    let s_pre: Vec<f64> = (0..n)
        .map(|j| b[j * r..(j + 1) * r].iter().map(|x| x.abs()).sum::<f64>() / r as f64)
        .collect();
    Ok(NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: a.iter().map(|&x| sign(x)).collect(),
        v: transpose(&b.iter().map(|&x| sign(x)).collect::<Vec<i8>>(), n, r),
        s_pre,
        s_post,
        admm_loss,
        u_latent: ul,
        v_latent: transpose(&vl, n, r),
    })
}

/// Settings of [`tune`]: the reference `tune_fact` optimiser (`compress_block.py:160-201`,
/// `main.py:90-94`): `optimi.AdamW` with `β = (0.9, 0.99)`, `ε = 1e-6`, weight decay 0, cosine
/// annealing over `steps` to `1e-4·lr_scale`.
#[derive(Debug, Clone, Copy)]
pub struct TuneConfig {
    /// Optimiser steps (reference: 128 samples × 8 epochs at batch 1 = 1024).
    pub steps: usize,
    /// Learning rate of the latents (reference `fact_binary_lr` `1e-5`).
    pub lr_binary: f64,
    /// Learning rate of `s_pre`, `s_post` (reference `fact_scale_lr` `1e-5`).
    pub lr_scale: f64,
    /// Multiplier of the per-token Gram: the reference loss sums one 2048-token sample, so `2048`
    /// puts the gradient on the reference's scale against `ε`.
    pub h_scale: f64,
}

impl TuneConfig {
    /// The reference settings at `steps` steps.
    #[must_use]
    pub fn reference(steps: usize) -> Self {
        Self {
            steps,
            lr_binary: 1e-5,
            lr_scale: 1e-5,
            h_scale: 2048.0,
        }
    }
}

/// The record of one [`tune`].
#[derive(Debug, Clone)]
pub struct TuneTrace {
    /// `tr(EHEᵀ)/tr(WHWᵀ)` of the deployed factor before the first step and after the last.
    pub rel_before: f64,
    /// As `rel_before`, after the last step's signs and scales.
    pub rel_after: f64,
    /// The lowest `rel` seen at any step, and its step.
    pub rel_best: (f64, usize),
    /// Entries of `U` and `V` whose sign changed.
    pub flips: usize,
}

/// One `optimi` Adam step on `p` with gradient `g` (moments `m`, `v`, 1-based step `t`):
/// `β̂ = (βᵗ − β)/(βᵗ − 1)`, `m ← lerp(m, g, 1 − β̂₁)`, `v ← β̂₂v + (1 − β̂₂)g²`,
/// `p ← p − lr·m/(√v + ε)`, `mv[i] = [m, v]`. Two [`par_fill`] passes, each entry alone, so the
/// result is the sequential one. `O(len)`.
fn adam(p: &mut [f64], g: &[f64], mv: &mut [[f64; 2]], t: usize, lr: f64) {
    const B1: f64 = 0.9;
    const B2: f64 = 0.99;
    const E: f64 = 1e-6;
    let debias = |b: f64| (b.powf(t as f64) - b) / (b.powf(t as f64) - 1.0);
    let (b1, b2) = (debias(B1), debias(B2));
    par_fill(mv, &|i, e| {
        e[0] = b1 * e[0] + (1.0 - b1) * g[i];
        e[1] = b2 * e[1] + (1.0 - b2) * g[i] * g[i];
    });
    let mv = &*mv;
    par_fill(p, &|i, x| *x -= lr * mv[i][0] / (mv[i][1].sqrt() + E));
}

/// The `±1` word of the sign of `e` (zero to `+1`).
fn sign_word(e: f64) -> i64 {
    if e < 0.0 {
        -(1i64 << 32)
    } else {
        1i64 << 32
    }
}

/// `±1` words of the signs of `x` (zero to `+1`).
fn sign_words(x: &[f64]) -> Vec<i64> {
    let mut out = vec![0i64; x.len()];
    par_fill(&mut out, &|c, o| *o = sign_word(x[c]));
    out
}

/// The four gradients of [`tune`]'s loss from `(EH)ᵀ = pt` (`n × o`), `M` and the signs:
/// `[∂U (o × r), ∂Vᵀ (n × r), ∂s_post, ∂s_pre]`. Two `o·n·r` kernel products and two
/// `rowdot_q32`s of `o·n`, plus `O(o·n)` element maps and transposes.
fn grads(
    pt: &[f64],
    m: &[f64],
    us: &[i64],
    vst: &[i64],
    sp: &[f64],
    sq: &[f64],
    (o, n, r): (usize, usize, usize),
) -> Result<[Vec<f64>; 4], NqError> {
    // `Gᵀ = 2·H·Eᵀ` (`n × o`) and `G`.
    let gt: Vec<f64> = pt.iter().map(|x| 2.0 * x).collect();
    let mut gsq = transpose(&gt, n, o);
    for i in 0..o {
        for j in 0..n {
            gsq[i * n + j] *= sq[j];
        }
    }
    let mut gsp = gt.clone();
    for j in 0..n {
        for i in 0..o {
            gsp[j * o + i] *= sp[i];
        }
    }
    // ∂U = diag(sp)·(G·diag(sq))·Vᵀ: `a·bᵀ` with `b = V` (`r × n`).
    let vs = transpose(vst, n, r);
    let mut ga: Vec<f64> = gemm_q32(&grid(&gsq)?, o, &vs, r, n, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    for i in 0..o {
        for k in 0..r {
            ga[i * r + k] *= sp[i];
        }
    }
    // ∂Vᵀ = diag(sq)·(Gᵀ·diag(sp))·U: `a·bᵀ` with `b = Uᵀ` (`r × o`).
    let ust = transpose(us, o, r);
    let mut gb: Vec<f64> = gemm_q32(&grid(&gsp)?, n, &ust, r, o, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    for j in 0..n {
        for k in 0..r {
            gb[j * r + k] *= sq[j];
        }
    }
    // ∂sp = rows of `G·diag(sq)` dotted with `M`; ∂sq = rows of `Gᵀ·diag(sp)` with `Mᵀ`.
    let grad_post: Vec<f64> = rowdot_q32(&grid(&gsq)?, &grid(m)?, o, n, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    let grad_pre: Vec<f64> = rowdot_q32(&grid(&gsp)?, &grid(&transpose(m, o, n))?, n, o, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    Ok([ga, gb, grad_post, grad_pre])
}

/// The layer-scope analogue of the reference's `tune_fact`: Adam on the latents (through the
/// straight-through sign, `binary_ste`) and the two scales against
/// `L = tr((Ŵ − W)·H·(Ŵ − W)ᵀ)`, the squared output error of the layer over the calibration
/// tokens whose Gram is `h` (`in × in`, Q32.32, per token), then the signs of the latents.
///
/// With `G = 2(Ŵ − W)H`, `M = U·V`: `∂L/∂U = diag(s_post)·G·diag(s_pre)·Vᵀ`,
/// `∂L/∂V = Uᵀ·diag(s_post)·G·diag(s_pre)`, `∂L/∂s_post[i] = Σ_j G_ij·M_ij·s_pre[j]`,
/// `∂L/∂s_pre[j] = Σ_i G_ij·M_ij·s_post[i]`.
///
/// # Errors
///
/// [`NqError::Shape`] on a mismatched `w`, `h` or latent; [`NqError::Range`] and
/// [`NqError::Kernel`] as the products refuse.
///
/// # Complexity
///
/// Per step `o·n² + 3·o·n·r` multiply-adds on the kernel (`EH`, `UV` and the two latent
/// gradients) plus `O(o·n)` element maps and two `rowdot_q32`s; one more `EH` before the first
/// step, one after the last, and one `WH`. Space `O(o·n + n²)` words.
pub fn tune(
    f: &mut NqFactor,
    w: &[i64],
    h: &[i64],
    cfg: &TuneConfig,
) -> Result<TuneTrace, NqError> {
    let (o, n, r) = (f.out, f.inn, f.rank);
    if w.len() != o * n
        || h.len() != n * n
        || f.u_latent.len() != o * r
        || f.v_latent.len() != r * n
    {
        return Err(NqError::Shape);
    }
    let hs: Vec<f64> = h.iter().map(|&x| from_grid(x) * cfg.h_scale).collect();
    let hp = prepare(&grid(&hs)?, n, n, LIMBS)?;
    let wf: Vec<f64> = w.iter().map(|&x| from_grid(x)).collect();
    // `tr(XHXᵀ)` and `(XH)ᵀ` for an `o × n` matrix `X`: `H·Xᵀ` on the prepared plane, then the
    // trace as row dots of `Xᵀ` against it.
    let energy = |x: &[f64]| -> Result<(f64, Vec<f64>), NqError> {
        let pt = hp.gemm(&grid(x)?, o)?;
        let d = rowdot_q32(&grid(&transpose(x, o, n))?, &pt, n, o, LIMBS)?;
        Ok((
            d.iter().map(|&e| from_grid(e)).sum(),
            pt.iter().map(|&e| from_grid(e)).collect(),
        ))
    };
    let base = energy(&wf)?.0.max(EPS);
    let mut ul = f.u_latent.clone();
    let mut vlt = transpose(&f.v_latent, r, n);
    let (mut sp, mut sq) = (f.s_post.clone(), f.s_pre.clone());
    let (u0, v0) = (sign_words(&ul), sign_words(&vlt));
    let mut mom = [
        vec![[0.0; 2]; o * r],
        vec![[0.0; 2]; n * r],
        vec![[0.0; 2]; o],
        vec![[0.0; 2]; n],
    ];
    // `Ŵ = diag(sp)·M·diag(sq)`, `M = sign(ul)·sign(vlt)ᵀ` exact on the kernel.
    let forward = |ul: &[f64], vlt: &[f64], sp: &[f64], sq: &[f64]| {
        let (us, vst) = (sign_words(ul), sign_words(vlt));
        let m: Vec<f64> = gemm_q32(&us, o, &vst, n, r, LIMBS)?
            .iter()
            .map(|&e| from_grid(e))
            .collect();
        let mut e = vec![0.0f64; o * n];
        for i in 0..o {
            for j in 0..n {
                e[i * n + j] = sp[i] * m[i * n + j] * sq[j] - wf[i * n + j];
            }
        }
        Ok::<_, NqError>((us, vst, m, e))
    };
    let (_, _, _, e0) = forward(&ul, &vlt, &sp, &sq)?;
    let rel_before = energy(&e0)?.0 / base;
    let mut rel_best = (rel_before, 0);
    let eta_min = 1e-4 * cfg.lr_scale;
    for t in 0..cfg.steps {
        let (us, vst, m, e) = forward(&ul, &vlt, &sp, &sq)?;
        let (en, pt) = energy(&e)?;
        if en / base < rel_best.0 {
            rel_best = (en / base, t);
        }
        let [ga, gb, grad_post, grad_pre] = grads(&pt, &m, &us, &vst, &sp, &sq, (o, n, r))?;
        let cos = f64::midpoint(
            1.0,
            (std::f64::consts::PI * t as f64 / cfg.steps as f64).cos(),
        );
        let lr_b = eta_min + (cfg.lr_binary - eta_min) * cos;
        let lr_s = eta_min + (cfg.lr_scale - eta_min) * cos;
        let [m0, m1, m2, m3] = &mut mom;
        adam(&mut ul, &ga, m0, t + 1, lr_b);
        adam(&mut vlt, &gb, m1, t + 1, lr_b);
        adam(&mut sp, &grad_post, m2, t + 1, lr_s);
        adam(&mut sq, &grad_pre, m3, t + 1, lr_s);
    }
    let (us, vst, _, e) = forward(&ul, &vlt, &sp, &sq)?;
    let rel_after = energy(&e)?.0 / base;
    if rel_after < rel_best.0 {
        rel_best = (rel_after, cfg.steps);
    }
    let flips = us.iter().zip(&u0).filter(|(a, b)| a != b).count()
        + vst.iter().zip(&v0).filter(|(a, b)| a != b).count();
    let sgn = |x: i64| -> i8 {
        if x < 0 {
            -1
        } else {
            1
        }
    };
    f.u = us.iter().map(|&x| sgn(x)).collect();
    f.v = transpose(&vst.iter().map(|&x| sgn(x)).collect::<Vec<i8>>(), n, r);
    f.u_latent = ul;
    f.v_latent = transpose(&vlt, n, r);
    f.s_post = sp;
    f.s_pre = sq;
    Ok(TuneTrace {
        rel_before,
        rel_after,
        rel_best,
        flips,
    })
}

/// A rank-`s` sketch of the calibration Gram: `Q` (`n × s`, orthonormal columns) spanning `H`'s
/// dominant subspace and `H_s = QᵀHQ` (`s × s`), so `QH_sQᵀ = PHP` with `P = QQᵀ`.
#[derive(Debug, Clone)]
pub struct HSketch {
    /// The sketch width `s`.
    pub s: usize,
    /// `Q`, row-major `n × s`.
    pub q: Vec<f64>,
    /// `H_s = QᵀHQ`, row-major `s × s`.
    pub hs: Vec<f64>,
}

/// The columns of `y` (`n × s`) orthonormalised by Cholesky QR: unit columns, `G = YᵀY` by
/// `syrk_q32`, `G⁻¹ = UᵀU` by [`inv_factor_upper`], `Q = Y·Uᵀ` (`QᵀQ = U·G·Uᵀ = I`). Three passes,
/// the first on `G + 2⁻¹²·I` (shifted `CholeskyQR3`, Fukaya et al. 2020): the 3-limb `H·Q`
/// carries its tail columns to about `10⁻⁴` relative, and at `s = 512` that leaves `G` singular
/// unshifted. `O(3·(n·s² + s³))` multiply-adds.
fn orth(y: &[f64], n: usize, s: usize) -> Result<Vec<f64>, NqError> {
    let mut q = y.to_vec();
    for pass in 0..3 {
        let mut yt = transpose(&q, n, s);
        let norms = rowdot_q32(&grid(&yt)?, &grid(&yt)?, s, n, LIMBS)?;
        for k in 0..s {
            let d = from_grid(norms[k]).sqrt().max(EPS);
            for e in &mut yt[k * n..(k + 1) * n] {
                *e /= d;
            }
        }
        let mut g = syrk_q32(&grid(&yt)?, s, n, LIMBS)?;
        if pass == 0 {
            for k in 0..s {
                g[k * s + k] += 1 << 20;
            }
        }
        let ui = inv_factor_upper(&g, s)?;
        q = gemm_q32(&grid(&transpose(&yt, s, n))?, n, &ui, s, s, LIMBS)?
            .iter()
            .map(|&w| from_grid(w))
            .collect();
    }
    Ok(q)
}

/// [`HSketch`] of `h` (`n × n`, Q32.32) by subspace iteration: a Gaussian `n × s` start, then
/// `passes` steps `Q ← orth(H·Q)` on one prepared plane of `H`, then `H_s = Qᵀ(HQ)`.
///
/// # Errors
///
/// [`NqError::Shape`] if `h.len() != n·n` or `s` is zero or above `n`; [`NqError::Range`],
/// [`NqError::Kernel`] and [`NqError::Solve`] as the products and the factor refuse.
///
/// # Complexity
///
/// `(passes + 1)·n²·s` kernel multiply-adds for the `H·Q` products, `(passes + 1)·3·(n·s² + s³)`
/// for [`orth`], `n·s²` for `H_s`. Space `O(n² + n·s)` words (the prepared plane of `H`).
pub fn h_sketch(
    h: &[i64],
    n: usize,
    s: usize,
    passes: usize,
    seed: u64,
) -> Result<HSketch, NqError> {
    if h.len() != n * n || s == 0 || s > n {
        return Err(NqError::Shape);
    }
    let hp = prepare(h, n, n, LIMBS)?;
    let mut rng = Rng(seed);
    let omega: Vec<f64> = (0..n * s).map(|_| rng.normal()).collect();
    let mut q = orth(&omega, n, s)?;
    // `hp.gemm(Qᵀ, s) = H·Q` (`n × s`).
    let hq = |q: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(hp
            .gemm(&grid(&transpose(q, n, s))?, s)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    for _ in 0..passes {
        q = orth(&hq(&q)?, n, s)?;
    }
    let y = hq(&q)?;
    let hs = gemm_q32(
        &grid(&transpose(&q, n, s))?,
        s,
        &grid(&transpose(&y, n, s))?,
        s,
        n,
        LIMBS,
    )?
    .iter()
    .map(|&w| from_grid(w))
    .collect();
    Ok(HSketch { s, q, hs })
}

impl HSketch {
    /// `tr(X·QH_sQᵀ·Xᵀ)` of a row-major `o × n` matrix `x`: `XQ` (`o·n·s`), `(XQ)·H_s` (`o·s²`) and
    /// the row dots.
    ///
    /// # Errors
    ///
    /// [`NqError::Shape`] on a mismatched `x`; [`NqError::Range`] and [`NqError::Kernel`] as the
    /// products refuse.
    pub fn energy(&self, x: &[f64], o: usize) -> Result<f64, NqError> {
        let s = self.s;
        let n = self.q.len() / s;
        if x.len() != o * n {
            return Err(NqError::Shape);
        }
        let xq = gemm_q32(&grid(x)?, o, &grid(&transpose(&self.q, n, s))?, s, n, LIMBS)?;
        let xqh = gemm_q32(&xq, o, &grid(&self.hs)?, s, s, LIMBS)?;
        let d = rowdot_q32(&xq, &xqh, o, s, LIMBS)?;
        Ok(d.iter().map(|&e| from_grid(e)).sum())
    }
}

/// Kernel words to reals, through [`par_fill`]. `O(len)`.
fn reals(x: &[i64]) -> Vec<f64> {
    let mut out = vec![0.0f64; x.len()];
    par_fill(&mut out, &|c, o| *o = from_grid(x[c]));
    out
}

/// Rows of [`col_grads`] per read of `xt`: eight words, one 64-byte cache line.
const COL_BLOCK: usize = 8;

/// From the `n × r` words `xt = Xᵀ`: the `r × n` reals `g[i·n + j] = from_grid(to_grid(X_ij·sq[j]))`
/// and the signed column sums `y[j] = Σ_i sign(v[i·n + j])·X_ij`, exact in `i128` (no product, no
/// rounding), so `y` is independent of the split. One row band of `g` per pool task, reading the
/// band's columns of `xt` at stride `r`; each band adds its exact partial sums into `y` under one
/// lock. [`NqError::Range`] if an entry leaves the grid or a sum leaves `i64`. `O(r·n)` reads,
/// multiplies and additions; `O(r·n)` space for `g` and `O(n)` per band.
fn col_grads(
    v: &[f64],
    xt: &[i64],
    sq: &[f64],
    r: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), NqError> {
    let bad = AtomicBool::new(false);
    let sums = Mutex::new(vec![0i128; n]);
    let mut g = vec![0.0f64; r * n];
    let row_band = |start: usize, piece: &mut [f64]| {
        let i0 = start / n;
        let rows = piece.len() / n;
        let mut acc = vec![0i128; n];
        // Rows in blocks of `COL_BLOCK`: one read of `xt[j·r + i..][..COL_BLOCK]` (one cache line)
        // serves the block's `COL_BLOCK` outputs in column `j`.
        let mut b0 = 0usize;
        while b0 < rows {
            let bl = COL_BLOCK.min(rows - b0);
            let i = i0 + b0;
            for j in 0..n {
                let line = &xt[j * r + i..j * r + i + bl];
                let mut a = 0i128;
                for (d, &w) in line.iter().enumerate() {
                    let c = (b0 + d) * n + j;
                    match to_grid(from_grid(w) * sq[j], GRID_BOUND) {
                        Some(q) => piece[c] = from_grid(q),
                        None => bad.store(true, Ordering::Relaxed),
                    }
                    let w = i128::from(w);
                    a += if v[start + c] < 0.0 { -w } else { w };
                }
                acc[j] += a;
            }
            b0 += bl;
        }
        let mut total = sums.lock().unwrap_or_else(PoisonError::into_inner);
        for (t, a) in total.iter_mut().zip(acc) {
            *t += a;
        }
    };
    if r * n < PAR_MIN {
        row_band(0, &mut g);
    } else {
        let chunk = r.div_ceil(width_global()).next_multiple_of(COL_BLOCK) * n;
        Pool::global().chunks_mut(&mut g, chunk, &|_, start, piece: &mut [f64]| {
            row_band(start, piece);
        });
    }
    let mut y = vec![0.0f64; n];
    for (o, s) in y
        .iter_mut()
        .zip(sums.into_inner().unwrap_or_else(PoisonError::into_inner))
    {
        match i64::try_from(s) {
            Ok(w) => *o = from_grid(w),
            Err(_) => bad.store(true, Ordering::Relaxed),
        }
    }
    if bad.into_inner() {
        Err(NqError::Range)
    } else {
        Ok((g, y))
    }
}

/// The fixed `s`-frame of [`tune_sketch`]: `Qᵀ` as kernel words, `Q` as a prepared kernel operand,
/// `C = h_scale·H_s` as kernel words and `W·Q`.
struct SkFrame {
    dims: (usize, usize, usize, usize),
    qt: Vec<i64>,
    qw: crate::engine::kern::limb::Prepared,
    c: Vec<i64>,
    wq: Vec<f64>,
}

/// One forward of [`tune_sketch`]: the sign words of `U`, `P`, `U·P`, `E_Q·C` and the
/// sketched loss `L_s`.
struct SkStep {
    us: Vec<i64>,
    p: Vec<i64>,
    upw: Vec<i64>,
    eqc: Vec<i64>,
    ls: f64,
}

impl SkFrame {
    /// `P = V·diag(sq)·Q` (`r × s`), `UP = U·P` (`o × s`), `E_Q = diag(sp)·UP − W·Q`,
    /// `E_Q·C` and `L_s = tr(E_Q·C·E_Qᵀ)`. `r·s·(n + o) + o·s²` kernel multiply-adds plus
    /// `O(r·n + o·r + o·s)` element maps. Words move between products without a round trip through
    /// reals (`from_grid` then `to_grid` returns the word), so every product sees the operands
    /// the round trip gave.
    fn step(&self, ul: &[f64], vl: &[f64], sp: &[f64], sq: &[f64]) -> Result<SkStep, NqError> {
        let (o, n, r, s) = self.dims;
        let us = sign_words(ul);
        // `sign(V)·diag(sq)`: `±sq[j]`, the words `scale_cols` gives on `±1.0`.
        let vsq = grid_map(r * n, &|c| {
            if vl[c] < 0.0 {
                -sq[c % n]
            } else {
                sq[c % n]
            }
        })?;
        let p = gemm_q32(&vsq, r, &self.qt, s, n, LIMBS)?;
        let upw = gemm_q32(&us, o, &transpose(&p, r, s), s, r, LIMBS)?;
        let eqw = grid_map(o * s, &|c| sp[c / s] * from_grid(upw[c]) - self.wq[c])?;
        let eqc = gemm_q32(&eqw, o, &self.c, s, s, LIMBS)?;
        let ls = reals(&rowdot_q32(&eqw, &eqc, o, s, LIMBS)?).iter().sum();
        Ok(SkStep {
            us,
            p,
            upw,
            eqc,
            ls,
        })
    }

    /// `[∂U, ∂V, ∂s_post, ∂s_pre]` at `st` (formulas at [`tune_sketch`]).
    /// `r·s·(2·o + n)` kernel multiply-adds plus `O(o·s)` row dots and one fused `O(r·n)` pass
    /// ([`col_grads`]) for `∂V` and `∂s_pre`.
    fn grads(
        &self,
        st: &SkStep,
        vl: &[f64],
        sp: &[f64],
        sq: &[f64],
    ) -> Result<[Vec<f64>; 4], NqError> {
        let (o, n, r, s) = self.dims;
        // `G = 2·E_Q·C` and `diag(s_post)·G`.
        let gw = grid_map(o * s, &|c| 2.0 * from_grid(st.eqc[c]))?;
        let gsp = grid_map(o * s, &|c| 2.0 * from_grid(st.eqc[c]) * sp[c / s])?;
        let ga = reals(&gemm_q32(&gsp, o, &st.p, r, s, LIMBS)?);
        let a = gemm_q32(
            &transpose(&st.us, o, r),
            r,
            &transpose(&gsp, o, s),
            s,
            o,
            LIMBS,
        )?;
        // `∂V = A·Qᵀ·diag(sq)` (`r × n`); `∂s_pre[j] = ⟨Q_j, (VᵀA)_j⟩ = Σ_i sign(V_ij)·(A·Qᵀ)_ij`,
        // exact on the words of `A·Qᵀ`, so `VᵀA` is never formed. The kernel takes `Q` as its
        // prepared operand and returns `(A·Qᵀ)ᵀ` (`n × r`); `col_grads` reads it transposed.
        let aqt = self.qw.gemm(&a, r)?;
        let (gb, grad_pre) = col_grads(vl, &aqt, sq, r, n)?;
        let grad_post = reals(&rowdot_q32(&gw, &st.upw, o, s, LIMBS)?);
        Ok([ga, gb, grad_post, grad_pre])
    }
}

/// [`tune`] on the sketched loss `L_s = tr(E·QH_sQᵀ·Eᵀ)` (`sk` from [`h_sketch`]): the same
/// optimiser, rates, schedule and STE, with every product taken in the `s`-dimensional frame.
/// With `P = V·diag(s_pre)·Q` (`r × s`), `E_Q = diag(s_post)·U·P − W·Q`, `G = 2·E_Q·C`
/// (`C = h_scale·H_s`): `∂U = diag(s_post)·G·Pᵀ`, `A = Uᵀ·diag(s_post)·G`,
/// `∂Vᵀ = diag(s_pre)·Q·Aᵀ`, `∂s_post[i] = ⟨G_i, (U·P)_i⟩`,
/// `∂s_pre[j] = ⟨Q_j, (Vᵀ·A)_j⟩ = Σ_i V_ij·(A·Qᵀ)_ij`.
/// `rel_before`/`rel_after` are the full `tr(EHEᵀ)/tr(WHWᵀ)` under `h`; `rel_best` tracks the
/// sketched ratio `L_s/tr(W·QH_sQᵀ·Wᵀ)`.
///
/// # Errors
///
/// As [`tune`]; [`NqError::Shape`] also when `sk` is not `n × s`.
///
/// # Complexity
///
/// Per step `r·s·(3·o + 2·n) + o·s²` kernel multiply-adds (`P`, `U·P`, `∂U`, `A`, `∂Vᵀ` and `G`)
/// plus `O(r·(o + n))` element maps; against [`tune`]'s `o·n² + 3·o·n·r`. One `W·Q`, one
/// prepared `Q` (`O(n·s)`) and the two full energies (`2·o·n² + o·n·r`) outside the loop.
/// Space `O(r·(o + n) + n·s + n²)`.
pub fn tune_sketch(
    f: &mut NqFactor,
    w: &[i64],
    h: &[i64],
    sk: &HSketch,
    cfg: &TuneConfig,
) -> Result<TuneTrace, NqError> {
    let (o, n, r, s) = (f.out, f.inn, f.rank, sk.s);
    if w.len() != o * n
        || h.len() != n * n
        || sk.q.len() != n * s
        || sk.hs.len() != s * s
        || f.u_latent.len() != o * r
        || f.v_latent.len() != r * n
    {
        return Err(NqError::Shape);
    }
    let qt = grid(&transpose(&sk.q, n, s))?;
    let qw = prepare(&grid(&sk.q)?, n, s, LIMBS)?;
    let c = grid(&sk.hs.iter().map(|x| x * cfg.h_scale).collect::<Vec<f64>>())?;
    let wq = reals(&gemm_q32(w, o, &qt, s, n, LIMBS)?);
    let wqc = gemm_q32(&grid(&wq)?, o, &c, s, s, LIMBS)?;
    let base_s: f64 = reals(&rowdot_q32(&grid(&wq)?, &wqc, o, s, LIMBS)?)
        .iter()
        .sum::<f64>()
        .max(EPS);
    let fr = SkFrame {
        dims: (o, n, r, s),
        qt,
        qw,
        c,
        wq,
    };
    let hp = prepare(h, n, n, LIMBS)?;
    let full = |x: &[i64]| -> Result<f64, NqError> {
        let pt = hp.gemm(x, o)?;
        let xt = grid(&transpose(&reals(x), o, n))?;
        Ok(reals(&rowdot_q32(&xt, &pt, n, o, LIMBS)?).iter().sum())
    };
    let base = full(w)?.max(EPS);
    let rel_full = |f: &NqFactor| -> Result<f64, NqError> {
        let e: Vec<i64> = f.weight()?.iter().zip(w).map(|(&a, &b)| a - b).collect();
        Ok(full(&e)? / base)
    };
    let rel_before = rel_full(f)?;
    let mut ul = f.u_latent.clone();
    let mut vl = f.v_latent.clone();
    let (mut sp, mut sq) = (f.s_post.clone(), f.s_pre.clone());
    let (u0, v0) = (sign_words(&ul), sign_words(&vl));
    let mut mom = [
        vec![[0.0; 2]; o * r],
        vec![[0.0; 2]; r * n],
        vec![[0.0; 2]; o],
        vec![[0.0; 2]; n],
    ];
    let mut rel_best = (f64::INFINITY, 0);
    let eta_min = 1e-4 * cfg.lr_scale;
    for t in 0..=cfg.steps {
        let st = fr.step(&ul, &vl, &sp, &sq)?;
        if st.ls / base_s < rel_best.0 {
            rel_best = (st.ls / base_s, t);
        }
        if t == cfg.steps {
            break;
        }
        let [ga, gb, grad_post, grad_pre] = fr.grads(&st, &vl, &sp, &sq)?;
        let cos = f64::midpoint(
            1.0,
            (std::f64::consts::PI * t as f64 / cfg.steps as f64).cos(),
        );
        let lr_b = eta_min + (cfg.lr_binary - eta_min) * cos;
        let lr_s = eta_min + (cfg.lr_scale - eta_min) * cos;
        let [m0, m1, m2, m3] = &mut mom;
        adam(&mut ul, &ga, m0, t + 1, lr_b);
        adam(&mut vl, &gb, m1, t + 1, lr_b);
        adam(&mut sp, &grad_post, m2, t + 1, lr_s);
        adam(&mut sq, &grad_pre, m3, t + 1, lr_s);
    }
    let (us, vs) = (sign_words(&ul), sign_words(&vl));
    let flips = us.iter().zip(&u0).filter(|(a, b)| a != b).count()
        + vs.iter().zip(&v0).filter(|(a, b)| a != b).count();
    let sgn = |x: i64| -> i8 {
        if x < 0 {
            -1
        } else {
            1
        }
    };
    f.u = us.iter().map(|&x| sgn(x)).collect();
    f.v = vs.iter().map(|&x| sgn(x)).collect();
    f.u_latent = ul;
    f.v_latent = vl;
    f.s_post = sp;
    f.s_pre = sq;
    let rel_after = rel_full(f)?;
    Ok(TuneTrace {
        rel_before,
        rel_after,
        rel_best,
        flips,
    })
}

impl NqFactor {
    /// The dense `Ŵ` (`out × in`, Q32.32): `U·V` on the kernel (exact: `±1` words), then the two
    /// diagonal scales. `O(out·in·r)` kernel products plus `O(out·in)`.
    ///
    /// # Errors
    ///
    /// [`NqError::Kernel`] if the product refuses; [`NqError::Range`] off the grid.
    pub fn weight(&self) -> Result<Vec<i64>, NqError> {
        let (o, n, r) = (self.out, self.inn, self.rank);
        let one = |s: i8| i64::from(s) << 32;
        let u: Vec<i64> = self.u.iter().map(|&s| one(s)).collect();
        let vt: Vec<i64> = transpose(&self.v, r, n).iter().map(|&s| one(s)).collect();
        let uv = gemm_q32(&u, o, &vt, n, r, LIMBS)?;
        let mut w = vec![0.0f64; o * n];
        for i in 0..o {
            for j in 0..n {
                w[i * n + j] = from_grid(uv[i * n + j]) * self.s_post[i] * self.s_pre[j];
            }
        }
        grid(&w)
    }
}

#[path = "nanoquant/step.rs"]
mod step;
use step::{device_iters, draw, rank1_step, resident, side_step};

#[cfg(test)]
#[path = "nanoquant/tests.rs"]
mod tests;
