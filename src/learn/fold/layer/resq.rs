//! The `-resq` production row-scale step for the MLP `up` plane: instead of rescaling the
//! reconstruction `Wp` (the `Wp *= s[:, None]` of `-res`), the per-row multiplier enters each
//! stored plane's hierarchical row scale, `rh = quant_r_hier(rh·s)`, costing no bits beyond
//! `quant_r_hier`'s r-bits. Ports `fold_transfer.py`'s `use_res and nm == "up"` block
//! (`spec/fold_transfer.py:1911-1945`).
//!
//! # Q32.32 carrier
//!
//! Every value is a Q32.32 `i64` word: [`ResqInputs`], [`ResqSolution`] and [`rescale_hier`]'s
//! `rh`/`s` arguments included, so nothing here reads or writes an `f64`. Products are the pooled
//! [`gemm_nt`]/[`gemm_nn`], SyrkAcc [`wsyrk`], the Hadamard [`hadamard_into`], `rhs = GoC·1`
//! [`row_dot_fixed`], and the solve [`chol_solve`] with its exact `sqrt(A_ii)` Jacobi column
//! scaling. An operand above `2³⁶` raw (`16` real) is scaled by one exact power of two and shifted
//! back; no `f64` division, no `crate::learn::ridge`.
//!
//! `G` and `P_mat` are SyrkAcc, so `G ⊙ P_mat` is PSD (Schur product theorem) and `GoP + lam·I` is
//! positive-definite for `lam > 0`; a non-positive pivot is [`ResqError::NotPositiveDefinite`].
//!
//! A shape against the declared dims, a zero `inter`/`n`/`hidden`, and any checked integer leaving
//! range are [`ResqError`] refusals; the word domain admits no non-finite value. The `1e-12` `lam`
//! floor is below the quantum `2⁻³² ≈ 2.3e-10`, the grid's zero: a zero diagonal is `lam == 0` and
//! the solve's own positivity refusal. No function here panics on caller input. Grade: `tested`
//! (`tests/learn_fold_resq.rs`); nothing here is `proved`.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the kernels index the row-major `[rows, n]`/`[inter, inter]` buffers by `i/j/r` and \
              name the carriers `W_g0`/`W_u0`/`Wq_g`/`Wq_u`/`P_mat`/`C_mat`/`GoP`/`GoC` as the \
              reference does"
)]

use crate::engine::row_dot_fixed;
use crate::engine::silu_fixed;
use crate::learn::fold::fit::quant::{quant_r_hier, QuantError, RHier};
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::gemm::{gemm_nn, gemm_nt};
use crate::learn::solve::q32::{ilog2_pos, round_q32_even, rshift_rne};
use crate::learn::solve::syrk::{hadamard_into, wsyrk};

/// `1.0` in raw Q32.32.
const ONE: i64 = 1_i64 << 32;

/// A refusal from the `-resq` row-scale step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResqError {
    /// A zero `inter`, `n` or `hidden`, or an empty rescale row set.
    Empty,
    /// A slice length that disagrees with the declared shape.
    ShapeMismatch,
    /// The elementwise-Gram system had a non-positive `Cholesky` pivot.
    NotPositiveDefinite,
    /// A checked integer accumulation, product or narrowing left its range.
    Overflow,
    /// The hierarchical row-scale quantiser refused its input.
    Quant(QuantError),
}

impl From<QuantError> for ResqError {
    fn from(e: QuantError) -> Self {
        ResqError::Quant(e)
    }
}

/// The `-resq` line-search inputs for one `up` plane: `x_slabs` (`rows_s × n` each), the teacher
/// gate/up `w_g0`/`w_u0` and student `wq_g`/`wq_u` (`inter × n`), and the teacher down `w_down`
/// (`hidden × inter`), all row-major.
#[derive(Clone, Copy, Debug)]
pub struct ResqInputs<'a> {
    /// The ffn-stage input slabs, each flattened row-major `[rows, n]`, Q32.32 words.
    pub x_slabs: &'a [&'a [i64]],
    /// The input width `n`.
    pub n: usize,
    /// The teacher gate weight, row-major `[inter, n]`, Q32.32 words.
    pub w_g0: &'a [i64],
    /// The teacher up weight, row-major `[inter, n]`, Q32.32 words.
    pub w_u0: &'a [i64],
    /// The student quantised gate weight, row-major `[inter, n]`, Q32.32 words.
    pub wq_g: &'a [i64],
    /// The current up reconstruction, row-major `[inter, n]`, Q32.32 words.
    pub wq_u: &'a [i64],
    /// The teacher down weight, row-major `[hidden, inter]`, Q32.32 words.
    pub w_down: &'a [i64],
    /// The intermediate width `inter`.
    pub inter: usize,
    /// The hidden width `hidden`.
    pub hidden: usize,
}

/// The `-resq` system and its solution: `gop = G ⊙ P_mat + lam·I`, `rhs = GoC·1 + lam`, `s_raw`
/// solves `gop·s = rhs`, `s_clamped` clamps it to `[0.5, 2.0]`.
#[derive(Clone, Debug)]
pub struct ResqSolution {
    /// `G ⊙ P_mat + lam·I`, row-major `[inter, inter]`, Q32.32 words.
    pub gop: Vec<i64>,
    /// `GoC·1 + lam`, length `inter`, Q32.32 words.
    pub rhs: Vec<i64>,
    /// The ridge `0.05·max(mean(diag(G ⊙ P_mat)), 1e-12)`, Q32.32.
    pub lam: i64,
    /// The unclamped solution `gop⁻¹·rhs`, length `inter`, Q32.32 words.
    pub s_raw: Vec<i64>,
    /// The solution clamped to `[0.5, 2.0]`, length `inter`, Q32.32 words.
    pub s_clamped: Vec<i64>,
}

/// `a` (row-major `[r, c]`) transposed to `[c, r]` on the word grid. `O(r·c)` moves and space.
fn transpose(a: &[i64], r: usize, c: usize) -> Vec<i64> {
    let mut out = vec![0_i64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// `round(a·b)` from the exact `i128` product, one round-to-nearest shift. `O(1)`.
fn mul_q32(a: i64, b: i64) -> Result<i64, ResqError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| ResqError::Overflow)
}

/// `round(num / den)` for a positive `den`, `None` when the result leaves `i64`. `O(1)`.
fn div_q32(num: i128, den: i64) -> Option<i64> {
    let d = i128::from(den);
    let n = if num >= 0 {
        num.checked_add(d / 2)?
    } else {
        num.checked_sub(d / 2)?
    };
    i64::try_from(n / d).ok()
}

/// `x` and the least `k ≥ 0` with `max|x|·2⁻ᵏ < 2³⁶` raw, the products' bound, by one exact power
/// of two; `O(len)`.
fn fit(x: &[i64]) -> Result<(Vec<i64>, u32), ResqError> {
    let m = x
        .iter()
        .fold(0_u128, |m, &v| m.max(u128::from(v.unsigned_abs())));
    let k = ilog2_pos(m).map_or(0, |b| (b + 1).saturating_sub(36));
    x.iter()
        .map(|&v| {
            rshift_rne(i128::from(v), k)
                .and_then(|y| i64::try_from(y).ok())
                .ok_or(ResqError::Overflow)
        })
        .collect::<Result<Vec<i64>, ResqError>>()
        .map(|y| (y, k))
}

/// `x·2ᵏ`, the exact scale [`fit`] removed (the shift is exact; `fit`'s rounding is not undone);
/// `O(len)`.
fn unfix(x: &[i64], k: u32) -> Result<Vec<i64>, ResqError> {
    x.iter()
        .map(|&v| v.checked_shl(k).ok_or(ResqError::Overflow))
        .collect()
}

/// `A·B` ([`gemm_nn`]) or `A·Bᵀ` ([`gemm_nt`]), both operands [`fit`] and the product unfitted;
/// `O(m·k·n)` time, `O(m·n)` space.
fn product(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    nt: bool,
) -> Result<Vec<i64>, ResqError> {
    let ((pa, ka), (pb, kb)) = (fit(a)?, fit(b)?);
    let c = if nt {
        gemm_nt(&pa, &pb, m, k, n)
    } else {
        gemm_nn(&pa, &pb, m, k, n)
    };
    unfix(&c.map_err(|_| ResqError::Overflow)?, ka + kb)
}

/// `xᵀ·x`: the pooled unit-weight [`wsyrk`] of a fitted panel; `O(o·n²)` time, `O(n²)` space.
fn gram(x: &[i64], o: usize, n: usize) -> Result<Vec<i64>, ResqError> {
    let (xs, k) = fit(x)?;
    let g = wsyrk(&xs, None, o, n).map_err(|_| ResqError::Overflow)?;
    unfix(&g, 2 * k)
}

/// One slab's `P` rows: `silu(X·W_gᵀ) ⊙ (X·W_uᵀ)`, row-major `[rows, inter]` Q32.32; `Overflow`
/// from a `SiLU`, product or pooled GEMM.
///
/// # Complexity
/// Time `O(rows·n·inter)` word ops in the two pooled products plus `O(rows·inter)` elementwise,
/// above the `10⁶` pool threshold at any production slab. Space `O(rows·inter)`.
fn project(
    slab: &[i64],
    rows: usize,
    n: usize,
    inter: usize,
    wg: &[i64],
    wu: &[i64],
) -> Result<Vec<i64>, ResqError> {
    let gate = product(slab, wg, rows, n, inter, true)?;
    let up = product(slab, wu, rows, n, inter, true)?;
    let mut out = Vec::with_capacity(rows * inter);
    for k in 0..rows * inter {
        let s = silu_fixed(gate[k]).map_err(|_| ResqError::Overflow)?;
        out.push(mul_q32(s, up[k])?);
    }
    Ok(out)
}

/// The clamped per-row multiplier `s_sol`, length `inter`.
///
/// # Errors
/// As [`resq_solve`].
pub fn resq_scale(inputs: &ResqInputs<'_>) -> Result<Vec<i64>, ResqError> {
    Ok(resq_solve(inputs)?.s_clamped)
}

/// Build and solve the `-resq` elementwise-Gram ridge system on the Q32.32 grid: `G` and the
/// per-slab `P_qᵀ·P_q`/`P_qᵀ·P_0`, `GoP = G ⊙ P_mat`, `GoC = G ⊙ C_mat`, the ridge
/// `0.05·max(mean(diag(GoP)), 1e-12)`, and `s = chol_solve(GoP + lam·I, GoC·1 + lam)`.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `inter`/`n`/`hidden`; [`ResqError::ShapeMismatch`] for a length
/// against the declared dims; [`ResqError::NotPositiveDefinite`] on a non-positive `Cholesky` pivot;
/// [`ResqError::Overflow`] on checked arithmetic.
///
/// # Complexity
/// Temporal: `O(hidden·inter² + Σ_s rows_s·(n·inter + inter²) + inter³)`, dominant term
/// `2·rows·inter²` in the pooled `P_qᵀ·P_q`/`P_qᵀ·P_0` pair. The per-row loops (`rhs = GoC·1` at
/// `inter²`, the row-scale fold at `o·g`) are serial: at or near `10⁶` word ops, cheaper than a
/// dispatch. Spatial: `O(inter² + rows_max·inter)` Q32.32 words.
pub fn resq_solve(inputs: &ResqInputs<'_>) -> Result<ResqSolution, ResqError> {
    let (n, inter, hidden) = (inputs.n, inputs.inter, inputs.hidden);
    if n == 0 || inter == 0 || hidden == 0 {
        return Err(ResqError::Empty);
    }
    let inter_n = inter.checked_mul(n).ok_or(ResqError::ShapeMismatch)?;
    let hidden_inter = hidden.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    let inter2 = inter.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    if inputs.w_g0.len() != inter_n
        || inputs.w_u0.len() != inter_n
        || inputs.wq_g.len() != inter_n
        || inputs.wq_u.len() != inter_n
    {
        return Err(ResqError::ShapeMismatch);
    }
    if inputs.w_down.len() != hidden_inter {
        return Err(ResqError::ShapeMismatch);
    }
    for slab in inputs.x_slabs {
        if slab.len() % n != 0 {
            return Err(ResqError::ShapeMismatch);
        }
    }

    let (down, g0, u0, qg, qu) = (
        inputs.w_down,
        inputs.w_g0,
        inputs.w_u0,
        inputs.wq_g,
        inputs.wq_u,
    );

    let g = gram(down, hidden, inter)?;

    let mut p_mat = vec![0_i64; inter2];
    let mut c_mat = vec![0_i64; inter2];
    for slab in inputs.x_slabs {
        let rows = slab.len() / n;
        if rows == 0 {
            continue;
        }
        let p0 = project(slab, rows, n, inter, g0, u0)?;
        let pq = project(slab, rows, n, inter, qg, qu)?;
        let pp = gram(&pq, rows, inter)?;
        let pqt = transpose(&pq, rows, inter);
        let pc = product(&pqt, &p0, inter, rows, inter, false)?;
        for idx in 0..inter2 {
            p_mat[idx] = p_mat[idx].checked_add(pp[idx]).ok_or(ResqError::Overflow)?;
            c_mat[idx] = c_mat[idx].checked_add(pc[idx]).ok_or(ResqError::Overflow)?;
        }
    }

    // GoP = G ⊙ P_mat; GoC = G ⊙ C_mat; rhs = GoC·1; the ridge from the GoP diagonal.
    let mut gop = g.clone();
    hadamard_into(&mut gop, &p_mat).map_err(|_| ResqError::Overflow)?;
    let mut goc = g;
    hadamard_into(&mut goc, &c_mat).map_err(|_| ResqError::Overflow)?;
    let ones = vec![ONE; inter];
    let mut rhs = vec![0_i64; inter];
    let mut diag_sum: i128 = 0;
    for i in 0..inter {
        rhs[i] = row_dot_fixed(&goc[i * inter..(i + 1) * inter], &ones)
            .map_err(|_| ResqError::Overflow)?;
        diag_sum = diag_sum
            .checked_add(i128::from(gop[i * inter + i]))
            .ok_or(ResqError::Overflow)?;
    }
    let mean = div_q32(
        diag_sum,
        i64::try_from(inter).map_err(|_| ResqError::Overflow)?,
    )
    .ok_or(ResqError::Overflow)?;
    let lam = div_q32(i128::from(mean.max(0)), 20).ok_or(ResqError::Overflow)?;
    for i in 0..inter {
        gop[i * inter + i] = gop[i * inter + i]
            .checked_add(lam)
            .ok_or(ResqError::Overflow)?;
        rhs[i] = rhs[i].checked_add(lam).ok_or(ResqError::Overflow)?;
    }

    let solved = chol_solve(&gop, &rhs, inter, 1).map_err(|e| match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            ResqError::NotPositiveDefinite
        }
        SolveError::Overflow => ResqError::Overflow,
        SolveError::ShapeMismatch | SolveError::Empty => ResqError::ShapeMismatch,
    })?;
    let half = ONE / 2;
    let twice = 2 * ONE;
    Ok(ResqSolution {
        gop,
        rhs,
        lam,
        s_raw: solved.x.clone(),
        s_clamped: solved.x.iter().map(|&v| v.clamp(half, twice)).collect(),
    })
}

/// Fold a per-row multiplier into a plane's stored hierarchical row scale: `rh = quant_r_hier(rh·s)`
/// with `rh` row-major `[o, g]` and `s` length `o`, both Q32.32 words; the product is exact-integer
/// on the grid and feeds [`quant_r_hier`] as words.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `o`/`g`; [`ResqError::ShapeMismatch`] for a length against the
/// shape; [`ResqError::Quant`] if [`quant_r_hier`] refuses.
///
/// # Complexity
/// Temporal: `O(o·g)` serial products plus `quant_r_hier`'s `O(o·g)` — one product per stored entry,
/// which a pool dispatch does not repay; spatial `O(o·g)`.
pub fn rescale_hier(rh: &[i64], s: &[i64], o: usize, g: usize) -> Result<RHier, ResqError> {
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n || s.len() != o {
        return Err(ResqError::ShapeMismatch);
    }
    let mut scaled = vec![0_i64; n];
    let mut err: Option<ResqError> = None;
    let mut i = 0;
    while i < o && err.is_none() {
        let mut j = 0;
        while j < g && err.is_none() {
            match mul_q32(rh[i * g + j], s[i]) {
                Ok(v) => scaled[i * g + j] = v,
                Err(e) => err = Some(e),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// The partial-plane (`pk2_top`) variant of [`rescale_hier`]: the same `s` gathered at `rows`,
/// the plane's global indices.
///
/// # Errors
/// As [`rescale_hier`], with an out-of-range gathered row for `s` as [`ResqError::ShapeMismatch`].
///
/// # Complexity
/// Temporal: `O(rows.len()·g)` serial products plus `quant_r_hier`'s `O(rows.len()·g)`; spatial
/// `O(rows.len()·g)`.
pub fn rescale_hier_top(
    rh: &[i64],
    s: &[i64],
    rows: &[usize],
    g: usize,
) -> Result<RHier, ResqError> {
    let o = rows.len();
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n {
        return Err(ResqError::ShapeMismatch);
    }
    let mut bad = false;
    let mut q = 0;
    while q < rows.len() {
        if rows[q] >= s.len() {
            bad = true;
        }
        q += 1;
    }
    if bad {
        return Err(ResqError::ShapeMismatch);
    }
    let mut scaled = vec![0_i64; n];
    let mut err: Option<ResqError> = None;
    let mut i = 0;
    while i < o && err.is_none() {
        let r = rows[i];
        let mut j = 0;
        while j < g && err.is_none() {
            match mul_q32(rh[i * g + j], s[r]) {
                Ok(v) => scaled[i * g + j] = v,
                Err(e) => err = Some(e),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// The K5 paired receipt for the resq call-site port: the live exact `solve::{gemm,syrk}` bodies
/// against a test-only `engine::kern::limb` port at the production group shape.
///
/// Two limb counts are the register's two rows. [`EXACT_LIMBS`] (`8`) carries the fitted `< 2³⁶`
/// operands exactly, so its `gop`/`rhs`/`lam`/`s_raw`/`s_clamped` are the exact words bit for bit.
/// [`DEFAULT_LIMBS`] is the K3 default (`3`, 21 digits): it is faster, and its outputs are measured
/// against the exact body and against the `2⁻²⁸` contract bound [`resq_solve`]'s own migration was
/// taken at — the per-entry error is printed and asserted.
///
/// The speed assertion is the K5 gate: the port must beat the exact body at this group shape.
/// Run in release: `cargo test --release --lib -- --ignored --nocapture resq_call_site_k5`.
#[cfg(test)]
mod call_site_k5 {
    #![allow(
        clippy::cast_precision_loss,
        reason = "the RNG maps the 53 high bits of a u64 into [0, 1) and the sample mean divides \
                  by a five-run count; both conversions are exact in f64 by construction"
    )]

    use super::*;
    use crate::engine::kern::limb::{prepare, Prepared};
    use crate::engine::{row_dot_fixed, silu_fixed};
    use crate::learn::solve::chol::{chol_solve, SolveError};
    use crate::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};
    use crate::learn::solve::syrk::hadamard_into;
    use std::time::Instant;

    /// The limb count that carries the fitted `< 2³⁶` operands exactly: `8` limbs are 56 bits of
    /// signed-byte digits, above the 36-bit operand and the block exponent's headroom.
    const EXACT_LIMBS: u32 = 8;

    /// The K3 default limb count: 21 digits.
    const DEFAULT_LIMBS: u32 = 3;

    /// The contract bound `resq_solve`'s Q32.32 migration is taken at (`2⁻²⁸`, relative to the
    /// reference's largest entry); see `agrees_with_f64_oracle` in `tests/learn_fold_resq.rs`.
    const BOUND: f64 = 3.725_290_298_461_914e-9;

    /// `a·bᵀ` on a prepared `a` (`m × k`) and the fitted `b` operand, the product unfitted.
    fn port_product_prepared(
        pa: &Prepared,
        ka: u32,
        b: &[i64],
        k: usize,
        n: usize,
        nt: bool,
    ) -> Result<Vec<i64>, ResqError> {
        let (pb, kb) = fit(b)?;
        let c = if nt {
            pa.gemm(&pb, n)
        } else {
            let bt = transpose(&pb, k, n);
            pa.gemm(&bt, n)
        };
        unfix(&c.map_err(|_| ResqError::Overflow)?, ka + kb)
    }

    fn port_syrk(x: &[i64], o: usize, n: usize, limbs: u32) -> Result<Vec<i64>, ResqError> {
        let (xs, k) = fit(x)?;
        let xt = transpose(&xs, o, n);
        let prepared = prepare(&xt, n, o, limbs).map_err(|_| ResqError::Overflow)?;
        let g = prepared.syrk().map_err(|_| ResqError::Overflow)?;
        unfix(&g, 2 * k)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the limb count joins the exact body's `project` parameters"
    )]
    fn port_project(
        slab: &[i64],
        rows: usize,
        n: usize,
        inter: usize,
        wg: &[i64],
        wu: &[i64],
        limbs: u32,
    ) -> Result<Vec<i64>, ResqError> {
        let (ps, ks) = fit(slab)?;
        let prepared = prepare(&ps, rows, n, limbs).map_err(|_| ResqError::Overflow)?;
        let gate = port_product_prepared(&prepared, ks, wg, n, inter, true)?;
        let up = port_product_prepared(&prepared, ks, wu, n, inter, true)?;
        let mut out = Vec::with_capacity(rows * inter);
        for k in 0..rows * inter {
            let s = silu_fixed(gate[k]).map_err(|_| ResqError::Overflow)?;
            out.push(mul_q32(s, up[k])?);
        }
        Ok(out)
    }

    /// One slab's `P_qᵀ·P_q` and `P_qᵀ·P_0` over the prepared limb products, unfixed to words.
    #[allow(
        clippy::too_many_arguments,
        reason = "one slab's gate/up planes and the limb count, as the exact body reads them"
    )]
    fn slab_accumulators(
        slab: &[i64],
        rows: usize,
        n: usize,
        inter: usize,
        wg: &[i64],
        wu: &[i64],
        qg: &[i64],
        qu: &[i64],
        limbs: u32,
    ) -> Result<(Vec<i64>, Vec<i64>), ResqError> {
        let p0 = port_project(slab, rows, n, inter, wg, wu, limbs)?;
        let pq = port_project(slab, rows, n, inter, qg, qu, limbs)?;
        let (pq_fit, kq) = fit(&pq)?;
        let pqt = transpose(&pq_fit, rows, inter);
        let pq_prepared = prepare(&pqt, inter, rows, limbs).map_err(|_| ResqError::Overflow)?;
        let pp = unfix(
            &pq_prepared.syrk().map_err(|_| ResqError::Overflow)?,
            2 * kq,
        )?;
        let (p0_fit, kp) = fit(&p0)?;
        let p0t = transpose(&p0_fit, rows, inter);
        let pc = unfix(
            &pq_prepared
                .gemm(&p0t, inter)
                .map_err(|_| ResqError::Overflow)?,
            kq + kp,
        )?;
        Ok((pp, pc))
    }

    /// [`resq_solve`]'s body with the limb products; the ridge and `Cholesky` solve are the live
    /// exact bodies.
    fn port_resq_solve(inputs: &ResqInputs<'_>, limbs: u32) -> Result<ResqSolution, ResqError> {
        let (n, inter, hidden) = (inputs.n, inputs.inter, inputs.hidden);
        if n == 0 || inter == 0 || hidden == 0 {
            return Err(ResqError::Empty);
        }
        let inter_n = inter.checked_mul(n).ok_or(ResqError::ShapeMismatch)?;
        let hidden_inter = hidden.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
        let inter2 = inter.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
        if inputs.w_g0.len() != inter_n
            || inputs.w_u0.len() != inter_n
            || inputs.wq_g.len() != inter_n
            || inputs.wq_u.len() != inter_n
        {
            return Err(ResqError::ShapeMismatch);
        }
        if inputs.w_down.len() != hidden_inter {
            return Err(ResqError::ShapeMismatch);
        }
        for slab in inputs.x_slabs {
            if slab.len() % n != 0 {
                return Err(ResqError::ShapeMismatch);
            }
        }
        let (down, g0, u0, qg, qu) = (
            inputs.w_down,
            inputs.w_g0,
            inputs.w_u0,
            inputs.wq_g,
            inputs.wq_u,
        );
        let g = port_syrk(down, hidden, inter, limbs)?;
        let mut p_mat = vec![0_i64; inter2];
        let mut c_mat = vec![0_i64; inter2];
        for slab in inputs.x_slabs {
            let rows = slab.len() / n;
            if rows == 0 {
                continue;
            }
            let (pp, pc) = slab_accumulators(slab, rows, n, inter, g0, u0, qg, qu, limbs)?;
            for idx in 0..inter2 {
                p_mat[idx] = p_mat[idx].checked_add(pp[idx]).ok_or(ResqError::Overflow)?;
                c_mat[idx] = c_mat[idx].checked_add(pc[idx]).ok_or(ResqError::Overflow)?;
            }
        }
        let mut gop = g.clone();
        hadamard_into(&mut gop, &p_mat).map_err(|_| ResqError::Overflow)?;
        let mut goc = g;
        hadamard_into(&mut goc, &c_mat).map_err(|_| ResqError::Overflow)?;
        let ones = vec![ONE; inter];
        let mut rhs = vec![0_i64; inter];
        let mut diag_sum: i128 = 0;
        for i in 0..inter {
            rhs[i] = row_dot_fixed(&goc[i * inter..(i + 1) * inter], &ones)
                .map_err(|_| ResqError::Overflow)?;
            diag_sum = diag_sum
                .checked_add(i128::from(gop[i * inter + i]))
                .ok_or(ResqError::Overflow)?;
        }
        let mean = div_q32(
            diag_sum,
            i64::try_from(inter).map_err(|_| ResqError::Overflow)?,
        )
        .ok_or(ResqError::Overflow)?;
        let lam = div_q32(i128::from(mean.max(0)), 20).ok_or(ResqError::Overflow)?;
        for i in 0..inter {
            gop[i * inter + i] = gop[i * inter + i]
                .checked_add(lam)
                .ok_or(ResqError::Overflow)?;
            rhs[i] = rhs[i].checked_add(lam).ok_or(ResqError::Overflow)?;
        }
        let solved = chol_solve(&gop, &rhs, inter, 1).map_err(|e| match e {
            SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
                ResqError::NotPositiveDefinite
            }
            SolveError::Overflow => ResqError::Overflow,
            SolveError::ShapeMismatch | SolveError::Empty => ResqError::ShapeMismatch,
        })?;
        let half = ONE / 2;
        let twice = 2 * ONE;
        Ok(ResqSolution {
            gop,
            rhs,
            lam,
            s_raw: solved.x.clone(),
            s_clamped: solved.x.iter().map(|&v| v.clamp(half, twice)).collect(),
        })
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0
        }
    }

    fn matrix(rng: &mut Rng, r: usize, c: usize) -> Vec<i64> {
        (0..r * c)
            .map(|_| to_grid(rng.next(), GRID_BOUND).expect("fixture on the grid"))
            .collect()
    }

    fn scale(x: &[i64], k: f64) -> Vec<i64> {
        x.iter()
            .map(|&w| to_grid(k * from_grid(w), GRID_BOUND).expect("scaled fixture on the grid"))
            .collect()
    }

    fn max_abs_diff(a: &[i64], b: &[i64]) -> u128 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (i128::from(x) - i128::from(y)).unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    /// The middle of an odd-count sample: the load-robust summary of a per-pair ratio.
    fn median(xs: &[f64]) -> f64 {
        let mut v = xs.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// The five published fields equal: the per-run reproducibility assertion.
    fn same(a: &ResqSolution, b: &ResqSolution) -> bool {
        a.gop == b.gop
            && a.rhs == b.rhs
            && a.lam == b.lam
            && a.s_raw == b.s_raw
            && a.s_clamped == b.s_clamped
    }

    /// The largest `|x|` of a word slice, in words.
    fn max_abs_words(x: &[i64]) -> f64 {
        x.iter()
            .fold(0.0_f64, |m, &v| m.max(v.unsigned_abs() as f64))
    }

    /// The exact body's contraction count at the group shape, the denominator of every ns/MAC
    /// figure here: `2·rows·n·inter` for the gate and up projections and `2·inter²·rows` for the
    /// `P_qᵀ·P_q`/`P_qᵀ·P_0` pair, per slab, plus `hidden²·inter` for `G`.
    fn word_macs(slabs: usize, rows: usize, n: usize, inter: usize, hidden: usize) -> f64 {
        (slabs * (2 * rows * n * inter + 2 * inter * inter * rows) + hidden * hidden * inter) as f64
    }

    #[test]
    #[ignore = "paired K5 receipt; run in release with --ignored"]
    #[allow(
        clippy::too_many_lines,
        reason = "one receipt: the limb-count settlement, the exact-count identity and the paired \
                  timing share the fixture and the four short assertions after it"
    )]
    fn resq_call_site_k5() {
        // One `up` plane's production group: 2 slabs x 128 rows at n=128, inter=16, hidden=8
        // (the 2 seqs x t=128 shape the transfer runs).
        const ROWS: usize = 128;
        const N: usize = 128;
        const INTER: usize = 16;
        const HIDDEN: usize = 8;
        const SLABS: usize = 2;
        const RUNS: usize = 9;

        let mut rng = Rng(0x5e51_0001);
        let xs: Vec<Vec<i64>> = (0..SLABS).map(|_| matrix(&mut rng, ROWS, N)).collect();
        let w_g0 = matrix(&mut rng, INTER, N);
        let w_u0 = matrix(&mut rng, INTER, N);
        let w_down = matrix(&mut rng, HIDDEN, INTER);
        let wq_g = w_g0.clone();
        let wq_u = scale(&w_u0, 0.5);
        let slabs: Vec<&[i64]> = xs.iter().map(std::vec::Vec::as_slice).collect();
        let inputs = ResqInputs {
            x_slabs: &slabs,
            n: N,
            w_g0: &w_g0,
            w_u0: &w_u0,
            wq_g: &wq_g,
            wq_u: &wq_u,
            w_down: &w_down,
            inter: INTER,
            hidden: HIDDEN,
        };

        let want = resq_solve(&inputs).expect("exact solve");
        let rel = |a: &[i64], b: &[i64]| {
            let d = max_abs_diff(a, b) as f64;
            (d, d / (BOUND * max_abs_words(b).max(1.0)))
        };
        // In words, `2^-28` of the clamp's absolute bound is `2^-28·2·2^32`.
        let clamp_bound = BOUND * 2.0 * 4_294_967_296.0;

        // The limb-count settlement: every count from the K3 default to the exact one, its
        // per-entry error against the exact body, and that error as a multiple of the `2^-28`
        // contract bound the Q32.32 migration was taken at.
        println!("resq contract settlement at {SLABS}x{ROWS} rows, n={N}, inter={INTER}, hidden={HIDDEN} (2^-28 = {BOUND:.3e}, clamp bound {clamp_bound:.1} words):");
        let mut worst_ok = 0u32;
        let mut exact_words: Option<ResqSolution> = None;
        let mut admissible = Vec::new();
        for limbs in DEFAULT_LIMBS..=EXACT_LIMBS {
            let got = port_resq_solve(&inputs, limbs).expect("limb solve");
            let (dg, rg) = rel(&got.gop, &want.gop);
            let (dr, rr) = rel(&got.rhs, &want.rhs);
            let (dl, rl) = rel(&[got.lam], &[want.lam]);
            let (ds, rs) = rel(&got.s_raw, &want.s_raw);
            let dcl = max_abs_diff(&got.s_clamped, &want.s_clamped) as f64;
            let inside = dg <= BOUND * max_abs_words(&want.gop)
                && dr <= BOUND * max_abs_words(&want.rhs)
                && dl <= BOUND * max_abs_words(&[want.lam])
                && ds <= BOUND * max_abs_words(&want.s_raw).max(1.0)
                && dcl <= clamp_bound;
            println!(
                "  limbs {limbs:>1}: gop {rg:8.2}x rhs {rr:8.2}x lam {rl:8.2}x s_raw {rs:8.2}x \
                 clamp {:8.2}x  {}",
                dcl / clamp_bound,
                if inside { "inside" } else { "outside" }
            );
            if inside {
                admissible.push(limbs);
                worst_ok = limbs;
            }
            if limbs == EXACT_LIMBS {
                exact_words = Some(got);
            }
        }
        let got8 = exact_words.expect("the exact limb count ran");
        assert_eq!(got8.gop, want.gop, "8-limb gop differs from the exact body");
        assert_eq!(got8.rhs, want.rhs, "8-limb rhs differs from the exact body");
        assert_eq!(got8.lam, want.lam, "8-limb lam differs");
        assert_eq!(got8.s_raw, want.s_raw, "8-limb s_raw differs");
        assert_eq!(got8.s_clamped, want.s_clamped, "8-limb s_clamped differs");
        println!(
            "8-limb per-entry error vs exact: 0 words (bit-identical); smallest admissible limb \
             count {worst_ok}, admissible set {admissible:?}"
        );
        let macs = word_macs(SLABS, ROWS, N, INTER, HIDDEN);
        // Five paired runs of the exact body and every limb count. The box is shared, so the
        // *per-pair* ratio is the reported quantity: each round trips both arms under the same
        // load, and the ratio of the means would credit one arm with a quiet round. The exact
        // arm leads on even rounds and trails on odd rounds, because a burst that lands inside
        // one round otherwise always hits the arms that run after it.
        let arms: Vec<u32> = (DEFAULT_LIMBS..=EXACT_LIMBS).collect();
        let mut te = Vec::with_capacity(RUNS);
        let mut tl: Vec<Vec<f64>> = vec![Vec::with_capacity(RUNS); arms.len()];
        let mut pair: Vec<Vec<f64>> = vec![Vec::with_capacity(RUNS); arms.len()];
        let exact_arm = || {
            let t = Instant::now();
            let a = resq_solve(&inputs).expect("exact solve");
            (t.elapsed().as_secs_f64(), a)
        };
        let limb_arms = |round: &mut Vec<f64>| {
            round.clear();
            for &limbs in &arms {
                let t = Instant::now();
                let b = port_resq_solve(&inputs, limbs).expect("limb solve");
                round.push(t.elapsed().as_secs_f64());
                if limbs == EXACT_LIMBS {
                    assert!(same(&b, &got8), "8-limb port drifted");
                }
            }
        };
        for run in 0..RUNS {
            let mut round = Vec::with_capacity(arms.len());
            let (de, a) = if run % 2 == 0 {
                let (de, a) = exact_arm();
                limb_arms(&mut round);
                (de, a)
            } else {
                limb_arms(&mut round);
                exact_arm()
            };
            assert!(same(&a, &want), "run {run}: exact body not reproducible");
            te.push(de);
            for (k, &dl) in round.iter().enumerate() {
                tl[k].push(dl);
                pair[k].push(dl / de);
            }
        }
        let spread = |xs: &[f64]| {
            (
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let me = mean(&te);
        println!(
            "resq K5 production shape: {SLABS}x{ROWS} rows, n={N}, inter={INTER}, hidden={HIDDEN}, \
             {macs:.0} word-MACs, {RUNS} alternating paired runs"
        );
        let (lo_e, hi_e) = spread(&te);
        println!(
            "exact (i128): mean {:.3} ms  spread [{:.3}, {:.3}] ms  {:.3} ns/word-MAC",
            me * 1e3,
            lo_e * 1e3,
            hi_e * 1e3,
            me * 1e9 / macs
        );
        let mut port_ratio = std::collections::HashMap::new();
        for (k, &limbs) in arms.iter().enumerate() {
            let m = mean(&tl[k]);
            let (lo, hi) = spread(&tl[k]);
            let (plo, phi) = spread(&pair[k]);
            let p = median(&pair[k]);
            port_ratio.insert(limbs, p);
            let ratios: Vec<String> = pair[k].iter().map(|r| format!("{r:.3}")).collect();
            println!(
                "limb {limbs:>1}: mean {:.3} ms  spread [{:.3}, {:.3}] ms  {:.3} ns/word-MAC  \
                 paired {:.3}x (median, spread [{:.3}, {:.3}]) per pair [{}]",
                m * 1e3,
                lo * 1e3,
                hi * 1e3,
                m * 1e9 / macs,
                p,
                plo,
                phi,
                ratios.join(" ")
            );
        }

        // Every K5 violation in one run, so the receipt is complete.
        let mut red: Vec<String> = Vec::new();
        if admissible.is_empty() {
            red.push(format!(
                "no limb count in {DEFAULT_LIMBS}..={EXACT_LIMBS} meets the 2^-28 contract bound"
            ));
        } else {
            let a = admissible[0];
            let p = port_ratio[&a];
            if p >= 1.0 {
                red.push(format!(
                    "at the smallest admissible limb count ({a}) the port is {p:.3}x the exact \
                     body's wall time (paired median): K5 not met"
                ));
            }
        }
        assert!(red.is_empty(), "K5 red: {}", red.join("; "));
    }
}
