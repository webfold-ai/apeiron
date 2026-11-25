//! Direct SPD solve in Q32.32 `i64`: diagonal (Jacobi) equilibration, a blocked right-looking
//! `Cholesky` factor, triangular substitution and one exact-residual refinement step, plus a
//! pooled entry that splits the trailing update across the engine pool.
//!
//! `i64` values are Q32.32: `v` denotes `v / 2^32`, so `ONE = 1 << 32`. Every product of two
//! Q32.32 values is formed in `i128` (Q64.64); a contraction over the contraction index is
//! accumulated in `i128` and rounded to nearest Q32.32 exactly once per stored entry. There is no
//! float in the executing path of this module and no `unsafe`; the only float is the reference
//! oracle in the Q32.32 test.
//!
//! # Method
//!
//! To solve `A X = B` with `A` SPD `[n, n]` and `B` `[n, k]`, row-major:
//!
//! 1. **Equilibration.** `d_i = isqrt(A_ii << 32)` is `sqrt(A_ii)` in Q32.32 (the exact integer
//!    square root, floored, no float). The scaled system is `Â = D⁻¹ A D⁻¹`, `B̂ = D⁻¹ B` with
//!    `D = diag(d_i)`, every quotient rounded to nearest. A non-positive diagonal is refused with
//!    [`SolveError::NonPositiveDiagonal`]; the scaled diagonal is unit to within the root's floor.
//! 2. **Factor.** Blocked right-looking `Cholesky` `Â = L Lᵀ` with block width [`CHOL_BLOCK`]. The
//!    diagonal pivot is the integer square root of the accumulated Q32.32 value; a non-positive
//!    pivot is refused with [`SolveError::NotPositiveDefinite`], never a panic.
//! 3. **Substitute.** `L Lᵀ Ŷ = B̂` by forward then back substitution, then `X = D⁻¹ Ŷ`.
//! 4. **Refine once.** The exact residual `R = B − A X` is accumulated in `i128` with no
//!    intermediate rounding and rounded once to Q32.32. The correction `Δ = A⁻¹ R` is solved through
//!    the same factor and added; [`Solved::residual_max`] is the max `|entry|` of the exact residual
//!    recomputed at the refined `X`.
//!
//! # Partiality
//!
//! Every dimension is checked and refused with [`SolveError::ShapeMismatch`]; `n == 0` is
//! [`SolveError::Empty`]; every operation is checked, so an `i128` accumulation that leaves its
//! range or an `i128 → i64` narrowing that does not fit is [`SolveError::Overflow`]. No function in
//! this module panics on caller input.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::manual_let_else,
    clippy::manual_map,
    clippy::module_name_repetitions,
    reason = "the narrowing `as i64` is range-checked against i64::MAX/MIN immediately before it \
              in `narrow`, and the single-letter names are the recurrence symbols of the blocked \
              `Cholesky` arithmetic (`i/j/k/p/gi/gj`). The closure-free `match` forms are what the \
              pinned Charon/Aeneas extraction fragment models, as in `crate::learn::grad::dense`. \
              The entry point is named `chol_solve` after the dispatch API while the module is \
              `chol`; the pedantic lints stay on for the rest of the crate."
)]

use super::q32::narrow;
use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::numeric::fixed_rsqrt::isqrt;

/// A refusal from a Q32.32 SPD solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// A slice length disagrees with the declared shape (`a` is `[n, n]` or `b`/`x` is `[n, k]`).
    ShapeMismatch,
    /// `n == 0`, so the system is empty and undefined.
    Empty,
    /// A diagonal entry of `A` is non-positive, so `sqrt(A_ii)` (the equilibration scale) is not a
    /// positive real.
    NonPositiveDiagonal,
    /// A `Cholesky` pivot of the equilibrated matrix is non-positive, so `A` is not SPD.
    NotPositiveDefinite,
    /// A checked `i128` accumulation left its range, or an `i128 → i64` narrowing did not fit.
    Overflow,
}

/// A solved SPD system `A X = B`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solved {
    /// The solution `X`, row-major `[n, k]` in Q32.32.
    pub x: Vec<i64>,
    /// The max `|entry|` of the exact residual `B − A X` at the returned `X`, in Q32.32.
    pub residual_max: i64,
}

/// Q32.32 block width of the right-looking `Cholesky` factor.
///
/// One block column of the working matrix is `CHOL_BLOCK` Q32.32 entries = `64 · 8 B = 512 B`, so a
/// block's rows pack into a few 64 B cache lines and the panel solve and trailing update stream a
/// few cache lines per block column while the `O(n²)` working matrix dominates. Block width fixes
/// only the memory traffic: the blocked form issues the same `O(n³/3)` contraction multiply-adds as
/// a flat factor. Chosen in `64..=128`; `64` is the width the retired f64 blocked factor used,
/// kept here as [`CHOL_BLOCK`].
const CHOL_BLOCK: usize = 64;

/// Round-to-nearest shift `(v + 2³¹) >> 32` of an `i128` Q64.64 value to Q32.32, or `None` if the
/// rounding add leaves `i128`.
fn round32(v: i128) -> Option<i128> {
    match v.checked_add(1_i128 << 31) {
        Some(s) => Some(s >> 32),
        None => None,
    }
}

/// Lift a checked `Option` to [`SolveError::Overflow`].
fn ov<T>(o: Option<T>) -> Result<T, SolveError> {
    match o {
        Some(v) => Ok(v),
        None => Err(SolveError::Overflow),
    }
}

/// Signed `round(v / d)` for `d > 0`, nearest with ties away from zero, or `None` if the result or
/// an intermediate leaves `i128`.
fn div_round(v: i128, d: i128) -> Option<i128> {
    if v >= 0 {
        match v.checked_add(d / 2) {
            Some(n) => Some(n / d),
            None => None,
        }
    } else {
        match v.checked_sub(d / 2) {
            Some(n) => Some(n / d),
            None => None,
        }
    }
}

/// [`div_round`] then [`narrow`], the round-to-nearest quotient as a Q32.32 `i64`. Promoted to
/// `pub(super)` for [`super::inv`]'s triangular inverse, which rounds one quotient per entry.
pub(super) fn div_to_i64(num: i128, den: i128) -> Option<i64> {
    match div_round(num, den) {
        Some(q) => narrow(q),
        None => None,
    }
}

/// [`round32`] then [`narrow`], the rounded Q32.32 value as an `i64`.
fn round_to_i64(v: i128) -> Option<i64> {
    match round32(v) {
        Some(r) => narrow(r),
        None => None,
    }
}

/// One stored Q32.32 entry of the factor or of the trailing update:
/// `round32((work_val << 32) − Σ_{p<kb} L[gi, kbase+p]·L[gj, kbase+p])`, or `None` on overflow.
///
/// The two `L` rows are read contiguously over the contraction index; the products and the
/// subtraction are `i128` and there is one rounding, at the end. Shared by the diagonal-block
/// factorization (`kbase = k`, `kb = j`), the panel solve (`kbase = k`, `kb = j`) and the trailing
/// update (`kbase = k`, `kb = kb`).
///
/// # Complexity
///
/// `kb` checked `i128` multiply-subtracts, `O(1)` space. Derivation: one loop over `p < kb`.
fn chol_entry(
    work_val: i64,
    l: &[i64],
    d: usize,
    gi: usize,
    gj: usize,
    kbase: usize,
    kb: usize,
) -> Option<i64> {
    let mut acc = i128::from(work_val) << 32;
    let mut p = 0usize;
    let mut ok = true;
    while p < kb && ok {
        // `i64 × i64` fits `i128` (|product| ≤ 2¹²⁶), so only the subtraction is checked.
        let pr = i128::from(l[gi * d + kbase + p]) * i128::from(l[gj * d + kbase + p]);
        match acc.checked_sub(pr) {
            Some(s) => acc = s,
            None => ok = false,
        }
        p += 1;
    }
    if ok {
        round_to_i64(acc)
    } else {
        None
    }
}

/// Diagonal (Jacobi) equilibration of SPD `a` (`[n, n]`, row-major Q32.32): returns the scale
/// `d_i = isqrt(a_ii << 32)` (the floored Q32.32 `sqrt(a_ii)`) and the scaled matrix
/// `Â_ij = round(a_ij << 64 / (d_i·d_j))`, so `Â = D⁻¹ A D⁻¹`.
///
/// Exposed so a test can assert the scaled diagonal is unit to within the root's floor
/// (`|Â_ii − ONE| ≤ 2⁻³⁰·ONE` for a `≤ 1e4` diagonal range): with `d_i` floored, `d_i²` is within
/// one raw unit of `a_ii << 32` whenever `sqrt(a_ii) ≥ 1`.
///
/// # Errors
///
/// [`SolveError::Empty`] if `n == 0`; [`SolveError::ShapeMismatch`] if `a.len() != n·n`;
/// [`SolveError::NonPositiveDiagonal`] if some `a_ii ≤ 0`; [`SolveError::Overflow`] if `n·n` leaves
/// `usize` or a quotient leaves `i64`.
///
/// # Complexity
///
/// `O(n²)` checked `i128` shifts, products and roundings, `O(n²)` words of output plus the `n`-word
/// scale. Derivation: one pass over the diagonal (`n` roots), one pass over `n²` entries.
pub fn equilibrate(a: &[i64], n: usize) -> Result<(Vec<i64>, Vec<i64>), SolveError> {
    if n == 0 {
        return Err(SolveError::Empty);
    }
    let nn = match n.checked_mul(n) {
        Some(v) => v,
        None => return Err(SolveError::Overflow),
    };
    if a.len() != nn {
        return Err(SolveError::ShapeMismatch);
    }
    // Flag loops: no `return`, `?`, `break` or `continue` may target the function from inside a
    // loop body (the pinned Aeneas fragment rejects it). Each loop deposits the first refusal in
    // `err` and later iterations are guarded off by `err.is_none()`; the `return` sits after.
    let mut d = vec![0i64; n];
    let mut err: Option<SolveError> = None;
    let mut i = 0usize;
    while i < n && err.is_none() {
        let aii = a[i * n + i];
        if aii <= 0 {
            err = Some(SolveError::NonPositiveDiagonal);
        } else {
            match narrow(isqrt(i128::from(aii) << 32)) {
                Some(v) => d[i] = v,
                None => err = Some(SolveError::Overflow),
            }
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut scaled = vec![0i64; nn];
    let mut err: Option<SolveError> = None;
    let mut i = 0usize;
    while i < n && err.is_none() {
        let mut j = 0usize;
        while j < n && err.is_none() {
            let num = i128::from(a[i * n + j]) << 64;
            let den = i128::from(d[i]) * i128::from(d[j]);
            match div_to_i64(num, den) {
                Some(v) => scaled[i * n + j] = v,
                None => err = Some(SolveError::Overflow),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok((scaled, d))
}

/// One row block of the right-looking trailing update:
/// `rows[ri][gj] ⇐ round32((rows[ri][gj] << 32) − Σ_{p<kb} L[gi, kbase+p]·L[gj, kbase+p])` for
/// `gj ∈ [kend, gi]`, `gi = first + ri`. The contraction runs over the just-factored block columns
/// `[kbase, kbase+kb) = [k, kend)`; the trailing columns updated are `[kend, n)`. `rows` is a
/// contiguous block of trailing rows each of stride `d`, columns indexed globally. A plain function
/// with slices in and out, so the pooled entry hands it to a task without a closure capture in the body.
///
/// # Errors
///
/// [`SolveError::Overflow`] if an `i128` accumulation or the stored narrowing leaves its range.
///
/// # Complexity
///
/// `O(nrows·(n−kend)·kb)` checked multiply-subtracts, `O(1)` extra space. Derivation: `nrows` rows,
/// each updating the lower triangle `gj ≤ gi` (at most `n−kend` columns), each `kb` products.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel takes the trailing rows plus the two block offsets (contraction base \
              `kbase`, trailing column start `kend`) and the geometry; a struct would relocate them"
)]
fn chol_trailing_update(
    rows: &mut [i64],
    first: usize,
    kbase: usize,
    kend: usize,
    l: &[i64],
    d: usize,
    kb: usize,
) -> Result<(), SolveError> {
    let nrows = rows.len() / d;
    // Flag loop: the contraction overflow and the store narrowing deposit the first refusal in
    // `err` and every enclosing loop condition and store branch tests `err.is_none()`; the `return`
    // sits after the outermost loop, so no `return`/`?` targets the function from inside a loop
    // body (the pinned Aeneas fragment rejects that). The four checked subtractions are applied one
    // at a time in the original order and all four refusals are `Overflow`, so the carried refusal
    // is the one the returning form produced. On a refusal the row stores are skipped, as the
    // returning form skipped them.
    let mut err: Option<SolveError> = None;
    let mut ri = 0usize;
    while ri < nrows && err.is_none() {
        let r2 = if ri + 1 < nrows { ri + 1 } else { ri };
        let (gi, gi2) = (first + ri, first + r2);
        let li0 = &l[gi * d + kbase..gi * d + kbase + kb];
        let li1 = &l[gi2 * d + kbase..gi2 * d + kbase + kb];
        let mut gj = kend;
        while gj <= gi2 && err.is_none() {
            let gc = if gj < gi2 { gj + 1 } else { gj };
            let lj0 = &l[gj * d + kbase..gj * d + kbase + kb];
            let lj1 = &l[gc * d + kbase..gc * d + kbase + kb];
            let mut c00 = i128::from(rows[ri * d + gj]) << 32;
            let mut c01 = i128::from(rows[ri * d + gc]) << 32;
            let mut c10 = i128::from(rows[r2 * d + gj]) << 32;
            let mut c11 = i128::from(rows[r2 * d + gc]) << 32;
            let mut p = 0usize;
            while p < kb && err.is_none() {
                let (a0, a1) = (i128::from(li0[p]), i128::from(li1[p]));
                let (b0, b1) = (i128::from(lj0[p]), i128::from(lj1[p]));
                // `i64 × i64` fits `i128`, so only the subtractions are checked.
                match c00.checked_sub(a0 * b0) {
                    Some(s) => c00 = s,
                    None => err = Some(SolveError::Overflow),
                }
                match c01.checked_sub(a0 * b1) {
                    Some(s) => c01 = s,
                    None => err = Some(SolveError::Overflow),
                }
                match c10.checked_sub(a1 * b0) {
                    Some(s) => c10 = s,
                    None => err = Some(SolveError::Overflow),
                }
                match c11.checked_sub(a1 * b1) {
                    Some(s) => c11 = s,
                    None => err = Some(SolveError::Overflow),
                }
                p += 1;
            }
            // Store only the lower-triangle entries `gj' ≤ gi'`; clamped duplicates are skipped.
            if err.is_none() {
                if gj <= gi {
                    match round_to_i64(c00) {
                        Some(v) => rows[ri * d + gj] = v,
                        None => err = Some(SolveError::Overflow),
                    }
                }
                if gc != gj && gc <= gi && err.is_none() {
                    match round_to_i64(c01) {
                        Some(v) => rows[ri * d + gc] = v,
                        None => err = Some(SolveError::Overflow),
                    }
                }
                if r2 != ri && err.is_none() {
                    match round_to_i64(c10) {
                        Some(v) => rows[r2 * d + gj] = v,
                        None => err = Some(SolveError::Overflow),
                    }
                    if gc != gj && err.is_none() {
                        match round_to_i64(c11) {
                            Some(v) => rows[r2 * d + gc] = v,
                            None => err = Some(SolveError::Overflow),
                        }
                    }
                }
            }
            gj += 2;
        }
        ri += 2;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The pool task context of [`trailing_update_pool`]: the factor and the block geometry.
struct TrailingRows<'a> {
    /// The factor `L`, row-major `[n, n]`.
    l: &'a [i64],
    /// Row stride of `l` and of the trailing block.
    n: usize,
    /// Column origin of the processed block (`kend − kb`).
    kbase: usize,
    /// First trailing row of the block.
    kend: usize,
    /// Width of the processed block.
    kb: usize,
    /// Lowest-offset refusal recorded by the tasks.
    first: &'a FirstErr<usize, SolveError>,
}

impl ChunkTask<i64> for TrailingRows<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let first = self.kend + start / self.n;
        if let Err(e) =
            chol_trailing_update(piece, first, self.kbase, self.kend, self.l, self.n, self.kb)
        {
            self.first.record(start, e);
        }
    }
}

/// Distribute the trailing update `work[kend.., kend..]` over the engine pool, one contiguous row
/// chunk per `width_global()` participant (disjoint writes, one write per entry, fixed contraction
/// order), so the factor is bit-identical to the serial body. Each entry is written by exactly one
/// task in the same `p` order the serial [`chol_trailing_update`] uses.
///
/// # Errors
///
/// [`SolveError::Overflow`] from the lowest offset that refused.
///
/// # Complexity
///
/// Same total work as the serial trailing update, `O((n−kend)²·kb)` multiply-subtracts, split
/// across the pool; `O(1)` extra space (the pool hands each task its slice in place).
fn trailing_update_pool(
    work: &mut [i64],
    l: &[i64],
    n: usize,
    kend: usize,
    kb: usize,
) -> Result<(), SolveError> {
    let nr = n - kend;
    let kbase = kend - kb;
    let trailing = &mut work[kend * n..n * n];
    let chunk_rows = nr.div_ceil(width_global()).max(1);
    let first: FirstErr<usize, SolveError> = FirstErr::new();
    chunks_mut_global(
        trailing,
        chunk_rows * n,
        &TrailingRows {
            l,
            n,
            kbase,
            kend,
            kb,
            first: &first,
        },
    );
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Blocked right-looking `Cholesky` factor `L` (lower-triangular, row-major `[n, n]`, Q32.32) of the
/// equilibrated SPD `a` (already `D⁻¹ A D⁻¹`). The trailing update runs on the engine pool.
///
/// # Errors
///
/// [`SolveError::ShapeMismatch`] if `a.len() != n·n`; [`SolveError::NotPositiveDefinite`] if a
/// pivot is non-positive; [`SolveError::Overflow`] on any checked operation.
///
/// # Complexity
///
/// `O(n³/3)` checked `i128` multiply-adds (diagonal block, panel solve and trailing update each
/// contract over the processed columns) and `O(n²)` space for `L` and the working copy.
/// Derivation: the trailing update contracts over `kb` per block column for `Σ_blocks (n−k)²`,
/// which is `n³/3`. Promoted to `pub(super)` so [`super::inv`] factors the reversed equilibrated
/// matrix through this body instead of a copy.
/// The lower Cholesky factor `L` of SPD `a` (`L·Lᵀ = a`), row-major `[n, n]`, in Q32.32.
///
/// [`chol_solve`] answers a system; this answers the factor itself, for a caller that needs the
/// factorisation and not the solution (GPTQ's error-feedback, a Heron-style refit, a determinant).
/// The factor is built the same way the solve builds it: Jacobi equilibration `Â = D⁻¹AD⁻¹`
/// ([`equilibrate`]), the blocked factor `Â = L̂L̂ᵀ` ([`factor_scaled`]), and the undo `L = D·L̂`, which
/// is exact because `D` is diagonal.
///
/// # Errors
///
/// [`SolveError::Empty`] if `n == 0`; [`SolveError::ShapeMismatch`] if `a.len() != n·n`;
/// [`SolveError::NonPositiveDiagonal`] if some `a_ii <= 0`; [`SolveError::NotPositiveDefinite`] if a
/// pivot of `Â` is non-positive; [`SolveError::Overflow`] on any checked `i128` operation.
///
/// # Complexity
///
/// `O(n³/3)` checked multiply-adds and `O(n²)` space - one equilibration pass, one factor, one
/// diagonal multiply.
pub fn chol_factor(a: &[i64], n: usize) -> Result<Vec<i64>, SolveError> {
    if n == 0 {
        return Err(SolveError::Empty);
    }
    if a.len() != n * n {
        return Err(SolveError::ShapeMismatch);
    }
    let (scaled, d) = equilibrate(a, n)?;
    let l_hat = factor_scaled(&scaled, n)?;
    let mut l = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let prod = i128::from(d[i]) * i128::from(l_hat[i * n + j]);
            l[i * n + j] = narrow(rshift_rne(prod, 32).ok_or(SolveError::Overflow)?)
                .ok_or(SolveError::Overflow)?;
        }
    }
    Ok(l)
}

pub(crate) fn factor_scaled(a: &[i64], n: usize) -> Result<Vec<i64>, SolveError> {
    if a.len() != n * n {
        return Err(SolveError::ShapeMismatch);
    }
    let mut work = a.to_vec();
    let mut l = vec![0i64; n * n];
    // Flag loop: the diagonal-block factor, the panel solve and the trailing update deposit their
    // first refusal in `err`; no `return`, `?` or `break` sits inside a loop body.
    let mut err: Option<SolveError> = None;
    let mut k = 0usize;
    while k < n && err.is_none() {
        let kb = (n - k).min(CHOL_BLOCK);
        // (1) Factor the diagonal block `L[k..k+kb, k..k+kb]` from `work`.
        let mut i = 0usize;
        while i < kb && err.is_none() {
            let mut j = 0usize;
            while j <= i && err.is_none() {
                let gi = k + i;
                let gj = k + j;
                match chol_entry(work[gi * n + gj], &l, n, gi, gj, k, j) {
                    Some(s) => {
                        if i == j {
                            if s <= 0 {
                                err = Some(SolveError::NotPositiveDefinite);
                            } else {
                                match narrow(isqrt(i128::from(s) << 32)) {
                                    Some(v) => l[gi * n + gj] = v,
                                    None => err = Some(SolveError::Overflow),
                                }
                            }
                        } else {
                            let ljj = i128::from(l[gj * n + gj]);
                            match div_to_i64(i128::from(s) << 32, ljj) {
                                Some(v) => l[gi * n + gj] = v,
                                None => err = Some(SolveError::Overflow),
                            }
                        }
                    }
                    None => err = Some(SolveError::Overflow),
                }
                j += 1;
            }
            i += 1;
        }
        let kend = k + kb;
        // (2) Panel solve `L[kend..n, k..kend]`: rows are independent of each other, columns are
        //     sequential (each uses the diagonal-block rows and its own earlier columns).
        let mut gi = kend;
        while gi < n && err.is_none() {
            let mut j = 0usize;
            while j < kb && err.is_none() {
                let gj = k + j;
                match chol_entry(work[gi * n + gj], &l, n, gi, gj, k, j) {
                    Some(s) => {
                        let ljj = i128::from(l[gj * n + gj]);
                        match div_to_i64(i128::from(s) << 32, ljj) {
                            Some(v) => l[gi * n + gj] = v,
                            None => err = Some(SolveError::Overflow),
                        }
                    }
                    None => err = Some(SolveError::Overflow),
                }
                j += 1;
            }
            gi += 1;
        }
        // (3) Trailing update `work[kend.., kend..] -= L21·L21ᵀ` (lower triangle).
        if kend < n && err.is_none() {
            match trailing_update_pool(&mut work, &l, n, kend, kb) {
                Ok(()) => {}
                Err(e) => err = Some(e),
            }
        }
        k = kend;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(l)
}

/// `(base << 32) − Σ_p xs[p]·ys[p]` in `i128` (`xs.len() == ys.len()`), or `None` on overflow.
///
/// # Complexity
///
/// `O(len)` checked `i128` multiply-subtracts, `O(1)` space.
fn dot_shifted_row(base: i64, xs: &[i64], ys: &[i64]) -> Option<i128> {
    let mut acc = i128::from(base) << 32;
    let mut p = 0usize;
    let mut ok = true;
    while p < xs.len() && ok {
        let pr = i128::from(xs[p]) * i128::from(ys[p]);
        match acc.checked_sub(pr) {
            Some(s) => acc = s,
            None => ok = false,
        }
        p += 1;
    }
    if ok {
        Some(acc)
    } else {
        None
    }
}

/// `(base << 32) − Σ_{p ∈ [p0, p1)} l[p·d + col]·ys[p]` in `i128`, or `None` on overflow; the
/// strided read of the transposed factor `Lᵀ` for the back substitution.
///
/// # Complexity
///
/// `O(p1 − p0)` checked `i128` multiply-subtracts, `O(1)` space.
fn dot_shifted_col(
    base: i64,
    l: &[i64],
    d: usize,
    col: usize,
    ys: &[i64],
    p0: usize,
    p1: usize,
) -> Option<i128> {
    let mut acc = i128::from(base) << 32;
    let mut p = p0;
    let mut ok = true;
    while p < p1 && ok {
        let pr = i128::from(l[p * d + col]) * i128::from(ys[p]);
        match acc.checked_sub(pr) {
            Some(s) => acc = s,
            None => ok = false,
        }
        p += 1;
    }
    if ok {
        Some(acc)
    } else {
        None
    }
}

/// Solve the columns `[c0, c0 + piece.len()/n)` of `A X = B` given the equilibration `d` and the
/// factor `L` of `Â`: equilibrate the right-hand side `B̂ = D⁻¹ B`, forward then back substitute
/// `L Lᵀ Ŷ = B̂`, then unscale `X = D⁻¹ Ŷ`. `b` is row-major `[n, k]` in the **original** scale;
/// the transposed solution `Xᵀ[c·n + i] = X_ic` is written to `piece`, one contiguous `n`-word
/// column per position, so a pool task owns a contiguous column range. The columns are independent:
/// each reads only `L`, `d` and its own gather of `b`, so the split changes no arithmetic.
///
/// # Errors
///
/// [`SolveError::Overflow`] on any checked operation or narrowing.
///
/// # Complexity
///
/// Time: `O((cols)·n²)` checked multiply-subtracts — per column the forward solve is `Σ_i i` and
/// the back solve `Σ_i (n−i)`, both `n²/2`. Space: `O(n)` per-task scratch; the output is in place.
fn solve_cols(
    l: &[i64],
    d: &[i64],
    b: &[i64],
    n: usize,
    k: usize,
    c0: usize,
    piece: &mut [i64],
) -> Result<(), SolveError> {
    let cols = piece.len() / n;
    let mut y = vec![0i64; n];
    // Flag loop: the four passes per column deposit the first refusal in `err`; the `return` sits
    // after the loop, so no `return`/`?` targets the function from inside a loop body.
    let mut err: Option<SolveError> = None;
    let mut t = 0usize;
    while t < cols && err.is_none() {
        let j = c0 + t;
        let yt = &mut piece[t * n..(t + 1) * n];
        // B̂ = D⁻¹ B.
        let mut i = 0usize;
        while i < n && err.is_none() {
            match div_to_i64(i128::from(b[i * k + j]) << 32, i128::from(d[i])) {
                Some(v) => y[i] = v,
                None => err = Some(SolveError::Overflow),
            }
            i += 1;
        }
        // Forward: y_i = (B̂_i − Σ_{p<i} L_ip y_p) / L_ii.
        let mut i = 0usize;
        while i < n && err.is_none() {
            match dot_shifted_row(y[i], &l[i * n..i * n + i], &y[0..i]) {
                Some(s) => match div_to_i64(s, i128::from(l[i * n + i])) {
                    Some(v) => y[i] = v,
                    None => err = Some(SolveError::Overflow),
                },
                None => err = Some(SolveError::Overflow),
            }
            i += 1;
        }
        // Back: y_i = (y_i − Σ_{p>i} L_pi y_p) / L_ii, reading Lᵀ.
        let mut i = n;
        while i > 0 && err.is_none() {
            i -= 1;
            match dot_shifted_col(y[i], l, n, i, &y, i + 1, n) {
                Some(s) => match div_to_i64(s, i128::from(l[i * n + i])) {
                    Some(v) => y[i] = v,
                    None => err = Some(SolveError::Overflow),
                },
                None => err = Some(SolveError::Overflow),
            }
        }
        // X = D⁻¹ Ŷ.
        let mut i = 0usize;
        while i < n && err.is_none() {
            match div_to_i64(i128::from(y[i]) << 32, i128::from(d[i])) {
                Some(v) => yt[i] = v,
                None => err = Some(SolveError::Overflow),
            }
            i += 1;
        }
        t += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The per-call checked multiply-subtract product below which [`solve_rhs`] keeps its columns on
/// the calling thread: one dispatch costs 86–292 µs on the 32-thread box, about `2^20` `i128`
/// multiply-subtracts, so a smaller system cannot repay it. The fit's production call carries
/// `n = 896`, `k = 4864` (`≈ 3.9·10^9`), four orders above the threshold.
const RHS_POOL_MIN_OPS: usize = 1 << 20;

/// The pool task context of [`solve_rhs`]: the factor, the equilibration, the right-hand side and
/// the refusal cell.
struct RhsColumns<'a> {
    l: &'a [i64],
    d: &'a [i64],
    b: &'a [i64],
    n: usize,
    k: usize,
    first: &'a FirstErr<usize, SolveError>,
}

impl ChunkTask<i64> for RhsColumns<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        if let Err(e) = solve_cols(
            self.l,
            self.d,
            self.b,
            self.n,
            self.k,
            start / self.n,
            piece,
        ) {
            self.first.record(start, e);
        }
    }
}

/// Solve `A X = B` given an equilibration `d` and the factor `L` of `Â`: [`solve_cols`] over every
/// right-hand-side column, split into `width_global()` contiguous column chunks through
/// [`chunks_mut_global`], then transposed into the row-major `[n, k]` result. Each column is
/// computed once by one task in the serial order, so the result is bit-identical to the serial body.
///
/// # Errors
///
/// [`SolveError::Overflow`] on any checked operation or narrowing, from the lowest refusing
/// column chunk.
///
/// # Complexity
///
/// Time: `O(n²·k)` checked multiply-subtracts (two triangular solves of `n²/2` per column) plus
/// `O(n·k)` transpose copies, split across the pool. The admission product `n²·k` is
/// [`RHS_POOL_MIN_OPS`]: `n = 512`, `k = 1536` (`4.0·10^8`, the digest fixture) and the production
/// `n = 896`, `k = 4864` both dispatch; a smaller solve runs [`solve_cols`] serially.
/// Space: `O(n·k)` for the transposed solution plus `O(n·k)` for the row-major result and `O(n)`
/// per-task scratch.
fn solve_rhs(l: &[i64], d: &[i64], b: &[i64], n: usize, k: usize) -> Result<Vec<i64>, SolveError> {
    let nk = n.checked_mul(k).ok_or(SolveError::Overflow)?;
    let mut yt = vec![0i64; nk];
    let ops = n.checked_mul(nk);
    if ops.is_none_or(|v| v < RHS_POOL_MIN_OPS) {
        solve_cols(l, d, b, n, k, 0, &mut yt)?;
    } else {
        let chunk_cols = k.div_ceil(width_global()).max(1);
        let first: FirstErr<usize, SolveError> = FirstErr::new();
        chunks_mut_global(
            &mut yt,
            chunk_cols * n,
            &RhsColumns {
                l,
                d,
                b,
                n,
                k,
                first: &first,
            },
        );
        if let Some((_, e)) = first.take() {
            return Err(e);
        }
    }
    let mut out = vec![0i64; nk];
    for c in 0..k {
        for i in 0..n {
            out[i * k + c] = yt[c * n + i];
        }
    }
    Ok(out)
}

/// The exact residual `R = B − A X`, rounded once to Q32.32: `R_ij = round32((B_ij << 32) −
/// Σ_t A_it·X_tj)` with the accumulation in `i128` and no intermediate rounding. `a` is `[n, n]`,
/// `b` and `x` are `[n, k]`, all row-major Q32.32.
///
/// # Errors
///
/// [`SolveError::Overflow`] on any checked operation or narrowing.
///
/// # Complexity
///
/// `O(n²·k)` checked multiply-subtracts and `O(n·k)` space for the residual plus `O(n)` scratch per
/// column. Derivation: `k` columns, each a gather (`n`) and `n` rows of an `n`-term contraction.
fn exact_residual(
    a: &[i64],
    b: &[i64],
    x: &[i64],
    n: usize,
    k: usize,
) -> Result<Vec<i64>, SolveError> {
    let mut r = vec![0i64; n * k];
    // Flag loop: the gather and the row contractions deposit the first refusal in `err`.
    let mut err: Option<SolveError> = None;
    let mut j = 0usize;
    while j < k && err.is_none() {
        let mut xcol = vec![0i64; n];
        let mut t = 0usize;
        while t < n {
            xcol[t] = x[t * k + j];
            t += 1;
        }
        let mut i = 0usize;
        while i < n && err.is_none() {
            match dot_shifted_row(b[i * k + j], &a[i * n..i * n + n], &xcol) {
                Some(acc) => match round_to_i64(acc) {
                    Some(v) => r[i * k + j] = v,
                    None => err = Some(SolveError::Overflow),
                },
                None => err = Some(SolveError::Overflow),
            }
            i += 1;
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(r)
}

/// The max `|entry|` of a Q32.32 residual slice, or `None` if it leaves `i64`.
///
/// # Complexity
///
/// `O(len)` `i128` absolute values, `O(1)` space.
fn max_abs(entries: &[i64]) -> Option<i64> {
    let mut mx: i128 = 0;
    for &v in entries {
        let a = i128::from(v).abs();
        if a > mx {
            mx = a;
        }
    }
    narrow(mx)
}

/// Solve the SPD system `A X = B` in Q32.32: equilibrate, blocked `Cholesky`, substitute, then one
/// exact-residual refinement step. `a` is row-major `[n, n]`, `b` is row-major `[n, k]`, the result
/// [`Solved::x`] is row-major `[n, k]`. The factor's trailing update runs on the engine pool; each
/// entry is accumulated once in a fixed order, so the result is bit-identical to the serial body.
///
/// # Errors
///
/// [`SolveError::Empty`] if `n == 0`; [`SolveError::ShapeMismatch`] if `a.len() != n·n` or
/// `b.len() != n·k`; [`SolveError::NonPositiveDiagonal`] if `A` has a non-positive diagonal;
/// [`SolveError::NotPositiveDefinite`] if the equilibrated matrix has a non-positive `Cholesky`
/// pivot; [`SolveError::Overflow`] on any checked `i128` operation or `i64` narrowing.
///
/// # Complexity
///
/// `O(n³ + n²·k)` checked multiply-adds and `O(n² + n·k)` space. Derivation: one `O(n³/3)` factor
/// ([`factor_scaled`]), two `O(n²·k)` triangular solves ([`solve_rhs`], for the solution and the
/// correction) and two exact residuals `O(n²·k)` ([`exact_residual`]).
pub fn chol_solve(a: &[i64], b: &[i64], n: usize, k: usize) -> Result<Solved, SolveError> {
    if n == 0 {
        return Err(SolveError::Empty);
    }
    let nn = match n.checked_mul(n) {
        Some(v) => v,
        None => return Err(SolveError::Overflow),
    };
    let nk = match n.checked_mul(k) {
        Some(v) => v,
        None => return Err(SolveError::Overflow),
    };
    if a.len() != nn || b.len() != nk {
        return Err(SolveError::ShapeMismatch);
    }
    let (a_hat, d) = equilibrate(a, n)?;
    let l = factor_scaled(&a_hat, n)?;
    let x0 = solve_rhs(&l, &d, b, n, k)?;
    // One refinement step: exact residual, solve the correction through the same factor, add.
    let r0 = exact_residual(a, b, &x0, n, k)?;
    let delta = solve_rhs(&l, &d, &r0, n, k)?;
    let mut x = vec![0i64; nk];
    // Flag loop: the refinement add deposits the first refusal in `err`.
    let mut err: Option<SolveError> = None;
    let mut idx = 0usize;
    while idx < nk && err.is_none() {
        match x0[idx].checked_add(delta[idx]) {
            Some(v) => x[idx] = v,
            None => err = Some(SolveError::Overflow),
        }
        idx += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let rf = exact_residual(a, b, &x, n, k)?;
    let residual_max = ov(max_abs(&rf))?;
    Ok(Solved { x, residual_max })
}

#[cfg(test)]
mod chol_factor_tests {
    use super::*;

    /// `L·Lᵀ = A`, in Q32.32, for a matrix with a spread diagonal (which exercises the
    /// equilibration undo).
    #[test]
    fn chol_factor_reconstructs() {
        let n = 4usize;
        let one = 1i64 << 32;
        // A = [[6,1,0,0],[1,9,1,0],[0,1,12,1],[0,0,1,20]] in Q32.32, scaled by `one`
        let raw = [6i64, 1, 0, 0, 1, 9, 1, 0, 0, 1, 12, 1, 0, 0, 1, 20];
        let a: Vec<i64> = raw.iter().map(|v| v * one).collect();
        let l = chol_factor(&a, n).expect("factor");
        for i in 0..n {
            for j in 0..n {
                let mut acc: i128 = 0;
                for k in 0..n {
                    acc += i128::from(l[i * n + k]) * i128::from(l[j * n + k]);
                }
                let got = narrow(rshift_rne(acc, 32).expect("shift")).expect("narrow");
                let want = a[i * n + j];
                let tol = one / 1024; // a few LSB of Q32.32 against an entry of order 6
                assert!(
                    (got - want).abs() <= tol,
                    "[{i}][{j}]: {got} vs {want} (tol {tol})"
                );
            }
        }
    }
}
