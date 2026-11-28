//! Fold-transfer refit steps over the 128-block Hadamard basis.
//!
//! Ports the continuous steps of the frozen `fold_transfer.py` (`spec/fold_transfer.py:426-490`) and
//! their transforms: `J` (434), `compose` (438), `fwht128` (441), `had_gram` (449), `c_step` (453),
//! `r_step` (463), `d_step` (478), with the `run_variant` forward model (886)
//! `recon(d) = compose(fwht128(S_rot ⊙ d), r, c)`. `W`, `Ws` are `O × n` row-major, `n = G·128`;
//! `S` is a sign matrix, `r` an `O × G` per-group scale, `c` a length-`n` column scale, `Hl` an
//! `n × n` metric.
//!
//! - `compose(S, r, c)[o,i] = S[o,i]·r[o, i/128]·c[i]`.
//! - `objective(Ws, Wp, Hl) = tr((Ws−Wp)·Hl·(Ws−Wp)ᵀ)`.
//! - `c_step`: `A·c = b`, `A[i,j] = Hl[i,j]·(Shᵀ diag(g) Sh)[i,j]`, `Sh = S⊙r_rep`,
//!   `b[i] = Σ_o (Ws·Hl)[o,i]·Sh[o,i]·g[o]`; `r_step`: one `G × G` system per out row.
//! - `d_step`: `H_rot = had_gram(Hl ⊙ c cᵀ)`, `G_rot = fwht128((Ws·Hl) ⊙ c)`,
//!   `A[i,j] = H_rot[i,j]·(Bᵀ diag(g) B)[i,j]` with `B = S_rot⊙r_rep`.
//! # The `Q32.32` carrier and the pool
//! Every operand, intermediate and result is a `Q32.32` `i64` word (`v/2³²` is the real). Every
//! **contraction** — each matrix product, Gram, batched dot and row energy — runs on the engine
//! kernel through [`limb`] at [`LIMBS`] = 3 block-float limbs per operand (`gemm_q32`, `gemv_q32`,
//! `rowdot_q32`); the elementwise word products (`mul_word`, `hadamard_into`, the row and column
//! scales) stay exact `Q32.32` and round once via
//! [`crate::learn::solve::q32::round_q32_even`]. There is no floating-point operation on the
//! computing path: the module's boundary is words in and words out. Pool dispatch is reserved for
//! matrix-sized work: [`limb::gemm_q32`], [`hadamard_into`], [`chol_solve`] and
//! [`crate::engine::fwht_fixed`] inside [`fwht_words`] pool in their own drivers (a nested
//! dispatch would deadlock); every per-row, per-column, per-entry and transpose loop is serial.
//!
//! # Accuracy, stated
//! The limb expansion is a 21-bit balanced-digit mantissa per 32-word block under one power-of-two
//! block scale, so a contraction entry carries a relative truncation of about `2⁻¹⁶` of the
//! block's `2^{Ea+Eb}` word instead of the exact body's `2⁻³²` grid quantum. Measured against the
//! pre-port exact body on this module's fixtures (`tests/learn_fold_steps.rs`, R5): `1.65e-5` on
//! `c_step`'s solution, `2.61e-5` on `d_step`'s, `9.48e-7` on `r_step`'s, `2.07e-6` on the
//! recovered `r`, `1.60e-5` on the recovered `c`. The bound itself is `tested`, not `proved`
//! (`src/engine/kern/limb.rs`).
//!
//! # Partiality and trust boundary
//! A shape disagreement, `n` not a multiple of 128 and a checked overflow are refused with
//! [`StepsError::ShapeMismatch`]; a non-SPD `Hl` is refused with
//! [`StepsError::NotPositiveDefinite`]. A normal system whose `Cholesky` factor refuses at the
//! script's ridge `1e-12·mean(diag A)` — a direction whose curvature is below the limb product's
//! own per-entry truncation is then supported by the ridge alone — is solved once more with the
//! ridge raised to [`limb_error_floor`]; a system that is still not positive-definite at that floor
//! is refused, so the raise is a stated ridge floor, not a swallowed refusal. Where the first
//! factor succeeds, the words are the script's. No function panics on caller input. The bodies are `tested`
//! (`tests/learn_fold_steps.rs`), not extracted, so not `proved`; the primitives are `tested` too.
//! The pool is the thread trust boundary; no performance claim is made.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "flat grid kernels index `[o, n]` buffers by `o/i/j/gg/k/m`; a shift exponent is a \
              small `i32`; steps take their operands directly rather than through a bundle struct."
)]

use crate::engine::kern::limb::{self, LimbError};
use crate::engine::{fwht_fixed, HadamardError};
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::q32::{self, ilog2_pos, rshift_rne};
use crate::learn::solve::syrk::{hadamard_into, SyrkError};

/// Hadamard block width: one normalised `H₁₂₈/√128` block per 128 columns.
pub const GROUP: usize = 128;

/// Exponent a shifted panel is filled to: `max|v| ∈ (2³⁴, 2³⁵]` raw, one bit below the historical
/// `2³⁶` operand bound, whether that is a down-shift or an up-shift.
const BOUND_BITS: u32 = 35;
/// `1.0` as a raw word: the unit row weight.
const UNIT_RAW: i64 = 1_i64 << 32;
/// `round(1e-12·2⁶²)`: the ridge coefficient in fixed point one scale coarser than `Q32.32`.
const EPS_Q: i64 = 4_611_686;
/// Block-float limbs per operand for every product routed through [`limb`] (the fp32 mantissa).
const LIMBS: u32 = 3;

fn map_limb(_: LimbError) -> StepsError {
    StepsError::ShapeMismatch
}

// Grid scalar ops, `O(1)` each.

/// `a·b` for two words: the exact `i128` product rounded once to `Q32.32`.
fn mul_word(a: i64, b: i64) -> Result<i64, StepsError> {
    q32::round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| StepsError::ShapeMismatch)
}

/// `v·2⁻ᵏ`: a round-to-nearest shift for `k > 0`, an exact checked left shift for `k < 0`.
#[allow(
    clippy::comparison_chain,
    reason = "a `match` on `k.cmp(&0)` would put `i32::cmp` and `Ordering` in the extracted body; \
              the three-arm chain is the extracted form"
)]
fn shift_word(v: i64, k: i32) -> Result<i64, StepsError> {
    if k > 0 {
        match rshift_rne(i128::from(v), k as u32) {
            Some(x) => q32::narrow(x).ok_or(StepsError::ShapeMismatch),
            None => Err(StepsError::ShapeMismatch),
        }
    } else if k < 0 {
        if k <= -128 {
            return Err(StepsError::ShapeMismatch);
        }
        q32::narrow(i128::from(v) << ((-k) as u32)).ok_or(StepsError::ShapeMismatch)
    } else {
        Ok(v)
    }
}

/// The signed `k` with `max·2⁻ᵏ ∈ (2³⁴, 2³⁵]` raw: the integer twin of the `f64` panel-shift rule,
/// extended to fill the range from below (`k < 0`). The shift is an exact power of two on the stored
/// words, so no product changes; filling from either side is what keeps a squared entry (`Sh²`,
/// still `Q32.32`) above the grid quantum: a panel left under-filled loses those squares to one
/// rounding. A panel with `max = 0` needs no shift.
fn panel_shift_words(max_mag: u128) -> i32 {
    match ilog2_pos(max_mag) {
        Some(bits) => bits as i32 + 1 - BOUND_BITS as i32,
        None => 0,
    }
}

/// The exact power-of-two down-shift `s` that keeps a Hadamard product `A ⊙ H` a `Q32.32` word:
/// with `bl_a`, `bl_h` the bit lengths of the two maxima, the product's word needs
/// `bl_a + bl_h − 32` bits and `s = max(0, bl_a + bl_h − 94)` leaves it at most `2^62`, one bit
/// below `i64::MAX`. `0` when the product already fits, so such an input is unchanged bit for bit.
/// The caller rescales the normal system's matrix and right-hand side by the same `2^-s`, which
/// leaves the solution unchanged: a rescale, never a clamp.
///
/// # Complexity
/// `O(1)`.
fn product_shift(a: u128, h: u128) -> i32 {
    let bl = |m: u128| ilog2_pos(m).map_or(0, |b| b as i32 + 1);
    (bl(a) + bl(h) - 32 - 62).max(0)
}

/// The ridge `round(eps·mean)` in raw units for a mean diagonal word `mean`.
fn ridge_word(mean: i128) -> Result<i64, StepsError> {
    let scaled = mean
        .checked_mul(i128::from(EPS_Q))
        .ok_or(StepsError::ShapeMismatch)?;
    match rshift_rne(scaled, 62) {
        Some(v) => q32::narrow(v).ok_or(StepsError::ShapeMismatch),
        None => Err(StepsError::ShapeMismatch),
    }
}

/// A refusal from a fold-transfer step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepsError {
    /// A length or dimension disagrees with the declared shape (a Hadamard step needs a multiple of
    /// [`GROUP`]), or a value left the grid's `i64` range.
    ShapeMismatch,
    /// A reduced matrix was not positive-definite: a non-SPD metric `hl`, or a rank-deficient
    /// normal system.
    NotPositiveDefinite,
}

/// [`SyrkError`] as a [`StepsError`]: every arm is the domain refusal.
fn map_syrk(e: SyrkError) -> StepsError {
    match e {
        SyrkError::ShapeMismatch
        | SyrkError::ZeroN
        | SyrkError::ZeroO
        | SyrkError::InputBound
        | SyrkError::Overflow => StepsError::ShapeMismatch,
    }
}

/// [`SolveError`] as a [`StepsError`]: a pivot or equilibration refusal is non-SPD, the rest shape.
fn map_solve(e: SolveError) -> StepsError {
    match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            StepsError::NotPositiveDefinite
        }
        SolveError::ShapeMismatch | SolveError::Empty | SolveError::Overflow => {
            StepsError::ShapeMismatch
        }
    }
}

/// The row weight on the grid: `None` is the unit weight `1` in raw words, `Some` is brought by
/// the exact `2⁻ᵏ` shift to the same `2³⁵` magnitude the panels are filled to. The shift cancels
/// between `A` and `b`, so the solution is unchanged. The shift is down-only: a weight at or below
/// the unit weight keeps its raw words, so the all-ones weight is `UNIT_RAW` in both arms and
/// `Some(&[1; o])` is the unweighted step bit for bit. A tiny weight therefore multiplies an
/// already-resolved Gram, where the panel fill is what keeps the square; up-shifting it would only
/// change the relative size of the ridge floor.
///
/// # Complexity
/// `O(o)` shifts, `O(o)` words.
fn row_weight(g_row: Option<&[i64]>, o: usize) -> Result<Vec<i64>, StepsError> {
    let Some(g) = g_row else {
        return Ok(vec![UNIT_RAW; o]);
    };
    if g.len() != o {
        return Err(StepsError::ShapeMismatch);
    }
    let mut v = g.to_vec();
    let k = panel_shift_words(max_mag(&v)).max(0);
    shift_in_place(&mut v, k)?;
    Ok(v)
}

/// # Complexity
/// Add `ridge` to the diagonal of a row-major `n × n` matrix: `O(n)` checked adds, `O(1)` space.
fn add_ridge(a: &mut [i64], n: usize, ridge: i64) -> Result<(), StepsError> {
    for i in 0..n {
        let d = &mut a[i * n + i];
        *d = d.checked_add(ridge).ok_or(StepsError::ShapeMismatch)?;
    }
    Ok(())
}

/// # Complexity
/// `max |x[i]|`: `O(len)` comparisons, `O(1)` extra space.
fn max_mag(x: &[i64]) -> u128 {
    x.iter()
        .fold(0u128, |m, &v| m.max(u128::from(v.unsigned_abs())))
}

/// # Complexity
/// `v·2⁻ᵏ` in place: `O(len)` shifts, `O(1)` extra space.
fn shift_in_place(x: &mut [i64], k: i32) -> Result<(), StepsError> {
    if k == 0 {
        return Ok(());
    }
    for v in x {
        *v = shift_word(*v, k)?;
    }
    Ok(())
}

/// # Complexity
/// `v·2⁻ᵏ` into a fresh slice: one copy plus [`shift_in_place`], `O(len)` time and words.
fn shift_copy(x: &[i64], k: i32) -> Result<Vec<i64>, StepsError> {
    let mut y = x.to_vec();
    shift_in_place(&mut y, k)?;
    Ok(y)
}

/// # Complexity
/// `Σ_i a[i·n + i]`: `O(n)` additions, `O(1)` space. No overflow: `n ≤ 2⁶³` `i64` terms fit `i128`.
fn diag_sum(a: &[i64], n: usize) -> i128 {
    (0..n).map(|i| i128::from(a[i * n + i])).sum()
}

/// The exact per-column fill of a shifted panel `x` (`o × n`, already at the global panel shift):
/// column `j` is shifted left by `e_j ≥ 0` so its own `max|x[·,j]| ∈ (2³⁴, 2³⁵]` raw. The Gram
/// becomes `D·A·D` with `D = diag(2^e)` (power-of-two equilibration, van der Sluis 1969): a column
/// orders of magnitude under the panel max keeps its square `x_j²` above the quantum, where the
/// global shift alone rounds its diagonal to zero while its cross terms survive and the rounded `A`
/// is indefinite. A column already at the panel max has `e_j = 0`; a zero column keeps `e_j = 0`.
///
/// # Complexity
/// `O(o·n)` comparisons and shifts, `O(n)` words for `e`.
fn fill_columns(x: &mut [i64], o: usize, n: usize) -> Result<Vec<i32>, StepsError> {
    let mut e = vec![0_i32; n];
    let mut ok = true;
    for j in 0..n {
        if ok {
            let mut m: u128 = 0;
            for q in 0..o {
                m = m.max(u128::from(x[q * n + j].unsigned_abs()));
            }
            let ej = (-panel_shift_words(m)).max(0);
            e[j] = ej;
            for q in 0..o {
                match shift_word(x[q * n + j], -ej) {
                    Ok(v) => x[q * n + j] = v,
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(e)
    } else {
        Err(StepsError::ShapeMismatch)
    }
}

/// The mean of `diag(A)` for `A'' = D·A·D` read back through `D⁻²`: `O(n)` rounded shifts.
fn filled_diag_mean(a: &[i64], e: &[i32], n: usize) -> Result<i128, StepsError> {
    let mut acc = 0i128;
    for (j, &ej) in e.iter().enumerate() {
        acc += i128::from(shift_word(a[j * n + j], 2 * ej)?);
    }
    Ok(acc / n as i128)
}

/// The ridge's floor on the filled diagonal of a normal system: the limb product's own per-entry
/// truncation bound at the matrix's maximum, in `Q32.32` words — `⌈P/2⌉ + 1 + B·(1 + ⌈C_L·2^{Ea+Eb−32}⌉)`
/// with `P = L(L+1)/2`, `C_L = (33 + 16(L−1))·2⁻⁷ᴸ` and `B = ⌈inner/32⌉`, the bound
/// [`gemm_q32`](crate::engine::kern::limb::gemm_q32) states for the product that built the matrix.
///
/// A ridge below this floor is smaller than the error the product carries into the same entries, so
/// the elimination cannot certify `s > 0` for a direction whose curvature is under that error: a
/// normal system the exact `i128` body factorises is then refused `NotPositiveDefinite`. The
/// alternating steps' Hadamard-domain matrices are the case: `A = G ⊙ H_rot` carries `H_rot`'s own
/// dynamic range, so its smallest diagonal is orders below the ridge `1e-12·mean(diag A)` the
/// reference script uses, and every direction at that scale is supported by the ridge alone. The
/// floor only binds where the reference ridge is below the port's own resolution; where the
/// system's curvature is above the floor the fit is unchanged to the ridge's relative size.
///
/// # Complexity
/// `O(1)`.
fn limb_error_floor(inner: usize, max_abs: u128) -> Result<i64, StepsError> {
    if max_abs == 0 {
        return Ok(1);
    }
    let pairs = u128::from(LIMBS) * u128::from(LIMBS + 1) / 2;
    let coeff = 33 + 16 * (u128::from(LIMBS) - 1);
    let trunc = ((coeff * max_abs) >> (7 * LIMBS)) + 1;
    let blocks = u128::from((inner.div_ceil(32)) as u64);
    let total = pairs.div_ceil(2) + 1 + blocks * (1 + trunc);
    i64::try_from(total).map_err(|_| StepsError::ShapeMismatch)
}

/// Add `ridge·D²` to the diagonal of `A'' = D·A·D`: the ridge `ridge·I` of `A`, exact.
/// `O(n)` checked shifts and adds, `O(1)` space.
fn add_filled_ridge(a: &mut [i64], e: &[i32], n: usize, ridge: i64) -> Result<(), StepsError> {
    for (j, &ej) in e.iter().enumerate() {
        let d = &mut a[j * n + j];
        *d = d
            .checked_add(shift_word(ridge, -2 * ej)?)
            .ok_or(StepsError::ShapeMismatch)?;
    }
    Ok(())
}

/// Raise the ridge a factorisation already carries — `ridge·2^{2e_j}` on column `j` — to `floor`
/// wherever it is below it: the smallest diagonal increment that puts every column's ridge above
/// the limb product's own per-entry truncation ([`limb_error_floor`]).
///
/// # Complexity
/// `O(n)` checked shifts and adds, `O(1)` space.
fn raise_ridge_to_floor(
    a: &mut [i64],
    e: &[i32],
    n: usize,
    ridge: i64,
    floor: i64,
) -> Result<(), StepsError> {
    for (j, &ej) in e.iter().enumerate() {
        let have = shift_word(ridge, -2 * ej)?;
        if have < floor {
            let d = &mut a[j * n + j];
            *d = d
                .checked_add(floor - have)
                .ok_or(StepsError::ShapeMismatch)?;
        }
    }
    Ok(())
}

/// The `Cholesky` solve of a topped normal system, retried once with the ridge raised to
/// [`limb_error_floor`] when the script's own ridge is below the port's resolution.
///
/// The script's ridge is `1e-12·mean(diag A)`; a direction whose curvature is under the product's
/// per-entry truncation is then supported by the ridge alone, and the elimination cannot certify a
/// positive pivot for it, so a system the exact `i128` body factorises is refused
/// [`StepsError::NotPositiveDefinite`]. The retry raises the ridge to the floor for exactly that
/// case and solves again. It is not a swallowed refusal: a system that is not positive-definite at
/// the floored ridge is still refused. Where the first factorisation succeeds the returned words
/// are the ones the script's ridge gives, unchanged.
///
/// # Complexity
/// `O(n³/3 + n²)` (one factorisation, two triangular solves, two exact residuals) plus, on the
/// refusal path only, `O(n)` diagonal raises and a second factorisation. Space `O(n·k)` for the
/// solution plus the factor's `O(n²)` inside [`chol_solve`].
fn solve_with_ridge_floor(
    a: &mut [i64],
    b: &[i64],
    e: &[i32],
    n: usize,
    inner: usize,
    ridge: i64,
) -> Result<Vec<i64>, StepsError> {
    match chol_solve(a, b, n, 1) {
        Ok(solved) => Ok(solved.x),
        Err(SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal) => {
            let floor = limb_error_floor(inner, max_mag(a))?;
            raise_ridge_to_floor(a, e, n, ridge, floor)?;
            match chol_solve(a, b, n, 1) {
                Ok(solved) => Ok(solved.x),
                Err(other) => Err(map_solve(other)),
            }
        }
        Err(other) => Err(map_solve(other)),
    }
}

/// The normalised 128-block transform on words, one `H₁₂₈/√128` image per 128-chunk: the engine's
/// [`fwht_fixed`] is the one implementation (its `C[7]` is `round(2³²/√128)`).
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a non-multiple length, a butterfly overflow or a rounding.
///
/// # Complexity
/// Temporal: `O(x.len())` — `448` add/sub plus `128` rounded multiplies per 128-chunk. One
/// `had_gram` pass at production `n = 4864` is `n²/128 ≈ 1.8·10^5` chunks × `576` word ops ≈
/// `1.1·10^8` word ops per call. Spatial: `O(1)` in place.
pub fn fwht_words(x: &mut [i64]) -> Result<(), StepsError> {
    if !x.len().is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    fwht_fixed(x, None, GROUP).map_err(map_hadamard)
}

/// [`HadamardError`] as a [`StepsError`]: every arm is the shape/range refusal.
fn map_hadamard(_: HadamardError) -> StepsError {
    StepsError::ShapeMismatch
}

/// # Complexity
/// Transpose of a row-major `cols × rows` matrix into `rows × cols`, serially:
/// `rows·cols` copies per call; an `n × n` transpose at production `n = 4864` is `2.4·10^7`
/// copies, two calls per [`had_gram`]. Spatial: `rows·cols` words.
///
/// The two arguments name the *output* shape (`t[r·cols + c] = m[c·rows + r]`); a square call is
/// its own inverse, a rectangular one needs the source's shape swapped.
fn transpose_grid(m: &[i64], rows: usize, cols: usize) -> Vec<i64> {
    let mut t = vec![0i64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[r * cols + c] = m[c * rows + r];
        }
    }
    t
}

/// The congruent Gram `R·H·R`, `R` the block-diagonal `H₁₂₈/√128`: two transforms about a transpose.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] from a transform.
///
/// # Complexity
/// Temporal: `O(n²·log 128)` — `7` stages over each of the `n²/128` row-blocks and column-blocks,
/// pooled. Spatial: `O(n²)` words per copy and transpose.
fn had_gram_grid(h: &[i64], n: usize) -> Result<Vec<i64>, StepsError> {
    let mut t = h.to_vec();
    fwht_words(&mut t)?;
    let mut u = transpose_grid(&t, n, n);
    fwht_words(&mut u)?;
    Ok(transpose_grid(&u, n, n))
}

/// The `f64` face of [`had_gram_grid`] is gone: the metric transform takes and returns words.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a length, a non-multiple `n` or a transform refusal.
///
/// # Complexity
/// Temporal: `O(n²·log 128)`; spatial `O(n²)` words.
pub fn had_gram(h: &[i64], n: usize) -> Result<Vec<i64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) || h.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    had_gram_grid(h, n)
}

/// # Complexity
/// `B[o,i] = s[o,i]·r[o, i/128]`: `O(o·n)` rounded multiplies, `O(o·n)` words of output.
fn group_scale(s: &[i64], r: &[i64], o: usize, n: usize, g: usize) -> Result<Vec<i64>, StepsError> {
    let mut b = vec![0i64; o * n];
    let mut ok = true;
    for oo in 0..o {
        for i in 0..n {
            if ok {
                match mul_word(s[oo * n + i], r[oo * g + i / GROUP]) {
                    Ok(v) => b[oo * n + i] = v,
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(b)
    } else {
        Err(StepsError::ShapeMismatch)
    }
}

/// # Complexity
/// [`group_scale`] then the column scale (`s·r` then `(s·r)·c`): two `O(o·n)` passes, `O(o·n)` words.
fn compose_grid(
    s: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
    g: usize,
) -> Result<Vec<i64>, StepsError> {
    col_scale(&group_scale(s, r, o, n, g)?, c, n)
}

/// The reconstruction `compose(S, r, c)[o,i] = S[o,i]·r[o, i/128]·c[i]`, `O × n` row-major: `S` is
/// `O × n`, `r` is `O × G` and `c` is length `n`, each entry the three-word product rounded once.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape or length refusal.
///
/// # Complexity
/// Temporal: `O(O·n)` rounded multiplies. Spatial: `O(O·n)` words.
pub fn compose(
    s: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if s.len() != o * n || r.len() != o * g || c.len() != n {
        return Err(StepsError::ShapeMismatch);
    }
    compose_grid(s, r, c, o, n, g)
}

/// # Complexity
/// `x ⊙ c`, `c` broadcast down the columns: `O(len)` rounded multiplies, `O(len)` words.
fn col_scale(x: &[i64], c: &[i64], n: usize) -> Result<Vec<i64>, StepsError> {
    let mut out = vec![0i64; x.len()];
    let mut ok = true;
    for i in 0..x.len() {
        if ok {
            match mul_word(x[i], c[i % n]) {
                Ok(v) => out[i] = v,
                Err(_) => ok = false,
            }
        }
    }
    if ok {
        Ok(out)
    } else {
        Err(StepsError::ShapeMismatch)
    }
}

/// # Complexity
/// `hl ⊙ c cᵀ`, one rounding per factor: `O(n²)` rounded multiplies, `O(n²)` words of output.
fn metric_scale(hl: &[i64], c: &[i64], n: usize) -> Result<Vec<i64>, StepsError> {
    let mut out = vec![0i64; n * n];
    let mut ok = true;
    for i in 0..n {
        for j in 0..n {
            if ok {
                match mul_word(hl[i * n + j], c[i]) {
                    Ok(a) => match mul_word(a, c[j]) {
                        Ok(v) => out[i * n + j] = v,
                        Err(_) => ok = false,
                    },
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(out)
    } else {
        Err(StepsError::ShapeMismatch)
    }
}

/// `A = Shᵀ diag(w) Sh` on the limb kernel: `Sh` is transposed once into an `n × o` panel whose
/// rows are the output's Gram rows, the weight scales that transpose by columns (the elementwise
/// word product, D5), and one [`limb::gemm_q32`] carries the `o`-term contraction.
///
/// The exact Gram is symmetric; the kernel's two operand planes take their block exponent per row,
/// so the two halves of the product can differ by a rounding when the weight is not the unit. The
/// upper triangle is therefore the result and the lower is its mirror, `O(n²)` copies.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] from the length check or a kernel refusal (`Range`, `Shape`).
///
/// # Complexity
/// Temporal `O(o·n)` transposes and column scales plus `O(o·n²)` word multiply-adds on the kernel
/// (pooled inside [`limb::gemm_q32`]); spatial `O(o·n)` for the two panels and `O(n²)` output.
fn wsyrk_gemm(sh: &[i64], w: &[i64], o: usize, n: usize) -> Result<Vec<i64>, StepsError> {
    let sh_t = transpose_grid(sh, n, o);
    let w_t = col_scale(&sh_t, w, o)?;
    let mut a = limb::gemm_q32(&w_t, n, &sh_t, n, o, LIMBS).map_err(map_limb)?;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n {
            a[j * n + i] = a[i * n + j];
            j += 1;
        }
        i += 1;
    }
    Ok(a)
}

/// # Complexity
/// `b[i] = Σ_o x[o,i]·y[o,i]·w[o]`, each `x·y` rounded once: `O(o·n)` multiplies, `O(n)` words;
/// the `o`-term sum is one [`limb::gemv_q32`] over the transposed product.
fn rhs_grid(x: &[i64], y: &[i64], w: &[i64], o: usize, n: usize) -> Result<Vec<i64>, StepsError> {
    // z[o,i] = round(x·y): the elementwise word product, transposed so each column `i` is a row.
    let mut z = vec![0i64; o * n];
    let mut ok = true;
    for oo in 0..o {
        for i in 0..n {
            if ok {
                match mul_word(x[oo * n + i], y[oo * n + i]) {
                    Ok(v) => z[oo * n + i] = v,
                    Err(_) => ok = false,
                }
            }
        }
    }
    if !ok {
        return Err(StepsError::ShapeMismatch);
    }
    let zt = transpose_grid(&z, n, o);
    limb::gemv_q32(w, &zt, n, o, LIMBS).map_err(map_limb)
}

/// # Complexity
/// One [`limb::rowdot_q32`] over `o·g` row pairs of [`GROUP`] words: `O(o·n)` word
/// multiply-adds on the kernel, `O(o·n)` limb bytes.
fn t_rows(p: &[i64], wshl: &[i64], o: usize, n: usize) -> Result<Vec<i64>, StepsError> {
    limb::rowdot_q32(p, wshl, o * n / GROUP, GROUP, LIMBS).map_err(map_limb)
}

/// # Complexity
/// One [`limb::rowdot_q32`] per group band over `o·g` row pairs of [`GROUP`] words: `O(o·n)` word
/// multiply-adds on the kernel, `O(o·g)` `i128` accumulators.
fn m_band(band: &mut [i128], kg: &[i64], p: &[i64]) -> Result<(), StepsError> {
    let y = limb::rowdot_q32(kg, p, band.len(), GROUP, LIMBS).map_err(map_limb)?;
    for i in 0..band.len() {
        band[i] += i128::from(y[i]);
    }
    Ok(())
}

/// The `c_step`: one exact update of the per-column scale against `Hl`, holding `S` and `r`.
/// Solves `A·c = b` with `Sh = S⊙r_rep`, `A[i,j] = Hl[i,j]·Σ_o g[o]·Sh[o,i]·Sh[o,j]` and
/// `b[i] = Σ_o (Ws·Hl)[o,i]·Sh[o,i]·g[o]`, adding the ridge `1e-12·mean(diag(A))`; `g_row` is the
/// row weight (`1` when `None`). `A` is symmetric, so the `Cholesky` solve is valid; when
/// `Ws = compose(S, r, c0)` the un-ridged system has `c0` as its exact solution.
///
/// # The algebra of `A` across steps
/// `A` depends on `S`, `r`, the row weight and `Hl` only — not on `c` or `Ws` — so two `c_step`s
/// with the same `(S, r, g, Hl)` are the same system bit for bit. A diagonal similarity
/// `A' = D A D`, `D = diag(ρ_{i/128})`, would need `r'[o,gg] = ρ_gg·r[o,gg]` for every out row,
/// and `r_step` solves one system per row, so the ratios are row-dependent and no such `ρ` exists.
/// With `Δ = Sh' − Sh` the weighted Gram changes by `Shᵀ diag(g) Δ + Δᵀ diag(g) Sh + Δᵀ diag(g) Δ`,
/// a rank `≤ 2·min(o, n)` update (`n` once `o ≥ n/2`), and `A' − A = Hl ⊙ (G' − G)`; the Hadamard
/// factor does not preserve rank. Each step factors its own `A`.
///
/// # The `Q32.32` boundary
/// `A` and `b` are built **directly on the grid**: `Sh` and `b` are words, no `f64` panel. The ridge
/// is `2⁻⁶²` fixed point ([`ridge_word`]) on the grid diagonal, floored at one raw unit where
/// positive. `A` and `b` carry the exact `2⁻²ᵏ`/`2⁻ᵏ` shift with `k` from `max|Sh|`, the solution
/// `c·2ᵏ` is shifted back, and the row-weight shift cancels.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape, length, grid-range or overflow refusal;
/// [`StepsError::NotPositiveDefinite`] if the metric reduces to a non-SPD `A`.
///
/// # Complexity
/// Temporal: one [`mul_metric`] at `m = O`, `k = n`, `n = n` (pooled) plus [`c_step_wshl`]'s cost.
/// Spatial: `O(O·n)` words for the product plus [`c_step_wshl`]'s footprint.
pub fn c_step(
    ws: &[i64],
    hl: &[i64],
    s: &[i64],
    r: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, StepsError> {
    let wshl = mul_metric(ws, hl, o, n)?;
    c_step_wshl(ws, hl, &wshl, s, r, o, n, g_row)
}

/// The [`c_step`] body with `wshl = mul_metric(ws, hl, o, n)` supplied: the same body, so a hoisted
/// product returns [`c_step`]'s vector bit for bit.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape, length, grid-range or overflow refusal;
/// [`StepsError::NotPositiveDefinite`] if the reduced `A` is not SPD.
///
/// # Complexity
/// Temporal: one [`wsyrk_gemm`] at `O(o·n²)` word multiply-adds on the kernel, one
/// `hadamard_into` at `O(n²)`, one `rhs_grid` at `O(O·n)`, one `chol_solve` at `O(n³/3 + n²)`,
/// `O(O·n²)` for the `Sh` scan and the transposes. The pooled primitives dispatch inside their own
/// drivers; the drivers here stay serial (a nested dispatch would deadlock). Spatial: `O(n²)` for
/// `A` and `O(O·n)` for `Sh`/`b` and the two transposed panels.
pub fn c_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s: &[i64],
    r: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s.len() != o * n
        || r.len() != o * g
    {
        return Err(StepsError::ShapeMismatch);
    }
    if g_row.is_some_and(|gr| gr.len() != o) {
        return Err(StepsError::ShapeMismatch);
    }
    // Sh = S⊙r_rep on the grid, then the exact `2⁻ᵏ` panel shift into the `2³⁵` fill range.
    let mut sh = group_scale(s, r, o, n, g)?;
    let k = panel_shift_words(max_mag(&sh));
    shift_in_place(&mut sh, k)?;
    let e = fill_columns(&mut sh, o, n)?;
    // The row weight carries its own exact power-of-two shift; `1` is the raw unit.
    let gr_g = row_weight(g_row, o)?;
    // A = hl ⊙ (Shᵀ diag(g_row) Sh) on the kernel, column-filled to D·A·D; nothing here is `f64`.
    let mut a = wsyrk_gemm(&sh, &gr_g, o, n)?;
    // A production Gram (`2^46` words at `o = 4864`) times the production metric (`2^53`) needs a
    // 67-bit product word, five above `i64`; the exact `2^-s` rescale of the whole system below
    // keeps the solution and leaves a fit whose product already fits (`s = 0`) bit for bit.
    let s = product_shift(max_mag(&a), max_mag(hl));
    shift_in_place(&mut a, s)?;
    hadamard_into(&mut a, hl).map_err(map_syrk)?;
    // The ridge `eps·mean(diag A)` on the grid diagonal, carried through `D²`.
    let mean = filled_diag_mean(&a, &e, n)?;
    let mut ridge = ridge_word(mean)?;
    if mean > 0 && ridge == 0 {
        ridge = 1;
    }
    add_filled_ridge(&mut a, &e, n, ridge)?;
    // b = D·Σ_o (Ws·Hl)⊙Sh·g_row, on the grid, at the matrix's `2^-s`.
    let mut b = rhs_grid(wshl, &sh, &gr_g, o, n)?;
    shift_in_place(&mut b, s)?;
    let solved = solve_with_ridge_floor(&mut a, &b, &e, n, o, ridge)?;
    // x = D·y, then the global shift: one rounding per entry.
    (0..n)
        .map(|j| shift_word(solved[j], k - e[j]))
        .collect::<Result<Vec<i64>, StepsError>>()
}

/// The `r_step`: one exact update of the per-group scale `r` (`O × G`), holding `S` and `c`. Per out
/// row `o` it solves the `G × G` SPD system `M[o]·r[o] = t[o]` with
/// `P[o,g,k] = S[o,g·128+k]·c[g·128+k]`, `M[o,g,g'] = Σ_{k,k'} P[o,g,k]·Hl[g·128+k, g'·128+k']·P[o,g',k']`
/// and `t[o,g] = Σ_k P[o,g,k]·(Ws·Hl)[o, g·128+k]`, adding the ridge `1e-12·mean(diag(M[o]))`.
/// When `Ws = compose(S, r0, c)` each `r0[o]` is the un-ridged solution of its system.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape or length refusal;
/// [`StepsError::NotPositiveDefinite`] if a reduced `M[o]` is not SPD.
///
/// # Complexity
/// Temporal: one [`mul_metric`] at `m = O`, `k = n`, `n = n` (pooled) plus [`r_step_wshl`]'s cost.
/// Spatial: `O(O·n)` words plus [`r_step_wshl`]'s footprint.
pub fn r_step(
    ws: &[i64],
    hl: &[i64],
    s: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, StepsError> {
    let wshl = mul_metric(ws, hl, o, n)?;
    r_step_wshl(ws, hl, &wshl, s, c, o, n)
}

/// The [`r_step`] body with `wshl = mul_metric(ws, hl, o, n)` supplied: the same body, so a hoisted
/// product returns [`r_step`]'s vector bit for bit.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape, length, grid-range or overflow refusal;
/// [`StepsError::NotPositiveDefinite`] if a reduced `M[o]` is not SPD.
///
/// # Complexity
/// Temporal: `G` `limb::gemm_q32` calls at `m = O`, `k = 128`, `n = n` (`O(O·n²)` in all, pooled), the
/// `t`/`M` accumulations at `O(O·n)`, and `O(O·G³)` for the per-row solves (`chol_solve` pools the
/// factor; its driver is serial here). Spatial: `O(O·n)` words for `P` and the panel, `O(O·G²)`
/// `i128` for the systems and `O(O·G)` output.
pub fn r_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s.len() != o * n
        || c.len() != n
    {
        return Err(StepsError::ShapeMismatch);
    }
    if o == 0 {
        return Ok(Vec::new());
    }
    // P = S⊙c and `Hl` scaled once: `M' = M·2⁻⁽²ᵏᵖ⁺ᵏʰ⁾`, `t' = t·2⁻ᵏᵖ`, so the solution carries
    // `2^⁽ᵏᵖ⁺ᵏʰ⁾` and is shifted back at the end.
    let mut p = col_scale(s, c, n)?;
    let kp = panel_shift_words(max_mag(&p));
    shift_in_place(&mut p, kp)?;
    let kh = panel_shift_words(max_mag(hl));
    let hl_s = shift_copy(hl, kh)?;
    let t = t_rows(&p, wshl, o, n)?;
    // M per out row in `Q32.32`: `ms[gg][o][gg2]`, so a group band is contiguous across rows.
    let mut ms = vec![0i128; g * o * g];
    let mut panel = vec![0i64; o * GROUP];
    let mut gg = 0;
    let mut ok = true;
    while gg < g {
        if ok {
            for oo in 0..o {
                panel[oo * GROUP..(oo + 1) * GROUP]
                    .copy_from_slice(&p[oo * n + gg * GROUP..oo * n + (gg + 1) * GROUP]);
            }
            let block_t = transpose_grid(&hl_s[gg * GROUP * n..(gg + 1) * GROUP * n], n, GROUP);
            match limb::gemm_q32(&panel, o, &block_t, n, GROUP, LIMBS) {
                Ok(kg) => match m_band(&mut ms[gg * o * g..(gg + 1) * o * g], &kg, &p) {
                    Ok(()) => {}
                    Err(_) => ok = false,
                },
                Err(_) => ok = false,
            }
        }
        gg += 1;
    }
    if !ok {
        return Err(StepsError::ShapeMismatch);
    }
    // The per-row solves: the driver is serial (see the complexity note), the factor is pooled.
    let mut out = vec![0i64; o * g];
    let mut m = vec![0i64; g * g];
    let mut err: Option<StepsError> = None;
    for oo in 0..o {
        if err.is_none() {
            for r0 in 0..g {
                for c0 in 0..g {
                    match q32::narrow(ms[(r0 * o + oo) * g + c0]) {
                        Some(v) => m[r0 * g + c0] = v,
                        None => err = Some(StepsError::ShapeMismatch),
                    }
                }
            }
            match ridge_word(diag_sum(&m, g) / g as i128) {
                Ok(ridge) => match add_ridge(&mut m, g, ridge) {
                    Ok(()) => match chol_solve(&m, &t[oo * g..(oo + 1) * g], g, 1) {
                        Ok(solved) => out[oo * g..(oo + 1) * g].copy_from_slice(&solved.x),
                        Err(e) => err = Some(map_solve(e)),
                    },
                    Err(e) => err = Some(e),
                },
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    shift_in_place(&mut out, kp + kh)?;
    Ok(out)
}

/// The `d_step`: the exact Hadamard-domain column-scale minimizer, holding `S_rot`, `r` and `c`.
/// Solves `A·d = b` with `B = S_rot⊙r_rep`, `B_g = B⊙g_row`, `G_rot = fwht128((Ws·Hl)⊙c)`,
/// `H_rot = had_gram(Hl⊙c cᵀ)`, `A[i,j] = H_rot[i,j]·Σ_o B[o,i]·B_g[o,j]` and
/// `b[i] = Σ_o B_g[o,i]·G_rot[o,i]`, adding the ridge `1e-12·max(mean(diag(A)), 1e-12)`.
///
/// The forward model is `recon(d) = compose(fwht128(S_rot⊙d), r, c)` (`spec/fold_transfer.py:886`).
/// Since `r_rep` is constant on each 128-block and `fwht128` is block-local, `r` commutes with the
/// transform, so `recon(d) = compose(fwht128(B⊙d), ones, c)` and `A`, `b` are the gradient of
/// `J(Ws, recon(d), Hl)`. `A` depends on `c` via `H_rot` and on `r` via `B`; every alternation
/// changes `c`, so no consecutive `d_step` pair shares the matrix (contrast [`c_step`]'s note).
///
/// # The `Q32.32` boundary
/// `G_rot`, `H_rot`, `A` and `b` are built **directly on the grid**: `B`, the rotated metric and `b`
/// are words, no `f64` panel. The solve is [`chol_solve`], the `f64` `A` loop and `solve_spd`
/// fallback gone. `B` carries the exact `2⁻ᵏ` panel shift with `k` from `max|B|`, so the solution
/// `d·2ᵏ` is shifted back; the ridge is the script's `1e-12·max(mean(diag A), 1e-12)`, floored at
/// one raw unit where it is positive.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape, length, grid-range or overflow refusal;
/// [`StepsError::NotPositiveDefinite`] if the reduced `A` is not SPD.
///
/// # Complexity
/// Temporal: one [`mul_metric`] at `m = O`, `k = n`, `n = n` (pooled) plus [`d_step_wshl`]'s cost.
/// Spatial: `O(O·n)` words plus [`d_step_wshl`]'s footprint.
pub fn d_step(
    ws: &[i64],
    hl: &[i64],
    s_rot: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, StepsError> {
    let wshl = mul_metric(ws, hl, o, n)?;
    d_step_wshl(ws, hl, &wshl, s_rot, r, c, o, n, g_row)
}

/// The [`d_step`] body with `wshl = mul_metric(ws, hl, o, n)` supplied: the same body, so a hoisted
/// product returns [`d_step`]'s vector bit for bit.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a shape, length, grid-range or overflow refusal;
/// [`StepsError::NotPositiveDefinite`] if the reduced `A` is not SPD.
///
/// # Complexity
/// Temporal: `O(O·n·log 128)` for `G_rot`'s transform (pooled) and `O(n²·log 128)` for `H_rot`
/// (pooled), one [`wsyrk_gemm`] at `O(o·n²)` on the kernel and `hadamard_into` at `O(n²)`
/// (pooled), `rhs_grid` at
/// `O(O·n)`, `chol_solve` at `O(n³/3 + n²)` (pooled); spatial `O(n²)` for `H_rot`/`A`.
pub fn d_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s_rot: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s_rot.len() != o * n
        || r.len() != o * g
        || c.len() != n
    {
        return Err(StepsError::ShapeMismatch);
    }
    if g_row.is_some_and(|gr| gr.len() != o) {
        return Err(StepsError::ShapeMismatch);
    }
    // G_rot = fwht128((Ws·Hl) ⊙ c) and H_rot = had_gram(Hl ⊙ c cᵀ), both on the grid.
    let mut g_rot = col_scale(wshl, c, n)?;
    fwht_words(&mut g_rot)?;
    let h_rot = had_gram_grid(&metric_scale(hl, c, n)?, n)?;
    // B = S_rot⊙r_rep on the grid, then the exact `2⁻ᵏ` panel shift.
    let mut b_scaled = group_scale(s_rot, r, o, n, g)?;
    let k = panel_shift_words(max_mag(&b_scaled));
    shift_in_place(&mut b_scaled, k)?;
    let e = fill_columns(&mut b_scaled, o, n)?;
    let gr_g = row_weight(g_row, o)?;
    let mut a = wsyrk_gemm(&b_scaled, &gr_g, o, n)?;
    // The same `2^-s` rescale as [`c_step_wshl`]: `Bᵀ diag(g) B` reaches `2^46` and `H_rot`
    // `2^53` at production, a 67-bit product word.
    let s = product_shift(max_mag(&a), max_mag(&h_rot));
    shift_in_place(&mut a, s)?;
    hadamard_into(&mut a, &h_rot).map_err(map_syrk)?;
    // The ridge `eps·max(mean(diag A), eps·2⁻²ᵏ)`: the clamp is below one raw unit for every shift,
    // so a non-positive mean still floors the ridge at one raw unit; carried through `D²`.
    let mean = filled_diag_mean(&a, &e, n)?;
    let ridge = ridge_word(mean.max(0))?.max(1);
    add_filled_ridge(&mut a, &e, n, ridge)?;
    let mut b = rhs_grid(&b_scaled, &g_rot, &gr_g, o, n)?;
    shift_in_place(&mut b, s)?;
    let solved = solve_with_ridge_floor(&mut a, &b, &e, n, o, ridge)?;
    (0..n)
        .map(|j| shift_word(solved[j], k - e[j]))
        .collect::<Result<Vec<i64>, StepsError>>()
}

/// `C[m,n] = A[m,k]·B[k,n]` for word operands: one exact power-of-two shift per operand into the
/// [`limb::gemm_q32`] input bound, the grid product, the exact undo.
///
/// # Errors
/// As [`limb::gemm_q32`] for the shape, plus an undo shift or a rounded entry leaving `i64`.
///
/// # Complexity
/// Temporal: `m·k·n` word multiply-adds on the kernel in [`limb::gemm_q32`] plus
/// `O(m·k + k·n + m·n)` shift and transpose bookkeeping; the product pools inside
/// [`limb::gemm_q32`], the shifts and the `B` transpose are serial. Space: `O(m·n)`.
fn metric_grid(a: &[i64], b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, StepsError> {
    let ka = panel_shift_words(max_mag(a));
    let kb = panel_shift_words(max_mag(b));
    let as_ = shift_copy(a, ka)?;
    let bs = shift_copy(b, kb)?;
    let bt = transpose_grid(&bs, n, n);
    let mut out = limb::gemm_q32(&as_, o, &bt, n, n, LIMBS).map_err(map_limb)?;
    shift_in_place(&mut out, -(ka + kb))?;
    Ok(out)
}

/// The metric objective `J(Ws, Wp, Hl) = Σ_{o,i,j} D[o,i]·Hl[i,j]·D[o,j] = tr(D·Hl·Dᵀ)`, `D =
/// Ws − Wp`, returned as the `Q32.32` word of the trace. `ws`, `wp` are `O × n` row-major, `hl` is
/// `n × n`; `D·Hl` is [`metric_grid`] and the trace one exact `i128` dot rounded once.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a length or product refusal. Never a panic.
///
/// # Complexity
/// Temporal: [`metric_grid`] at `m = O`, `k = n`, `n = n` (`O(O·n²)` multiply-adds, pooled inside
/// `limb::gemm_q32`), then `O(O·n)` for the subtraction and dot; spatial `O(O·n)` words.
pub fn objective(
    ws: &[i64],
    wp: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
) -> Result<i64, StepsError> {
    if ws.len() != o * n || wp.len() != o * n || hl.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    let mut d = vec![0i64; o * n];
    for i in 0..d.len() {
        d[i] = ws[i].checked_sub(wp[i]).ok_or(StepsError::ShapeMismatch)?;
    }
    if o == 0 || n == 0 {
        return Ok(0);
    }
    let y = metric_grid(&d, hl, o, n)?;
    flat_pair_sum(&d, &y)
}

/// `Σ aᵢ·bᵢ` over the kernel's grouped row dot.
///
/// The kernel's single-call inner bound is [`MAX_INNER`](crate::engine::kern::limb::MAX_INNER)
/// words; the fit's flat length is `o·n` and exceeds it at every tensor whose area is past
/// `128×128` (the `896×896` projection is 802,816 words), where one `gemv_q32` call refuses with
/// [`LimbError::Shape`]. The dot runs as `len/GROUP` row pairs of [`GROUP`] words through
/// [`limb::rowdot_q32`], the group words are summed exactly in `i128`, and the total is narrowed
/// once.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] on an empty, unequal or non-[`GROUP`] length, or a total outside
/// the Q32.32 words.
fn flat_pair_sum(a: &[i64], b: &[i64]) -> Result<i64, StepsError> {
    let len = a.len();
    if len == 0 || len != b.len() || !len.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let parts = limb::rowdot_q32(a, b, len / GROUP, GROUP, LIMBS).map_err(map_limb)?;
    let mut acc: i128 = 0;
    for &v in &parts {
        acc += i128::from(v);
    }
    q32::narrow(acc).ok_or(StepsError::ShapeMismatch)
}

/// `Ws·Hl` through [`limb::gemm_q32`], with [`metric_grid`]'s operand shift and exact undo.
///
/// # Errors
/// [`StepsError::ShapeMismatch`] on a length, grid-range or product refusal. Never a panic.
///
/// # Complexity
/// As [`limb::gemm_q32`] at `m = O`, `k = n`, `n = n` (`O(O·n²)` multiply-accumulates), pooled inside
/// `limb::gemm_q32`; one call per step. Spatial `O(O·n)` words.
pub fn mul_metric(ws: &[i64], hl: &[i64], o: usize, n: usize) -> Result<Vec<i64>, StepsError> {
    if ws.len() != o * n || hl.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    if o == 0 || n == 0 {
        return Ok(vec![0i64; o * n]);
    }
    metric_grid(ws, hl, o, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `896×896` projection's flat length is 802,816 words, past the kernel's 16,384-word
    /// single-call inner bound: one `gemv_q32` call refuses, and the grouped route must sum
    /// within the module's stated `2⁻¹³` relative bound.
    #[test]
    fn flat_pair_sum_past_the_kernel_inner_bound() {
        let len = 192 * 192;
        let mut a = vec![0i64; len];
        let mut b = vec![0i64; len];
        let mut exact: i128 = 0;
        for i in 0..len {
            a[i] = ((i % 997 + 1) as i64) << 8;
            b[i] = ((i * 7 % 991 + 1) as i64) << 10;
            exact += i128::from(a[i]) * i128::from(b[i]);
        }
        let want = q32::round_q32_even(exact).expect("reference in words");
        let got = flat_pair_sum(&a, &b).expect("grouped route runs past 16,384 words");
        let err = (i128::from(got) - i128::from(want)).abs();
        let bound = (i128::from(want).abs() >> 13).max(1);
        assert!(
            err <= bound,
            "grouped {got} vs exact {want}: error {err} over bound {bound}"
        );
    }
}
