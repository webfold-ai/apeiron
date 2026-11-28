//! Iterated-conditional-modes (ICM) refinement of the one-bit fold transfer's `E8 ∪ O8` planes in the
//! `(r, c, d)`-scaled Hadamard–Gram metric.
//!
//! Port of `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 585–720. A plane is the pack
//! `(s, r, c, d)` with weight `plane_w = fwht128(s·d) · r_rep · c`, its block codes one of the 256 rows
//! of the [`crate::learn::code::e8`] codebook; both routines are exact coordinate descent on
//! `J(Ws, Wp, Hl) = Σ_o (Ws_o − Wp_o)ᵀ Hl (Ws_o − Wp_o)`.
//!
//! # Carrier and boundary
//!
//! Every value is Q32.32: `x` is `round(x·2³²)` in an `i64`; matrix products, SyrkAcc and dots run on
//! [`crate::engine::kern::limb`] at [`LIMBS`] limbs (block-floating, the bound in that module's
//! `bound_ulps`, reported by [`product_bound`]), every other product is exact in `i128`, and
//! block/pair costs compare as exact Q96.96 `i128`, so an argmin is the integer argmin and a tie goes
//! to the first index. The 128-point transform is the engine's [`fwht_fixed`]; `chol` owns the solves;
//! the kernel dispatches the engine pool, so the sweep bands' products run on the caller, and the one
//! pass that itself dispatches the pool is the joint pair decision ([`pair_decisions`]: the rows are
//! independent, so it runs in row chunks); every other elementwise/transform pass is a serial integer
//! loop. Every `pub`
//! entry here takes and returns words: the planes, the metric and both `J` values are `Q32.32`, so
//! a caller that holds reals converts once at its own boundary ([`codebook`] and [`JointStats`] are
//! the read-out exceptions). No step pays a `from_grid`/`to_grid` round trip.
//!
//! Grade: `tested` (`tests/learn_fold_icm.rs`) for the band and sweep bodies; `best_code_q`'s argmin
//! is `proved` through `proof/corpus/Learn/FitIcm.lean` (`witnessArgmin_all`,
//! `witnessStepSpec_exists`), and the Rust binding is the unit test
//! `tests::learn_fold_fit_icm_witness_best_code_q_is_row0`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/k/l/m` and narrow a \
              code index already bounded by `CODE_LEN` to `u16`, reading closer to the reference \
              `fold_transfer.py` arithmetic than enumerated iterators would"
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::engine::kern::limb::{gemm_q32, gemv_q32, rowdot_q32, LimbError};
use crate::engine::ONE;
use crate::engine::{fwht_fixed, HadamardError};
use crate::learn::code::e8;
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::q32::{self, ilog2_pos, rshift_rne};

/// Columns per `E8 ∪ O8` block: the code dimension.
pub const BLOCK: usize = 8;
/// Columns per `r` group scale, and the transform block of [`fwht_rows`].
pub const GROUP: usize = 128;
/// Codebook rows: the 256 `E8 ∪ O8` codes of [`crate::learn::code::e8`].
pub const CODE_LEN: usize = e8::CODEBOOK_LEN;

/// A refusal from an ICM refinement step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcmError {
    /// A slice length disagrees with the declared shape (`o`, `n` and the group/block widths).
    Shape,
    /// `n` is not a positive multiple of both [`BLOCK`] and [`GROUP`].
    Width,
    /// `row_chunk == 0`.
    ZeroChunk,
    /// A block index in `blocks` is `>= n / BLOCK`.
    BlockOutOfRange,
    /// A value left the `Q32.32` grid or a checked integer operation left its range.
    Overflow,
    /// The `r` normal system was not positive-definite (only reachable on a degenerate metric).
    NotPositiveDefinite,
}

/// A borrowed plane `(s, idx, r, c, d)` on the `Q32.32` grid.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a> {
    /// Plane code values (`Q32.32` words), row-major `[o, n]`: block `m` of row `o` is `s[o*n+8m .. +8]`.
    pub s: &'a [i64],
    /// Codebook row per block, row-major `[o, n / 8]`, each in `0..256`.
    pub idx: &'a [u16],
    /// Group scale, row-major `[o, n / GROUP]`.
    pub r: &'a [i64],
    /// Column scale, length `n`.
    pub c: &'a [i64],
    /// Hadamard column scale, length `n`.
    pub d: &'a [i64],
}

/// An owned plane: the result of a refinement, with `r`, `c`, `d` passed through unchanged.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Plane code values (`Q32.32` words), row-major `[o, n]`.
    pub s: Vec<i64>,
    /// Codebook row per block, row-major `[o, n / 8]`.
    pub idx: Vec<u16>,
    /// Group scale, row-major `[o, n / GROUP]`.
    pub r: Vec<i64>,
    /// Column scale, length `n`.
    pub c: Vec<i64>,
    /// Hadamard column scale, length `n`.
    pub d: Vec<i64>,
}

/// Counters from [`e8_joint_icm_2plane`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointStats {
    /// `J` of the two planes before the sweeps, in the ICM-ridged metric, row-weighted by `g_row`,
    /// as a Q32.32 word.
    pub j_before: i64,
    /// `J` after the sweeps, same metric, as a Q32.32 word.
    pub j_after: i64,
    /// Sum over blocks of the rows in which at least one plane's block code changed.
    pub moves: usize,
}

/// `round(v / 2^s)` as an `i64`, or [`IcmError::Overflow`] when the rounded value leaves `i64`.
fn round_shift(v: i128, s: u32) -> Result<i64, IcmError> {
    match rshift_rne(v, s) {
        Some(x) => i64::try_from(x).map_err(|_| IcmError::Overflow),
        None => Err(IcmError::Overflow),
    }
}

/// `a + b`, refusing an `i64` overflow.
fn add_q(a: i64, b: i64) -> Result<i64, IcmError> {
    a.checked_add(b).ok_or(IcmError::Overflow)
}

/// `a − b`, refusing an `i64` overflow.
fn sub_q(a: i64, b: i64) -> Result<i64, IcmError> {
    a.checked_sub(b).ok_or(IcmError::Overflow)
}

/// `round(a·b / 2³²)`.
pub(super) fn mul_q(a: i64, b: i64) -> Result<i64, IcmError> {
    round_shift(i128::from(a) * i128::from(b), 32)
}

/// Checked `i128` sum.
fn qsum(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_add(b).ok_or(IcmError::Overflow)
}

/// Checked `i128` difference.
fn qsub(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_sub(b).ok_or(IcmError::Overflow)
}

/// Checked `i128` product.
fn qmul(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_mul(b).ok_or(IcmError::Overflow)
}

/// Limbs per local (per-row, per-block) kernel product. The 3-limb route (`24`-bit mantissa per
/// 32-element block) leaves a residual of about `2^-15` of the operand peak, which the joint
/// two-plane sweep's `CODE_LEN²` pair search and the `r_step2` normal system turn into a decision
/// that can be worse than the pre-change one; the R5 report carries the measured figures, so the
/// local products carry 5 limbs (`2^-31`).
const LIMBS: u32 = 5;

/// Limbs of the public row-panel product [`matmul_q`]: the wave default, the R8 measured route and
/// the operand of every tolerance derived from [`product_bound`].
const LIMBS_MATMUL: u32 = 3;

/// `Σ a[k]·b[k]` at Q32.32 on the limb kernel; `O(len)` multiply-adds, `O(1)` space.
fn dot_q(a: &[i64], b: &[i64]) -> Result<i64, IcmError> {
    let y = gemv_q(a, b, 1, a.len())?;
    y.first().copied().ok_or(IcmError::Shape)
}

/// `gemm_q32` over row-major `[m, n] · [p, n]ᵀ` with this file's error map.
fn nt_q(a: &[i64], m: usize, b: &[i64], p: usize, n: usize) -> Result<Vec<i64>, IcmError> {
    gemm_q32(a, m, b, p, n, LIMBS).map_err(map_limb)
}

/// [`matmul_q`] at [`LIMBS`]: the decision-state products (the pre-change residual `G − B·H`), where
/// a `2^-15` entry error of the operand peak survives the cancellation and moves the code choice.
fn matmul_l(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
    nt: bool,
) -> Result<Vec<i64>, IcmError> {
    if nt {
        nt_l(a, rows, b, cols, inner)
    } else {
        nt_l(a, rows, &transpose_q(b, inner, cols), cols, inner)
    }
}

/// [`nt_q`] at [`LIMBS`].
fn nt_l(a: &[i64], m: usize, b: &[i64], p: usize, n: usize) -> Result<Vec<i64>, IcmError> {
    gemm_q32(a, m, b, p, n, LIMBS).map_err(map_limb)
}

/// One row against `p` kernel rows: `y[j] = Σ_l x[l]·b[j·n + l]`.
fn gemv_q(x: &[i64], b: &[i64], p: usize, n: usize) -> Result<Vec<i64>, IcmError> {
    gemv_q32(x, b, p, n, LIMBS).map_err(map_limb)
}

/// The kernel's per-entry `Q32.32` word bound for a [`LIMBS_MATMUL`]-limb product of `inner` terms
/// whose operands are bounded by the real magnitudes `max_a` and `max_b`
/// ([`crate::engine::kern::limb`]'s `bound_ulps`, the assertion a product test runs against), as a
/// real magnitude.
#[must_use]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the two casts convert a finite non-negative real magnitude to its Q32.32 word count \
              (range-checked) and the bound back; the bound is a tolerance, not an exact count"
)]
pub fn product_bound(inner: usize, max_a: f64, max_b: f64) -> f64 {
    let words = |v: f64| -> u64 {
        let w = v.abs() * 4_294_967_296.0;
        if w >= 9.2e18 {
            u64::MAX
        } else {
            w as u64
        }
    };
    let b = crate::engine::kern::limb::bound_ulps(inner, LIMBS_MATMUL, words(max_a), words(max_b));
    b as f64 / 4_294_967_296.0
}

/// The row scale's constant products `(a², 2a)`, exact in `i128`.
fn row_scale(a: i64) -> (i128, i128) {
    let av = i128::from(a);
    (av * av, av * 2)
}

/// The Q96.96 block cost `a²·q − 2·a·dot` (`dot` at Q32.32 from the kernel, lifted to Q64.64 by an
/// exact shift), compared unrounded. `aa = a²` and `two_a = 2a` are the row scale's constant
/// products, hoisted once per row-block.
fn block_cost(aa: i128, two_a: i128, q: i64, dot: i64) -> Result<i128, IcmError> {
    qsub(
        qmul(aa, i128::from(q))?,
        qmul(two_a, i128::from(dot) << 32)?,
    )
}

/// The exact Q96.96 pair cost `2·a₁a₂·x + A + B` with `two_a12 = 2·a₁a₂` hoisted once per
/// row-block.
fn pair_cost(two_a12: i128, x: i64, av: i128, bv: i128) -> Result<i128, IcmError> {
    qsum(qsum(qmul(two_a12, i128::from(x))?, av)?, bv)
}

/// The `CODE_LEN`-way argmin of the exact Q96.96 block cost over one row-block's codebook dots:
/// `q[j]` is the block's quadratic form, `dot[j]` the kernel product `s·c_j` at Q32.32. Kept out of
/// the band loop: the pinned Aeneas refuses to compute a loop fixed point over the band's whole
/// carried state.
pub(crate) fn best_code_q(
    aa: i128,
    two_a: i128,
    q: &[i64],
    dot: &[i64],
) -> Result<usize, IcmError> {
    let (mut best, mut best_cost) = (0_u16, i128::MAX);
    let mut err: Option<IcmError> = None;
    let mut j = 0;
    while j < CODE_LEN && err.is_none() {
        match block_cost(aa, two_a, q[j], dot[j]) {
            Ok(cost) => {
                if cost < best_cost {
                    best_cost = cost;
                    best = j as u16;
                }
            }
            Err(e) => err = Some(e),
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(usize::from(best))
}

/// The exact Q96.96 joint pair cost table of one row-block, row-major `[i₁·CODE_LEN + i₂]`:
/// `2·a₁a₂·x[i₁,i₂] + A[i₁] + B[i₂]`, with `two_a12 = 2·a₁a₂` hoisted by the caller. `out` is
/// cleared first, so one buffer serves every row of a band.
///
/// # Errors
/// [`IcmError::Shape`] if `xm` is not `CODE_LEN²`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(CODE_LEN²)` checked operations and `CODE_LEN²` `i128` words in `out`.
fn pair_costs_into(
    two_a12: i128,
    av: &[i128; CODE_LEN],
    xm: &[i64],
    bv: &[i128; CODE_LEN],
    out: &mut Vec<i128>,
) -> Result<(), IcmError> {
    if xm.len() != CODE_LEN * CODE_LEN {
        return Err(IcmError::Shape);
    }
    out.clear();
    let mut i1 = 0;
    while i1 < CODE_LEN {
        let x_row = &xm[i1 * CODE_LEN..][..CODE_LEN];
        let a = av[i1];
        let mut i2 = 0;
        while i2 < CODE_LEN {
            out.push(pair_cost(two_a12, x_row[i2], a, bv[i2])?);
            i2 += 1;
        }
        i1 += 1;
    }
    Ok(())
}

/// The zero-temperature joint pair decision: the `CODE_LEN²`-way argmin of [`pair_costs_into`],
/// returned as `i₁·CODE_LEN + i₂`, first index on a tie. This is the `T = 0` control the finite
/// temperature schedule is measured against; it is the decision the joint sweep made before the
/// schedule existed.
///
/// # Errors
/// [`IcmError::Shape`] if `xm` is not `CODE_LEN²`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(CODE_LEN²)` checked operations, `O(1)` space beyond the caller's operands.
pub fn best_pair_q(
    two_a12: i128,
    av: &[i128; CODE_LEN],
    xm: &[i64],
    bv: &[i128; CODE_LEN],
) -> Result<usize, IcmError> {
    if xm.len() != CODE_LEN * CODE_LEN {
        return Err(IcmError::Shape);
    }
    let (mut flat, mut best) = (0_usize, i128::MAX);
    let mut i1 = 0;
    while i1 < CODE_LEN {
        let x_row = &xm[i1 * CODE_LEN..][..CODE_LEN];
        let mut i2 = 0;
        while i2 < CODE_LEN {
            let t = pair_cost(two_a12, x_row[i2], av[i1], bv[i2])?;
            if t < best {
                best = t;
                flat = i1 * CODE_LEN + i2;
            }
            i2 += 1;
        }
        i1 += 1;
    }
    Ok(flat)
}

/// One finite-temperature joint pair decision: the `CODE_LEN²` [`pair_costs_into`] table drawn with
/// [`gibbs_choice`] at inverse temperature `beta` (Q32.32), returned as `i₁·CODE_LEN + i₂`.
///
/// `beta = 0` is the hot limit (a uniform draw), not the `T = 0` control; the control is
/// [`best_pair_q`].
///
/// # Errors
/// [`IcmError::Shape`] on a negative `beta` or an `xm` that is not `CODE_LEN²`;
/// [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(CODE_LEN²)` checked operations and `O(CODE_LEN²)` `i128` scratch words per draw.
pub fn best_pair_q_beta(
    two_a12: i128,
    av: &[i128; CODE_LEN],
    xm: &[i64],
    bv: &[i128; CODE_LEN],
    beta: i64,
    rng: &mut u64,
) -> Result<usize, IcmError> {
    if beta < 0 {
        return Err(IcmError::Shape);
    }
    let mut costs = Vec::with_capacity(CODE_LEN * CODE_LEN);
    pair_costs_into(two_a12, av, xm, bv, &mut costs)?;
    gibbs_choice(&costs, beta, rng)
}

/// A deterministic Q32.32 inverse-temperature schedule for the joint code sweep: `beta` ramps
/// linearly from `beta0` at sweep `0` to `beta1` at sweep `sweeps − 1`. `beta0 < beta1` is a cooling
/// schedule (hot first, so the sweep can leave a coordinate local minimum, cold last, so it settles);
/// `beta0 = beta1` is isothermal.
///
/// The draws use [`gibbs_choice`] with one xorshift state per row, seeded from `seed` and the row
/// index only, so a run is reproducible and its draws do not depend on the pool banding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BetaSchedule {
    /// Inverse temperature of sweep `0`, Q32.32.
    beta0: i64,
    /// Inverse temperature of sweep `sweeps − 1`, Q32.32 (`>= beta0`).
    beta1: i64,
    /// Sweeps the ramp spans; at least one.
    sweeps: usize,
    /// xorshift seed; row `o` starts at `seed ^ mix64(o)`.
    seed: u64,
}

impl BetaSchedule {
    /// A cooling schedule from `beta0` to `beta1` over `sweeps` sweeps.
    ///
    /// # Errors
    /// [`IcmError::Shape`] if `sweeps == 0`, `beta0 < 0`, or `beta1 < beta0`.
    pub fn new(beta0: i64, beta1: i64, sweeps: usize, seed: u64) -> Result<Self, IcmError> {
        if sweeps == 0 || beta0 < 0 || beta1 < beta0 {
            return Err(IcmError::Shape);
        }
        Ok(Self {
            beta0,
            beta1,
            sweeps,
            seed,
        })
    }

    /// The inverse temperature of `sweep` (Q32.32), clamped to the last sweep's value.
    #[must_use]
    pub fn beta_at(&self, sweep: usize) -> i64 {
        if self.sweeps <= 1 {
            return self.beta1;
        }
        let span = self.sweeps - 1;
        let k = i128::try_from(sweep.min(span)).unwrap_or(i128::MAX);
        let span = i128::try_from(span).unwrap_or(i128::MAX);
        let b0 = i128::from(self.beta0);
        let delta = i128::from(self.beta1) - b0;
        (b0 + delta * k / span) as i64
    }

    /// The sweep count the ramp spans.
    #[must_use]
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// The xorshift seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// `SplitMix64`'s finaliser: spreads a row index into a non-degenerate xorshift seed.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One Boltzmann draw from a joint code-cost table, at inverse temperature `beta` (Q32.32).
///
/// The finite-temperature move is the folding `gibbs_variational_principle` prescription: the
/// Boltzmann weights `exp(-beta * (cost - min))` are the unique minimiser of the variational free
/// energy, and `beta = infinity` is the zero-temperature [`best_code_q`] argmin. A positive `beta`
/// admits the uphill move the coordinate descent cannot take; an annealing schedule is the
/// `funnelFreeEnergy` sweep. The costs are exact Q96.96 `i128` (the same scale as [`pair_cost`]).
///
/// # Errors
/// [`IcmError::Shape`] on an empty cost table; [`IcmError::Overflow`] if a cost or the weight sum
/// leaves the grid. Weights below `exp(-16)` clamp to that floor, so a very cold table still has a
/// positive total.
///
/// # Complexity
/// `O(costs.len())` checked operations and `O(costs.len())` scratch words per draw; one draw is
/// `O(CODE_LEN²)` for the two-plane table.
pub fn gibbs_choice(costs: &[i128], beta: i64, rng: &mut u64) -> Result<usize, IcmError> {
    if costs.is_empty() || beta < 0 {
        return Err(IcmError::Shape);
    }
    let min = *costs.iter().min().ok_or(IcmError::Shape)?;
    let floor = -16 * ONE;
    let mut weights = Vec::with_capacity(costs.len());
    let mut total = 0_i64;
    for &cost in costs {
        let gap = cost - min;
        let gap32 = i64::try_from(gap >> 64).map_err(|_| IcmError::Overflow)?;
        let prod = i128::from(beta) * i128::from(gap32);
        let neg = i64::try_from(-(prod >> 32)).map_err(|_| IcmError::Overflow)?;
        let x = neg.max(floor);
        let w = crate::numeric::fixed_exp::fixed_exp(x).map_err(|_| IcmError::Overflow)?;
        total = total.checked_add(w).ok_or(IcmError::Overflow)?;
        weights.push(w);
    }
    if total <= 0 {
        return Err(IcmError::Overflow);
    }
    let mut state = *rng;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    *rng = state;
    let total_u = u64::try_from(total).map_err(|_| IcmError::Overflow)?;
    let draw = i64::try_from(state % total_u).map_err(|_| IcmError::Overflow)?;
    let mut acc = 0_i64;
    for (index, &w) in weights.iter().enumerate() {
        acc = acc.checked_add(w).ok_or(IcmError::Overflow)?;
        if draw < acc {
            return Ok(index);
        }
    }
    Ok(weights.len() - 1)
}

/// An operand shape the kernel refuses is [`IcmError::Shape`]; a range or missing-kernel refusal is [`IcmError::Overflow`].
fn map_limb(e: LimbError) -> IcmError {
    match e {
        LimbError::Shape => IcmError::Shape,
        LimbError::Range | LimbError::NoKernel => IcmError::Overflow,
    }
}

/// A non-positive pivot (or diagonal) is the normal-system refusal, every other cause shape.
fn map_solve(e: SolveError) -> IcmError {
    match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            IcmError::NotPositiveDefinite
        }
        SolveError::ShapeMismatch | SolveError::Empty | SolveError::Overflow => IcmError::Shape,
    }
}

/// Row-major `[rows, inner] × [inner, cols]` product (`nt`: `b` is `[cols, inner]`) on the limb kernel.
/// # Complexity
/// `O(rows·inner·cols)` kernel multiply-accumulates times the limb pairs, `O(rows·cols)` space plus an
/// `O(inner·cols)` transpose when `!nt`.
fn matmul_q(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
    nt: bool,
) -> Result<Vec<i64>, IcmError> {
    let (Some(ri), Some(bi)) = (rows.checked_mul(inner), cols.checked_mul(inner)) else {
        return Err(IcmError::Shape);
    };
    if rows == 0 || inner == 0 || cols == 0 || a.len() != ri || b.len() != bi {
        return Err(IcmError::Shape);
    }
    if nt {
        gemm_q32(a, rows, b, cols, inner, LIMBS_MATMUL).map_err(map_limb)
    } else {
        gemm_q32(
            a,
            rows,
            &transpose_q(b, inner, cols),
            cols,
            inner,
            LIMBS_MATMUL,
        )
        .map_err(map_limb)
    }
}

/// The 256 `E8 ∪ O8` codewords as Q32.32 words.
fn codebook_q() -> [[i64; BLOCK]; CODE_LEN] {
    let mut cb = [[0_i64; BLOCK]; CODE_LEN];
    let mut j = 0;
    while j < CODE_LEN {
        match e8::codeword(j) {
            Ok(w) => cb[j] = w,
            Err(_) => cb[j] = [0_i64; BLOCK],
        }
        j += 1;
    }
    cb
}

/// The `m`-th `8 × 8` diagonal block of a row-major `n × n` buffer, as `H[a][b]`.
fn block_of_q(h: &[i64], n: usize, m: usize) -> [[i64; BLOCK]; BLOCK] {
    let mut b = [[0_i64; BLOCK]; BLOCK];
    let mut a = 0;
    while a < BLOCK {
        let mut c = 0;
        while c < BLOCK {
            b[a][c] = h[(m * BLOCK + a) * n + m * BLOCK + c];
            c += 1;
        }
        a += 1;
    }
    b
}

/// Rows per pool task of a pooled row band.
fn rows_per_task(rows: usize) -> usize {
    rows.div_ceil(width_global()).max(1)
}

/// The elementwise binary op of [`bin_into`].
#[derive(Clone, Copy)]
pub(super) enum BinOp {
    /// `dst + src`.
    Add,
    /// `dst − src`.
    Sub,
}

/// `dst op= src`, checked; `O(len)` adds, `O(1)` space.
pub(super) fn bin_into(dst: &mut [i64], src: &[i64], op: BinOp) -> Result<(), IcmError> {
    if dst.len() != src.len() {
        return Err(IcmError::Shape);
    }
    let mut i = 0;
    while i < dst.len() {
        let s = src[i];
        dst[i] = match op {
            BinOp::Add => add_q(dst[i], s)?,
            BinOp::Sub => sub_q(dst[i], s)?,
        };
        i += 1;
    }
    Ok(())
}

/// What a [`scale_by`] multiplies a column by.
#[derive(Clone, Copy)]
pub(super) enum ScaleBy {
    /// `v[col]`, the column scale `c`.
    Column,
    /// `v[row·g + col/GROUP]`, the group scale `r`.
    Group,
    /// `v[row]`, the row scale `x`.
    Row,
}

/// `x[o, n] *= v` along the selected axis, checked; `O(o·n)` multiplies, `O(1)` space.
pub(super) fn scale_by(
    x: &mut [i64],
    o: usize,
    n: usize,
    v: &[i64],
    g: usize,
    by: ScaleBy,
) -> Result<(), IcmError> {
    let need = match by {
        ScaleBy::Column => n,
        ScaleBy::Group => o * g,
        ScaleBy::Row => o,
    };
    let group_ok = !matches!(by, ScaleBy::Group) || g * GROUP == n;
    if o == 0 || n == 0 || x.len() != o * n || v.len() != need || !group_ok {
        return Err(IcmError::Shape);
    }
    let mut err: Option<IcmError> = None;
    let mut row = 0;
    while row < o && err.is_none() {
        let mut col = 0;
        while col < n && err.is_none() {
            let f = match by {
                ScaleBy::Column => v[col],
                ScaleBy::Group => v[row * g + col / GROUP],
                ScaleBy::Row => v[row],
            };
            match mul_q(x[row * n + col], f) {
                Ok(z) => x[row * n + col] = z,
                Err(e) => err = Some(e),
            }
            col += 1;
        }
        row += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `fwht128` over each [`GROUP`]-column block of `x`, in place: the engine's normalized integer
/// rotation `x' = round(H₁₂₈ · x / √128)`, the `1/√128` folded into one Q32.32 constant.
/// # Complexity
/// `O(o·n·log₂(GROUP))` checked adds and one `i128` multiply per entry, `O(1)` space in place.
fn fwht_rows(x: &mut [i64], o: usize, n: usize) -> Result<(), IcmError> {
    if o == 0 || n == 0 || !n.is_multiple_of(GROUP) || x.len() != o * n {
        return Err(IcmError::Shape);
    }
    fwht_fixed(x, None, GROUP).map_err(|e| match e {
        HadamardError::Overflow => IcmError::Overflow,
        _ => IcmError::Shape,
    })
}

/// The transpose of a row-major `[r, c]` buffer, as `[c, r]`; `O(r·c)` copies, `O(r·c)` space.
fn transpose_q(h: &[i64], r: usize, c: usize) -> Vec<i64> {
    let mut t = vec![0_i64; r * c];
    let mut i = 0;
    while i < r {
        let mut j = 0;
        while j < c {
            t[j * r + i] = h[i * c + j];
            j += 1;
        }
        i += 1;
    }
    t
}

/// `R·h·R` for `R = H₁₂₈/√128`: row transform, transpose, row transform, transpose back.
/// # Complexity
/// `O(n²·log₂(GROUP))` checked adds plus two `O(n²)` transposes, `O(n²)` space.
fn had_gram_q(h: &[i64], n: usize) -> Result<Vec<i64>, IcmError> {
    if n == 0 || !n.is_multiple_of(GROUP) || h.len() != n * n {
        return Err(IcmError::Width);
    }
    let mut t = h.to_vec();
    fwht_rows(&mut t, n, n)?;
    let mut u = transpose_q(&t, n, n);
    fwht_rows(&mut u, n, n)?;
    Ok(transpose_q(&u, n, n))
}

/// `H_jk = had_gram(Hl ∘ c_j c_kᵀ) ∘ d_j d_kᵀ`, the scaled Hadamard–Gram metric of one (or two) plane.
/// # Complexity
/// As [`had_gram_q`] plus four `O(n²)` scalings, `O(n²)` space.
fn cross_metric_q(
    hl: &[i64],
    n: usize,
    c1: &[i64],
    c2: &[i64],
    d1: &[i64],
    d2: &[i64],
) -> Result<Vec<i64>, IcmError> {
    if hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    let mut h = hl.to_vec();
    scale_by(&mut h, n, n, c2, 1, ScaleBy::Column)?;
    scale_by(&mut h, n, n, c1, 1, ScaleBy::Row)?;
    let mut h = had_gram_q(&h, n)?;
    scale_by(&mut h, n, n, d2, 1, ScaleBy::Column)?;
    scale_by(&mut h, n, n, d1, 1, ScaleBy::Row)?;
    Ok(h)
}

/// Validate `ws`, `hl` and `g` against `o`/`n`.
fn check_inputs(
    ws: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g: Option<&[i64]>,
) -> Result<(), IcmError> {
    if ws.len() != o * n || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    if let Some(g) = g {
        if g.len() != o {
            return Err(IcmError::Shape);
        }
    }
    Ok(())
}

/// An owned [`Plane`] from its refined code and the input's unchanged scales; `O(o·n)` time and space.
fn into_plane(s: &[i64], idx: Vec<u16>, p: PlaneRef<'_>) -> Plane {
    Plane {
        s: s.to_vec(),
        idx,
        r: p.r.to_vec(),
        c: p.c.to_vec(),
        d: p.d.to_vec(),
    }
}

/// The 256 `E8 ∪ O8` codewords as Q32.32 words `codeword(j)[k]`. Constant size.
#[must_use]
pub fn codebook() -> [[i64; BLOCK]; CODE_LEN] {
    codebook_q()
}

/// Euclidean nearest codebook row of a grid 8-vector, first index on ties, via the codebook's own
/// `Q32.32` rule ([`e8::nearest`]). The ICM block argmin is not this rule: it minimises the block cost
/// in the `H_blk`-weighted metric.
/// # Errors
/// [`IcmError::Overflow`] for an exact `i128` distance overflow.
/// # Complexity
/// `O(256·8)` exact `i128` operations, `O(1)` space: constant in the caller's dimensions.
pub fn nearest_index(code: &[i64; BLOCK]) -> Result<usize, IcmError> {
    e8::nearest(code).map_err(|_| IcmError::Overflow)
}

/// The metric the ICM cost is minimised in: `Hl`, or `Hl` plus the `icm_ridge` diagonal shift
/// `icm_ridge · (n / 896) · mean(diag(Hl))` on the grid when `icm_ridge > 0`; a positive shift that
/// rounds below one raw unit is kept at one raw unit, as `steps::c_step` does.
/// # Errors
/// [`IcmError::Shape`] if `hl.len() != n*n` or `n == 0`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(n²)` time and space: one copy, then `n` diagonal adds.
pub fn ridged_metric(hl: &[i64], n: usize, icm_ridge: i64) -> Result<Vec<i64>, IcmError> {
    if n == 0 || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    ridged_metric_q(hl, n, icm_ridge)
}

/// # Complexity
/// `O(n²)` time and space: one copy, `n` checked diagonal adds.
fn ridged_metric_q(hl: &[i64], n: usize, icm_ridge: i64) -> Result<Vec<i64>, IcmError> {
    let mut h = hl.to_vec();
    if icm_ridge > 0 {
        let mut sum: i128 = 0;
        let mut err: Option<IcmError> = None;
        let mut i = 0;
        while i < n && err.is_none() {
            match qsum(sum, i128::from(hl[i * n + i])) {
                Ok(v) => sum = v,
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mean = q32::narrow(div_round(sum, n as i128)?).ok_or(IcmError::Overflow)?;
        // `n / 896` on the grid, then `icm_ridge · (n/896) · mean`.
        let nq = (n as i128) * 4_294_967_296;
        let frac = q32::narrow(div_round(nq, 896)?).ok_or(IcmError::Overflow)?;
        let add = mul_q(mul_q(icm_ridge, frac)?, mean)?;
        let add = if mean > 0 && add < 1 { ONE } else { add };
        let mut i = 0;
        while i < n && err.is_none() {
            match add_q(h[i * n + i], add) {
                Ok(v) => h[i * n + i] = v,
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
    }
    Ok(h)
}

/// Signed `round(v / d)` (ties away from zero) for `d > 0`; [`IcmError::Overflow`] if `d <= 0` or an
/// intermediate leaves `i128`.
fn div_round(v: i128, d: i128) -> Result<i128, IcmError> {
    if d <= 0 {
        return Err(IcmError::Overflow);
    }
    let half = d / 2;
    if v >= 0 {
        v.checked_add(half).map(|x| x / d)
    } else {
        v.checked_sub(half).map(|x| x / d)
    }
    .ok_or(IcmError::Overflow)
}

/// `plane_w(P) = fwht128(s·d) · r_rep · c`, the plane's weight matrix, row-major `[o, n]`, on the
/// `Q32.32` grid.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`, or a value the grid cannot carry;
/// [`IcmError::Width`] if `n` is not a positive multiple of [`BLOCK`] and [`GROUP`].
/// # Complexity
/// `O(o·n·log2(GROUP))` additions for the transform plus `O(o·n)` checked multiplies; `O(o·n)` space.
pub fn plane_w(p: PlaneRef<'_>, o: usize, n: usize) -> Result<Vec<i64>, IcmError> {
    check_plane(p, o, n)?;
    plane_w_q(p.s, p.r, p.c, p.d, o, n)
}

/// # Complexity
/// `O(o·n)` checked multiplies plus `O(o·n·log2(GROUP))` transform additions; `O(o·n)` space.
pub(super) fn code_transform_q(
    s: &[i64],
    d: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let mut t = s.to_vec();
    scale_by(&mut t, o, n, d, n / GROUP, ScaleBy::Column)?;
    fwht_rows(&mut t, o, n)?;
    Ok(t)
}

/// # Complexity
/// As [`plane_w`].
fn plane_w_q(
    s: &[i64],
    r: &[i64],
    c: &[i64],
    d: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let g = n / GROUP;
    let mut t = code_transform_q(s, d, o, n)?;
    scale_by(&mut t, o, n, c, g, ScaleBy::Column)?;
    scale_by(&mut t, o, n, r, g, ScaleBy::Group)?;
    Ok(t)
}

/// `J(Ws, W, Hl) = Σ_o g_row[o] · (D_o Hl · D_o)`, `D = Ws − W` on the grid, as the `Q32.32`
/// word of the sum; unweighted when `g_row` is `None`.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(o·n²)` multiply-adds (the `D·Hl` product runs on the engine pool), `O(o·n)` space.
pub fn j_rows(
    ws: &[i64],
    w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<i64, IcmError> {
    check_inputs(ws, hl, o, n, g_row)?;
    if w.len() != o * n {
        return Err(IcmError::Shape);
    }
    j_rows_q(ws, w, hl, o, n, g_row)
}

/// # Complexity
/// `O(o·n)` multiply-adds in the row reduction (serial in row order, so the sum is deterministic); the
/// `D·Hl` product it consumes is the pooled [`matmul_q`].
fn j_rows_q(
    ws: &[i64],
    w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<i64, IcmError> {
    let mut d = ws.to_vec();
    bin_into(&mut d, w, BinOp::Sub)?;
    let prod = matmul_q(&d, hl, o, n, n, false)?;
    let mut total: i128 = 0;
    let mut err: Option<IcmError> = None;
    let mut row = 0;
    while row < o && err.is_none() {
        let (dr, pr) = (&d[row * n..(row + 1) * n], &prod[row * n..(row + 1) * n]);
        match dot_q(dr, pr) {
            Ok(j0) => {
                let mut j = j0;
                if let Some(g) = g_row {
                    match mul_q(g[row], j) {
                        Ok(v) => j = v,
                        Err(e) => err = Some(e),
                    }
                }
                if err.is_none() {
                    match qsum(total, i128::from(j)) {
                        Ok(v) => total = v,
                        Err(e) => err = Some(e),
                    }
                }
            }
            Err(e) => err = Some(e),
        }
        row += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    q32::narrow(total).ok_or(IcmError::Overflow)
}

/// One plane's iterated conditional modes: `sweeps` passes over the `n / 8` blocks, each replacing one
/// block's code by the codebook row minimising the block cost in the (optionally `icm_ridge`d) metric,
/// block order reversed on even sweeps. `r`, `c` and `d` are returned unchanged.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not a positive multiple of [`BLOCK`] `*` [`GROUP`]; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `sweeps · (n/8) · O · (BLOCK² + CODE_LEN·(BLOCK+2) + [BLOCK·n] changed)` checked operations over
/// the `n/8` independent blocks of each row, `O(sweeps·n²)` words copied for the block windows, plus
/// `O(o·n²)` setup for `Ws@Hl` and the transforms;
/// `O(n²)` for `H_rot`/`q_all` plus `O(o·n)`.
pub fn e8_icm_step(
    ws: &[i64],
    hl: &[i64],
    p: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
) -> Result<Plane, IcmError> {
    check_inputs(ws, hl, o, n, None)?;
    check_plane(p, o, n)?;
    let metric = ridged_metric_q(hl, n, icm_ridge)?;
    let (g, nb, cb) = (n / GROUP, n / BLOCK, codebook_q());
    // G_rot = fwht128((Ws @ Hl) * c) * d, H_rot = had_gram(Hl ∘ c c^T) ∘ d d^T
    let mut g_rot = matmul_l(ws, &metric, o, n, n, false)?;
    scale_by(&mut g_rot, o, n, p.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g_rot, o, n)?;
    scale_by(&mut g_rot, o, n, p.d, g, ScaleBy::Column)?;
    let h_rot = cross_metric_q(&metric, n, p.c, p.c, p.d, p.d)?;
    // B = s * r_rep, F_res = G_rot - B @ H_rot
    let mut b = p.s.to_vec();
    scale_by(&mut b, o, n, p.r, g, ScaleBy::Group)?;
    let mut f_res = g_rot;
    bin_into(
        &mut f_res,
        &matmul_l(&b, &h_rot, o, n, n, false)?,
        BinOp::Sub,
    )?;
    let q_all = block_quads_q(&h_rot, &cb, nb, n)?;
    let (mut s_cur, mut idx_cur) = (p.s.to_vec(), p.idx.to_vec());
    let cx = SweepCtx {
        h_rot: &h_rot,
        q_all: &q_all,
        cb: &cb,
        n,
        g,
        sweeps,
    };
    // The kernel dispatches the engine pool itself over the operand rows; the per-row decision work is
    // serial on the caller.
    sweep_blocks(cx, &mut f_res, &mut s_cur, &mut idx_cur, p.r)?;
    Ok(into_plane(&s_cur, idx_cur, p))
}

/// `X_m[i₁,i₂] = C_i₁ᵀ H_blk[m] C_i₂` at Q32.32, row-major `[CODE_LEN, CODE_LEN]`, as two kernel
/// products; `hbf`/`cbf` are the flat `[BLOCK, BLOCK]` block and the flat codebook.
/// # Complexity
/// `CODE_LEN·BLOCK² + CODE_LEN²·BLOCK` kernel multiply-accumulates, `O(CODE_LEN²)` space.
fn block_x(hb: &[[i64; BLOCK]; BLOCK], cbf: &[i64]) -> Result<Vec<i64>, IcmError> {
    let mut hbf = vec![0_i64; BLOCK * BLOCK];
    let mut a = 0;
    while a < BLOCK {
        let mut b = 0;
        while b < BLOCK {
            hbf[a * BLOCK + b] = hb[a][b];
            b += 1;
        }
        a += 1;
    }
    let hbt = transpose_q(&hbf, BLOCK, BLOCK);
    let t = nt_q(cbf, CODE_LEN, &hbt, BLOCK, BLOCK)?;
    nt_q(&t, CODE_LEN, cbf, CODE_LEN, BLOCK)
}

/// `Q_all[m][j] = C_jᵀ H_blk[m] C_j` over the `n/8` diagonal blocks: the diagonal of [`block_x`].
fn block_quads_q(
    h: &[i64],
    cb: &[[i64; BLOCK]; CODE_LEN],
    nb: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let cbf = flatten_codebook_q(cb);
    let mut out = vec![0_i64; nb * CODE_LEN];
    let mut err: Option<IcmError> = None;
    let mut m = 0;
    while m < nb && err.is_none() {
        let hb = block_of_q(h, n, m);
        match block_x(&hb, &cbf) {
            Ok(x) => {
                let mut j = 0;
                while j < CODE_LEN {
                    out[m * CODE_LEN + j] = x[j * CODE_LEN + j];
                    j += 1;
                }
            }
            Err(e) => err = Some(e),
        }
        m += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// `X_m[i₁,i₂] = C_i₁ᵀ H_blk[m] C_i₂` for every block, concatenated; `(n/8)·` [`block_x`].
fn block_cross_q(
    h: &[i64],
    cb: &[[i64; BLOCK]; CODE_LEN],
    nb: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let cbf = flatten_codebook_q(cb);
    let mut out = vec![0_i64; nb * CODE_LEN * CODE_LEN];
    let mut err: Option<IcmError> = None;
    let mut m = 0;
    while m < nb && err.is_none() {
        let hb = block_of_q(h, n, m);
        match block_x(&hb, &cbf) {
            Ok(x) => {
                let mut i = 0;
                while i < x.len() {
                    out[m * CODE_LEN * CODE_LEN + i] = x[i];
                    i += 1;
                }
            }
            Err(e) => err = Some(e),
        }
        m += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// The `256 × 8` codebook row-major, flat: the kernel operand of [`block_x`].
fn flatten_codebook_q(cb: &[[i64; BLOCK]; CODE_LEN]) -> Vec<i64> {
    let mut out = vec![0_i64; CODE_LEN * BLOCK];
    let mut j = 0;
    while j < CODE_LEN {
        let mut k = 0;
        while k < BLOCK {
            out[j * BLOCK + k] = cb[j][k];
            k += 1;
        }
        j += 1;
    }
    out
}

/// The read-only operands of [`sweep_blocks`]. Passed by value: the pinned Aeneas refuses a shared
/// reference to a struct holding a slice across the nested sweep loops (`Internal error`).
#[derive(Clone, Copy)]
struct SweepCtx<'a> {
    h_rot: &'a [i64],
    q_all: &'a [i64],
    cb: &'a [[i64; BLOCK]; CODE_LEN],
    n: usize,
    g: usize,
    sweeps: usize,
}

/// The single-plane sweep: `sweeps` passes over the `n / 8` blocks (block order reversed on even
/// sweeps), every product on the limb kernel. Rows are independent — a row's residual and codes
/// depend only on that row — so the row order is free and all `o` rows go to one kernel dispatch:
/// the kernel dispatches the engine pool over its operand rows, and a band per call would pay the
/// call's fixed cost once per band. The per-row decision work is serial on the caller.
///
/// The residual is updated in **lazy batches** of `GROUP / BLOCK` blocks (one `r`-scale period,
/// `GROUP` columns): a block's change is added to the group's `[o, GROUP]` accumulator, and only at
/// the group's last block is that accumulator contracted once against the `[n, GROUP]` transposed
/// window and subtracted from `f_res`. A block's decision reads the residual as of the last batch
/// flush, so the in-batch part of it (the earlier blocks of its own group) is applied to that block's
/// `BLOCK` columns as a separate `[o, BLOCK]` product against the running accumulator — the term that
/// keeps the sweep a coordinate descent. The group is exactly one `r` period, so the row scale is a
/// per-row constant inside it and factors into the accumulator once.
///
/// # Complexity
/// Per sweep and group: one `O(n·GROUP)` window copy, and one `O(o·GROUP)` accumulator clear. Per
/// sweep and block: one `O(o·BLOCK)` gather, one `O(o·BLOCK²)` block product and one
/// `O(o·CODE_LEN·BLOCK)` code-dot product (both times the limb pairs), the `O(o·CODE_LEN)` argmin,
/// and one `O(o·BLOCK·GROUP)` in-batch product — zero on the group's first block. Per sweep, at each
/// group's last block: one `O(o·n·GROUP)` kernel product, i.e. `O(o·n²)` per sweep, plus `O(o·n)`
/// elementwise subtractions. Space: `O(o·BLOCK)` gather, block and correction buffers,
/// `O(o·GROUP)` accumulator and `O(n·GROUP)` window.
fn sweep_blocks(
    cx: SweepCtx<'_>,
    f_res: &mut [i64],
    s_cur: &mut [i64],
    idx: &mut [u16],
    r: &[i64],
) -> Result<(), IcmError> {
    let (n, nb, g) = (cx.n, cx.n / BLOCK, cx.g);
    let o = f_res.len() / n;
    let per_group = GROUP / BLOCK;
    let cbf = flatten_codebook_q(cx.cb);
    let mut s_blk = vec![0_i64; o * BLOCK];
    let mut f_blk = vec![0_i64; o * BLOCK];
    let mut blk_op = vec![0_i64; BLOCK * BLOCK];
    let mut corr = vec![0_i64; o * BLOCK];
    let mut grp_cols = vec![0_i64; n * GROUP];
    let mut grp_chg = vec![0_i64; o * GROUP];
    let mut err: Option<IcmError> = None;
    let mut sw = 0;
    while sw < cx.sweeps && err.is_none() {
        let mut step = 0;
        while step < nb && err.is_none() {
            let mut m = step;
            if sw % 2 == 0 {
                m = nb - 1 - step;
            }
            let b0 = m * BLOCK;
            let gi = b0 / GROUP;
            let j0 = b0 - gi * GROUP;
            let first = step % per_group == 0;
            let last = step % per_group == per_group - 1;
            if first {
                // The group's `GROUP × n` columns transposed to `[col, j]`: the kernel's `b` rows are
                // both this block's `[b0, b0 + BLOCK)` sub-block and the whole batch window.
                let mut col = 0;
                while col < n {
                    let mut j = 0;
                    while j < GROUP {
                        grp_cols[col * GROUP + j] = cx.h_rot[(gi * GROUP + j) * n + col];
                        j += 1;
                    }
                    col += 1;
                }
                let mut i = 0;
                while i < o * GROUP {
                    grp_chg[i] = 0;
                    i += 1;
                }
            }
            // The block's diagonal `BLOCK × BLOCK` metric block, row `j` entry `k` = `H[b0+k][b0+j]`.
            let mut jj = 0;
            while jj < BLOCK {
                let mut kk = 0;
                while kk < BLOCK {
                    blk_op[jj * BLOCK + kk] = grp_cols[(b0 + jj) * GROUP + j0 + kk];
                    kk += 1;
                }
                jj += 1;
            }
            // The in-batch part of this block's residual: the group's earlier blocks against the
            // block's own columns. Zero on the group's first block, which reads the flushed residual.
            if !first && err.is_none() {
                match nt_q(
                    &grp_chg,
                    o,
                    &grp_cols[b0 * GROUP..(b0 + BLOCK) * GROUP],
                    BLOCK,
                    GROUP,
                ) {
                    Ok(v) => {
                        let mut i = 0;
                        while i < o * BLOCK {
                            corr[i] = v[i];
                            i += 1;
                        }
                    }
                    Err(e) => err = Some(e),
                }
            }
            let mut row = 0;
            while row < o {
                let mut k = 0;
                while k < BLOCK {
                    s_blk[row * BLOCK + k] = s_cur[row * n + b0 + k];
                    k += 1;
                }
                row += 1;
            }
            match nt_q(&s_blk, o, &blk_op, BLOCK, BLOCK) {
                Ok(v) => {
                    let mut row = 0;
                    while row < o && err.is_none() {
                        let a = r[row * g + gi];
                        let mut k = 0;
                        while k < BLOCK && err.is_none() {
                            match mul_q(a, v[row * BLOCK + k]) {
                                Ok(w) => match add_q(f_res[row * n + b0 + k], w) {
                                    Ok(z) => f_blk[row * BLOCK + k] = z,
                                    Err(e) => err = Some(e),
                                },
                                Err(e) => err = Some(e),
                            }
                            k += 1;
                        }
                        row += 1;
                    }
                }
                Err(e) => err = Some(e),
            }
            if !first && err.is_none() {
                let mut row = 0;
                while row < o && err.is_none() {
                    let mut k = 0;
                    while k < BLOCK && err.is_none() {
                        match sub_q(f_blk[row * BLOCK + k], corr[row * BLOCK + k]) {
                            Ok(z) => f_blk[row * BLOCK + k] = z,
                            Err(e) => err = Some(e),
                        }
                        k += 1;
                    }
                    row += 1;
                }
            }
            if err.is_none() {
                match nt_q(&f_blk, o, &cbf, CODE_LEN, BLOCK) {
                    Ok(dots) => {
                        let mut row = 0;
                        while row < o && err.is_none() {
                            let a = r[row * g + gi];
                            let (aa, two_a) = row_scale(a);
                            let qb = m * CODE_LEN;
                            let db = row * CODE_LEN;
                            let mut best = 0_usize;
                            match best_code_q(
                                aa,
                                two_a,
                                &cx.q_all[qb..qb + CODE_LEN],
                                &dots[db..db + CODE_LEN],
                            ) {
                                Ok(v) => best = v,
                                Err(e) => err = Some(e),
                            }
                            let mut k = 0;
                            while k < BLOCK && err.is_none() {
                                match sub_q(cx.cb[best][k], s_cur[row * n + b0 + k]) {
                                    Ok(d) => {
                                        match mul_q(a, d) {
                                            Ok(w) => {
                                                grp_chg[row * GROUP + j0 + k] = w;
                                            }
                                            Err(e) => err = Some(e),
                                        }
                                        s_cur[row * n + b0 + k] = cx.cb[best][k];
                                    }
                                    Err(e) => err = Some(e),
                                }
                                k += 1;
                            }
                            if err.is_none() {
                                idx[row * nb + m] = best as u16;
                            }
                            row += 1;
                        }
                    }
                    Err(e) => err = Some(e),
                }
            }
            if last && err.is_none() {
                // The batch flush: the whole group's change, contracted once against the group's
                // columns and subtracted from every column of the residual. One call over all `o`
                // rows: the kernel's `p × GROUP` weight plane is built once per call, so the rows
                // amortise it and the band split would rebuild it per band.
                match nt_q(&grp_chg, o, &grp_cols, n, GROUP) {
                    Ok(e) => {
                        let mut row = 0;
                        while row < o && err.is_none() {
                            let mut col = 0;
                            while col < n && err.is_none() {
                                match sub_q(f_res[row * n + col], e[row * n + col]) {
                                    Ok(z) => f_res[row * n + col] = z,
                                    Err(e2) => err = Some(e2),
                                }
                                col += 1;
                            }
                            row += 1;
                        }
                    }
                    Err(e) => err = Some(e),
                }
            }
            step += 1;
        }
        sw += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Joint block ICM over two planes: `sweeps` passes over the `n / 8` blocks (or the `blocks` subset, in
/// the given order), each replacing both planes' codes of one block by the pair `(i₁, i₂)` minimising
/// `2 a₁a₂ X_m[i₁,i₂] + A[i₁] + B[i₂]`. `row_chunk` bounds the rows one pool task works on for a
/// `blocks` subset; a full sweep runs one band, because the group flush contracts the whole band
/// against `n` columns in one dispatch.
///
/// This is the `T = 0` control: it is [`e8_joint_icm_2plane_annealed`] with no schedule, so every
/// decision is the exact [`best_pair_q`] argmin and `J` is non-increasing per sweep.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not
/// a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::ZeroChunk`] if `row_chunk == 0`;
/// [`IcmError::BlockOutOfRange`] for a block index `>= n / BLOCK`; [`IcmError::Overflow`] off the grid.
#[allow(
    clippy::too_many_arguments,
    reason = "the option surface (sweeps, icm_ridge, g_row, row_chunk, blocks) mirrors the Python \
              signature; a bag struct would hide which arguments are metric options and which are \
              shapes"
)]
pub fn e8_joint_icm_2plane(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
    g_row: Option<&[i64]>,
    row_chunk: usize,
    blocks: Option<&[usize]>,
) -> Result<(Plane, Plane, JointStats), IcmError> {
    e8_joint_icm_2plane_annealed(
        ws, hl, p1, p2, o, n, sweeps, icm_ridge, g_row, row_chunk, blocks, None,
    )
}

/// [`e8_joint_icm_2plane`] with a finite-temperature schedule: with `Some(schedule)` every block's
/// pair is drawn from the exact `CODE_LEN²` Boltzmann table at that sweep's `beta` instead of taken
/// by argmin, so the sweep can take an uphill move a coordinate descent cannot. `None` is the
/// `T = 0` control, decision for decision identical to [`e8_joint_icm_2plane`].
///
/// With a schedule `J` is no longer monotone: the draw is the folding `gibbs_variational_principle`
/// conditional, and a cooling ramp is its `funnelFreeEnergy` sweep. The rows' xorshift states are
/// seeded from the schedule's seed and the row index, so the same schedule and inputs reproduce the
/// same planes.
/// # Errors
/// As [`e8_joint_icm_2plane`], plus [`IcmError::Overflow`] if a cost table or a Boltzmann weight
/// leaves the grid, and [`IcmError::Shape`] if `schedule.sweeps() == 0` (refused at construction).
/// # Complexity
/// `sweeps · (n/8) · O · (4·CODE_LEN·BLOCK + 2·BLOCK² + CODE_LEN² + [2·BLOCK·n] changed)` checked
/// operations for the pair decision and the two residual updates, plus `O(n²·log2(GROUP))` for the
/// three `had_gram` passes and `(n/8)·CODE_LEN²·BLOCK` for `X`; the pair decision's rows are split
/// across the engine pool in `width_global()` chunks, and every other pass runs on the caller. The
/// scheduled path adds one `CODE_LEN²` `i128` table rebuild per row and block, reused per chunk of a
/// block's rows.
/// `O(o·n)` for `s`, `F`, `G`, `B`, `O(n²)` for `H11/H22/H12`, `(n/8)·CODE_LEN²` `i64` for `X`,
/// `O(rows)` picks plus `O(rows·CODE_LEN²)` `i128` for the cost buffers.
#[allow(
    clippy::too_many_arguments,
    reason = "the option surface (sweeps, icm_ridge, g_row, row_chunk, blocks, schedule) mirrors the \
              coordinate-descent signature plus the one annealing parameter; a bag struct would hide \
              which arguments are metric options and which are shapes"
)]
pub fn e8_joint_icm_2plane_annealed(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
    g_row: Option<&[i64]>,
    row_chunk: usize,
    blocks: Option<&[usize]>,
    schedule: Option<&BetaSchedule>,
) -> Result<(Plane, Plane, JointStats), IcmError> {
    check_inputs(ws, hl, o, n, g_row)?;
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    if row_chunk == 0 {
        return Err(IcmError::ZeroChunk);
    }
    if let Some(s) = schedule {
        if s.sweeps() == 0 {
            return Err(IcmError::Shape);
        }
    }
    let nb = n / BLOCK;
    if let Some(bs) = blocks {
        for &m in bs {
            if m >= nb {
                return Err(IcmError::BlockOutOfRange);
            }
        }
    }
    let metric = ridged_metric_q(hl, n, icm_ridge)?;
    let (g, cb) = (n / GROUP, codebook_q());
    let mut wsum = plane_w_q(p1.s, p1.r, p1.c, p1.d, o, n)?;
    bin_into(
        &mut wsum,
        &plane_w_q(p2.s, p2.r, p2.c, p2.d, o, n)?,
        BinOp::Add,
    )?;
    let j_before = j_rows_q(ws, &wsum, &metric, o, n, g_row)?;
    let (mut s1, mut s2) = (p1.s.to_vec(), p2.s.to_vec());
    let (mut idx1, mut idx2) = (p1.idx.to_vec(), p2.idx.to_vec());
    // G_k = fwht128((Ws @ Hl) * c_k) * d_k
    let wh = matmul_l(ws, &metric, o, n, n, false)?;
    let mut g1 = wh.clone();
    scale_by(&mut g1, o, n, p1.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g1, o, n)?;
    scale_by(&mut g1, o, n, p1.d, g, ScaleBy::Column)?;
    let mut g2 = wh;
    scale_by(&mut g2, o, n, p2.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g2, o, n)?;
    scale_by(&mut g2, o, n, p2.d, g, ScaleBy::Column)?;
    // H_jk = had_gram(Hl ∘ c_j c_k^T) ∘ d_j d_k^T
    let h11 = cross_metric_q(&metric, n, p1.c, p1.c, p1.d, p1.d)?;
    let h22 = cross_metric_q(&metric, n, p2.c, p2.c, p2.d, p2.d)?;
    let h12 = cross_metric_q(&metric, n, p1.c, p2.c, p1.d, p2.d)?;
    // F_1 = G_1 - B_1 H_11 - B_2 H_12^T, F_2 = G_2 - B_2 H_22 - B_1 H_12
    let mut b1 = s1.clone();
    scale_by(&mut b1, o, n, p1.r, g, ScaleBy::Group)?;
    let mut b2 = s2.clone();
    scale_by(&mut b2, o, n, p2.r, g, ScaleBy::Group)?;
    let mut f1 = g1;
    bin_into(&mut f1, &matmul_l(&b1, &h11, o, n, n, false)?, BinOp::Sub)?;
    bin_into(&mut f1, &matmul_l(&b2, &h12, o, n, n, true)?, BinOp::Sub)?;
    let mut f2 = g2;
    bin_into(&mut f2, &matmul_l(&b2, &h22, o, n, n, false)?, BinOp::Sub)?;
    bin_into(&mut f2, &matmul_l(&b1, &h12, o, n, n, false)?, BinOp::Sub)?;
    let (q1v, q2v) = (
        block_quads_q(&h11, &cb, nb, n)?,
        block_quads_q(&h22, &cb, nb, n)?,
    );
    let x_all = block_cross_q(&h12, &cb, nb, n)?;
    let mut moves = 0usize;
    let mut rows: Vec<JointRow<'_>> = Vec::with_capacity(o);
    let mut rest_f1: &mut [i64] = f1.as_mut_slice();
    let mut rest_f2: &mut [i64] = f2.as_mut_slice();
    let mut rest_s1: &mut [i64] = s1.as_mut_slice();
    let mut rest_s2: &mut [i64] = s2.as_mut_slice();
    let mut rest_i1: &mut [u16] = idx1.as_mut_slice();
    let mut rest_i2: &mut [u16] = idx2.as_mut_slice();
    let mut r = 0;
    while r < o {
        let (c_f1, t_f1) = rest_f1.split_at_mut(n);
        let (c_f2, t_f2) = rest_f2.split_at_mut(n);
        let (c_s1, t_s1) = rest_s1.split_at_mut(n);
        let (c_s2, t_s2) = rest_s2.split_at_mut(n);
        let (c_i1, t_i1) = rest_i1.split_at_mut(nb);
        let (c_i2, t_i2) = rest_i2.split_at_mut(nb);
        rows.push(JointRow {
            f1: c_f1,
            f2: c_f2,
            s1: c_s1,
            s2: c_s2,
            idx1: c_i1,
            idx2: c_i2,
        });
        rest_f1 = t_f1;
        rest_f2 = t_f2;
        rest_s1 = t_s1;
        rest_s2 = t_s2;
        rest_i1 = t_i1;
        rest_i2 = t_i2;
        r += 1;
    }
    // The kernel owns the engine pool, so the bands' products run on the caller; the pair decision is
    // the one pass inside a band that dispatches the pool itself, in row chunks. A full sweep runs one
    // band: the group flush contracts every row of the band against `n` columns in one dispatch, and a
    // band per `row_chunk` would pay the call's fixed cost 32 times over at `o = 896`. A `blocks`
    // subset — whose order is caller-given — keeps the band split.
    let per = if blocks.is_some() {
        row_chunk.min(rows_per_task(o)).max(1)
    } else {
        rows.len().max(1)
    };
    let task = JointSweep {
        h11: &h11,
        h22: &h22,
        h12: &h12,
        q1: &q1v,
        q2: &q2v,
        x_all: &x_all,
        cb: &cb,
        r1: p1.r,
        r2: p2.r,
        blocks,
        n,
        g,
        nb,
        sweeps,
    };
    // As in [`e8_icm_step`]: the kernel owns the engine pool, so the bands run on the caller.
    // One xorshift state per row, from the schedule's seed and the row index, so a draw does not
    // depend on where the band boundaries fall. No schedule leaves the states unused (the `T = 0`
    // argmin path draws nothing).
    let mut row_rng = vec![0_u64; rows.len()];
    if let Some(s) = schedule {
        let mut r = 0_usize;
        while r < row_rng.len() {
            let mut state = s.seed() ^ mix64(r as u64);
            if state == 0 {
                state = 0x9E37_79B9_7F4A_7C15;
            }
            row_rng[r] = state;
            r += 1;
        }
    }
    let mut lo = 0usize;
    while lo < rows.len() {
        let hi = (lo + per).min(rows.len());
        moves += joint_band(&task, &mut rows[lo..hi], lo, schedule, &mut row_rng[lo..hi])?;
        lo = hi;
    }
    let mut wsum = plane_w_q(&s1, p1.r, p1.c, p1.d, o, n)?;
    bin_into(
        &mut wsum,
        &plane_w_q(&s2, p2.r, p2.c, p2.d, o, n)?,
        BinOp::Add,
    )?;
    let stats = JointStats {
        j_before,
        j_after: j_rows_q(ws, &wsum, &metric, o, n, g_row)?,
        moves,
    };
    Ok((into_plane(&s1, idx1, p1), into_plane(&s2, idx2, p2), stats))
}

/// One row of the joint sweep's mutable state: two residual fields, two sign fields, two code rows.
struct JointRow<'a> {
    f1: &'a mut [i64],
    f2: &'a mut [i64],
    s1: &'a mut [i64],
    s2: &'a mut [i64],
    idx1: &'a mut [u16],
    idx2: &'a mut [u16],
}

/// The read-only operands of the joint sweep.
struct JointSweep<'a> {
    h11: &'a [i64],
    h22: &'a [i64],
    h12: &'a [i64],
    q1: &'a [i64],
    q2: &'a [i64],
    x_all: &'a [i64],
    cb: &'a [[i64; BLOCK]; CODE_LEN],
    r1: &'a [i64],
    r2: &'a [i64],
    blocks: Option<&'a [usize]>,
    n: usize,
    g: usize,
    nb: usize,
    sweeps: usize,
}

/// One row's joint pair decision: the `CODE_LEN²`-way `flat` index and the row's xorshift state after
/// the draw. Written per row by [`pair_decisions`], so the decision phase can run in row chunks on the
/// engine pool while both quantities stay row-local.
#[derive(Clone, Copy)]
struct PairPick {
    /// `i₁·CODE_LEN + i₂` as [`best_pair_q`] or [`gibbs_choice`] returns it, below `CODE_LEN² = 2¹⁶`.
    flat: u16,
    /// The row's xorshift state (unchanged on the `T = 0` path, which draws nothing).
    rng: u64,
}

/// One block's per-row decision over a chunk of the band's rows. Row `row` reads only
/// `z1/z2[row·CODE_LEN..]`, `r1/r2[(base+row)·g + gidx]` and its own `picks[row].rng`, and writes only
/// `picks[row]`, so a chunk's decisions are the serial loop's.
struct PairRows<'a> {
    /// Block quadratic forms, `[nb·CODE_LEN]`, first plane.
    q1: &'a [i64],
    /// Block quadratic forms, `[nb·CODE_LEN]`, second plane.
    q2: &'a [i64],
    /// The `[nb, CODE_LEN, CODE_LEN]` cross tables of the two planes.
    x_all: &'a [i64],
    /// Group scales `[o·g]`, first plane.
    r1: &'a [i64],
    /// Group scales `[o·g]`, second plane.
    r2: &'a [i64],
    /// The band's `[rows, CODE_LEN]` codebook dots, first plane.
    z1: &'a [i64],
    /// The band's `[rows, CODE_LEN]` codebook dots, second plane.
    z2: &'a [i64],
    /// Group scales per row.
    g: usize,
    /// The band's first row in the plane (the `r` index offset).
    base: usize,
    /// The block's `r` group index.
    gidx: usize,
    /// The block.
    m: usize,
    /// The inverse temperature in force, `None` for the `T = 0` argmin.
    beta: Option<i64>,
    /// Lowest-offset refusal recorded by the tasks.
    first: &'a FirstErr<usize, IcmError>,
}

impl PairRows<'_> {
    /// Rows `start .. start + piece.len()`: one exact `CODE_LEN²` cost table per row and, with `beta`,
    /// one [`gibbs_choice`] draw of that table.
    ///
    /// # Errors
    /// [`IcmError::Overflow`] off the grid.
    /// # Complexity
    /// Time: `rows·(2·CODE_LEN + CODE_LEN²)` checked operations. Space: one reused `CODE_LEN²` `i128`
    /// table per task on the scheduled path, `O(1)` on the `T = 0` path.
    fn pick_rows(&self, start: usize, piece: &mut [PairPick]) -> Result<(), IcmError> {
        let xm = &self.x_all[self.m * CODE_LEN * CODE_LEN..(self.m + 1) * CODE_LEN * CODE_LEN];
        let mut costs: Vec<i128> = Vec::new();
        for (i, pick) in piece.iter_mut().enumerate() {
            let row = start + i;
            let r = self.base + row;
            let (av1, av2) = (
                i128::from(self.r1[r * self.g + self.gidx]),
                i128::from(self.r2[r * self.g + self.gidx]),
            );
            let (aa1, two_a1) = (av1 * av1, av1 * 2);
            let (aa2, two_a2) = (av2 * av2, av2 * 2);
            let two_a12 = qmul(qmul(av1, av2)?, 2)?;
            let (mut av, mut bv) = ([0_i128; CODE_LEN], [0_i128; CODE_LEN]);
            for j in 0..CODE_LEN {
                av[j] = block_cost(
                    aa1,
                    two_a1,
                    self.q1[self.m * CODE_LEN + j],
                    self.z1[row * CODE_LEN + j],
                )?;
                bv[j] = block_cost(
                    aa2,
                    two_a2,
                    self.q2[self.m * CODE_LEN + j],
                    self.z2[row * CODE_LEN + j],
                )?;
            }
            let flat = match self.beta {
                Some(beta) => {
                    pair_costs_into(two_a12, &av, xm, &bv, &mut costs)?;
                    gibbs_choice(&costs, beta, &mut pick.rng)?
                }
                None => best_pair_q(two_a12, &av, xm, &bv)?,
            };
            // `flat < CODE_LEN² == 2¹⁶`, so the narrowing never leaves `u16`.
            pick.flat = u16::try_from(flat).map_err(|_| IcmError::Overflow)?;
        }
        Ok(())
    }
}

impl ChunkTask<PairPick> for PairRows<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [PairPick]) {
        if let Err(e) = self.pick_rows(start, piece) {
            self.first.record(start, e);
        }
    }
}

/// The row-wise pair decision of one block over the whole band: every row's `CODE_LEN²` cost table and
/// its draw, in `width_global()` row chunks on the engine pool.
///
/// The rows are independent, so the chunking changes no decision: row `row` reads only its own `z`,
/// `r` and xorshift state, so the returned `flat` and the states written back are the serial row
/// loop's, bit for bit, on the `T = 0` argmin and the finite-temperature draw alike.
///
/// # Errors
/// The refusal of the lowest-offset failing chunk, e.g. [`IcmError::Overflow`].
/// # Complexity
/// Time: `rows·(2·CODE_LEN + CODE_LEN²)` checked operations, split across the pool, plus `O(rows)` for
/// the xorshift state copy in and out. Space: `O(rows)` picks plus one `CODE_LEN²` `i128` table per
/// participant on the scheduled path.
fn pair_decisions(
    cx: &JointSweep<'_>,
    z1: &[i64],
    z2: &[i64],
    start: usize,
    m: usize,
    beta: Option<i64>,
    rng: &mut [u64],
    picks: &mut [PairPick],
) -> Result<(), IcmError> {
    let rows = picks.len();
    for (pick, &state) in picks.iter_mut().zip(rng.iter()) {
        pick.rng = state;
    }
    let first: FirstErr<usize, IcmError> = FirstErr::new();
    let chunk = rows.div_ceil(width_global()).max(1);
    chunks_mut_global(
        picks,
        chunk,
        &PairRows {
            q1: cx.q1,
            q2: cx.q2,
            x_all: cx.x_all,
            r1: cx.r1,
            r2: cx.r2,
            z1,
            z2,
            g: cx.g,
            base: start,
            gidx: m * BLOCK / GROUP,
            m,
            beta,
            first: &first,
        },
    );
    if let Some((_, e)) = first.take() {
        return Err(e);
    }
    for (pick, state) in picks.iter().zip(rng.iter_mut()) {
        *state = pick.rng;
    }
    Ok(())
}

/// One band of the joint sweep, every product on the limb kernel. The band's per-row slices are
/// gathered into flat `[rows, ·]` buffers so one kernel product serves the whole band: the two block
/// products are `rows × 2·BLOCK × BLOCK`, the codebook product `rows × CODE_LEN × BLOCK` per plane,
/// and the residual one `GROUP`-wide batch per group.
///
/// The residual is updated in **lazy batches** of `GROUP / BLOCK` blocks (`GROUP` columns, one `r`
/// period, as in [`sweep_blocks`]). A block's change is written unscaled into the group's
/// `[rows, GROUP]` accumulator; the block's decision reads the in-batch part of its own `BLOCK`
/// columns as one `[rows, BLOCK] × [BLOCK, GROUP]` product per contraction against that accumulator
/// — the term that keeps the sweep a coordinate descent — and at the group's last block one
/// `[moved, GROUP] × [n, GROUP]` product per contraction applies the whole group's change to every
/// column. Both planes' changes contract both residuals, so a group is four contractions (`H₁₁Δ₁`,
/// `H₁₂Δ₂`, `H₂₂Δ₂`, `H₁₂ᵀΔ₁`). The deltas stay unscaled and the `a₁`/`a₂` combine is the same exact
/// `i128` sum [`round_shift`]ed once per element, so the scaling is unchanged.
///
/// A `blocks` subset keeps the per-block residual ([`joint_bands_per_block`]): its order is
/// caller-given, so a `GROUP`'s blocks need not be visited consecutively or completely, and the
/// in-batch prefix is only the coordinate-descent term when they are.
///
/// # Complexity
/// Per sweep, per group: four `n × GROUP` window copies, one `O(rows·GROUP)` accumulator clear and
/// four `moved × n × GROUP` flush products. Per sweep, per block: one `O(BLOCK²)` diagonal gather,
/// two `rows × 2·BLOCK × BLOCK` block products, two `rows × CODE_LEN × BLOCK` codebook products, the
/// `CODE_LEN²`-way pair search (`O(rows·CODE_LEN²)` checked operations with `2·a₁a₂` hoisted, in
/// `width_global()` row chunks on the engine pool: see [`pair_decisions`]) and,
/// off the group's first block with a change in the group, four `rows × BLOCK × GROUP` in-batch
/// products. `O(rows·n)` for the gathered band, `O(n·GROUP)` for the windows and `O(rows·GROUP)` for
/// the flush gathers; a scheduled band adds `O(rows·CODE_LEN²)` `i128` for the per-task cost tables.
///
/// The pair decision is [`best_pair_q`] when `beta` is `None` and one [`gibbs_choice`] draw of the
/// same table at `beta.beta_at(sw)` otherwise; `rng[row]` is that row's xorshift state. Both are row-
/// local, so the pool split runs them per row without changing a decision.
fn joint_band(
    cx: &JointSweep<'_>,
    piece: &mut [JointRow<'_>],
    start: usize,
    beta: Option<&BetaSchedule>,
    rng: &mut [u64],
) -> Result<usize, IcmError> {
    let n = cx.n;
    let rows = piece.len();
    let mut f1 = vec![0_i64; rows * n];
    let mut f2 = vec![0_i64; rows * n];
    let mut s1 = vec![0_i64; rows * n];
    let mut s2 = vec![0_i64; rows * n];
    for (rr, row) in piece.iter().enumerate() {
        f1[rr * n..][..n].copy_from_slice(row.f1);
        f2[rr * n..][..n].copy_from_slice(row.f2);
        s1[rr * n..][..n].copy_from_slice(row.s1);
        s2[rr * n..][..n].copy_from_slice(row.s2);
    }
    let moved = if cx.blocks.is_some() {
        joint_bands_per_block(
            cx, piece, start, beta, rng, &mut f1, &mut f2, &mut s1, &mut s2,
        )?
    } else {
        joint_bands_batched(
            cx, piece, start, beta, rng, &mut f1, &mut f2, &mut s1, &mut s2,
        )?
    };
    for (rr, row) in piece.iter_mut().enumerate() {
        row.f1.copy_from_slice(&f1[rr * n..][..n]);
        row.f2.copy_from_slice(&f2[rr * n..][..n]);
        row.s1.copy_from_slice(&s1[rr * n..][..n]);
        row.s2.copy_from_slice(&s2[rr * n..][..n]);
    }
    Ok(moved)
}

/// The per-block residual of [`joint_band`], kept for a caller-given `blocks` subset.
#[allow(
    clippy::too_many_arguments,
    reason = "the band's four carried fields are separate slices, as in `sweep_blocks`: the pinned \
              Aeneas refuses a reference to a struct holding a slice across the nested sweep loops"
)]
fn joint_bands_per_block(
    cx: &JointSweep<'_>,
    piece: &mut [JointRow<'_>],
    start: usize,
    beta: Option<&BetaSchedule>,
    rng: &mut [u64],
    f1: &mut [i64],
    f2: &mut [i64],
    s1: &mut [i64],
    s2: &mut [i64],
) -> Result<usize, IcmError> {
    let (n, nb) = (cx.n, cx.nb);
    let rows = piece.len();
    let cb = cx.cb.as_flattened();
    let mut hc11 = vec![0_i64; BLOCK * n];
    let mut hc22 = vec![0_i64; BLOCK * n];
    let mut hc12 = vec![0_i64; BLOCK * n];
    let mut cx12 = vec![0_i64; BLOCK * n];
    let mut bb1 = vec![0_i64; 2 * BLOCK * BLOCK];
    let mut bb2 = vec![0_i64; 2 * BLOCK * BLOCK];
    let mut s1b = vec![0_i64; rows * BLOCK];
    let mut s2b = vec![0_i64; rows * BLOCK];
    let mut f1b = vec![0_i64; rows * BLOCK];
    let mut f2b = vec![0_i64; rows * BLOCK];
    let mut d1b = vec![0_i64; rows * BLOCK];
    let mut d2b = vec![0_i64; rows * BLOCK];
    let mut c1b = vec![0_i64; rows * 2 * BLOCK];
    let mut c2b = vec![0_i64; rows * 2 * BLOCK];
    let mut m1: Vec<(usize, i128, i128)> = Vec::with_capacity(rows);
    let mut m2: Vec<(usize, i128, i128)> = Vec::with_capacity(rows);
    let mut picks = vec![PairPick { flat: 0, rng: 0 }; rows];
    let mut moved = 0_usize;
    for sw in 0..cx.sweeps {
        let count = cx.blocks.map_or(nb, <[usize]>::len);
        for step in 0..count {
            let m = match cx.blocks {
                Some(bs) => bs[step],
                None if sw % 2 == 0 => nb - 1 - step,
                None => step,
            };
            let (b0, gidx) = (m * BLOCK, m * BLOCK / GROUP);
            for k in 0..BLOCK {
                for col in 0..n {
                    hc11[col * BLOCK + k] = cx.h11[(b0 + k) * n + col];
                    hc22[col * BLOCK + k] = cx.h22[(b0 + k) * n + col];
                    hc12[col * BLOCK + k] = cx.h12[(b0 + k) * n + col];
                    cx12[col * BLOCK + k] = cx.h12[col * n + b0 + k];
                }
            }
            // `b[k][l] = h[(b0+k)·n + b0+l]` for the three blocks the gradient needs. Row `k` of
            // the kernel operand is the contraction against `s`, so: `bb1` is `[col_k(H₁₁);
            // col_k(H₁₂)]` for `d₁₁` and `d₁₂`, `bb2` is `[row_k(H₁₂); col_k(H₂₂)]` for `d₁₂ᵀ`
            // and `d₂₂` (`hb12[k][l]` is a row, `hb12[l][k]` a column; `H₁₂` is not symmetric).
            for j in 0..BLOCK {
                for l in 0..BLOCK {
                    // Written directly rather than through a shared scratch block: the second half
                    // of `bb1` is a column of `H₁₂`, the first half of `bb2` its row.
                    bb1[j * BLOCK + l] = hc11[(b0 + j) * BLOCK + l];
                    bb1[(BLOCK + j) * BLOCK + l] = hc12[(b0 + j) * BLOCK + l];
                    bb2[j * BLOCK + l] = hc12[(b0 + l) * BLOCK + j];
                    bb2[(BLOCK + j) * BLOCK + l] = hc22[(b0 + j) * BLOCK + l];
                }
            }
            for row in 0..rows {
                s1b[row * BLOCK..][..BLOCK].copy_from_slice(&s1[row * n + b0..][..BLOCK]);
                s2b[row * BLOCK..][..BLOCK].copy_from_slice(&s2[row * n + b0..][..BLOCK]);
            }
            let p1v = nt_q(&s1b, rows, &bb1, 2 * BLOCK, BLOCK)?;
            let p2v = nt_q(&s2b, rows, &bb2, 2 * BLOCK, BLOCK)?;
            for row in 0..rows {
                let (a1, a2) = (
                    cx.r1[(start + row) * cx.g + gidx],
                    cx.r2[(start + row) * cx.g + gidx],
                );
                let (av1, av2) = (i128::from(a1), i128::from(a2));
                let o1 = row * 2 * BLOCK;
                for k in 0..BLOCK {
                    // `d₁₁ = s₁·H₁₁ᵀ`, `d₁₂ = s₁·H₁₂`, `d₁₂ᵀ = s₂·H₁₂`, `d₂₂ = s₂·H₂₂ᵀ`.
                    let u1 = qsum(
                        qmul(av1, i128::from(p1v[o1 + k]))?,
                        qmul(av2, i128::from(p2v[o1 + k]))?,
                    )?;
                    let u2 = qsum(
                        qmul(av2, i128::from(p2v[o1 + BLOCK + k]))?,
                        qmul(av1, i128::from(p1v[o1 + BLOCK + k]))?,
                    )?;
                    f1b[row * BLOCK + k] = add_q(f1[row * n + b0 + k], round_shift(u1, 32)?)?;
                    f2b[row * BLOCK + k] = add_q(f2[row * n + b0 + k], round_shift(u2, 32)?)?;
                }
            }
            let z1 = nt_q(&f1b, rows, cb, CODE_LEN, BLOCK)?;
            let z2 = nt_q(&f2b, rows, cb, CODE_LEN, BLOCK)?;
            pair_decisions(
                cx,
                &z1,
                &z2,
                start,
                m,
                beta.map(|sched| sched.beta_at(sw)),
                rng,
                &mut picks,
            )?;
            m1.clear();
            m2.clear();
            for row in 0..rows {
                let (a1, a2) = (
                    cx.r1[(start + row) * cx.g + gidx],
                    cx.r2[(start + row) * cx.g + gidx],
                );
                let (av1, av2) = (i128::from(a1), i128::from(a2));
                let flat = usize::from(picks[row].flat);
                let (i1s, i2s) = (flat / CODE_LEN, flat % CODE_LEN);
                let mut ch1 = false;
                let mut ch2 = false;
                for k in 0..BLOCK {
                    let n1 = sub_q(cx.cb[i1s][k], s1[row * n + b0 + k])?;
                    let n2 = sub_q(cx.cb[i2s][k], s2[row * n + b0 + k])?;
                    ch1 |= n1 != 0;
                    ch2 |= n2 != 0;
                    d1b[row * BLOCK + k] = n1;
                    d2b[row * BLOCK + k] = n2;
                    s1[row * n + b0 + k] = cx.cb[i1s][k];
                    s2[row * n + b0 + k] = cx.cb[i2s][k];
                }
                piece[row].idx1[m] = i1s as u16;
                piece[row].idx2[m] = i2s as u16;
                if ch1 || ch2 {
                    // Both residuals contract *both* planes' deltas, so a row whose other plane
                    // moved still needs its own residual updated (a zero delta where it did not).
                    m1.push((row, av1, av2));
                    m2.push((row, av2, av1));
                    moved += 1;
                }
            }
            if !m1.is_empty() {
                // `f₁ -= a₁·(H₁₁ Δ₁) + a₂·(H₁₂ Δ₂)`: the second term contracts the *other*
                // plane's delta against the transposed window `H₁₂[col, b0+k]`.
                for (i, &(row, ..)) in m1.iter().enumerate() {
                    let at = i * BLOCK;
                    c1b[at..at + BLOCK].copy_from_slice(&d1b[row * BLOCK..][..BLOCK]);
                    c1b[m1.len() * BLOCK + at..][..BLOCK]
                        .copy_from_slice(&d2b[row * BLOCK..][..BLOCK]);
                }
                let e11 = nt_q(&c1b[..m1.len() * BLOCK], m1.len(), &hc11, n, BLOCK)?;
                let e12 = nt_q(
                    &c1b[m1.len() * BLOCK..m1.len() * 2 * BLOCK],
                    m1.len(),
                    &cx12,
                    n,
                    BLOCK,
                )?;
                for (i, &(row, s11, s12)) in m1.iter().enumerate() {
                    for col in 0..n {
                        let u = qsum(
                            qmul(s11, i128::from(e11[i * n + col]))?,
                            qmul(s12, i128::from(e12[i * n + col]))?,
                        )?;
                        f1[row * n + col] = sub_q(f1[row * n + col], round_shift(u, 32)?)?;
                    }
                }
            }
            if !m2.is_empty() {
                // `f₂ -= a₂·(H₂₂ Δ₂) + a₁·(H₁₂ᵀ Δ₁)`: the row window `H₁₂[b0+k, col]` contracts the
                // first plane's delta.
                for (i, &(row, ..)) in m2.iter().enumerate() {
                    let at = i * BLOCK;
                    c2b[at..at + BLOCK].copy_from_slice(&d2b[row * BLOCK..][..BLOCK]);
                    c2b[m2.len() * BLOCK + at..][..BLOCK]
                        .copy_from_slice(&d1b[row * BLOCK..][..BLOCK]);
                }
                let e22 = nt_q(&c2b[..m2.len() * BLOCK], m2.len(), &hc22, n, BLOCK)?;
                let e21 = nt_q(
                    &c2b[m2.len() * BLOCK..m2.len() * 2 * BLOCK],
                    m2.len(),
                    &hc12,
                    n,
                    BLOCK,
                )?;
                for (i, &(row, s22, s21)) in m2.iter().enumerate() {
                    for col in 0..n {
                        let u = qsum(
                            qmul(s22, i128::from(e22[i * n + col]))?,
                            qmul(s21, i128::from(e21[i * n + col]))?,
                        )?;
                        f2[row * n + col] = sub_q(f2[row * n + col], round_shift(u, 32)?)?;
                    }
                }
            }
        }
    }
    Ok(moved)
}

/// The group-batched residual of [`joint_band`]: see its doc for the shape. A block's change is
/// accumulated in the group's `[rows, GROUP]` delta matrices and applied in one flush per group; the
/// block's own eight columns read the in-batch prefix.
#[allow(
    clippy::too_many_arguments,
    reason = "the band's four carried fields are separate slices, as in `sweep_blocks`: the pinned \
              Aeneas refuses a reference to a struct holding a slice across the nested sweep loops"
)]
fn joint_bands_batched(
    cx: &JointSweep<'_>,
    piece: &mut [JointRow<'_>],
    start: usize,
    beta: Option<&BetaSchedule>,
    rng: &mut [u64],
    f1: &mut [i64],
    f2: &mut [i64],
    s1: &mut [i64],
    s2: &mut [i64],
) -> Result<usize, IcmError> {
    let (n, nb, g) = (cx.n, cx.nb, cx.g);
    let rows = piece.len();
    let cb = cx.cb.as_flattened();
    let per_group = GROUP / BLOCK;
    // The group's `GROUP` rows of the three metrics as `[col, in-group index]`: the kernel's `b`
    // rows are the flush's whole `n` columns and, sliced at a block's eight columns, the in-batch
    // contraction's operand rows, so one copy serves both.
    let mut u11 = vec![0_i64; n * GROUP];
    let mut u22 = vec![0_i64; n * GROUP];
    let mut u12c = vec![0_i64; n * GROUP];
    let mut u12r = vec![0_i64; n * GROUP];
    // The group's accumulated column changes, unscaled: `d1a[row, jj] = Δ₁` of in-group column `jj`.
    let mut d1a = vec![0_i64; rows * GROUP];
    let mut d2a = vec![0_i64; rows * GROUP];
    let mut seen = vec![0_u8; rows];
    let mut mv: Vec<usize> = Vec::with_capacity(rows);
    let mut g1 = vec![0_i64; rows * GROUP];
    let mut g2 = vec![0_i64; rows * GROUP];
    let mut bb1 = vec![0_i64; 2 * BLOCK * BLOCK];
    let mut bb2 = vec![0_i64; 2 * BLOCK * BLOCK];
    let mut s1b = vec![0_i64; rows * BLOCK];
    let mut s2b = vec![0_i64; rows * BLOCK];
    let mut f1b = vec![0_i64; rows * BLOCK];
    let mut f2b = vec![0_i64; rows * BLOCK];
    let mut d1b = vec![0_i64; rows * BLOCK];
    let mut d2b = vec![0_i64; rows * BLOCK];
    let mut picks = vec![PairPick { flat: 0, rng: 0 }; rows];
    let mut moved = 0_usize;
    for sw in 0..cx.sweeps {
        for step in 0..nb {
            let m = if sw % 2 == 0 { nb - 1 - step } else { step };
            let (b0, gidx) = (m * BLOCK, m * BLOCK / GROUP);
            let j0 = b0 - gidx * GROUP;
            let first = step % per_group == 0;
            let last = step % per_group == per_group - 1;
            if first {
                let mut col = 0;
                while col < n {
                    let mut jj = 0;
                    while jj < GROUP {
                        u11[col * GROUP + jj] = cx.h11[(gidx * GROUP + jj) * n + col];
                        u22[col * GROUP + jj] = cx.h22[(gidx * GROUP + jj) * n + col];
                        u12c[col * GROUP + jj] = cx.h12[col * n + gidx * GROUP + jj];
                        u12r[col * GROUP + jj] = cx.h12[(gidx * GROUP + jj) * n + col];
                        jj += 1;
                    }
                    col += 1;
                }
                let mut i = 0;
                while i < rows * GROUP {
                    d1a[i] = 0;
                    d2a[i] = 0;
                    i += 1;
                }
                let mut r = 0;
                while r < rows {
                    seen[r] = 0;
                    r += 1;
                }
                mv.clear();
            }
            // `b[k][l] = h[(b0+k)·n + b0+l]` for the three diagonal metric blocks the gradient
            // needs, in the operand layout the kernel takes: `bb1` is `[col_k(H₁₁); col_k(H₁₂)]` for
            // `d₁₁` and `d₁₂`, `bb2` is `[row_k(H₁₂); col_k(H₂₂)]` for `d₁₂ᵀ` and `d₂₂`.
            for j in 0..BLOCK {
                for l in 0..BLOCK {
                    bb1[j * BLOCK + l] = cx.h11[(b0 + l) * n + b0 + j];
                    bb1[(BLOCK + j) * BLOCK + l] = cx.h12[(b0 + l) * n + b0 + j];
                    bb2[j * BLOCK + l] = cx.h12[(b0 + j) * n + b0 + l];
                    bb2[(BLOCK + j) * BLOCK + l] = cx.h22[(b0 + l) * n + b0 + j];
                }
            }
            for row in 0..rows {
                s1b[row * BLOCK..][..BLOCK].copy_from_slice(&s1[row * n + b0..][..BLOCK]);
                s2b[row * BLOCK..][..BLOCK].copy_from_slice(&s2[row * n + b0..][..BLOCK]);
            }
            let p1v = nt_q(&s1b, rows, &bb1, 2 * BLOCK, BLOCK)?;
            let p2v = nt_q(&s2b, rows, &bb2, 2 * BLOCK, BLOCK)?;
            for row in 0..rows {
                let (a1, a2) = (
                    cx.r1[(start + row) * g + gidx],
                    cx.r2[(start + row) * g + gidx],
                );
                let (av1, av2) = (i128::from(a1), i128::from(a2));
                let o1 = row * 2 * BLOCK;
                for k in 0..BLOCK {
                    // `d₁₁ = s₁·H₁₁ᵀ`, `d₁₂ = s₁·H₁₂`, `d₁₂ᵀ = s₂·H₁₂`, `d₂₂ = s₂·H₂₂ᵀ`.
                    let u1 = qsum(
                        qmul(av1, i128::from(p1v[o1 + k]))?,
                        qmul(av2, i128::from(p2v[o1 + k]))?,
                    )?;
                    let u2 = qsum(
                        qmul(av2, i128::from(p2v[o1 + BLOCK + k]))?,
                        qmul(av1, i128::from(p1v[o1 + BLOCK + k]))?,
                    )?;
                    f1b[row * BLOCK + k] = add_q(f1[row * n + b0 + k], round_shift(u1, 32)?)?;
                    f2b[row * BLOCK + k] = add_q(f2[row * n + b0 + k], round_shift(u2, 32)?)?;
                }
            }
            // The in-batch part of the residual: the group's changes so far against this block's own
            // eight columns. Zero while no block of the group has moved, so it is skipped there.
            if !first && !mv.is_empty() {
                let jt = b0 * GROUP;
                let jb = (b0 + BLOCK) * GROUP;
                let p11 = nt_q(&d1a, rows, &u11[jt..jb], BLOCK, GROUP)?;
                let p12 = nt_q(&d2a, rows, &u12c[jt..jb], BLOCK, GROUP)?;
                let p22 = nt_q(&d2a, rows, &u22[jt..jb], BLOCK, GROUP)?;
                let p21 = nt_q(&d1a, rows, &u12r[jt..jb], BLOCK, GROUP)?;
                for row in 0..rows {
                    let (a1, a2) = (
                        cx.r1[(start + row) * g + gidx],
                        cx.r2[(start + row) * g + gidx],
                    );
                    let (av1, av2) = (i128::from(a1), i128::from(a2));
                    for k in 0..BLOCK {
                        let at = row * BLOCK + k;
                        let u1 = qsum(
                            qmul(av1, i128::from(p11[at]))?,
                            qmul(av2, i128::from(p12[at]))?,
                        )?;
                        let u2 = qsum(
                            qmul(av2, i128::from(p22[at]))?,
                            qmul(av1, i128::from(p21[at]))?,
                        )?;
                        f1b[at] = sub_q(f1b[at], round_shift(u1, 32)?)?;
                        f2b[at] = sub_q(f2b[at], round_shift(u2, 32)?)?;
                    }
                }
            }
            let z1 = nt_q(&f1b, rows, cb, CODE_LEN, BLOCK)?;
            let z2 = nt_q(&f2b, rows, cb, CODE_LEN, BLOCK)?;
            pair_decisions(
                cx,
                &z1,
                &z2,
                start,
                m,
                beta.map(|sched| sched.beta_at(sw)),
                rng,
                &mut picks,
            )?;
            for row in 0..rows {
                let flat = usize::from(picks[row].flat);
                let (i1s, i2s) = (flat / CODE_LEN, flat % CODE_LEN);
                let mut ch1 = false;
                let mut ch2 = false;
                for k in 0..BLOCK {
                    let w1 = sub_q(cx.cb[i1s][k], s1[row * n + b0 + k])?;
                    let w2 = sub_q(cx.cb[i2s][k], s2[row * n + b0 + k])?;
                    ch1 |= w1 != 0;
                    ch2 |= w2 != 0;
                    d1b[row * BLOCK + k] = w1;
                    d2b[row * BLOCK + k] = w2;
                    s1[row * n + b0 + k] = cx.cb[i1s][k];
                    s2[row * n + b0 + k] = cx.cb[i2s][k];
                }
                piece[row].idx1[m] = i1s as u16;
                piece[row].idx2[m] = i2s as u16;
                if ch1 || ch2 {
                    // Both residuals contract *both* planes' deltas, so a row whose other plane
                    // moved still needs its own residual updated (a zero delta where it did not).
                    moved += 1;
                    for k in 0..BLOCK {
                        d1a[row * GROUP + j0 + k] = d1b[row * BLOCK + k];
                        d2a[row * GROUP + j0 + k] = d2b[row * BLOCK + k];
                    }
                    if seen[row] == 0 {
                        seen[row] = 1;
                        mv.push(row);
                    }
                }
            }
            if last && !mv.is_empty() {
                // The batch flush: the whole group's change, contracted once per plane pair against
                // the group's columns and subtracted from every column of the accumulated rows.
                let cnt = mv.len();
                for (i, &row) in mv.iter().enumerate() {
                    g1[i * GROUP..][..GROUP].copy_from_slice(&d1a[row * GROUP..][..GROUP]);
                    g2[i * GROUP..][..GROUP].copy_from_slice(&d2a[row * GROUP..][..GROUP]);
                }
                // `f₁ -= a₁·(H₁₁ Δ₁) + a₂·(H₁₂ Δ₂)`.
                let x11 = nt_q(&g1[..cnt * GROUP], cnt, &u11, n, GROUP)?;
                let x12 = nt_q(&g2[..cnt * GROUP], cnt, &u12c, n, GROUP)?;
                for (i, &row) in mv.iter().enumerate() {
                    let (a1, a2) = (
                        cx.r1[(start + row) * g + gidx],
                        cx.r2[(start + row) * g + gidx],
                    );
                    let (av1, av2) = (i128::from(a1), i128::from(a2));
                    for col in 0..n {
                        let u = qsum(
                            qmul(av1, i128::from(x11[i * n + col]))?,
                            qmul(av2, i128::from(x12[i * n + col]))?,
                        )?;
                        f1[row * n + col] = sub_q(f1[row * n + col], round_shift(u, 32)?)?;
                    }
                }
                // `f₂ -= a₂·(H₂₂ Δ₂) + a₁·(H₁₂ᵀ Δ₁)`.
                let x22 = nt_q(&g2[..cnt * GROUP], cnt, &u22, n, GROUP)?;
                let x21 = nt_q(&g1[..cnt * GROUP], cnt, &u12r, n, GROUP)?;
                for (i, &row) in mv.iter().enumerate() {
                    let (a1, a2) = (
                        cx.r1[(start + row) * g + gidx],
                        cx.r2[(start + row) * g + gidx],
                    );
                    let (av1, av2) = (i128::from(a1), i128::from(a2));
                    for col in 0..n {
                        let u = qsum(
                            qmul(av2, i128::from(x22[i * n + col]))?,
                            qmul(av1, i128::from(x21[i * n + col]))?,
                        )?;
                        f2[row * n + col] = sub_q(f2[row * n + col], round_shift(u, 32)?)?;
                    }
                }
            }
        }
    }
    Ok(moved)
}

/// Joint least-squares solve for both planes' group scales `(r1, r2)`: per row a `2G × 2G` normal
/// system `M x = t` minimising `J(Ws, plane_w(s1,r1,c1,d1) + plane_w(s2,r2,c2,d2), Hl)`, with a
/// `1e-12 · mean(diag M)` diagonal ridge (kept at one raw unit when it would round below the grid).
/// `s1`/`c1` and `s2`/`c2` are the raw code values and column scales; `d` does not enter (the group
/// scale is applied after the transform; the ported reference passes `s_k · c_k` and the metric `Hl`).
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not
/// a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::NotPositiveDefinite`] if the per-row
/// normal matrix is not positive-definite after the ridge; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `g` band products of `rows·2` windows against `n × GROUP` (`rows·2g·n·GROUP` word multiply-adds
/// times the limb pairs), `rows·g²` `2 × GROUP` normal-matrix products, one `rows·2g` row dot against
/// `wh`, and `rows` solves of `2G × 2G` (`rows·(2G)³/3`); the kernel dispatches the engine pool, so
/// the bands run on the caller. `O(rows·(2G)·n)` for the metric buffer plus `O(rows·(2G)²)`.
pub fn r_step2(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>), IcmError> {
    check_inputs(ws, hl, o, n, None)?;
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    let (g, per) = (n / GROUP, rows_per_task(o));
    let wh = matmul_l(ws, hl, o, n, n, false)?;
    let mut r1_out = vec![0_i64; o * g];
    let mut r2_out = vec![0_i64; o * g];
    // The kernel owns the engine pool, so the bands run on the caller.
    let per = per.max(1);
    let mut row0 = 0usize;
    while row0 < o {
        let take = per.min(o - row0);
        r_band_q(
            hl,
            &wh,
            p1,
            p2,
            n,
            g,
            row0,
            &mut r1_out[row0 * g..(row0 + take) * g],
            &mut r2_out[row0 * g..(row0 + take) * g],
        )?;
        row0 += take;
    }
    Ok((r1_out, r2_out))
}

/// The same `2^k` scaling of a `Q32.32` word already on the grid (a kernel product's output).
fn lift_q(t: i64, k: u32) -> Result<i64, IcmError> {
    q32::narrow(qmul(i128::from(t), 1_i128 << k)?).ok_or(IcmError::Overflow)
}

/// The panel rule's `2^k` scaling of a `Q64.64` elementwise product onto the `Q32.32` grid.
fn fill_q(v: i128, k: u32) -> Result<i64, IcmError> {
    if k <= 32 {
        round_shift(v, 32 - k)
    } else {
        q32::narrow(qmul(v, 1_i128 << (k - 32))?).ok_or(IcmError::Overflow)
    }
}

/// Copy one `GROUP`-wide block. The only body here that walks a window.
///
/// # Complexity
/// `O(GROUP)` words moved, no allocation.
fn copy_group(src: &[i64], dst: &mut [i64]) {
    let mut t = 0;
    while t < GROUP {
        dst[t] = src[t];
        t += 1;
    }
}

/// `P[a_, j] = s[row, a_ % g, j] · c[a_ % g, j]` over both planes' blocks.
///
/// # Complexity
/// `2g·GROUP` integer products, no allocation.
fn band_prod_fill(
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    row: usize,
    n: usize,
    g: usize,
    prod: &mut [i128],
) {
    let two_g = 2 * g;
    let mut a_ = 0;
    while a_ < two_g {
        let (s, c) = if a_ < g { (p1.s, p1.c) } else { (p2.s, p2.c) };
        let aa = a_ % g;
        let mut j = 0;
        while j < GROUP {
            let sv = s[row * n + aa * GROUP + j];
            let cv = c[aa * GROUP + j];
            let p: i128 = i128::from(sv) * i128::from(cv);
            prod[a_ * GROUP + j] = p;
            j += 1;
        }
        a_ += 1;
    }
}

/// `max|P|` over the band's product buffer.
///
/// # Complexity
/// `2g·GROUP` magnitude reads, no allocation.
fn band_prod_max(prod: &[i128]) -> u128 {
    let mut max: u128 = 0;
    let mut idx = 0;
    while idx < prod.len() {
        // `unsigned_abs` is an unmodeled external in the pinned Aeneas model; the cast of a
        // non-negative magnitude is exact.
        #[allow(
            clippy::cast_abs_to_unsigned,
            clippy::cast_sign_loss,
            reason = "the pinned model carries no `I128.unsigned_abs`; `abs()` is non-negative \
                      here and the cast preserves every bit"
        )]
        let m = prod[idx].abs() as u128;
        if m > max {
            max = m;
        }
        idx += 1;
    }
    max
}

/// The panel rule's `k` from `max|P|` raw: `k = 66 − log2(max)`, so `2²ᵏ·M` lands in `(2³⁴, 2³⁵]`.
///
/// # Complexity
/// `O(2g·GROUP)` for the scan, no allocation.
fn band_scale_k(prod: &[i128]) -> u32 {
    match ilog2_pos(band_prod_max(prod)) {
        Some(l) => 66_u32.saturating_sub(l),
        None => 0_u32,
    }
}

/// `fill_q` over the band's product buffer into the `pv` window row.
///
/// # Complexity
/// `2g·GROUP` shifts, no allocation.
fn band_fill_pv(prod: &[i128], k: u32, dst: &mut [i64]) -> Result<(), IcmError> {
    let mut err: Option<IcmError> = None;
    let mut idx = 0;
    while idx < dst.len() && err.is_none() {
        match fill_q(prod[idx], k) {
            Ok(v) => dst[idx] = v,
            Err(e) => err = Some(e),
        }
        idx += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The two `q` panels of one band row, contiguous: the normal matrix's `b` rows.
///
/// # Complexity
/// `2g·GROUP` words copied, no allocation.
fn band_pv2_gather(pv: &[i64], rr: usize, g: usize, two_g: usize, pv2: &mut [i64]) {
    let mut w = 0;
    while w < g {
        let mut q = 0;
        while q < 2 {
            let a_ = w + q * g;
            let dst = ((rr * g + w) * 2 + q) * GROUP;
            let src = (rr * two_g + a_) * GROUP;
            copy_group(&pv[src..src + GROUP], &mut pv2[dst..dst + GROUP]);
            q += 1;
        }
        w += 1;
    }
}

/// `hb[j, k] = Hl[ga·GROUP + k, j]`: the metric's `b` rows transposed for this row block.
///
/// # Complexity
/// `n·GROUP` word copies at stride `n`, no allocation.
fn band_hb_fill(hl: &[i64], ga: usize, n: usize, hb: &mut [i64]) {
    let mut j = 0;
    while j < n {
        let mut k = 0;
        while k < GROUP {
            hb[j * GROUP + k] = hl[(ga * GROUP + k) * n + j];
            k += 1;
        }
        j += 1;
    }
}

/// The band's `a` rows for row block `ga`: both planes' window rows stacked in `q` order.
///
/// # Complexity
/// `rows·2·GROUP` word copies, no allocation.
fn band_arows_fill(pv: &[i64], ga: usize, rows: usize, g: usize, two_g: usize, arows: &mut [i64]) {
    let mut rr = 0;
    while rr < rows {
        let mut q = 0;
        while q < 2 {
            let a_ = ga + q * g;
            let src = (rr * two_g + a_) * GROUP;
            let dst = (rr * 2 + q) * GROUP;
            copy_group(&pv[src..src + GROUP], &mut arows[dst..dst + GROUP]);
            q += 1;
        }
        rr += 1;
    }
}

/// One `(rr, w)` block's gathered `a2` window: the two `q` panels of `kk`'s row.
///
/// # Complexity
/// `2·GROUP` word copies, no allocation.
fn band_a2_gather(kk: &[i64], rr: usize, w: usize, n: usize, a2: &mut [i64]) {
    let mut q = 0;
    while q < 2 {
        let src = (rr * 2 + q) * n + w * GROUP;
        copy_group(&kk[src..src + GROUP], &mut a2[q * GROUP..(q + 1) * GROUP]);
        q += 1;
    }
}

/// One band row's `ga` row block of the normal matrix: `nt_q` per window against the two panels.
///
/// # Complexity
/// `g` window products of `2×2` on `GROUP`-wide dots, `O(g·GROUP·n)` word multiply-adds.
fn band_mmat_row(
    kk: &[i64],
    pv2: &[i64],
    a2: &mut [i64],
    rr: usize,
    n: usize,
    g: usize,
    two_g: usize,
    ga: usize,
    out: &mut [i64],
) -> Result<(), IcmError> {
    let mut err: Option<IcmError> = None;
    let mut w = 0;
    while w < g && err.is_none() {
        band_a2_gather(kk, rr, w, n, a2);
        let bs = ((rr * g + w) * 2) * GROUP;
        match nt_q(a2, 2, &pv2[bs..bs + 2 * GROUP], 2, GROUP) {
            Ok(c) => {
                let base = ga * two_g;
                out[base + w] = c[0];
                out[base + w + g] = c[1];
                out[base + two_g * g + w] = c[2];
                out[base + two_g * g + w + g] = c[3];
            }
            Err(e) => err = Some(e),
        }
        w += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The band's `wh` window rows: `whrows[rr, a_, ·] = wh[row0 + rr, a_ % g, ·]`.
///
/// # Complexity
/// `rows·2g·GROUP` word copies, no allocation.
fn band_whrows_fill(
    wh: &[i64],
    row0: usize,
    rows: usize,
    n: usize,
    g: usize,
    two_g: usize,
    whrows: &mut [i64],
) {
    let mut rr = 0;
    while rr < rows {
        let row = row0 + rr;
        let mut a_ = 0;
        while a_ < two_g {
            let src = row * n + (a_ % g) * GROUP;
            let dst = (rr * two_g + a_) * GROUP;
            copy_group(&wh[src..src + GROUP], &mut whrows[dst..dst + GROUP]);
            a_ += 1;
        }
        rr += 1;
    }
}

/// One band of [`r_step2`]'s rows, every product on the limb kernel. `P[a_, j]` (`a_ < G` plane 1's
/// block `a_`, `a_ ≥ G` plane 2's) is filled from below so `max|P| ∈ (2³⁴, 2³⁵]` raw (`steps`' panel
/// rule), `M' = 2²ᵏM` and `t` is read at `2²ᵏ`, so the solve returns `x` itself at the grid quantum.
///
/// The three phases are one band each: the scaled product window, the per-`ga` normal matrix, and
/// the right-hand-side dot plus the `2G × 2G` solve. Every loop body is a separate function, so a
/// phase carries only its own buffers (AGENTS.md §11, Aeneas loop shapes).
///
/// # Complexity
/// `rows·2g·n·GROUP` metric word multiply-adds, `rows·g²·4·GROUP` normal-matrix word multiply-adds
/// and `rows·2g·GROUP` right-hand-side dots, times the limb pairs, plus the `2G × 2G` solves
/// (`rows·(2G)³/3`); the block windows are gathered once per band. `O(rows·2G·n)` for the metric
/// buffer and `O(rows·(2G)²)` for the normal systems.
fn r_band_q(
    hl: &[i64],
    wh: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    n: usize,
    g: usize,
    row0: usize,
    r1_out: &mut [i64],
    r2_out: &mut [i64],
) -> Result<(), IcmError> {
    let two_g = 2 * g;
    let rows = r1_out.len() / g;
    let mut pv = vec![0_i64; rows * two_g * GROUP];
    let mut pv2 = vec![0_i64; rows * g * 2 * GROUP];
    let mut ks = vec![0_u32; rows];
    let mut prod = vec![0_i128; two_g * GROUP];
    let mut err: Option<IcmError> = None;
    let mut rr = 0;
    while rr < rows && err.is_none() {
        let row = row0 + rr;
        band_prod_fill(p1, p2, row, n, g, &mut prod);
        let k = band_scale_k(&prod);
        ks[rr] = k;
        match band_fill_pv(
            &prod,
            k,
            &mut pv[rr * two_g * GROUP..(rr + 1) * two_g * GROUP],
        ) {
            Ok(()) => {}
            Err(e) => err = Some(e),
        }
        if err.is_none() {
            band_pv2_gather(&pv, rr, g, two_g, &mut pv2);
        }
        rr += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut mmat = vec![0_i64; rows * two_g * two_g];
    let mut a2 = vec![0_i64; 2 * GROUP];
    let mut arows = vec![0_i64; rows * 2 * GROUP];
    let mut hb = vec![0_i64; n * GROUP];
    let mut ga = 0;
    while ga < g && err.is_none() {
        band_hb_fill(hl, ga, n, &mut hb);
        band_arows_fill(&pv, ga, rows, g, two_g, &mut arows);
        match nt_q(&arows, rows * 2, &hb, n, GROUP) {
            Ok(kk) => {
                let mut rr = 0;
                while rr < rows && err.is_none() {
                    match band_mmat_row(
                        &kk,
                        &pv2,
                        &mut a2,
                        rr,
                        n,
                        g,
                        two_g,
                        ga,
                        &mut mmat[rr * two_g * two_g..(rr + 1) * two_g * two_g],
                    ) {
                        Ok(()) => {}
                        Err(e) => err = Some(e),
                    }
                    rr += 1;
                }
            }
            Err(e) => err = Some(e),
        }
        ga += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut whrows = vec![0_i64; rows * two_g * GROUP];
    band_whrows_fill(wh, row0, rows, n, g, two_g, &mut whrows);
    let tvals = rowdot_q32(&pv, &whrows, rows * two_g, GROUP, LIMBS).map_err(map_limb)?;
    let mut rr = 0;
    while rr < rows && err.is_none() {
        match r_ridge_solve_q(
            &mut mmat[rr * two_g * two_g..(rr + 1) * two_g * two_g],
            &tvals[rr * two_g..(rr + 1) * two_g],
            ks[rr],
            two_g,
            g,
            &mut r1_out[rr * g..(rr + 1) * g],
            &mut r2_out[rr * g..(rr + 1) * g],
        ) {
            Ok(()) => {}
            Err(e) => err = Some(e),
        }
        rr += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// One row of [`r_band_q`]'s normal system: the `1e-12 · mean(diag M)` ridge, the `2^k` undo of the
/// batched right-hand side, and the `2G × 2G` solve into the two group scales. `O((2G)²)` space.
fn r_ridge_solve_q(
    mm: &mut [i64],
    t: &[i64],
    k: u32,
    two_g: usize,
    g: usize,
    r1_out: &mut [i64],
    r2_out: &mut [i64],
) -> Result<(), IcmError> {
    let mut err: Option<IcmError> = None;
    let mut mean: i128 = 0;
    let mut i = 0;
    while i < two_g && err.is_none() {
        match qsum(mean, i128::from(mm[i * two_g + i])) {
            Ok(v) => mean = v,
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mean = q32::narrow(div_round(mean, two_g as i128)?).ok_or(IcmError::Overflow)?;
    // `1e-12 · mean(diag)`, floored at one raw unit when a positive ridge would round to zero.
    let add =
        q32::narrow(div_round(i128::from(mean), 1_000_000_000_000)?).ok_or(IcmError::Overflow)?;
    let add = if mean > 0 && add < 1 { 1 } else { add };
    let mut i = 0;
    while i < two_g && err.is_none() {
        match add_q(mm[i * two_g + i], add) {
            Ok(v) => mm[i * two_g + i] = v,
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut tvec = vec![0_i64; two_g];
    let mut i = 0;
    while i < two_g && err.is_none() {
        match lift_q(t[i], k) {
            Ok(v) => tvec[i] = v,
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    match chol_solve(mm, &tvec, two_g, 1) {
        Ok(solved) => {
            let mut i = 0;
            while i < g {
                r1_out[i] = solved.x[i];
                r2_out[i] = solved.x[g + i];
                i += 1;
            }
            Ok(())
        }
        Err(e) => Err(map_solve(e)),
    }
}

/// Validate `o`, `n` and every slice length of a plane.
fn check_plane(p: PlaneRef<'_>, o: usize, n: usize) -> Result<(), IcmError> {
    if o == 0 || n == 0 {
        return Err(IcmError::Shape);
    }
    if !n.is_multiple_of(BLOCK) || !n.is_multiple_of(GROUP) {
        return Err(IcmError::Width);
    }
    if p.s.len() != o * n
        || p.idx.len() != o * (n / BLOCK)
        || p.r.len() != o * (n / GROUP)
        || p.c.len() != n
        || p.d.len() != n
    {
        return Err(IcmError::Shape);
    }
    Ok(())
}

/// Row-major `[rows, inner] × [inner, cols]` product on the shared `Q32.32` path: the operands go to
/// [`crate::engine::kern::limb`] at [`LIMBS_MATMUL`] limbs after one transpose when `b` is row-major
/// `[inner, cols]`, so a product outside the `i64` grid range is carried by the limb block scales
/// rather than refused.
/// # Errors
/// [`IcmError::Shape`] for an empty dimension or a length that disagrees with the shape;
/// [`IcmError::Overflow`] for a product that leaves the grid.
/// # Complexity
/// `O(rows·inner·cols)` kernel word multiply-accumulates times the limb pairs, `O(rows·cols)` space
/// plus the `O(inner·cols)` transpose.
pub fn matmul(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<i64>, IcmError> {
    matmul_q(a, b, rows, inner, cols, false)
}

#[cfg(test)]
mod tests {
    use super::{
        best_code_q, codebook, codebook_q, flatten_codebook_q, nt_l, nt_q, rows_per_task, BLOCK,
        CODE_LEN, LIMBS, LIMBS_MATMUL,
    };
    use crate::engine::kern::limb::{bound_ulps, gemm_q32};
    use crate::engine::ONE;

    /// Production row count of the `Kind::Down` fold profile (`tests/learn_fold_profile_down.rs`).
    const PROD_O: usize = 896;

    /// Production column count of the `Kind::Down` fold profile.
    const PROD_N: usize = 4864;

    /// Operand peaks of one real `e8_icm_step` at `o = 896, n = 4864, sweeps = 1`, in Q32.32 words,
    /// from the repository's `Down` fixture (`tests/learn_fold_profile_down.rs`: `Ws ~ N(0, 1/√n)`,
    /// `H = XᵀX/512` with eight columns × 20, `s` the sign plane, `r = c = d = 1`, production
    /// `icm_ridge`): `Ws` `0.0667`, `metric` `439.86`, `H_rot` `14.08`, `f_blk` `123.47`.
    const PROD_WS_PEAK: u64 = 286_328_950;
    const PROD_METRIC_PEAK: u64 = 1_889_194_637_885;
    const PROD_H_ROT_PEAK: u64 = 60_466_383_651;
    const PROD_F_PEAK: u64 = 530_310_743_293;
    /// The codebook's own peak (`2√2`), the `dot` product's `b` operand.
    const PROD_CB_PEAK: u64 = 12_148_002_000;

    /// A non-negative peak constant as the `i64` operand bound [`word`] takes.
    fn wpeak(p: u64) -> i64 {
        i64::try_from(p).expect("peak fits i64")
    }

    /// Fixed xorshift64 seed for the operand streams; the same seed reproduces every number below.
    const SEED: u64 = 0x5DEE_CE66_D1CE_4F1B;

    /// One xorshift64 word.
    fn xs(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// One Q32.32 operand word in `[-peak, peak]`.
    fn word(state: &mut u64, peak: i64) -> i64 {
        let span = u64::try_from(peak).expect("positive peak") * 2 + 1;
        i64::try_from(xs(state) % span).expect("span fits i64") - peak
    }

    /// `C[i·p + j] = Σ_l a[i·n + l]·b[j·n + l]` in Q32.32, exactly: the `i128` sum rounded once with
    /// the kernel's own `round_shift_32` convention (`(v + 2^31) >> 32`, ties upward). This is the
    /// oracle the `gemm_q32` entries are compared against.
    fn exact_nt(a: &[i64], m: usize, b: &[i64], p: usize, n: usize) -> Vec<i128> {
        let mut out = vec![0_i128; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut acc = 0_i128;
                for l in 0..n {
                    acc += i128::from(a[i * n + l]) * i128::from(b[j * n + l]);
                }
                out[i * p + j] = (acc + (1_i128 << 31)) >> 32;
            }
        }
        out
    }

    /// `(max |kernel − exact| in Q32.32 words, max relative error against max(1, |exact|), entries)`.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the relative-error metric is a reported diagnostic, not a cost or a decision"
    )]
    fn compare(k: &[i64], e: &[i128]) -> (i128, f64, usize) {
        let mut max_abs = 0_i128;
        let mut max_rel = 0_f64;
        for (&kv, &ev) in k.iter().zip(e) {
            let d = (i128::from(kv) - ev).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d as f64 / ev.abs().max(1) as f64);
        }
        (max_abs, max_rel, k.len())
    }

    /// `Σ_p C_j(p)` of `Learn.FitIcm.witnessDotList` (`proof/corpus/Learn/FitIcm.lean:1011-1013`),
    /// transcribed from `Learn.E8Code.codewordSpec` (`proof/corpus/Learn/E8Code.lean:304-321`):
    /// `evenKeys`/`signAt` for the 128 spinor rows, `rootPairs`/`rootSpec` for the 112 D8 rows and
    /// `o8Spec` for the 16 O8 rows.
    fn witness_dot() -> Vec<i64> {
        const ONE: i64 = 1 << 32;
        const SQRT8: i64 = 12_148_002_000;
        let even_keys: Vec<usize> = (0..256_usize).filter(|i| i.count_ones() % 2 == 0).collect();
        let pairs: Vec<(usize, usize)> = (0..8)
            .flat_map(|a| ((a + 1)..8).map(move |b| (a, b)))
            .collect();
        (0..CODE_LEN)
            .map(|j| {
                if j < 128 {
                    let key = even_keys[j];
                    let mut sum = 0;
                    for k in 0..8 {
                        sum += if key / (1 << (7 - k)) % 2 == 1 {
                            -ONE
                        } else {
                            ONE
                        };
                    }
                    sum
                } else if j < 240 {
                    let sub = (j - 128) % 4;
                    let (a, b) = pairs[(j - 128) / 4];
                    let mut row = [0_i64; 8];
                    row[a] = if sub / 2 == 0 { -2 * ONE } else { 2 * ONE };
                    row[b] = if sub % 2 == 0 { -2 * ONE } else { 2 * ONE };
                    row.iter().sum()
                } else if (j - 240) % 2 == 0 {
                    -SQRT8
                } else {
                    SQRT8
                }
            })
            .collect()
    }

    /// The Lean witness `best_code_q witnessAA witnessTwoA witnessQ witnessDot` (`FitIcm.lean:981-1023`)
    /// returns row `0` on this body: `aa = 2^64`, `two_a = 2^33`, `q = 2^35` on all 256 rows (the O8
    /// rows' `SQRT8²/2^32 = 2^35 + 0.54` narrows to `2^35`) and `dot = Σ_p C_j(p)`. Row `0` (all `+1`)
    /// is the unique maximum of `dot`, hence the unique minimum of `2^64·q − 2^65·dot`.
    #[test]
    fn learn_fold_fit_icm_witness_best_code_q_is_row0() {
        let q = vec![1_i64 << 35; CODE_LEN];
        let dot = codebook()
            .iter()
            .map(|c| c.iter().sum::<i64>())
            .collect::<Vec<_>>();
        let witness = witness_dot();
        for j in 0..CODE_LEN {
            assert_eq!(dot[j], witness[j], "dot[{j}] differs from witnessDotList");
        }
        assert_eq!(
            best_code_q(1_i128 << 64, 1_i128 << 33, &q, &dot),
            Ok(0),
            "the witness argmin is not row 0"
        );
    }

    /// The limb-path error at the shapes [`sweep_blocks`] actually calls, against the exact `i128` dot.
    ///
    /// `sweep_blocks` makes two band products per block, both with contraction `BLOCK`: the `q`
    /// product `rows × BLOCK × BLOCK` (`nt_q(&s_blk, rows, …, BLOCK, BLOCK)`) and the code-dot
    /// product `rows × CODE_LEN × BLOCK` (`nt_q(&f_blk, rows, &cbf, CODE_LEN, BLOCK)`). Both run at
    /// [`LIMBS`]; the public row-panel product `matmul_q` runs at [`LIMBS_MATMUL`], so both limb
    /// counts are measured here. Operands are the production codebook on the code side and a
    /// deterministic xorshift stream (seed [`SEED`]) on the data side, at two magnitude bands.
    ///
    /// The printed words are the `εq` (q product) and `εd` (dot product) of
    /// `Learn.FitIcm.word_argmin_cost_bound`, in Q32.32 units; divide by `2^32` for the real ε.
    #[test]
    fn learn_fold_fit_icm_limb_error_at_sweep_shapes() {
        let rows = rows_per_task(PROD_O);
        assert!(rows >= 1, "no rows in a sweep band");
        let cbf = flatten_codebook_q(&codebook_q());
        let mut state = SEED;
        // (band label, data peak) with the asserted 5-limb max-abs bound in words: the observed
        // maxima at seed `SEED` rounded up (q 6 / dot 4 at |x| ≤ 3; q 478 / dot 287 at |x| ≤ 256).
        let bands = [
            ("3", 3 * ONE, 64_i128, 64_i128),
            ("256", 256 * ONE, 4096_i128, 4096_i128),
        ];
        let limbs = [LIMBS, LIMBS_MATMUL, 2];
        println!("sweep band rows = {rows} (o = {PROD_O}, BLOCK = {BLOCK}, CODE_LEN = {CODE_LEN})");
        for &(band, peak, q_stated, d_stated) in &bands {
            let mut at_limbs = (0_i128, 0_i128);
            // q: `s_blk` codes (|x| ≤ 2·2^32) against the block's `H_rot` window.
            let a_q: Vec<i64> = (0..rows * BLOCK)
                .map(|_| word(&mut state, 2 * ONE))
                .collect();
            let b_q: Vec<i64> = (0..BLOCK * BLOCK).map(|_| word(&mut state, peak)).collect();
            // dot: `f_blk` residuals against the real `CODE_LEN × BLOCK` codebook.
            let a_d: Vec<i64> = (0..rows * BLOCK).map(|_| word(&mut state, peak)).collect();
            let peak_of = |v: &[i64]| v.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
            let (qa_peak, qb_peak) = (peak_of(&a_q), peak_of(&b_q));
            let (da_peak, db_peak) = (peak_of(&a_d), peak_of(&cbf));
            for &l in &limbs {
                // The kernel's own documented per-entry bound, at this contraction and these peaks.
                let q_bound = bound_ulps(BLOCK, l, qa_peak, qb_peak);
                let d_bound = bound_ulps(BLOCK, l, da_peak, db_peak);
                let kq = gemm_q32(&a_q, rows, &b_q, BLOCK, BLOCK, l).expect("q product");
                let (qa, qr, qn) = compare(&kq, &exact_nt(&a_q, rows, &b_q, BLOCK, BLOCK));
                let kd = gemm_q32(&a_d, rows, &cbf, CODE_LEN, BLOCK, l).expect("dot product");
                let (da, dr, dn) = compare(&kd, &exact_nt(&a_d, rows, &cbf, CODE_LEN, BLOCK));
                println!(
                    "band |x|<={band}  limbs={l}  q: shape {rows}x{BLOCK}x{BLOCK} entries {qn} \
                     max_abs {qa} words (kernel bound {q_bound}) max_rel {qr:.3e}  dot: shape \
                     {rows}x{CODE_LEN}x{BLOCK} entries {dn} max_abs {da} words (kernel bound \
                     {d_bound}) max_rel {dr:.3e}"
                );
                assert!(
                    qa <= q_bound && da <= d_bound,
                    "limbs={l} band {band}: q {qa} / dot {da} words exceed the kernel bound \
                     {q_bound} / {d_bound}"
                );
                if l == LIMBS {
                    // The measured product is the production call: `nt_q` is `gemm_q32` at `LIMBS`.
                    assert_eq!(nt_q(&a_q, rows, &b_q, BLOCK, BLOCK), Ok(kq.clone()));
                    assert_eq!(nt_q(&a_d, rows, &cbf, CODE_LEN, BLOCK), Ok(kd.clone()));
                    assert!(
                        qa <= q_stated && da <= d_stated,
                        "band {band}: q {qa} / dot {da} words exceed the asserted bound \
                         {q_stated} / {d_stated} at the production {LIMBS} limbs"
                    );
                    at_limbs = (qa, da);
                }
                if l == 2 {
                    // The control: dropping to 2 limbs must be strictly worse on both products.
                    assert!(
                        qa > at_limbs.0 && da > at_limbs.1,
                        "2-limb error {qa}/{da} is not strictly above the LIMBS error {}/{} at \
                         band {band}",
                        at_limbs.0,
                        at_limbs.1
                    );
                }
            }
        }
    }

    /// The `moved × n × BLOCK` row-update product of `sweep_blocks` at the production `Kind::Down`
    /// shape: `n = 4864`, `moved = rows_per_task(896)` (the largest band the sweep hands the
    /// product), against the exact `i128` dot.
    ///
    /// `sweep_blocks` runs it as `nt_q(&chg[..moved·BLOCK], moved, &cols, n, BLOCK)` at [`LIMBS`],
    /// with `chg` a codeword difference (`|x| ≤ 4·2^32`) and `cols` the block-transposed `H_rot`
    /// window. The `b` band is the peak the real run reaches ([`PROD_H_ROT_PEAK`]); the 3-limb and
    /// 2-limb arms are the controls. Printed errors are Q32.32 words; divide by `2^32` for the real
    /// magnitude.
    ///
    /// Measured at the seed and peaks above, 5 limbs: `78` words on the row update (bound `6218`)
    /// and `141` on the `dot` product at [`PROD_F_PEAK`] (bound `24842`); the 3- and 2-limb
    /// controls are `1.0e6`/`1.3e8` and `2.0e6`/`2.4e8` words. Release wall clock is `0.05 s`.
    ///
    /// # Complexity
    /// `moved·n·BLOCK` kernel multiply-adds times the limb pairs plus the same exact-`i128` count;
    /// `O(moved·n)` space. `≈ 1.5e6` multiply-adds at the production shape.
    #[test]
    fn learn_fold_fit_icm_limb_error_at_production_row_update() {
        let n = PROD_N;
        let rows = rows_per_task(PROD_O);
        let peak = 4 * ONE;
        let peak_u = u64::try_from(peak).expect("peak is non-negative");
        let mut state = SEED;
        let a: Vec<i64> = (0..rows * BLOCK).map(|_| word(&mut state, peak)).collect();
        let b: Vec<i64> = (0..BLOCK * n)
            .map(|_| word(&mut state, wpeak(PROD_H_ROT_PEAK)))
            .collect();
        let exact = exact_nt(&a, rows, &b, n, BLOCK);
        println!(
            "row update: shape {rows}x{n}x{BLOCK} (Down) a |x| <= {peak}, b |x| <= {PROD_H_ROT_PEAK}"
        );
        let mut errs = [0_i128; 3];
        for (slot, &l) in [LIMBS, LIMBS_MATMUL, 2].iter().enumerate() {
            // The `LIMBS` arm is the production wrapper itself; the controls name their limb count.
            let k = if l == LIMBS {
                nt_q(&a, rows, &b, n, BLOCK).expect("row update")
            } else {
                gemm_q32(&a, rows, &b, n, BLOCK, l).expect("row update")
            };
            let (abs, rel, cnt) = compare(&k, &exact);
            let bound = bound_ulps(BLOCK, l, peak_u, PROD_H_ROT_PEAK);
            println!(
                "  limbs={l} entries {cnt} max_abs {abs} words (kernel bound {bound}) max_rel \
                 {rel:.3e}"
            );
            assert!(
                abs <= bound,
                "limbs={l}: row-update {abs} words exceed the kernel bound {bound}"
            );
            if l == LIMBS {
                // The measured 5-limb level at this seed and these peaks, rounded up.
                assert!(
                    abs <= 128,
                    "row update at the production limb count: {abs} words exceed the measured level 128"
                );
            }
            errs[slot] = abs;
        }
        assert!(
            errs[0] > 0,
            "the production path is bit-exact against the oracle: no limb error to bound"
        );
        assert!(
            errs[1] > errs[0] && errs[2] > errs[1],
            "controls are not strictly worse: 5/3/2 limbs = {errs:?}"
        );
        // The same band's `dot` product (`nt_q(&f_blk, rows, &cbf, CODE_LEN, BLOCK)`) at the real
        // `f_blk` peak against the real codebook: εd at production magnitude.
        let cbf = flatten_codebook_q(&codebook_q());
        let f: Vec<i64> = (0..rows * BLOCK)
            .map(|_| word(&mut state, wpeak(PROD_F_PEAK)))
            .collect();
        let dexact = exact_nt(&f, rows, &cbf, CODE_LEN, BLOCK);
        let mut derrs = [0_i128; 3];
        for (slot, &l) in [LIMBS, LIMBS_MATMUL, 2].iter().enumerate() {
            let k = if l == LIMBS {
                nt_q(&f, rows, &cbf, CODE_LEN, BLOCK).expect("dot product")
            } else {
                gemm_q32(&f, rows, &cbf, CODE_LEN, BLOCK, l).expect("dot product")
            };
            let (abs, rel, cnt) = compare(&k, &dexact);
            let bound = bound_ulps(BLOCK, l, PROD_F_PEAK, PROD_CB_PEAK);
            println!(
                "  dot limbs={l} entries {cnt} max_abs {abs} words (kernel bound {bound}) max_rel \
                 {rel:.3e}"
            );
            assert!(
                abs <= bound,
                "dot limbs={l}: {abs} words exceed the kernel bound {bound}"
            );
            if l == LIMBS {
                // The measured 5-limb level at this seed and these peaks, rounded up.
                assert!(
                    abs <= 256,
                    "dot product at the production limb count: {abs} words exceed the measured level 256"
                );
            }
            derrs[slot] = abs;
        }
        assert!(
            derrs[1] > derrs[0] && derrs[2] > derrs[1],
            "dot controls are not strictly worse: 5/3/2 limbs = {derrs:?}"
        );
    }

    /// `matmul_l` at the production `Kind::Down` shape (`o = 896`, `n = 4864`, contraction `n`):
    /// the two `O(o·n²)` products of `e8_icm_step` (`G_rot = Ws @ metric`, `B @ H_rot`), against
    /// the exact `i128` dot, at the operand peaks the real run reaches ([`PROD_WS_PEAK`],
    /// [`PROD_METRIC_PEAK`]). The 3-limb and 2-limb arms are the controls.
    ///
    /// Measured at the seed and peaks above: `2138` words (rel `2.2e-4`) at 5 limbs,
    /// `2.8e7` at 3, `2.8e9` at 2; the kernel bound at these peaks is `472120`. Release wall clock
    /// is `11.5 s` for the oracle plus `1.3 s` for the three products.
    ///
    /// Ignored by default: the exact oracle is `o·n² = 2.1e10` `i128` multiply-adds, which a debug
    /// build carries far past the 540 s gate. Run it with
    /// `cargo test --release --locked --lib -- --ignored --nocapture \
    /// learn_fold_fit_icm_limb_error_at_production_matmul`.
    ///
    /// # Complexity
    /// Three kernel products (`o·n²` multiply-adds times 15/6/3 limb pairs) plus one `o·n²`
    /// exact-`i128` pass; `O(o·n)` for the `i128` oracle buffer (70 MB) and `O(n²)` for the `b`
    /// operand (189 MB).
    #[test]
    #[ignore = "production O(o·n²) oracle: 2.1e10 i128 multiply-adds; --release --ignored"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the relative-error and elapsed-second diagnostics are reported, not decided on"
    )]
    fn learn_fold_fit_icm_limb_error_at_production_matmul() {
        let (o, n) = (PROD_O, PROD_N);
        let mut state = SEED;
        let a: Vec<i64> = (0..o * n)
            .map(|_| word(&mut state, wpeak(PROD_WS_PEAK)))
            .collect();
        let b: Vec<i64> = (0..n * n)
            .map(|_| word(&mut state, wpeak(PROD_METRIC_PEAK)))
            .collect();
        let t = std::time::Instant::now();
        let mut exact = vec![0_i128; o * n];
        for i in 0..o {
            for j in 0..n {
                let mut acc = 0_i128;
                for l in 0..n {
                    acc += i128::from(a[i * n + l]) * i128::from(b[j * n + l]);
                }
                exact[i * n + j] = (acc + (1_i128 << 31)) >> 32;
            }
        }
        println!(
            "matmul: shape {o}x{n}x{n} (Down) a |x| <= {PROD_WS_PEAK}, b |x| <= \
             {PROD_METRIC_PEAK}; exact oracle {:.1} s",
            t.elapsed().as_secs_f64()
        );
        let mut errs = [0_i128; 3];
        for (slot, &l) in [LIMBS, LIMBS_MATMUL, 2].iter().enumerate() {
            let t = std::time::Instant::now();
            // The `LIMBS` arm is the production wrapper `nt_l`; the controls name their limb count.
            let k = if l == LIMBS {
                nt_l(&a, o, &b, n, n).expect("matmul")
            } else {
                gemm_q32(&a, o, &b, n, n, l).expect("matmul")
            };
            let el = t.elapsed().as_secs_f64();
            let (abs, rel, cnt) = compare(&k, &exact);
            let bound = bound_ulps(n, l, PROD_WS_PEAK, PROD_METRIC_PEAK);
            println!(
                "  limbs={l} entries {cnt} max_abs {abs} words (kernel bound {bound}) max_rel \
                 {rel:.3e} in {el:.1} s"
            );
            assert!(
                abs <= bound,
                "limbs={l}: matmul {abs} words exceed the kernel bound {bound}"
            );
            if l == LIMBS {
                // The measured 5-limb level at this seed and these peaks, rounded up.
                assert!(
                    abs <= 4096,
                    "matmul at the production limb count: {abs} words exceed the measured level 4096"
                );
            }
            errs[slot] = abs;
        }
        assert!(
            errs[1] > errs[0] && errs[2] > errs[1],
            "controls are not strictly worse: 5/3/2 limbs = {errs:?}"
        );
    }
}
