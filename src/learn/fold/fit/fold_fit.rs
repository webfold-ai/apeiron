//! `fold_fit` in its production configuration: the alternation of the discrete E8 `gptq_e8` step
//! and the exact continuous `c`/`r`/`d` steps plus the E8 `e8_icm_step` refinement, followed by
//! the fp16/u8 pack of the winning plane.
//!
//! Port of `fold_fit` in `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 839–977,
//! specialised to the call site the transfer actually runs (`fold_transfer.py` 1733–1736):
//! `refit=True`, `had=True`, `act_order=True`, `wscale=True`, `hier_scale=True`, `bscale=True`,
//! `e8_code=True`, `e8_icm=True`, `iters=1`, `sweeps=2`. `iters > 1` (the `cs`/`Wc`
//! requantisation) is outside the ported domain: [`FoldFitError::Unsupported`].
//!
//! 1. `Hl = metric(H, ridge)`: `H` with `ridge · mean(diag(H))` added to its diagonal.
//! 2. `gptq_e8(fwht128(Ws), had_gram(H), beam, damp, act_order, wscale)` gives the code plane
//!    `S_rot`, the group scale `r` and the candidate `compose(fwht128(S_rot), r, ones)`; its `J`
//!    is `trace[0]`.
//! 3. One `refit0` candidate from `inner_iters` passes of `c_step → mean-|c| normalisation →
//!    r_step → d_step → 8-block sign parity fix → 128-group `d` normalisation → e8_icm_step →
//!    fwht128(S_rot·d) → r_step`; its `J` is `trace[1]`.
//! 4. `best = argmin J` (first on a tie). The `bscale and had` arm quantises the winning `d` to 5
//!    bits, refits `r` and `c` against the re-materialised `S_dh`, tries the `e8_icm_step`
//!    re-quantisation candidate and the un-`d` fallback, keeps the smallest `J`, and returns the
//!    pack `(S_rot, rh, ch, dh)` whose recomposition equals `Wp` word for word.
//!
//! # Carrier and the boundary
//!
//! Every quantity between [`fold_fit`]'s arguments and its result fields is a Q32.32 `i64` word
//! `round(x·2³²)`; products are `i128` rounded once by [`round_q32_even`]. The step helpers
//! ([`c_step_wshl`], [`r_step_wshl`], [`d_step_wshl`], [`compose`], [`objective`], [`fwht_words`],
//! [`had_gram`], [`mul_metric`], [`e8_icm_step`]) take and return words, and so do [`fold_fit`]'s
//! `ws`, `h` and `g_row` inputs, its [`FoldFit`]/[`PackOut`] outputs and [`FoldFitConfig`]'s
//! `ridge`/`damp`/`icm_ridge`: the code plane is carried as words from [`gptq_e8`] through the
//! alternation with no grid round trip, and nothing in this file reads an `f64`.
//!
//! # Complexity and the pool
//!
//! The matrix-sized products run in the crate's pooled kernels inside the helpers: `mul_metric`
//! (`O(o·n²)`), the `c`/`r`/`d` normal systems and the ICM sweeps. This file's own passes are the
//! `O(n²)` metric copy (one `memcpy` plus `n` diagonal adds) and the `O(o·n)` column scalings
//! (`6·o·n` rounded products per fit): each entry is one integer multiply over a contiguous
//! buffer, so a row-band dispatch buys no per-entry work and they stay serial. The one-word-per
//! column passes (parity fix, group normalisation, unit scaling, recomposition compare) are `O(n)`
//! or `O(o·n)` compares and stay serial by the same rule.
//!
//! # Deviations from the Python
//!
//! The `assert torch.equal(...)` on the winning pack is a typed [`FoldFitError::PackMismatch`],
//! never a panic; every helper refusal is a typed [`FoldFitError`] arm; the `torch` `float64`/
//! `float32` ridge branch is the group-maximum floor `2⁻³²` (the Python's `1e-12` rounds to `0` on
//! the grid and exists only to keep the group quotient defined); a finite input outside the grid
//! range, or a checked quantity leaving its range, is [`FoldFitError::Overflow`].
//!
//! `tested` (`tests/learn_fold_fold_fit.rs`); nothing here is extracted or linked to the canonical
//! object, so nothing is `proved`, and it makes no hardware or performance claim.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::must_use_candidate,
    clippy::float_cmp,
    reason = "the body indexes row-major `[o, n]` and `[o, g]` buffers by `r/o/i/j/k/m/g` and names \
              the carriers `c`/`r`/`d`/`s_rot`/`dh`/`rh`/`ch` as the reference does; the `usize` to \
              `i64` casts are the shape counts widened for one division; the recomposition check \
              and the candidate tie are the reference's exact comparisons (`torch.equal`, `min`)"
)]

use crate::learn::fold::fit::gptq_e8::{gptq_e8, GptqE8Error};
use crate::learn::fold::fit::icm::{e8_icm_step, IcmError, PlaneRef};
use crate::learn::fold::fit::quant::{quant_c_u8, quant_r_hier, QuantError};
use crate::learn::fold::fit::steps::{
    c_step_wshl, compose, d_step_wshl, fwht_words, had_gram, mul_metric, objective, r_step_wshl,
    StepsError,
};
use crate::learn::solve::q32::round_q32_even;

/// Columns per `r` group scale: the Python `GROUP`.
pub const GROUP: usize = 128;

/// Columns per `E8 ∪ O8` code: the Python block width.
pub const BLOCK: usize = 8;

/// The Q32.32 unit `1.0`.
const ONE: i64 = 1 << 32;

/// The smallest positive Q32.32 word, `2⁻³²`: the `group_max` floor (the Python's `1e-12`, which
/// rounds to `0` on the grid, exists only to keep the group quotient defined).
const EPS_W: i64 = 1;

/// A refusal from [`fold_fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldFitError {
    /// A slice length disagrees with the declared `o × n` / `n × n` shape, or an inner step refused
    /// the shape.
    ShapeMismatch,
    /// `o == 0` or `n == 0`.
    ZeroDim,
    /// `n` is not a positive multiple of [`GROUP`].
    Width,
    /// `iters != 1`: the `cs`/`Wc` requantisation arm is out of the ported domain.
    Unsupported,
    /// A weight, Hessian or row weight is NaN or infinite.
    NonFinite,
    /// A reduced normal system or the damped Hessian is not positive-definite.
    NotPositiveDefinite,
    /// A checked integer quantity left its range, or a finite value left the Q32.32 grid range.
    Overflow,
    /// A code index does not decode: a codebook refusal.
    Codebook,
    /// A quantisation carrier refused its bit width.
    BadBits,
    /// The winning pack does not recompose to `Wp` word for word.
    PackMismatch,
}

impl core::fmt::Display for FoldFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "matrix shape or buffer length mismatch"),
            Self::ZeroDim => write!(f, "output or column count must be non-zero"),
            Self::Width => write!(f, "column count is not a positive multiple of 128"),
            Self::Unsupported => write!(f, "only iters = 1 is ported"),
            Self::NonFinite => write!(f, "weight or Hessian entry is not finite"),
            Self::NotPositiveDefinite => write!(f, "a reduced system is not positive-definite"),
            Self::Overflow => write!(f, "a checked integer quantity left its range"),
            Self::Codebook => write!(f, "codebook search left its admitted range"),
            Self::BadBits => write!(f, "quantisation bit width is out of range"),
            Self::PackMismatch => write!(f, "the winning pack does not recompose to Wp"),
        }
    }
}

impl std::error::Error for FoldFitError {}

impl From<StepsError> for FoldFitError {
    fn from(e: StepsError) -> Self {
        match e {
            StepsError::ShapeMismatch => Self::ShapeMismatch,
            StepsError::NotPositiveDefinite => Self::NotPositiveDefinite,
        }
    }
}

impl From<IcmError> for FoldFitError {
    fn from(e: IcmError) -> Self {
        match e {
            IcmError::Shape | IcmError::ZeroChunk | IcmError::BlockOutOfRange => {
                Self::ShapeMismatch
            }
            IcmError::Width => Self::Width,
            IcmError::Overflow => Self::Overflow,
            IcmError::NotPositiveDefinite => Self::NotPositiveDefinite,
        }
    }
}

impl From<QuantError> for FoldFitError {
    fn from(e: QuantError) -> Self {
        match e {
            QuantError::Empty | QuantError::ShapeMismatch => Self::ShapeMismatch,
            QuantError::BadBits => Self::BadBits,
            QuantError::LengthOverflow => Self::Overflow,
        }
    }
}

impl From<GptqE8Error> for FoldFitError {
    fn from(e: GptqE8Error) -> Self {
        match e {
            GptqE8Error::ShapeMismatch
            | GptqE8Error::InvalidDamping
            | GptqE8Error::BeamOutOfRange => Self::ShapeMismatch,
            GptqE8Error::ZeroDim => Self::ZeroDim,
            GptqE8Error::BlockMisaligned => Self::Width,
            GptqE8Error::NotPositiveDefinite => Self::NotPositiveDefinite,
            GptqE8Error::NonFinite => Self::NonFinite,
            GptqE8Error::Overflow => Self::Overflow,
            GptqE8Error::Codebook => Self::Codebook,
        }
    }
}

/// The configuration of the production [`fold_fit`]. The fixed flags (`had`, `e8_code`, `bscale`,
/// `hier_scale`, `e8_icm`, `act_order`, `wscale`) are not parameters: the production arm sets them
/// all true.
#[derive(Clone, Copy, Debug)]
pub struct FoldFitConfig {
    /// Outer iterations of the alternation (line 863). Only `1` is ported.
    pub iters: usize,
    /// Metric ridge of `metric(H, ridge)`: `ridge · mean(diag(H))` on the diagonal, Q32.32.
    pub ridge: i64,
    /// Damping passed to [`gptq_e8`] (the Python `damp`), Q32.32.
    pub damp: i64,
    /// Inner refit passes per outer iteration (line 864; the production call site uses `3`).
    pub inner_iters: usize,
    /// Beam width passed to [`gptq_e8`] (the production call site uses `4`).
    pub e8_beam: usize,
    /// `icm_ridge` of [`e8_icm_step`] (`icm_ridge · (n/896) · mean(diag(H))`, line 590), Q32.32.
    pub icm_ridge: i64,
    /// `sweeps` of each [`e8_icm_step`] call (the production call site uses `2`).
    pub sweeps: usize,
}

/// Which candidate of the alternation supplied the winning plane's code, scale and Hadamard scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldFitTag {
    /// The plain `gptq_e8` candidate (`trace[0]`).
    Gptq,
    /// The `refit0` candidate (`trace[1]`).
    Refit0,
}

impl FoldFitTag {
    /// The Python tag string (`"gptq"` / `"refit0"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gptq => "gptq",
            Self::Refit0 => "refit0",
        }
    }
}

/// The stored plane of the winning branch: `Wp = compose(fwht128(s_rot · dh), rh, ch)`, all four
/// buffers Q32.32 words.
#[derive(Clone, Debug)]
pub struct PackOut {
    /// Hadamard-domain code plane, row-major `[o, n]`.
    pub s_rot: Vec<i64>,
    /// Per-group row scale from [`quant_r_hier`], row-major `[o, n/128]`.
    pub rh: Vec<i64>,
    /// Per-column scale from [`quant_c_u8`], length `n`.
    pub ch: Vec<i64>,
    /// Hadamard column scale from [`quant_c_u8`], length `n`; all ones on the fallback branch.
    pub dh: Vec<i64>,
}

/// The result of [`fold_fit`]: the dequantised weight and the pack that recomposes to it.
#[derive(Clone, Debug)]
pub struct FoldFit {
    /// The dequantised weight `Wp`, row-major `[o, n]`, as Q32.32 words.
    pub wp: Vec<i64>,
    /// The winning plane `(s_rot, rh, ch, dh)`.
    pub pack: PackOut,
    /// The smallest candidate `J` (`min` over `trace`, the first one on a tie), as a word.
    pub j_best: i64,
    /// `J` per candidate: `[J(gptq), J(refit0)]`, as words.
    pub trace: Vec<i64>,
    /// Bit budget of the winning `rh` (`4·o·g + 8·o + 64`).
    pub bits_r: u64,
    /// Bit budget of the winning `ch` (`6·n + 32`).
    pub bits_c: u64,
    /// Bit budget of `dh` (`5·n + 32`).
    pub bits_d: u64,
    /// The candidate that supplied the winning plane.
    pub tag: FoldFitTag,
}

/// A candidate of the alternation, holding the plane it produced (all Q32.32 words).
#[derive(Clone, Debug)]
struct Candidate {
    j: i64,
    tag: FoldFitTag,
    s_rot: Vec<i64>,
    r: Vec<i64>,
    c: Vec<i64>,
    d: Vec<i64>,
}

/// The fit's shared carriers and shape: the converted inputs, the hoisted `Ws·Hl` product and the
/// dims every step takes.
struct Fit<'a> {
    wsq: &'a [i64],
    hlq: &'a [i64],
    wshlq: &'a [i64],
    grq: Option<&'a [i64]>,
    o: usize,
    n: usize,
    g: usize,
    /// `e8_icm_step` sweeps.
    sweeps: usize,
    /// `e8_icm_step` ridge, Q32.32.
    ridge: i64,
}

/// `round(num / den)` to nearest, ties to even; `None` when `den == 0` or the quotient leaves
/// `i64`. `O(1)`.
fn div_int(num: i128, den: i128) -> Option<i64> {
    if den == 0 {
        return None;
    }
    let mut q = num / den;
    let r = num % den;
    let twice = r.unsigned_abs().checked_mul(2)?;
    let up = match twice.cmp(&den.unsigned_abs()) {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Less => false,
        core::cmp::Ordering::Equal => q % 2 != 0,
    };
    if up {
        q = if r > 0 {
            q.checked_add(1)?
        } else {
            q.checked_sub(1)?
        };
    }
    i64::try_from(q).ok()
}

/// `a·b` as a Q32.32 word, one rounding to nearest, ties to even. `O(1)`.
fn mul_word(a: i64, b: i64) -> Result<i64, FoldFitError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| FoldFitError::Overflow)
}

/// `a/b` as a Q32.32 word, one rounding to nearest, ties to even. `O(1)`.
fn div_word(a: i64, b: i64) -> Result<i64, FoldFitError> {
    div_int(i128::from(a) << 32, i128::from(b)).ok_or(FoldFitError::Overflow)
}

/// The 128-block transform of an owned word buffer; the engine's [`fwht_words`] in place. `O(len)`.
fn rotate(mut x: Vec<i64>) -> Result<Vec<i64>, FoldFitError> {
    fwht_words(&mut x)?;
    Ok(x)
}

/// `metric(H, ridge) = H` with `ridge · mean(diag(H))` added to the diagonal (line 431), in words:
/// one `memcpy`, the diagonal mean by one rounded quotient of the exact `i128` sum, and `n`
/// checked adds. Time `O(n²)`, space `O(n²)`; the copy is a `memcpy` and the ridge touches `n`
/// entries, so the pool would buy nothing.
fn metric(h: &[i64], n: usize, ridge: i64) -> Result<Vec<i64>, FoldFitError> {
    let mut sum: i128 = 0;
    let mut ok = true;
    for i in 0..n {
        if ok {
            match sum.checked_add(i128::from(h[i * n + i])) {
                Some(v) => sum = v,
                None => ok = false,
            }
        }
    }
    if !ok {
        return Err(FoldFitError::Overflow);
    }
    let add = mul_word(
        ridge,
        div_int(sum, n as i128).ok_or(FoldFitError::Overflow)?,
    )?;
    let mut hl = h.to_vec();
    let mut ok = true;
    for i in 0..n {
        if ok {
            match hl[i * n + i].checked_add(add) {
                Some(v) => hl[i * n + i] = v,
                None => ok = false,
            }
        }
    }
    if ok {
        Ok(hl)
    } else {
        Err(FoldFitError::Overflow)
    }
}

/// `x[:, i] *= v[i]` for a row-major `[o, n]` buffer, one rounded product per entry. Time
/// `O(o·n)`, space `O(1)`. Six calls per fit over the whole plane: `6·o·n` integer multiplies
/// (`4.8·10⁶` at `o = n = 896`), each one round of a contiguous word pair, so a row-band dispatch
/// would add a task per band without cutting per-entry work; serial.
fn scale_columns(x: &mut [i64], o: usize, n: usize, v: &[i64]) -> Result<(), FoldFitError> {
    let mut ok = true;
    for row in 0..o {
        for k in 0..n {
            if ok {
                match mul_word(x[row * n + k], v[k]) {
                    Ok(y) => x[row * n + k] = y,
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(())
    } else {
        Err(FoldFitError::Overflow)
    }
}

/// The per-column sign of `d` with the 8-block parity fix (lines 871–878): the sign vector has an
/// even number of `−1` in every 8-block, and a block with an odd count has the sign at its
/// least-`|d|` index (first on a tie) flipped. Time `O(n)`, space `O(n)`; one word per column,
/// three orders below a dispatch's fixed cost.
fn parity_signs(d: &[i64], n: usize) -> Vec<i64> {
    let mut sign = vec![ONE; n];
    for (m, blk) in sign.chunks_mut(BLOCK).enumerate() {
        let b0 = m * BLOCK;
        let mut negatives = 0_usize;
        for k in 0..blk.len() {
            if d[b0 + k] < 0 {
                blk[k] = -ONE;
                negatives += 1;
            }
        }
        if negatives % 2 == 1 {
            let mut min_i = 0_usize;
            let mut min_abs = u64::MAX;
            for k in 0..blk.len() {
                let a = d[b0 + k].unsigned_abs();
                if a < min_abs {
                    min_abs = a;
                    min_i = k;
                }
            }
            blk[min_i] = -blk[min_i];
        }
    }
    sign
}

/// The 128-group normalisation of `d` (lines 878–889): `d_new[k] = d[k]·sign[k]` divided by its
/// group's maximum (floored at [`EPS_W`]), returned with the `n/128` maxima. Time `O(n)`, space
/// `O(n)`; one pass over `n/128 ≈ 2`–`32` groups, below the pool's task count.
fn group_norm(d: &[i64], sign: &[i64], n: usize) -> Result<(Vec<i64>, Vec<i64>), FoldFitError> {
    let g = n / GROUP;
    let mut dn = vec![0_i64; n];
    let mut gmax = vec![0_i64; g];
    let mut ok = true;
    for gi in 0..g {
        if ok {
            let b0 = gi * GROUP;
            let mut mx = 0_u64;
            for k in 0..GROUP {
                if ok {
                    match mul_word(d[b0 + k], sign[b0 + k]) {
                        Ok(v) => {
                            dn[b0 + k] = v;
                            let a = v.unsigned_abs();
                            if a > mx {
                                mx = a;
                            }
                        }
                        Err(_) => ok = false,
                    }
                }
            }
            let mx = i64::try_from(mx).unwrap_or(i64::MAX).max(EPS_W);
            for k in 0..GROUP {
                if ok {
                    match div_word(dn[b0 + k], mx) {
                        Ok(v) => dn[b0 + k] = v,
                        Err(_) => ok = false,
                    }
                }
            }
            gmax[gi] = mx;
        }
    }
    if ok {
        Ok((dn, gmax))
    } else {
        Err(FoldFitError::Overflow)
    }
}

/// `r[:, i] *= v[i]` for a row-major `[o, g]` buffer. Time `O(o·g)`, space `O(1)`; `g = n/128`
/// leaves a handful of columns per row, `o·n/128` words, below a dispatch.
fn scale_group_columns(r: &mut [i64], o: usize, g: usize, v: &[i64]) -> Result<(), FoldFitError> {
    let mut ok = true;
    for row in 0..o {
        for k in 0..g {
            if ok {
                let i = row * g + k;
                match mul_word(r[i], v[k]) {
                    Ok(y) => r[i] = y,
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(())
    } else {
        Err(FoldFitError::Overflow)
    }
}

/// `x[*] /= d` for a word buffer, one rounded quotient per entry. Time `O(len)`, space `O(1)`;
/// `len = n` columns per refit iteration, the same order as a dispatch's fixed cost; serial.
fn scale_inv(x: &mut [i64], d: i64) -> Result<(), FoldFitError> {
    for w in x.iter_mut() {
        *w = div_word(*w, d)?;
    }
    Ok(())
}

/// One stored-format candidate: `S_dh = fwht128(s_rot · dh)`, the `r`/`c` refit against it, the
/// recomposition `Wp`, its `J` and the two carriers' bit budgets.
struct Refit {
    wp: Vec<i64>,
    rh: Vec<i64>,
    ch: Vec<i64>,
    j: i64,
    bits_r: u64,
    bits_c: u64,
}

/// The `bscale and had` arm's `fwht → r_step → quant_r_hier → c_step → quant_c_u8 → compose →
/// objective` sequence for one `(s_rot, c, dh)`. Time `O(o·n·log n + o·n²)` (one `fwht` and two
/// `O(o·n²)` steps), space `O(o·n)`.
fn pack_words(f: &Fit, s_rot: &[i64], c: &[i64], dh: &[i64]) -> Result<Refit, FoldFitError> {
    let mut rotated = s_rot.to_vec();
    scale_columns(&mut rotated, f.o, f.n, dh)?;
    let s_dh = rotate(rotated)?;
    let r = r_step_wshl(f.wsq, f.hlq, f.wshlq, &s_dh, c, f.o, f.n)?;
    let hier = quant_r_hier(&r, f.o, f.g)?;
    let rh = hier.rq;
    let c_ref = c_step_wshl(f.wsq, f.hlq, f.wshlq, &s_dh, &rh, f.o, f.n, f.grq)?;
    let (ch, bits_c) = quant_c_u8(&c_ref, 6)?;
    let wp = compose(&s_dh, &rh, &ch, f.o, f.n)?;
    let j = objective(f.wsq, &wp, f.hlq, f.o, f.n)?;
    Ok(Refit {
        wp,
        rh,
        ch,
        j,
        bits_r: hier.bits,
        bits_c,
    })
}

/// The production [`fold_fit`]: the `gptq_e8` candidate, the `inner_iters` refit passes, the
/// candidate selection and the `bscale and had` pack.
///
/// `ws` is the pristine weight, row-major `[o, n]` Q32.32 words; `h` the calibration Gram, row-major
/// `[n, n]` words; `g_row` the optional row weight (length `o`), words. `n` must be a positive
/// multiple of [`GROUP`], `iters` must be `1`. The arguments, the returned struct and
/// [`FoldFitConfig`]'s `ridge`/`damp`/`icm_ridge` are words: the code plane never leaves the
/// carrier.
///
/// # Errors
///
/// [`FoldFitError::Unsupported`] if `iters != 1`; [`FoldFitError::ZeroDim`] if `o == 0` or
/// `n == 0`; [`FoldFitError::Width`] if `n` is not a positive multiple of [`GROUP`];
/// [`FoldFitError::ShapeMismatch`] on a buffer length that disagrees with `o`/`n`;
/// [`FoldFitError::NotPositiveDefinite`] if a reduced system is not positive-definite;
/// [`FoldFitError::Overflow`], [`FoldFitError::Codebook`] or [`FoldFitError::BadBits`] from a
/// helper; [`FoldFitError::PackMismatch`] if the winning pack does not recompose to `Wp`.
///
/// # Complexity
///
/// Time `O(o·n² + gptq + inner_iters·(o·n² + n³ + e8_icm) + q)`: the metric product `Ws·Hl` is
/// formed once and each refit step consumes it; the dominant terms are the `gptq_e8` beam search,
/// the per-pass `c`/`d` normal systems and the [`e8_icm_step`] sweeps, all pooled inside the
/// helpers. This file adds `O(n²)` for the metric copy and `O(o·n)` per column scaling. Space
/// `O(n² + o·n)`.
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the body is the ported `fold_fit` alternation; splitting it would only relocate the \
              step sequence"
)]
pub fn fold_fit(
    ws: &[i64],
    h: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
    cfg: &FoldFitConfig,
) -> Result<FoldFit, FoldFitError> {
    if cfg.iters != 1 {
        return Err(FoldFitError::Unsupported);
    }
    if o == 0 || n == 0 {
        return Err(FoldFitError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(FoldFitError::Width);
    }
    if ws.len() != o * n || h.len() != n * n {
        return Err(FoldFitError::ShapeMismatch);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(FoldFitError::ShapeMismatch);
        }
    }

    let ridge_q = cfg.ridge;
    let damp_q = cfg.damp;
    let g = n / GROUP;
    let nb = n / BLOCK;
    let hlq = metric(h, n, ridge_q)?;
    // One metric product `Ws·Hl` for the whole fit: every `c`/`r`/`d` step reuses it.
    let wshlq = mul_metric(ws, &hlq, o, n)?;
    let f = Fit {
        wsq: ws,
        hlq: &hlq,
        wshlq: &wshlq,
        grq: g_row,
        o,
        n,
        g,
        sweeps: cfg.sweeps,
        ridge: cfg.icm_ridge,
    };
    let idx_zero = vec![0_u16; o * nb];

    // Step 2: the `gptq_e8` candidate (lines 843–860), words in and out.
    let ws_rot_q = rotate(ws.to_vec())?;
    let h_syrk_q = had_gram(&hlq, n)?;
    let q = gptq_e8(&ws_rot_q, &h_syrk_q, o, n, damp_q, true, true, cfg.e8_beam)?;
    let s_rot0 = q.s;
    let r0 = q.scale;
    let c0 = vec![ONE; n];
    let d0 = vec![ONE; n];
    let s0 = rotate(s_rot0.clone())?;
    let wq0 = compose(&s0, &r0, &c0, o, n)?;
    let j0 = objective(f.wsq, &wq0, f.hlq, o, n)?;

    let mut cands: Vec<Candidate> = vec![Candidate {
        j: j0,
        tag: FoldFitTag::Gptq,
        s_rot: s_rot0,
        r: r0,
        c: c0,
        d: d0,
    }];
    let mut trace = vec![j0];

    // Step 3: the single refit pass (`iters = 1`, lines 862–889), on owned copies of the
    // `gptq` candidate's plane.
    let mut s_rot = cands[0].s_rot.clone();
    let mut s_cur = rotate(s_rot.clone())?;
    let mut r_cur = cands[0].r.clone();
    let mut c_cur = cands[0].c.clone();
    let mut d_cur = cands[0].d.clone();
    for _ in 0..cfg.inner_iters {
        c_cur = c_step_wshl(f.wsq, f.hlq, f.wshlq, &s_cur, &r_cur, o, n, f.grq)?;
        let mut sum: i128 = 0;
        for w in &c_cur {
            sum = sum
                .checked_add(i128::from(w.unsigned_abs()))
                .ok_or(FoldFitError::Overflow)?;
        }
        let s_mean = div_int(sum, n as i128).ok_or(FoldFitError::Overflow)?;
        if s_mean == 0 {
            return Err(FoldFitError::Overflow);
        }
        scale_inv(&mut c_cur, s_mean)?;
        let s_vec = vec![s_mean; g];
        scale_group_columns(&mut r_cur, o, g, &s_vec)?;
        r_cur = r_step_wshl(f.wsq, f.hlq, f.wshlq, &s_cur, &c_cur, o, n)?;

        let d_new = d_step_wshl(f.wsq, f.hlq, f.wshlq, &s_rot, &r_cur, &c_cur, o, n, f.grq)?;
        let sign = parity_signs(&d_new, n);
        scale_columns(&mut s_rot, o, n, &sign)?;
        let (d_next, gmax) = group_norm(&d_new, &sign, n)?;
        d_cur = d_next;
        scale_group_columns(&mut r_cur, o, g, &gmax)?;

        let plane = PlaneRef {
            s: &s_rot,
            idx: &idx_zero,
            r: &r_cur,
            c: &c_cur,
            d: &d_cur,
        };
        s_rot = e8_icm_step(f.wsq, f.hlq, plane, o, n, f.sweeps, f.ridge)?.s;
        let mut rotated = s_rot.clone();
        scale_columns(&mut rotated, o, n, &d_cur)?;
        s_cur = rotate(rotated)?;
        r_cur = r_step_wshl(f.wsq, f.hlq, f.wshlq, &s_cur, &c_cur, o, n)?;
    }
    let wq1 = compose(&s_cur, &r_cur, &c_cur, o, n)?;
    let j1 = objective(f.wsq, &wq1, f.hlq, o, n)?;
    cands.push(Candidate {
        j: j1,
        tag: FoldFitTag::Refit0,
        s_rot,
        r: r_cur,
        c: c_cur,
        d: d_cur,
    });
    trace.push(j1);

    // Step 4: `best = argmin J`, first on a tie (line 910).
    let mut best = 0_usize;
    for i in 1..cands.len() {
        if cands[i].j < cands[best].j {
            best = i;
        }
    }
    let j_best = cands[best].j;
    let tag = cands[best].tag;
    let s_rot_best = cands[best].s_rot.clone();
    let c_best = cands[best].c.clone();
    let d_best = cands[best].d.clone();

    // The `bscale and had` arm (lines 932–968): quantise the winning `d` to 5 bits, refit `r` and
    // `c` against the re-materialised `S_dh`, then the `e8_icm` re-quantisation and un-`d`
    // candidates.
    let (dh, bits_d) = quant_c_u8(&d_best, 5)?;
    let ones = vec![ONE; n];
    let first = pack_words(&f, &s_rot_best, &c_best, &dh)?;
    let (bits_r, bits_c) = (first.bits_r, first.bits_c);
    let mut wp = first.wp;
    let mut j_wp = first.j;
    let mut pack_s_rot = s_rot_best.clone();
    let mut pack_rh = first.rh;
    let mut pack_ch = first.ch;
    let mut pack_dh = dh;
    let mut s_rot_cur = s_rot_best;

    let plane = PlaneRef {
        s: &pack_s_rot,
        idx: &idx_zero,
        r: &pack_rh,
        c: &pack_ch,
        d: &pack_dh,
    };
    let s_rot_q = e8_icm_step(f.wsq, f.hlq, plane, o, n, f.sweeps, f.ridge)?.s;
    let again = pack_words(&f, &s_rot_q, &pack_ch, &pack_dh)?;
    if again.j < j_wp {
        j_wp = again.j;
        wp = again.wp;
        s_rot_cur.clone_from(&s_rot_q);
        pack_s_rot = s_rot_q;
        pack_rh = again.rh;
        pack_ch = again.ch;
    }

    let fall = pack_words(&f, &s_rot_cur, &c_best, &ones)?;
    if fall.j < j_wp {
        wp = fall.wp;
        pack_s_rot = s_rot_cur;
        pack_rh = fall.rh;
        pack_ch = fall.ch;
        pack_dh = ones;
    }

    // The winning pack must recompose to `Wp` exactly (the Python's `assert`, line 968).
    let mut rotated = pack_s_rot.clone();
    scale_columns(&mut rotated, o, n, &pack_dh)?;
    let recomposed = compose(&rotate(rotated)?, &pack_rh, &pack_ch, o, n)?;
    if recomposed.iter().zip(&wp).any(|(a, b)| a != b) {
        return Err(FoldFitError::PackMismatch);
    }

    Ok(FoldFit {
        wp,
        pack: PackOut {
            s_rot: pack_s_rot,
            rh: pack_rh,
            ch: pack_ch,
            dh: pack_dh,
        },
        j_best,
        trace,
        bits_r,
        bits_c,
        bits_d,
        tag,
    })
}
