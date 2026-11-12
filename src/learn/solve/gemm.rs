//! Blocked, pooled, exact-accumulation Q32.32 matrix products: [`gemm_nn`] (`C = A·B`) and
//! [`gemm_nt`] (`C = A·Bᵀ`), the one GEMM the fold's low-rank corrector needs.
//!
//! Entry for entry, `C[i,j] = round32(Σ_l A[i,l]·B[l,j])` ([`gemm_nn`]) or
//! `C[i,j] = round32(Σ_l A[i,l]·B[j,l])` ([`gemm_nt`]): each Q32.32 `i64` product is formed exactly
//! in `i128`, the contraction sum is accumulated exactly in `i128`, and a single round-to-nearest
//! shift by 32 is applied once to the finished sum. This is the product `eigen_parts`
//! (`src/learn/fold/fit/corrector.rs`) forms as `w1 = ΔW·Hl` (`o×n×n`, [`gemm_nn`]) and
//! `m = w1·ΔWᵀ` (`o×o×n`, [`gemm_nt`]) in scalar `f64` — 140 s at `o = 896`, `n = 4864` — and the
//! one the later `stage`/`gptq_e8` steps need. [`gemm_nt`] is the cache-friendly form: both
//! operands are read as contiguous dot products.
//!
//! # Reuse
//!
//! The rounding ([`super::syrk::round32`]), the `i64` narrowing ([`super::q32::narrow`]) and the
//! magnitude-bound scan ([`super::syrk::within_bound`] over [`super::syrk::BOUND_BITS`]) are the
//! `syrk` module's, promoted to `pub(super)` rather than copied, and [`MAX_K`] is derived from that
//! same `BOUND_BITS`. The blocked shape is `syrk`'s too: a plain `fn` block body driven by the engine
//! pool over disjoint output row chunks, one first-error return, the chunk count from
//! [`crate::engine::chart::pool::width_global`].
//!
//! # Relation to `dense::linear_fwd`
//!
//! `crate::learn::grad::dense::linear_fwd(x, w, n, i, o)` already computes exactly
//! `gemm_nt(x, w, n, i, o)`: `round32(Σ_k x[r,k]·w[c,k])` with `x` `n×i`, `w` `o×i`, the same
//! Q32.32 product, the same `(v + 2³¹) >> 32` rounding and the same `k`-ascending accumulation — as
//! a single-threaded, unbounded triple loop. [`gemm_nt`] is its blocked, pooled, bound-checked
//! generalisation and returns the same bits on the shared domain, so **`linear_fwd` should delegate
//! to `gemm_nt(x, w, n, i, o, 1)`** and map [`GemmError`], keeping only its `Empty`
//! zero-dimension policy; `dense.rs` is not edited here.
//!
//! # Exactness and the accepted domain
//!
//! Every `a` and `b` entry satisfies `|v| < 2³⁶` in raw Q32.32 units (real magnitudes below 16) and
//! the contraction length satisfies `k < MAX_K` with [`MAX_K`] `= 1 << (127 − 2·BOUND_BITS) = 2⁵⁵`.
//! The product of two entries is then below `2⁷²`, so the exact `i128` contraction sum is below
//! `k·2⁷² < 2¹²⁷`, and the round-half add `2³¹` keeps it inside `i128`
//! (`(2⁵⁵−1)·(2³⁶−1)² + 2³¹ < 2¹²⁷`). A violated bound is refused, not wrapped:
//! [`GemmError::InputBound`] for an entry at `2³⁶`, [`GemmError::Overflow`] for `k ≥ MAX_K`, for a
//! dimension product that leaves `usize`, or for a rounded entry that leaves `i64`.
//!
//! # Blocking and the tile
//!
//! The contraction is cut into `KC`-deep cache blocks; within a block an `MC × NC` output tile
//! accumulates in a `i128` array while the packed `MC × KC` A panel and `NC × KC` B panel are
//! read. With `MC = NC = 64`, `KC = 128` the A and B panels are `64 KiB` each and the accumulator
//! tile `64 KiB` — a `192 KiB` working set that fits this box's per-core L2 (`32 MiB` across 12
//! instances, `≥ 2 MiB` per core). The inner kernel is a `4 × 4` tile of `i128` accumulators
//! ([`MR`] `×` [`NR`]) held across the block's `kk` loop, a plain safe-Rust loop with no `unsafe`
//! and no intrinsics; the compiler lowers `i64 × i64 → i128` to a widening multiply and a
//! two-word add and does not vectorise it, which the `#[ignore]` timing test reports as a measured
//! GMAC/s rather than assumes away.
//!
//! Grade: `tested` (`tests/learn_solve_gemm.rs`); no body here is extracted or refined.

#![allow(
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::similar_names,
    reason = "`usize::from` is not usable in the `const MAX_K` initialiser, so the `BOUND_BITS as \
              usize` widening stays; the single-letter names are the operands' index symbols (`m`, `k`, `n`, `i`, `j`, \
              `l`, `kb`, `kc`, `ii`, `jj`) that read closer to the arithmetic than spelled-out \
              names; the `4 × 4` register tile names `c00..c33`, `a0..a3`, `b0..b3`. The `Result` \
              returns are already `#[must_use]`."
)]

use super::q32::narrow;
use super::syrk::{round32, within_bound, BOUND_BITS};
use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;

/// The largest accepted contraction length: `k < MAX_K` keeps the exact `i128` sum plus the
/// round-half add inside `i128` (`(MAX_K−1)·(2³⁶−1)² + 2³¹ < 2¹²⁷`). `127 − 2·BOUND_BITS = 55`.
///
/// On a target whose `usize` is narrower than 56 bits (32-bit, WebAssembly) `2⁵⁵` is not
/// representable, and no slice there can be that long, so the bound is `usize::MAX`: the check
/// `k < MAX_K` still holds for every length that can exist.
pub const MAX_K: usize = {
    const BITS: usize = 127 - 2 * BOUND_BITS as usize;
    if BITS < usize::BITS as usize {
        1 << BITS
    } else {
        usize::MAX
    }
};

/// Rows of the A cache panel: `MC · KC · 8 B = 64 KiB`.
const MC: usize = 64;
/// Columns of the B cache panel: `NC · KC · 8 B = 64 KiB`.
const NC: usize = 64;
/// Contraction depth of a cache block: `(MC + NC) · KC · 8 B = 128 KiB` of panel.
const KC: usize = 128;
/// Rows of the register tile of `i128` accumulators.
const MR: usize = 4;
/// Columns of the register tile of `i128` accumulators.
const NR: usize = 4;

/// A refused Q32.32 matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmError {
    /// A dimension is zero (`m == 0`, `k == 0` or `n == 0`), so the product has no entries.
    ZeroDim,
    /// A slice length disagrees with the declared shape (`a` is `[m, k]`, `b` is `[k, n]` for
    /// [`gemm_nn`] and `[n, k]` for [`gemm_nt`]).
    ShapeMismatch,
    /// An entry of `a` or `b` reaches the documented magnitude bound `2³⁶`.
    InputBound,
    /// `k ≥ `[`MAX_K`], a dimension product left `usize`, or a rounded entry left `i64`.
    Overflow,
}

/// Accumulate one `MR × NR` register tile of the `acc` block into `acc` over the packed A and B
/// panels: `acc[i,j] += Σ_kk ap[i·kc+kk]·bp[j·kc+kk]` for `i ∈ [ii, ii+MR)`, `j ∈ [jj, jj+NR)`.
///
/// `acc` is row-major `[mc, NC]`, `ap` row-major `[mc, kc]` and `bp` row-major `[nc, kc]`, with
/// `kc` the packed block depth. The read rows and columns are in range because `MC`/`NC` are
/// multiples of `MR`/`NR` and `ii ≤ mc ≤ MC`, `jj ≤ nc ≤ NC`. Rows and columns past `mc`/`nc` hold
/// pack padding and touch only accumulator entries the caller does not store. The `i128` products
/// and adds cannot overflow under the [`MAX_K`] contract, so they are unchecked here; the caller
/// checks the rounding add and the narrowing.
///
/// # Complexity
/// Time: `MR·NR·kc` `i128` multiply-accumulates and `MR·NR·(kc+1)` `i128` loads/stores — the `kk`
/// loop carries the `MR·NR` accumulators, one A-row element and one B-row element per step.
/// Space: `O(1)` — the accumulators are register-resident and nothing is allocated.
fn micro_4x4(ap: &[i64], bp: &[i64], acc: &mut [i128], kc: usize, ii: usize, jj: usize) {
    let a0 = &ap[ii * kc..ii * kc + kc];
    let a1 = &ap[(ii + 1) * kc..(ii + 1) * kc + kc];
    let a2 = &ap[(ii + 2) * kc..(ii + 2) * kc + kc];
    let a3 = &ap[(ii + 3) * kc..(ii + 3) * kc + kc];
    let b0 = &bp[jj * kc..jj * kc + kc];
    let b1 = &bp[(jj + 1) * kc..(jj + 1) * kc + kc];
    let b2 = &bp[(jj + 2) * kc..(jj + 2) * kc + kc];
    let b3 = &bp[(jj + 3) * kc..(jj + 3) * kc + kc];
    let base = ii * NC + jj;
    let mut c00 = acc[base];
    let mut c01 = acc[base + 1];
    let mut c02 = acc[base + 2];
    let mut c03 = acc[base + 3];
    let mut c10 = acc[base + NC];
    let mut c11 = acc[base + NC + 1];
    let mut c12 = acc[base + NC + 2];
    let mut c13 = acc[base + NC + 3];
    let mut c20 = acc[base + 2 * NC];
    let mut c21 = acc[base + 2 * NC + 1];
    let mut c22 = acc[base + 2 * NC + 2];
    let mut c23 = acc[base + 2 * NC + 3];
    let mut c30 = acc[base + 3 * NC];
    let mut c31 = acc[base + 3 * NC + 1];
    let mut c32 = acc[base + 3 * NC + 2];
    let mut c33 = acc[base + 3 * NC + 3];
    let mut kk = 0usize;
    while kk < kc {
        let x0 = i128::from(a0[kk]);
        let x1 = i128::from(a1[kk]);
        let x2 = i128::from(a2[kk]);
        let x3 = i128::from(a3[kk]);
        let y0 = i128::from(b0[kk]);
        let y1 = i128::from(b1[kk]);
        let y2 = i128::from(b2[kk]);
        let y3 = i128::from(b3[kk]);
        c00 += x0 * y0;
        c01 += x0 * y1;
        c02 += x0 * y2;
        c03 += x0 * y3;
        c10 += x1 * y0;
        c11 += x1 * y1;
        c12 += x1 * y2;
        c13 += x1 * y3;
        c20 += x2 * y0;
        c21 += x2 * y1;
        c22 += x2 * y2;
        c23 += x2 * y3;
        c30 += x3 * y0;
        c31 += x3 * y1;
        c32 += x3 * y2;
        c33 += x3 * y3;
        kk += 1;
    }
    acc[base] = c00;
    acc[base + 1] = c01;
    acc[base + 2] = c02;
    acc[base + 3] = c03;
    acc[base + NC] = c10;
    acc[base + NC + 1] = c11;
    acc[base + NC + 2] = c12;
    acc[base + NC + 3] = c13;
    acc[base + 2 * NC] = c20;
    acc[base + 2 * NC + 1] = c21;
    acc[base + 2 * NC + 2] = c22;
    acc[base + 2 * NC + 3] = c23;
    acc[base + 3 * NC] = c30;
    acc[base + 3 * NC + 1] = c31;
    acc[base + 3 * NC + 2] = c32;
    acc[base + 3 * NC + 3] = c33;
}

/// Pack rows `[i0, i0+mc)` and contraction columns `[kb, kb+kc)` of the row-major `[m, k]` `a` into
/// the contiguous `ap`, row stride `kc`.
///
/// # Complexity
/// `mc·kc` `i64` copies and `O(1)` extra space; derivation: one `counter`-free row copy per row, one
/// [`copy_from_slice`](slice::copy_from_slice) of `kc` words each.
fn pack_a(a: &[i64], k: usize, i0: usize, mc: usize, kb: usize, kc: usize, ap: &mut [i64]) {
    let mut i = 0usize;
    while i < mc {
        let src = (i0 + i) * k + kb;
        let dst = i * kc;
        ap[dst..dst + kc].copy_from_slice(&a[src..src + kc]);
        i += 1;
    }
}

/// Pack rows `[j0, j0+nc)` and contraction columns `[kb, kb+kc)` of the row-major `[n, k]` `b` (the
/// [`gemm_nt`] layout) into the contiguous `bp`, row stride `kc`.
///
/// # Complexity
/// `nc·kc` `i64` copies and `O(1)` extra space; derivation as [`pack_a`].
fn pack_b_nt(b: &[i64], k: usize, j0: usize, nc: usize, kb: usize, kc: usize, bp: &mut [i64]) {
    let mut j = 0usize;
    while j < nc {
        let src = (j0 + j) * k + kb;
        let dst = j * kc;
        bp[dst..dst + kc].copy_from_slice(&b[src..src + kc]);
        j += 1;
    }
}

/// Pack output columns `[j0, j0+nc)` and contraction rows `[kb, kb+kc)` of the row-major `[k, n]`
/// `b` (the [`gemm_nn`] layout) into the `[nc, kc]` `bp`, transposing the block so the inner kernel
/// reads B as contiguous rows.
///
/// # Complexity
/// `nc·kc` `i64` copies and `O(1)` extra space; derivation: one `kc`-long row segment of `b` per
/// column `j`, transposed on the copy.
fn pack_b_nn(b: &[i64], n: usize, j0: usize, nc: usize, kb: usize, kc: usize, bp: &mut [i64]) {
    let mut j = 0usize;
    while j < nc {
        let mut kk = 0usize;
        while kk < kc {
            bp[j * kc + kk] = b[(kb + kk) * n + j0 + j];
            kk += 1;
        }
        j += 1;
    }
}

/// Compute all `rows` output rows of one worker's `c` (row stride `n`) for `C = A·B` (`nn`) or
/// `C = A·Bᵀ` (`nt`) with `B` in the `nn` (`[k, n]`, `true`) or `nt` (`[n, k]`, `false`) layout, in
/// `MC × NC` output blocks over `KC`-deep packed panels with the [`micro_4x4`] register kernel. `a` is this worker's `rows × k`
/// row block, so the local row `i` of `c` is row `i` of `a`.
///
/// A given output entry is accumulated over the `k`-blocks in order and within a block over `kk`
/// in order — that is ascending `k`, exactly once and by this worker alone — so the result is
/// independent of the block geometry and the pool width. `c.len()` must be `rows·n`.
///
/// # Complexity
/// Time: `rows·k·n` `i128` multiply-accumulates plus `rows·k·n·(1/MC + 1/NC)` `i64` panel copies
/// (`rows·k·n/32` at the chosen tiles). Derivation: the output is `⌈rows/MC⌉·⌈n/NC⌉` blocks, each
/// cost `k·MC·NC` MACs and `k·(MC + NC)` panel copies. Space: `O(MC·KC + NC·KC) i64` for the panels
/// plus `O(MC·NC) i128` for the accumulator tile, per worker, reused across the whole block.
fn block_rows(
    c: &mut [i64],
    rows: usize,
    a: &[i64],
    b: &[i64],
    n: usize,
    k: usize,
    nn: bool,
) -> Result<(), GemmError> {
    if rows == 0 {
        return Ok(());
    }
    let mut acc = vec![0i128; MC * NC];
    let mut ap = vec![0i64; MC * KC];
    let mut bp = vec![0i64; NC * KC];
    // Flag loop: the only refusal source is the rounding/narrowing of an accumulator entry in the
    // store loop; the packing and the register kernel are total. Every enclosing loop is guarded by
    // `err.is_none()`, no `return`, `?` or `break` sits inside a loop body, and the `return` sits
    // after the outermost loop.
    let mut err: Option<GemmError> = None;
    let mut i0 = 0usize;
    while i0 < rows && err.is_none() {
        let mc = (i0 + MC).min(rows) - i0;
        let mut jb = 0usize;
        while jb < n && err.is_none() {
            let nc = (jb + NC).min(n) - jb;
            acc[..].fill(0);
            let mut kb = 0usize;
            while kb < k && err.is_none() {
                let kc = (kb + KC).min(k) - kb;
                pack_a(a, k, i0, mc, kb, kc, &mut ap);
                if nn {
                    pack_b_nn(b, n, jb, nc, kb, kc, &mut bp);
                } else {
                    pack_b_nt(b, k, jb, nc, kb, kc, &mut bp);
                }
                let mut ii = 0usize;
                while ii < mc && err.is_none() {
                    let mut jj = 0usize;
                    while jj < nc && err.is_none() {
                        micro_4x4(&ap, &bp, &mut acc, kc, ii, jj);
                        jj += NR;
                    }
                    ii += MR;
                }
                kb += kc;
            }
            let mut i = 0usize;
            while i < mc && err.is_none() {
                let mut j = 0usize;
                while j < nc && err.is_none() {
                    match round32(acc[i * NC + j]) {
                        Some(r) => match narrow(r) {
                            Some(v) => c[(i0 + i) * n + jb + j] = v,
                            None => err = Some(GemmError::Overflow),
                        },
                        None => err = Some(GemmError::Overflow),
                    }
                    j += 1;
                }
                i += 1;
            }
            jb += nc;
        }
        i0 += mc;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The pool task context of [`block_rows_pool`]: the shared operands and geometry, the row stride
/// and the refusal cell.
struct GemmRows<'a> {
    /// The `A` panel, row-major `[m, k]`.
    a: &'a [i64],
    /// The `B` panel, row-major `[k, n]` (`nn`) or `[n, k]` (`!nn`).
    b: &'a [i64],
    /// Contraction depth.
    k: usize,
    /// Output row stride.
    n: usize,
    /// The `B` panel layout: `true` for `A·B`, `false` for `A·Bᵀ`.
    nn: bool,
    /// Lowest-offset refusal recorded by the tasks.
    first: &'a FirstErr<usize, GemmError>,
}

impl ChunkTask<i64> for GemmRows<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let rows = piece.len() / self.n;
        let row0 = start / self.n;
        let ablock = &self.a[row0 * self.k..(row0 + rows) * self.k];
        if let Err(e) = block_rows(piece, rows, ablock, self.b, self.n, self.k, self.nn) {
            self.first.record(start, e);
        }
    }
}

/// Split `c` (`m × n`, row stride `n`) into `width_global()` contiguous row chunks and run
/// [`block_rows`] on each through the engine pool; returns the first refusal. Every task owns a
/// disjoint row range of `c` and is handed the matching `a` row block (`&a[lo·k .. hi·k]`), so no
/// two tasks write one output entry and none reads another's A rows. Each entry is accumulated by
/// exactly one task in ascending `k`, so the result is bit-identical to the serial body.
///
/// # Complexity
/// Time: [`block_rows`] plus `O(m)` split bookkeeping. Space: `O(1)` extra (the pool hands each task
/// its slice in place) plus each task's `O(MC·KC + NC·KC + MC·NC)` scratch.
fn block_rows_pool(
    c: &mut [i64],
    m: usize,
    n: usize,
    a: &[i64],
    b: &[i64],
    k: usize,
    nn: bool,
) -> Result<(), GemmError> {
    let chunk_rows = m.div_ceil(width_global()).max(1);
    let first: FirstErr<usize, GemmError> = FirstErr::new();
    chunks_mut_global(
        c,
        chunk_rows * n,
        &GemmRows {
            a,
            b,
            k,
            n,
            nn,
            first: &first,
        },
    );
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Check the shared domain of [`gemm_nn`] and [`gemm_nt`]: a non-zero `m`, `k`, `n`; `k < `[`MAX_K`]
/// and dimension products that fit `usize`; `a.len() == m·k` and `b.len() == k·n` (the length is the
/// same in either layout); and every entry of `a` and `b` below `2³⁶`.
///
/// # Complexity
/// `O(m·k + k·n)` bound comparisons and `O(1)` extra space.
fn check(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Result<(), GemmError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(GemmError::ZeroDim);
    }
    if k >= MAX_K {
        return Err(GemmError::Overflow);
    }
    let Some(ak) = m.checked_mul(k) else {
        return Err(GemmError::Overflow);
    };
    let Some(kn) = k.checked_mul(n) else {
        return Err(GemmError::Overflow);
    };
    if n.checked_mul(m).is_none() {
        return Err(GemmError::Overflow);
    }
    if a.len() != ak || b.len() != kn {
        return Err(GemmError::ShapeMismatch);
    }
    if !within_bound(a, Some(b)) {
        return Err(GemmError::InputBound);
    }
    Ok(())
}

/// `C[m,n] = A[m,k]·B[k,n]` in Q32.32: `a` row-major `[m, k]`, `b` row-major `[k, n]`, `c`
/// row-major `[m, n]` with `C[i,j] = round32(Σ_l a[i,l]·b[l,j])`, one rounding per entry.
///
/// The output rows are split across the engine pool, one chunk per `width_global()` participants.
/// Each entry is accumulated by exactly one task over ascending `k`, so the result is bit-identical
/// to the serial body.
///
/// # Errors
/// [`GemmError::ZeroDim`] if `m`, `k` or `n` is zero; [`GemmError::ShapeMismatch`] if `a.len() !=
/// m·k` or `b.len() != k·n`; [`GemmError::InputBound`] if an entry reaches `2³⁶`;
/// [`GemmError::Overflow`] if `k ≥ `[`MAX_K`], a dimension product leaves `usize`, or a rounded
/// entry leaves `i64`.
///
/// # Complexity
/// Time: `m·k·n` `i128` multiply-accumulates plus `m·k·n·(1/MC + 1/NC)` panel copies, split across
/// pool; the derivation is in [`block_rows`]. Space: `O(m·n)` for `c` plus each worker's
/// `O(MC·KC + NC·KC + MC·NC)` scratch.
pub fn gemm_nn(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Result<Vec<i64>, GemmError> {
    run(a, b, m, k, n, true)
}

/// `C[m,n] = A[m,k]·B[n,k]ᵀ` in Q32.32: `a` row-major `[m, k]`, `b` row-major `[n, k]`, `c`
/// row-major `[m, n]` with `C[i,j] = round32(Σ_l a[i,l]·b[j,l])`, one rounding per entry. Both
/// operand rows are contiguous in the contraction index, so this is the cache-friendly form and the
/// one `m = w1·ΔWᵀ` needs.
///
/// The output rows are split exactly as [`gemm_nn`] does, with the same bit-identity.
///
/// # Errors
/// As [`gemm_nn`], with `b` of shape `[n, k]`.
///
/// # Complexity
/// Time and space as [`gemm_nn`], with `B` read as contiguous rows.
pub fn gemm_nt(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Result<Vec<i64>, GemmError> {
    run(a, b, m, k, n, false)
}

/// The shared body of [`gemm_nn`] and [`gemm_nt`]: validate, allocate `C`, and drive
/// [`block_rows_pool`] with `nn` selecting the B-panel layout.
///
/// # Complexity
/// As [`gemm_nn`].
fn run(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    nn: bool,
) -> Result<Vec<i64>, GemmError> {
    check(a, b, m, k, n)?;
    let mut c = vec![0i64; m * n];
    block_rows_pool(&mut c, m, n, a, b, k, nn)?;
    Ok(c)
}
