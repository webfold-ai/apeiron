//! k-column int8-activation projection (L1): [`matmat_q8act`], the int8-activation matrix product
//! whose column `c` is [`matvec_q8act`] on `qs[c]` bit for bit.
//!
//! For `Q8_0` and `Q5_0` each weight block is decoded once and dotted against all `k` activation
//! blocks while its bytes are in registers, so the weight payload is read once per call instead of
//! once per column. `Q4_K` and `Q6_K` walk the same rows through the packed eight-column lane of
//! `chart::kcols`, with the exact `i128` carrier as the lane's past-limit fallback. The exact
//! carriers use the additive decomposition of [`matvec_q8act`]'s fast row —
//! `Σ_b S_b·(d_b·Σ_l q_l·xq_{c,l})` in exact `i128` — so the integers are equal, the single
//! [`round_shift_32`] epilogue is equal, and the `i64` range check is equal. A fast kernel runs
//! only when [`matvec_q8act`] itself would take its fast row for every column and the call-level
//! bound rules out its `i64`/`i128` overflow; otherwise, and for every other family, the call is
//! the per-column [`matvec_q8act`], which is the contract itself.
//!
//! The AVX2 cores are reached only behind `is_x86_feature_detected!("avx2")` and are the only
//! unsafe code here. `Q8_0`/`Q5_0` have a portable twin; every carrier is compared with the
//! per-column oracle by the equality tests. That comparison is `tested`, not extracted (trust
//! boundary).
#![allow(
    dead_code,
    reason = "the L1 entry point and its cores land in this wave; their W2 consumers (spec_greedy, \
              batched prefill) are still to be wired, and the tests call them under cfg(test)"
)]

use crate::engine::chart::pool::PBand;
use crate::engine::chart::task::BandTask;
#[cfg(target_arch = "x86_64")]
use crate::engine::chart::{
    lane_spread_limit, pack_cols, q4_k_row_lanes_avx2, q6_k_row_lanes_avx2, PackedCols, Q4KRow,
    Q6KRow,
};
use crate::engine::chart::{matvec_q8act, ActMagnitude, QuantAct};
use crate::engine::f16_to_fixed;
use crate::engine::num::value::round_shift_32;
use crate::engine::MatvecError;
use crate::format::quant::{
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, Q4_K_BLOCK_BYTES,
    Q5_0_BLOCK_BYTES, Q5_0_QK, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_QK, QK_K,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_abs_epi8, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set_m128i,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sign_epi8,
    _mm256_slli_epi16, _mm256_sub_epi16, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32,
    _mm_loadu_si128, _mm_set1_epi8, _mm_shuffle_epi32, _mm_srli_epi16,
};

/// Lanes of one activation quantization block.
const ACT_QK: usize = 32;

/// The largest column count the portable row kernel handles: its per-row accumulators are a
/// `MAX_K`-wide `i128` array on the stack. Larger `k` takes the per-column reference unless the
/// pooled `AVX2` kernel takes the call.
const MAX_K: usize = 16;

/// The largest column count the pooled `AVX2` kernel handles. It keeps its accumulators on the
/// heap, so the bound is a placement budget rather than a register or stack width; `16` is
/// the portable kernel's, not this one's. A factorisation's `X`-update projects with one column
/// per rank (`k = r`, 576–1024 here), which is exactly the call the portable bound rejects.
const MAX_POOLED_K: usize = 4096;

/// The largest finite `f16` magnitude in Q32.32: `65504 · 2^32`, as in the chart's bound.
const F16_MAX_FIXED: i128 = 65504i128 << 32;

/// The four families with a shared weight-block decode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `Q8_0`: `d` at 0, 32 signed code bytes at 2.
    Q8_0,
    /// `Q5_0`: `d` at 0, `qh` at 2, 16 packed `qs` bytes at 6.
    Q5_0,
    /// `Q4_K`: 256 weights per 144-byte super-block, an affine `d·sc·dot − dmin·m·xsum` group.
    #[allow(
        non_camel_case_types,
        reason = "`Q4_K` is the format's GGML name, as in the carrier tables"
    )]
    Q4_K,
    /// `Q6_K`: 256 weights per 210-byte super-block, an offset-free `d·sc·dot` group.
    #[allow(
        non_camel_case_types,
        reason = "`Q6_K` is the format's GGML name, as in the carrier tables"
    )]
    Q6_K,
}

/// The weight payload geometry of one call: `n_in` is a whole number of 32-wide activation blocks.
#[derive(Clone, Copy)]
struct Geom {
    /// The carrier's weight blocks per row: `n_in / 32` for `Q8_0`/`Q5_0`, `n_in / 256` for
    /// `Q4_K`/`Q6_K`.
    blocks: usize,
    /// The raw bytes per weight block.
    block_bytes: usize,
    /// The raw bytes per weight row, `blocks · block_bytes`.
    row_bytes: usize,
}

/// The per-column terminal state of a fast-kernel call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColumnState {
    /// Still matching [`matvec_q8act`]'s fast row and within `i64` so far.
    Live,
    /// A row's rounded value left `i64`: the column's refusal is [`MatvecError::OutputOverflow`].
    Overflow,
    /// A row's fast value was `None` (a non-finite block scale): the column falls back to
    /// [`matvec_q8act`] itself.
    Fallback,
}

/// The fast kernel's per-row outcome.
#[allow(
    clippy::large_enum_variant,
    reason = "the fixed `MAX_K`-wide `i128` array is stack scratch on the row hot path; boxing it \
              would add an allocation per weight row, and the unit variants carry no payload"
)]
enum RowOutcome {
    /// Every live column's exact `i128` accumulator for this row.
    Values([i128; MAX_K]),
    /// A block scale was `Inf`/`NaN`, so [`matvec_q8act`]'s fast row returns `None` for the row.
    NonFinite,
    /// A checked `i64`/`i128` step left range, which the call-level bound makes unreachable; the
    /// caller takes the per-column reference rather than guess.
    Bail,
}

/// The k-column int8-activation projection: `out[c] = matvec_q8act(ggml_type, raw, n_out, n_in,
/// &qs[c])` for every `c`, including the same `Err`.
///
/// `Q8_0`, `Q5_0`, `Q4_K` and `Q6_K` with a whole-block width, exact activation magnitudes and a
/// call-level overflow bound run the pooled carrier row kernel (`q8_0_row_kcols_avx2`,
/// `q5_0_row_kcols_avx2`, or `q4_k_row_lanes_avx2`/`q6_k_row_lanes_avx2` on its packed panel)
/// under AVX2 over any `k` within [`MAX_POOLED_K`] and the output budget; without AVX2 only the
/// portable shared-decode kernel runs, over `k ≤ MAX_K`, for `Q8_0`/`Q5_0`; every other input is
/// the per-column [`matvec_q8act`]. Either way column `c` equals the per-column call bit for bit,
/// so a caller may read the k columns as k independent projections.
///
/// # Errors
///
/// The first column's [`MatvecError`], in column order: the call is `Ok` exactly when every column
/// is `Ok`, and otherwise refuses with the refusal of the lowest-indexed refusing column — the
/// same [`MatvecError`] [`matvec_q8act`] returns for that column.
///
/// # Complexity
///
/// Temporal `O(n_out·n_in·k)` exact integer multiply-adds — the same count as `k` separate calls,
/// with the weight payload's `O(n_out·n_in)` bytes read once per call instead of `k` times and the
/// `k·n_out·n_in` activation code bytes read per column. Spatial `O(n_out·k)` for the output plus
/// `O(k)` column state, and `O(k·n_in)` bytes of `Q4_K`/`Q6_K` pack; no other carrier keeps
/// `O(n_in)` scratch.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check; `TryFrom` emits an unmodeled external the pinned \
              fail-closed extraction audit rejects"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the family dispatch, the row loop and the per-column refusal assembly are one \
              column-state machine: each live/overflow/fallback transition is one branch, and \
              factoring a branch out would move the state it decides away from the loop"
)]
pub(crate) fn matmat_q8act(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Result<Vec<Vec<i64>>, MatvecError> {
    // The power-of-two-scale GEMM (`super::gemm`) computes the same values when it can; it
    // defers every refusal to the path below.
    if let Some(cols) = super::gemm::gemm_q8_0(ggml_type, raw, n_out, n_in, qs) {
        return Ok(cols);
    }
    let k = qs.len();
    let Some(family) = fast_family(ggml_type, n_in, qs) else {
        return per_column(ggml_type, raw, n_out, n_in, qs);
    };
    let (blocks, block_bytes) = match family {
        Family::Q8_0 => (n_in / ACT_QK, Q8_0_BLOCK_BYTES),
        Family::Q5_0 => (n_in / ACT_QK, Q5_0_BLOCK_BYTES),
        Family::Q4_K => (n_in / QK_K, Q4_K_BLOCK_BYTES),
        Family::Q6_K => (n_in / QK_K, Q6_K_BLOCK_BYTES),
    };
    let row_bytes = blocks
        .checked_mul(block_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    let need = n_out
        .checked_mul(row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    if raw.len() < need {
        return Err(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        });
    }
    let geom = Geom {
        blocks,
        block_bytes,
        row_bytes,
    };
    let avx = avx2_available();
    #[cfg(target_arch = "x86_64")]
    if avx && k > 1 {
        if let Some(cols) = pooled(family, raw, n_out, geom, qs) {
            return Ok(cols);
        }
    }
    // The serial row loop has no `Q4_K`/`Q6_K` packed-lane carrier; a declined call, and decode
    // `k = 1`, take the per-column reference, which is the contract itself.
    if matches!(family, Family::Q4_K | Family::Q6_K) {
        return per_column(ggml_type, raw, n_out, n_in, qs);
    }
    // The portable row kernel's accumulators are a `MAX_K`-wide array on the stack; a larger `k`
    // that the pooled kernel declined (a non-finite scale, an `i64` word out of range, a budget
    // overrun) takes the per-column reference, which carries the same refusals.
    if k > MAX_K {
        return per_column(ggml_type, raw, n_out, n_in, qs);
    }
    let mut columns: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut c = 0usize;
    while c < k {
        columns.push(Vec::with_capacity(n_out));
        c += 1;
    }
    let mut state = vec![ColumnState::Live; k];
    let mut bail = false;
    let mut o = 0usize;
    while o < n_out && !bail {
        if !state.contains(&ColumnState::Live) {
            break;
        }
        let start = o * row_bytes;
        let row = &raw[start..start + row_bytes];
        match row_accs(family, row, geom, qs, k, &state, avx) {
            RowOutcome::NonFinite => {
                let mut c = 0usize;
                while c < k {
                    if state[c] == ColumnState::Live {
                        state[c] = ColumnState::Fallback;
                    }
                    c += 1;
                }
            }
            RowOutcome::Bail => bail = true,
            RowOutcome::Values(accs) => {
                let mut c = 0usize;
                while c < k {
                    if state[c] == ColumnState::Live {
                        let rounded = round_shift_32(accs[c]);
                        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                            state[c] = ColumnState::Overflow;
                        } else {
                            columns[c].push(rounded as i64);
                        }
                    }
                    c += 1;
                }
            }
        }
        o += 1;
    }
    if bail {
        return per_column(ggml_type, raw, n_out, n_in, qs);
    }
    let mut result: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut err: Option<MatvecError> = None;
    let mut c = 0usize;
    while c < k && err.is_none() {
        match state[c] {
            ColumnState::Live => result.push(std::mem::take(&mut columns[c])),
            ColumnState::Overflow => err = Some(MatvecError::OutputOverflow),
            ColumnState::Fallback => match matvec_q8act(ggml_type, raw, n_out, n_in, &qs[c]) {
                Ok(col) => result.push(col),
                Err(e) => err = Some(e),
            },
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(result)
}

/// Rows `r0..r0 + band.rows()` of the `k` output columns, writing each column's value through the
/// band, or `Err` when a row hits a non-finite scale or an output leaves `i64`.
///
/// # Complexity
///
/// `O(rows·n_in·k)` multiply-adds; `k` `i128` words of scratch per task, one `k`-word row for the
/// `Q4_K` lane, and the packed row decoder's `O(n_in)` words per task.
#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "k columns, r rows, o and c indices are the kernel's standard symbols; the context, \
              packed panel, limit and band are the task's disjoint inputs"
)]
fn pooled_rows(
    family: Family,
    raw: &[u8],
    geom: Geom,
    qs: &[QuantAct],
    packed: Option<&PackedCols>,
    limit: u32,
    r0: usize,
    band: &mut PBand<i64>,
) -> Result<(), ()> {
    let k = qs.len();
    // `Q4_K`/`Q6_K` take their packed lane: one row decode per row, reused by every column group,
    // eight columns per `i64` lane, one rounding per output. The exact `i128` carrier is the
    // lane's own past-`limit` fallback inside the lane kernel, not a second row loop here. The
    // lane fills one contiguous row, so the band receives it column by column afterwards.
    if matches!(family, Family::Q4_K | Family::Q6_K) {
        let Some(packed) = packed else {
            return Err(());
        };
        let mut q4 = (family == Family::Q4_K).then(|| Q4KRow::new(geom.blocks));
        let mut q6 = (family == Family::Q6_K).then(|| Q6KRow::new(geom.blocks));
        let mut out = vec![0i64; k];
        let mut r = 0usize;
        while r < band.rows() {
            let o = r0 + r;
            let row = &raw[o * geom.row_bytes..(o + 1) * geom.row_bytes];
            let bounds = match family {
                // SAFETY: the caller established AVX2 and the row's `geom.blocks` blocks.
                Family::Q4_K => unsafe { q4.as_mut().unwrap().fill(row, geom.blocks) },
                Family::Q6_K => unsafe { q6.as_mut().unwrap().fill(row, geom.blocks) },
                _ => unreachable!("only the lane carriers reach this branch"),
            };
            let Some((kmin, kmax)) = bounds else {
                return Err(());
            };
            // SAFETY: as `fill`, and `packed` covers every column with `out.len() == k`.
            let done = match family {
                Family::Q4_K => unsafe {
                    q4_k_row_lanes_avx2(
                        row,
                        geom.blocks,
                        qs,
                        packed,
                        q4.as_ref().unwrap(),
                        kmin,
                        kmax - kmin,
                        limit,
                        &mut out,
                    )
                },
                Family::Q6_K => unsafe {
                    q6_k_row_lanes_avx2(
                        row,
                        geom.blocks,
                        qs,
                        packed,
                        q6.as_ref().unwrap(),
                        kmin,
                        kmax - kmin,
                        limit,
                        &mut out,
                    )
                },
                _ => unreachable!("only the lane carriers reach this branch"),
            };
            if !done {
                return Err(());
            }
            let mut c = 0usize;
            while c < k {
                band.push(c, out[c]);
                c += 1;
            }
            r += 1;
        }
        return Ok(());
    }
    // Allocated once per task, not per row: `k` is a rank in the factorisation's projection and
    // `MAX_K` is the portable kernel's stack width, not this one's.
    let mut acc = vec![0i128; k];
    let mut stop = false;
    let mut r = 0usize;
    while r < band.rows() && !stop {
        let o = r0 + r;
        let row = &raw[o * geom.row_bytes..(o + 1) * geom.row_bytes];
        acc.fill(0);
        // SAFETY: the caller established AVX2, `raw.len() >= n_out·row_bytes`, and
        // `fast_family` checked every column's geometry and the call-level bound.
        let ok = match family {
            Family::Q8_0 => unsafe {
                crate::engine::chart::q8_0_row_kcols_avx2(row, geom.blocks, qs, &mut acc)
            },
            Family::Q5_0 => unsafe {
                crate::engine::chart::q5_0_row_kcols_avx2(row, geom.blocks, qs, &mut acc)
            },
            Family::Q4_K | Family::Q6_K => {
                unreachable!("the lane carriers return through their packed lane above")
            }
        };
        stop = ok.is_none();
        let mut c = 0usize;
        while c < k && !stop {
            if let Ok(v) = i64::try_from(round_shift_32(acc[c])) {
                band.push(c, v);
            } else {
                stop = true;
            }
            c += 1;
        }
        r += 1;
    }
    if stop {
        Err(())
    } else {
        Ok(())
    }
}

/// The pooled band-task context: the payload, its geometry, the carrier family, the activation
/// columns, and the packed panel and shift limit.
#[cfg(target_arch = "x86_64")]
struct PooledRowsCtx<'a> {
    family: Family,
    raw: &'a [u8],
    geom: Geom,
    qs: &'a [QuantAct],
    packed: Option<&'a PackedCols>,
    limit: u32,
}

#[cfg(target_arch = "x86_64")]
impl BandTask<i64, ()> for PooledRowsCtx<'_> {
    fn run(&self, _t: usize, first_row: usize, band: &mut PBand<i64>) -> Result<(), ()> {
        pooled_rows(
            self.family,
            self.raw,
            self.geom,
            self.qs,
            self.packed,
            self.limit,
            first_row,
            band,
        )
    }
}

/// The 32-wide chart fast path split by rows across the persistent pool: each task runs the
/// carrier's row kernel — `q8_0_row_kcols_avx2`, `q5_0_row_kcols_avx2`, or the `Q4_K`/`Q6_K`
/// packed `q4_k_row_lanes_avx2`/`q6_k_row_lanes_avx2` — over its rows for all `k` columns, so
/// every weight row is read from memory once per call. Each row value is written straight into
/// its output column through the row band, so the `n_out × k` result is never staged and
/// transposed.
///
/// Returns `None` when any row hits a non-finite scale, an output leaves `i64`, or a `Q4_K`/`Q6_K`
/// scale is not a power of two; the caller then runs the serial kernel, which carries the exact
/// refusal and fallback semantics.
///
/// # Complexity
///
/// `O(n_out·n_in·k)` multiply-adds over the pool; `O(n_out·k)` output words, written once in
/// place; `O(k·n_in)` bytes of `Q4_K`/`Q6_K` activation pack, the only carriers that build one.
#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::many_single_char_names,
    reason = "k columns, r rows, o and c indices are the kernel's standard symbols"
)]
fn pooled(
    family: Family,
    raw: &[u8],
    n_out: usize,
    geom: Geom,
    qs: &[QuantAct],
) -> Option<Vec<Vec<i64>>> {
    let k = qs.len();
    if k > MAX_POOLED_K {
        return None;
    }
    // The `Q4_K`/`Q6_K` packed lanes read the call's columns as `x + 128` code quads, four
    // consecutive inputs of each column in one 32-bit lane: `pack_cols` builds the panel once per
    // call, and `lane_spread_limit` is the largest `i64` shift spread the group grid admits.
    // `Q6_K` has no offset term, so it passes `need_xs = false`. Every other carrier uses the row
    // kernels' per-column codes and needs neither.
    let (packed, limit) = match family {
        Family::Q4_K => {
            let n32 = geom.blocks * 8;
            (Some(pack_cols(qs, n32, true)?), lane_spread_limit(n32))
        }
        Family::Q6_K => {
            let n32 = geom.blocks * 8;
            (
                Some(pack_cols(qs, n32, false)?),
                lane_spread_limit(geom.blocks * 16),
            )
        }
        Family::Q8_0 | Family::Q5_0 => (None, 0),
    };
    let mut cols: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut c = 0usize;
    while c < k {
        cols.push(Vec::new());
        c += 1;
    }
    let rows_per = n_out
        .div_ceil(crate::engine::ops::par::row_chunk_budget(n_out))
        .max(1);
    let rows_ctx = PooledRowsCtx {
        family,
        raw,
        geom,
        qs,
        packed: packed.as_ref(),
        limit,
    };
    if crate::engine::chart::pool::for_pbands(&mut cols, n_out, rows_per, &rows_ctx).is_err() {
        return None;
    }
    Some(cols)
}

/// The per-column [`matvec_q8act`] fallback, which is the contract itself.
///
/// Refuses with the lowest-indexed column's refusal, matching [`matmat_q8act`]'s error contract.
///
/// # Complexity
///
/// The sum of the per-column [`matvec_q8act`] costs, `O(k)` calls.
fn per_column(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Result<Vec<Vec<i64>>, MatvecError> {
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(qs.len());
    let mut c = 0usize;
    while c < qs.len() {
        out.push(matvec_q8act(ggml_type, raw, n_out, n_in, &qs[c])?);
        c += 1;
    }
    Ok(out)
}

/// The fast kernel's family, or `None` when any column must take the per-column reference.
///
/// True exactly when [`matvec_q8act`]'s `fast` flag is true for every column: a `Q8_0`, `Q5_0`,
/// `Q4_K` or `Q6_K` type, a whole number of 32-wide activation blocks (`Q4_K`/`Q6_K` also a whole
/// 256-weight super-block), `1 ≤ k ≤ MAX_POOLED_K`, each column's length matching `n_in` with one
/// scale per 32-lane group, an exact [`ActMagnitude`] (`NoQ32` and the `i128` total overflow both
/// route [`matvec_q8act`] elsewhere), and [`fast_bound_ok`] — the chart's private
/// `q8act_fast_bound_ok(…, narrow)` predicate, re-derived here from the public `scales()` accessor
/// so this file edits no other module.
///
/// [`fast_bound_ok`] subsumes the exact-magnitude test:
/// - With `scales.len() == blocks` and `q.len() == n_in`, every scale lookup is in-bounds.
/// - The `s_max·127 ≤ i64::MAX` check is exactly `NoQ32`'s block bound: every code obeys
///   `|xq_j| ≤ 127`, so every checked `i64` product `S_b·xq_j` is `Some`.
/// - The narrow condition `s_max·(32·top·127) ≤ i64::MAX` (checked for `Q8_0`'s `top = 128` and
///   `Q5_0`'s `top = 16`) additionally keeps each block's `i64` product in range; `Q4_K`
///   (`top = 1008`) and `Q6_K` (`top = 4096`) run their group terms in `i128`, so they take the
///   wide branch.
/// - The sum of at most `2^14` terms with `fast_bound_ok`'s
///   `bound = n_in·F16_MAX_FIXED·top·127·s_max < i128::MAX` guarantees
///   `Σ_j |S_b·xq_j| ≤ n_in·128·s_max < bound < i128::MAX`.
///   Thus no checked add overflows `i128` (ruling out `TotalOverflow`).
///
/// `ActMagnitude::Exact` is therefore strictly implied by [`fast_bound_ok`] for all four families.
/// `debug_assertions` builds re-check the implication on every accepted column.
///
/// # Complexity
///
/// `O(k·blocks)` scalar `i128` compares in [`fast_bound_ok`], `O(1)` extra space.
/// The removed `magnitude()` pass was `O(k·n_in)` checked `i64`/`i128` operations on the calling thread.
fn fast_family(ggml_type: u32, n_in: usize, qs: &[QuantAct]) -> Option<Family> {
    let family = match ggml_type {
        GGML_TYPE_Q8_0 => Family::Q8_0,
        GGML_TYPE_Q5_0 => Family::Q5_0,
        GGML_TYPE_Q4_K => Family::Q4_K,
        GGML_TYPE_Q6_K => Family::Q6_K,
        _ => return None,
    };
    if !n_in.is_multiple_of(ACT_QK) || qs.is_empty() || qs.len() > MAX_POOLED_K {
        return None;
    }
    if matches!(family, Family::Q4_K | Family::Q6_K) && !n_in.is_multiple_of(QK_K) {
        return None;
    }
    let blocks = n_in / ACT_QK;
    let (top, narrow): (i128, bool) = match family {
        Family::Q8_0 => (128, true),
        Family::Q5_0 => (16, true),
        Family::Q4_K => (1008, false),
        Family::Q6_K => (4096, false),
    };
    let mut c = 0usize;
    let mut admitted = true;
    while c < qs.len() && admitted {
        let q = &qs[c];
        admitted =
            q.len() == n_in && q.scales().len() == blocks && fast_bound_ok(n_in, top, narrow, q);
        if admitted {
            debug_assert!(
                matches!(q.magnitude(), ActMagnitude::Exact(_)),
                "fast_bound_ok implies an exact magnitude"
            );
        }
        c += 1;
    }
    if admitted {
        Some(family)
    } else {
        None
    }
}

/// The chart's `q8act_fast_bound_ok(n_in, top, q, narrow)`, re-derived here.
///
/// `S_max = max_b |S_b|`; `S_max·127 ≤ i64::MAX` keeps every `S_b·xq_j` in `i64` (the exact
/// magnitude `matvec_q8act` requires), the narrow condition `S_max·(32·top·127) ≤ i64::MAX` keeps
/// each block's `S_b·I_b` in `i64`, and `n_in·(65504·2^32)·top·127·S_max < i128::MAX` keeps every
/// running `i128` partial sum in range. The strict `<` mirrors the chart's saturated comparison.
///
/// # Complexity
///
/// `O(n_in/32)` time, `O(1)` extra space.
fn fast_bound_ok(n_in: usize, top: i128, narrow: bool, q: &QuantAct) -> bool {
    let mut s_max: i128 = 0;
    let mut b = 0usize;
    while b < q.scales().len() {
        let a = i128::from(q.scales()[b]).abs();
        if a > s_max {
            s_max = a;
        }
        b += 1;
    }
    if s_max.saturating_mul(127) > i128::from(i64::MAX) {
        return false;
    }
    if narrow && s_max.saturating_mul(32 * top * 127) > i128::from(i64::MAX) {
        return false;
    }
    let bound = (n_in as i128)
        .saturating_mul(F16_MAX_FIXED)
        .saturating_mul(top)
        .saturating_mul(127)
        .saturating_mul(s_max);
    bound < i128::MAX
}

/// One weight row's `i128` accumulators for every live column: `Σ_b S_b·(d_b·Σ_l q_l·xq_{c,l})`.
///
/// The block loop decodes each weight block once ([`q8_0_block_kdots`]/[`q5_0_block_kdots`]) and
/// folds its contribution into every live column in the same `i64`/`i128` op order as
/// [`matvec_q8act`]'s fast row. A non-finite block scale returns [`RowOutcome::NonFinite`], and an
/// unreachable checked overflow returns [`RowOutcome::Bail`].
///
/// # Complexity
///
/// `O(blocks·k)` exact multiply-adds plus the per-block decode; `O(1)` extra space.
fn row_accs(
    family: Family,
    row: &[u8],
    geom: Geom,
    acts: &[QuantAct],
    k: usize,
    state: &[ColumnState],
    avx: bool,
) -> RowOutcome {
    let mut totals = [0i128; MAX_K];
    let mut early: Option<RowOutcome> = None;
    let mut blk = 0usize;
    while blk < geom.blocks && early.is_none() {
        let off = blk * geom.block_bytes;
        match f16_to_fixed(u16::from_le_bytes([row[off], row[off + 1]])) {
            Err(_) => early = Some(RowOutcome::NonFinite),
            Ok(d) => {
                let dots = match family {
                    Family::Q8_0 => q8_0_block_kdots(avx, row, off + 2, acts, blk, k),
                    Family::Q5_0 => q5_0_block_kdots(avx, row, off, acts, blk, k),
                    // The affine `Q4_K` fold lives in the pooled carrier only; a call that
                    // reaches the serial row loop takes the per-column reference instead.
                    Family::Q4_K | Family::Q6_K => return RowOutcome::Bail,
                };
                let mut c = 0usize;
                while c < k && early.is_none() {
                    if state[c] == ColumnState::Live {
                        let scale = acts[c].scales()[blk];
                        match scale.checked_mul(i64::from(dots[c])) {
                            None => early = Some(RowOutcome::Bail),
                            Some(scaled) => match i128::from(d).checked_mul(i128::from(scaled)) {
                                None => early = Some(RowOutcome::Bail),
                                Some(term) => match totals[c].checked_add(term) {
                                    Some(total) => totals[c] = total,
                                    None => early = Some(RowOutcome::Bail),
                                },
                            },
                        }
                    }
                    c += 1;
                }
            }
        }
        blk += 1;
    }
    match early {
        Some(outcome) => outcome,
        None => RowOutcome::Values(totals),
    }
}

/// The `k` exact 32-lane dots of one `Q8_0` weight block: AVX2 when the host has it, else the
/// portable lane loop.
///
/// `weight_off` is the block's first code byte; the caller guarantees 32 code bytes there and 32
/// activation lanes per column.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q8_0_block_kdots(
    avx: bool,
    row: &[u8],
    weight_off: usize,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present, `weight_off + 32 ≤ row.len()` by the caller's geometry and each
        // `acts[c].codes()` holds the block's `blk·32 + 32` lanes by [`fast_family`].
        return unsafe { q8_0_block_kdots_avx2(row.as_ptr().add(weight_off), acts, blk, k) };
    }
    let _ = avx;
    q8_0_block_kdots_portable(&row[weight_off..weight_off + Q8_0_QK], acts, blk, k)
}

/// The `k` exact 32-lane dots of one `Q5_0` weight block: AVX2 when the host has it, else the
/// portable lane loop.
///
/// `block_off` is the block's `d` byte; the caller guarantees [`Q5_0_BLOCK_BYTES`] bytes there and
/// 32 activation lanes per column.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q5_0_block_kdots(
    avx: bool,
    row: &[u8],
    block_off: usize,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present, `block_off + Q5_0_BLOCK_BYTES ≤ row.len()` by the caller's
        // geometry and each `acts[c].codes()` holds the block's `blk·32 + 32` lanes by
        // [`fast_family`].
        return unsafe { q5_0_block_kdots_avx2(row.as_ptr().add(block_off), acts, blk, k) };
    }
    let _ = avx;
    q5_0_block_kdots_portable(&row[block_off..block_off + Q5_0_BLOCK_BYTES], acts, blk, k)
}

/// Exact `Σ_{l<32} (bytes[l] as i8)·xq[l]` for a `Q8_0` code window: the portable twin of the
/// AVX2 core and the no-AVX2 path.
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
fn portable_u8_i8_dot(codes: &[u8], xq: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    let mut l = 0usize;
    while l < codes.len() {
        acc += i32::from(i8::from_ne_bytes([codes[l]])) * i32::from(xq[l]);
        l += 1;
    }
    acc
}

/// Exact `Σ_{l<32} c_l·xq[l]` for one `Q5_0` block from its `qh`/`qs` bytes: the portable twin of
/// the AVX2 core. The lane order is `quant::q5_0_codes`'s: `qs` low nibbles are lanes `0..16`,
/// high nibbles lanes `16..32`, and `qh` bit `l` is lane `l`'s fifth bit.
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
fn portable_q5_0_dot(qh: &[u8], qs: &[u8], xq: &[i8]) -> i32 {
    let qh_u32 = u32::from_le_bytes([qh[0], qh[1], qh[2], qh[3]]);
    let mut acc: i32 = 0;
    let mut j = 0usize;
    while j < 16 {
        let nib = qs[j];
        let lo_bit = ((qh_u32 >> j) & 1) as u8;
        let hi_bit = ((qh_u32 >> (16 + j)) & 1) as u8;
        let lo = i32::from((nib & 0x0F) | (lo_bit << 4)) - 16;
        let hi = i32::from((nib >> 4) | (hi_bit << 4)) - 16;
        acc += lo * i32::from(xq[j]);
        acc += hi * i32::from(xq[16 + j]);
        j += 1;
    }
    acc
}

/// The portable [`q8_0_block_kdots`]: one 32-byte window dotted against each column's 32 lanes.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q8_0_block_kdots_portable(
    window: &[u8],
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    let xbase = blk * ACT_QK;
    let mut out = [0i32; MAX_K];
    let mut c = 0usize;
    while c < k {
        out[c] = portable_u8_i8_dot(window, &acts[c].codes()[xbase..xbase + ACT_QK]);
        c += 1;
    }
    out
}

/// The portable [`q5_0_block_kdots`]: one block decoded and dotted against each column's 32 lanes.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q5_0_block_kdots_portable(
    block: &[u8],
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    let qh = &block[2..6];
    let qs = &block[6..6 + Q5_0_QK / 2];
    let xbase = blk * ACT_QK;
    let mut out = [0i32; MAX_K];
    let mut c = 0usize;
    while c < k {
        out[c] = portable_q5_0_dot(qh, qs, &acts[c].codes()[xbase..xbase + ACT_QK]);
        c += 1;
    }
    out
}

/// Whether the executing host exposes AVX2.
#[inline]
#[must_use]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Horizontal sum of the four `i32` lanes of `v`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum128(v: __m128i) -> i32 {
    // SAFETY: register arithmetic only; no memory and no lanes outside `v`.
    unsafe {
        let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
        let t = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        _mm_cvtsi128_si32(t)
    }
}

/// Horizontal sum of the eight `i32` lanes of `v`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256(v: __m256i) -> i32 {
    // SAFETY: register arithmetic only; the lane extraction reads no memory.
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum128(_mm_add_epi32(lo, hi))
    }
}

/// AVX2 [`q8_0_block_kdots`]: the weight vector is loaded and its magnitudes formed once, then
/// each column's activation vector is sign-folded into it with one `vpmaddubsw` pair sum.
///
/// # Safety
///
/// `weight` must point at 32 readable code bytes; each `acts[c].codes()` must hold at least
/// `blk·32 + 32` lanes; the caller must have AVX2.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_block_kdots_avx2(
    weight: *const u8,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    // SAFETY: the caller guarantees the 32-byte load and AVX2.
    unsafe {
        let w = _mm256_loadu_si256(weight.cast());
        let mag = _mm256_abs_epi8(w);
        let ones = _mm256_set1_epi16(1);
        let xbase = blk * ACT_QK;
        let mut out = [0i32; MAX_K];
        let mut c = 0usize;
        while c < k {
            let x = _mm256_loadu_si256(acts[c].codes().as_ptr().add(xbase).cast());
            let signed = _mm256_sign_epi8(x, w);
            let pair = _mm256_maddubs_epi16(mag, signed);
            out[c] = hsum256(_mm256_madd_epi16(pair, ones));
            c += 1;
        }
        out
    }
}

/// AVX2 [`q5_0_block_kdots`]: the fifth-bit decode (`nibble + 16·bit`, the lane bias removed with
/// `16·Σ xq`) forms the unsigned code vector once, then each column is one `vpmaddubsw` pair sum.
///
/// # Safety
///
/// `block` must point at [`Q5_0_BLOCK_BYTES`] readable bytes; each `acts[c].codes()` must hold at
/// least `blk·32 + 32` lanes; the caller must have AVX2.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q5_0_block_kdots_avx2(
    block: *const u8,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    // SAFETY: the caller guarantees the block's 22 readable bytes and AVX2.
    unsafe {
        let qh = block.add(2);
        let qs = block.add(6);
        let bytes = _mm_loadu_si128(qs.cast());
        let nib = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(bytes, nib);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), nib);
        let codes = _mm256_set_m128i(hi, lo); // lanes 0..16 low nibbles, 16..32 high nibbles

        // The fifth-bit plane: byte `l` of `hbit` is bit `l` of the little-endian `qh` u32.
        let lo16 = _mm256_set1_epi16(i16::from_le_bytes([*qh, *qh.add(1)]));
        let hi16 = _mm256_set1_epi16(i16::from_le_bytes([*qh.add(2), *qh.add(3)]));
        let pair = _mm256_set_m128i(_mm256_castsi256_si128(hi16), _mm256_castsi256_si128(lo16));
        let spread = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1,
        );
        let spread_bytes = _mm256_shuffle_epi8(pair, spread);
        let bitmask = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let masked = _mm256_and_si256(spread_bytes, bitmask);
        let hbit = _mm256_andnot_si256(
            _mm256_cmpeq_epi8(masked, _mm256_setzero_si256()),
            _mm256_set1_epi8(1),
        );
        let bit16 = _mm256_slli_epi16(hbit, 4); // 0 or 16 per byte, no carry across a byte
        let unsigned = _mm256_add_epi8(codes, bit16); // 0..31

        let ones = _mm256_set1_epi16(1);
        let xbase = blk * ACT_QK;
        let mut out = [0i32; MAX_K];
        let mut c = 0usize;
        while c < k {
            let x = _mm256_loadu_si256(acts[c].codes().as_ptr().add(xbase).cast());
            let prod = _mm256_maddubs_epi16(unsigned, x);
            let xsum = _mm256_maddubs_epi16(_mm256_set1_epi8(1), x);
            let comb = _mm256_sub_epi16(prod, _mm256_slli_epi16(xsum, 4));
            out[c] = hsum256(_mm256_madd_epi16(comb, ones));
            c += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        clippy::many_single_char_names,
        reason = "test-only fixture arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::chart::quant_act;
    use crate::engine::matvec_fixed;
    use crate::format::quant::{
        block_geom, GGML_TYPE_F32, GGML_TYPE_PQ2_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    };
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    /// xorshift64: a deterministic code/activation stream for the equality tests.
    fn next_rand(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// A `rows × n_in` payload of random bytes with every block's `f16` scale(s) overwritten by
    /// `d` in every row, so each block is well formed while its codes and packed scales are random.
    fn random_payload(ggml_type: u32, rows: usize, n_in: usize, d: u16, seed: u64) -> Vec<u8> {
        let (qk, bb) = block_geom(ggml_type).expect("supported geometry");
        let blocks = n_in.div_ceil(qk);
        let row_bytes = blocks * bb;
        let mut raw = vec![0u8; rows * row_bytes];
        let mut state = seed;
        let mut i = 0usize;
        while i < raw.len() {
            raw[i] = (next_rand(&mut state) & 0xff) as u8;
            i += 1;
        }
        let mut r = 0usize;
        while r < rows {
            let mut block = 0usize;
            while block < blocks {
                let o = r * row_bytes + block * bb;
                match ggml_type {
                    GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_PQ2_0 => {
                        raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    }
                    GGML_TYPE_Q4_K => {
                        raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        raw[o + 2..o + 4].copy_from_slice(&d.to_le_bytes());
                    }
                    GGML_TYPE_Q6_K => {
                        raw[o + bb - 2..o + bb].copy_from_slice(&d.to_le_bytes());
                    }
                    _ => {}
                }
                block += 1;
            }
            r += 1;
        }
        raw
    }

    /// `n` small Q32.32 activations in `[-2048, 2047]`, from a deterministic stream.
    fn random_x(n: usize, seed: u64) -> Vec<i64> {
        let mut state = seed;
        let mut x: Vec<i64> = Vec::with_capacity(n);
        while x.len() < n {
            x.push((next_rand(&mut state) % 4096) as i64 - 2048);
        }
        x
    }

    /// `k` distinct small activation quantizations of length `n_in`.
    fn acts(n_in: usize, k: usize, seed: u64) -> Vec<QuantAct> {
        let mut out: Vec<QuantAct> = Vec::with_capacity(k);
        let mut c = 0usize;
        while c < k {
            let s = seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(c as u64 + 1);
            out.push(quant_act(&random_x(n_in, s)).expect("small activations dequantize"));
            c += 1;
        }
        out
    }

    /// `matmat_q8act` on `qs`: `Ok` and column-equal to the per-column [`matvec_q8act`] when every
    /// column is `Ok`, else the lowest-indexed column's refusal.
    fn check_columns(ggml_type: u32, raw: &[u8], n_out: usize, n_in: usize, qs: &[QuantAct]) {
        let expected: Vec<Result<Vec<i64>, MatvecError>> = qs
            .iter()
            .map(|q| matvec_q8act(ggml_type, raw, n_out, n_in, q))
            .collect();
        match matmat_q8act(ggml_type, raw, n_out, n_in, qs) {
            Ok(cols) => {
                assert_eq!(cols.len(), qs.len(), "column count");
                for (c, e) in expected.iter().enumerate() {
                    assert_eq!(
                        &cols[c],
                        e.as_ref().expect("every column must be Ok"),
                        "type {ggml_type} shape ({n_out},{n_in}) column {c} diverged"
                    );
                }
            }
            Err(e) => {
                let first = expected.iter().find_map(|r| r.as_ref().err());
                assert_eq!(
                    Some(&e),
                    first,
                    "type {ggml_type} shape ({n_out},{n_in}) refusal must be the first column's"
                );
            }
        }
    }

    /// `Q8_0`/`Q5_0` (shared-decode kernel), `Q4_K`/`Q6_K` (packed lane on a whole 256-weight
    /// width, per-column fallback otherwise) and `PQ2_0` (per-column fallback) over several shapes,
    /// including ragged widths, and `k ∈ {1, 2, 3, 8, 16}`: every column equals the per-column
    /// [`matvec_q8act`] bit for bit.
    #[test]
    fn columns_match_per_column_matvec_q8act() {
        const TYPES: [u32; 5] = [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_PQ2_0,
        ];
        const SHAPES: [(usize, usize); 6] =
            [(4, 64), (3, 96), (2, 33), (2, 300), (1, 128), (2, 256)];
        const KS: [usize; 5] = [1, 2, 3, 8, 16];
        let mut seed = 0x1234_5678_9abc_def0u64;
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            for &(rows, n_in) in &SHAPES {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw = random_payload(ggml_type, rows, n_in, 0x3800, seed);
                for &k in &KS {
                    let qs = acts(n_in, k, seed ^ ((k as u64) << 17) ^ ti as u64);
                    check_columns(ggml_type, &raw, rows, n_in, &qs);
                }
            }
        }
    }

    /// `k` past the portable kernel's `MAX_K` still equals the per-column contract: for both
    /// carriers the pooled `AVX2` kernel takes the call (asserted on this host) and is
    /// bit-identical to the reference.
    #[test]
    fn columns_past_max_k_match_per_column() {
        const SHAPES: [(usize, usize); 3] = [(4, 64), (3, 96), (2, 128)];
        const KS: [usize; 6] = [17, 24, 33, 42, 64, 96];
        let seed = 0x0f1e_2d3c_4b5a_6978u64;
        for (si, &(rows, n_in)) in SHAPES.iter().enumerate() {
            for &(ggml_type, block_bytes) in &[
                (GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES),
                (GGML_TYPE_Q5_0, Q5_0_BLOCK_BYTES),
            ] {
                let raw = random_payload(ggml_type, rows, n_in, 0x3800, seed ^ si as u64);
                for &k in &KS {
                    let qs = acts(n_in, k, seed ^ ((k as u64) << 17));
                    check_columns(ggml_type, &raw, rows, n_in, &qs);
                    #[cfg(target_arch = "x86_64")]
                    if avx2_available() {
                        let geom = Geom {
                            blocks: n_in / ACT_QK,
                            block_bytes,
                            row_bytes: (n_in / ACT_QK) * block_bytes,
                        };
                        let family = match ggml_type {
                            GGML_TYPE_Q5_0 => Family::Q5_0,
                            _ => Family::Q8_0,
                        };
                        assert!(
                            pooled(family, &raw, rows, geom, &qs).is_some(),
                            "type {ggml_type} k = {k} must take the pooled kernel"
                        );
                    }
                }
            }
        }
    }

    /// Extremes: maximal finite weight scales against `±127` codes at unit scale (the kernel's
    /// fast path), and activations that exhaust `i64` (the gate falls back to the per-column
    /// reference). Equality holds in both.
    #[test]
    fn columns_match_on_extreme_codes_and_scales() {
        const TYPES: [u32; 2] = [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0];
        let n_in = 256usize;
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            let raw = random_payload(ggml_type, 2, n_in, 0x7BFF, 0x51ff_0000 + ti as u64);
            let x: Vec<i64> = (0..n_in)
                .map(|j| if j % 2 == 0 { 127 } else { -127 })
                .collect();
            let extreme = quant_act(&x).expect("±127 dequantizes");
            assert!(
                extreme.scales().iter().all(|&s| s == 1),
                "±127 must quantize at unit scale"
            );
            check_columns(ggml_type, &raw, 2, n_in, &[extreme]);

            let huge = quant_act(&vec![127i64 << 56; n_in])
                .expect("127·2^56 dequantizes on the power-of-two grid");
            check_columns(ggml_type, &raw, 2, n_in, &[huge]);
        }
    }

    /// The fast path's columns also equal `matvec_fixed` on the dequantized activation, the
    /// definition `matvec_q8act` itself refines.
    #[test]
    fn columns_match_matvec_fixed_on_dequantized_activations() {
        const TYPES: [u32; 2] = [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0];
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            let (rows, n_in) = (3usize, 96usize);
            let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0xbeef_0000 + ti as u64);
            let qs = acts(n_in, 4, 0x0dd0 + ti as u64);
            let cols = matmat_q8act(ggml_type, &raw, rows, n_in, &qs).expect("fast payload");
            for (c, q) in qs.iter().enumerate() {
                let xhat = q.dequant().expect("small activations dequantize");
                assert_eq!(
                    cols[c],
                    matvec_fixed(ggml_type, &raw, rows, n_in, &xhat).expect("reference"),
                    "type {ggml_type} column {c} vs matvec_fixed"
                );
            }
        }
    }

    /// Every refusal class agrees with the per-column reference, including the refusal value.
    #[test]
    fn refusals_match_per_column() {
        let q8 = GGML_TYPE_Q8_0;
        let raw = random_payload(q8, 2, 32, 0x3800, 1);
        let qs = acts(32, 1, 7);

        // Shape mismatch: the dequantized activation length differs from `n_in`.
        check_columns(q8, &raw, 2, 31, &qs);

        // Short payload.
        check_columns(q8, &raw[..raw.len() - 1], 2, 32, &qs);

        // Unsupported family.
        check_columns(GGML_TYPE_F32, &[], 1, 1, &qs);

        // Row-stride / payload length overflow.
        check_columns(q8, &raw, usize::MAX, 32, &qs);

        // Non-finite weight scale.
        let mut inf = random_payload(q8, 1, 32, 0x3800, 2);
        inf[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        check_columns(q8, &inf, 1, 32, &qs);

        // Output overflow: weight 1.0 against `127·2^56` rounds past `i64`.
        let mut one = random_payload(q8, 1, 32, 0x3C00, 3);
        let mut l = 0usize;
        while l < 32 {
            one[2 + l] = 1;
            l += 1;
        }
        let huge = quant_act(&vec![127i64 << 56; 32])
            .expect("127·2^56 dequantizes on the power-of-two grid");
        check_columns(q8, &one, 1, 32, &[huge]);
    }

    /// `matmat_q8act` on `qs` equals `k` per-column [`matvec_q8act`] calls, or the lowest-indexed
    /// column's refusal; a divergence names the type, the width, the column count and the column.
    fn check_kquant_columns(
        ggml_type: u32,
        raw: &[u8],
        n_out: usize,
        n_in: usize,
        k: usize,
        qs: &[QuantAct],
    ) {
        assert_eq!(qs.len(), k, "fixture column count");
        let expected: Vec<Result<Vec<i64>, MatvecError>> = qs
            .iter()
            .map(|q| matvec_q8act(ggml_type, raw, n_out, n_in, q))
            .collect();
        match matmat_q8act(ggml_type, raw, n_out, n_in, qs) {
            Ok(cols) => {
                assert_eq!(
                    cols.len(),
                    k,
                    "type {ggml_type} n_in {n_in} k {k}: column count"
                );
                for (c, e) in expected.iter().enumerate() {
                    assert_eq!(
                        &cols[c],
                        e.as_ref().expect("every column must be Ok"),
                        "type {ggml_type} n_in {n_in} k {k} column {c} diverged"
                    );
                }
            }
            Err(e) => {
                let first = expected.iter().find_map(|r| r.as_ref().err());
                assert_eq!(
                    Some(&e),
                    first,
                    "type {ggml_type} n_in {n_in} k {k} refusal must be the first column's"
                );
            }
        }
    }

    /// `Q4_K`/`Q6_K` on the pooled multi-column kernel: `n_out = 7` rows, `n_in ∈ {256, 1024}` and
    /// `k ∈ {1, 3, 33, 64}` (crossing the 32-column group), random codes and packed scales from a
    /// fixed seed. `matmat_q8act` equals `k` per-column `matvec_q8act` calls word for word, including
    /// the refusal class (a payload one byte short), and a 133-row payload crosses the pool's row
    /// split.
    #[test]
    fn kquant_columns_match_per_column_matvec_q8act() {
        const KS: [usize; 4] = [1, 3, 33, 64];
        let n_out = 7usize;
        let mut seed = 0x0ddc_0ffe_e123_4567u64;
        for &ggml_type in &[GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
            for &n_in in &[256usize, 1024] {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw = random_payload(ggml_type, n_out, n_in, 0x3800, seed);
                for &k in &KS {
                    let qs = acts(n_in, k, seed ^ ((k as u64) << 19));
                    check_kquant_columns(ggml_type, &raw, n_out, n_in, k, &qs);
                }
                // The refusal class travels: one byte short of the payload every column refuses on.
                let qs = acts(n_in, 3, seed ^ 0xfeed);
                check_kquant_columns(ggml_type, &raw[..raw.len() - 1], n_out, n_in, 3, &qs);
            }
            // 133 rows is two pool pieces at `row_chunk_budget`'s minimum chunk of 64 rows.
            let (n_in, k) = (256usize, 3usize);
            let raw = random_payload(ggml_type, 133, n_in, 0x3800, seed ^ 0x99);
            let qs = acts(n_in, k, seed ^ 0xaa);
            check_kquant_columns(ggml_type, &raw, 133, n_in, k, &qs);
        }
    }

    /// The AVX2 block cores return the same exact `i32` dots as their portable twins on random
    /// weight bytes and random activations, for `k ∈ {1, 2, 3, 8, 16}`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_block_kdots_match_the_portable_twin() {
        const KS: [usize; 5] = [1, 2, 3, 8, 16];
        if !avx2_available() {
            return;
        }
        let n_in = 64usize;
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        for &ggml_type in &[GGML_TYPE_Q8_0, GGML_TYPE_Q5_0] {
            let (qk, bb) = block_geom(ggml_type).expect("supported geometry");
            let blocks = n_in / qk;
            let raw = random_payload(ggml_type, 1, n_in, 0x3800, seed);
            for &k in &KS {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let qs = acts(n_in, k, seed);
                let mut blk = 0usize;
                while blk < blocks {
                    let off = blk * bb;
                    if ggml_type == GGML_TYPE_Q8_0 {
                        let port = q8_0_block_kdots_portable(&raw[off + 2..off + bb], &qs, blk, k);
                        // SAFETY: AVX2 is present, `raw[off + 2 .. off + 34]` holds 32 code bytes
                        // and each column's codes hold `blk·32 + 32` lanes.
                        let avx = unsafe {
                            q8_0_block_kdots_avx2(raw.as_ptr().add(off + 2), &qs, blk, k)
                        };
                        assert_eq!(&port[..k], &avx[..k], "Q8_0 block {blk} k {k} diverged");
                    } else {
                        let port = q5_0_block_kdots_portable(&raw[off..off + bb], &qs, blk, k);
                        // SAFETY: AVX2 is present, `raw[off .. off + 22]` holds the block and each
                        // column's codes hold `blk·32 + 32` lanes.
                        let avx =
                            unsafe { q5_0_block_kdots_avx2(raw.as_ptr().add(off), &qs, blk, k) };
                        assert_eq!(&port[..k], &avx[..k], "Q5_0 block {blk} k {k} diverged");
                    }
                    blk += 1;
                }
            }
        }
    }

    /// `fast_family`'s per-column predicate equals the former `len ∧ scales ∧ Exact ∧ bound`
    /// conjunction for both `Q8_0` (top = 128) and `Q5_0` (top = 16):
    /// - on random columns across scale shifts,
    /// - at the adversarial bound edge (`s_max = i64::MAX / (32·top·127)` and `s_max + 1`),
    /// - with zero scales,
    /// - with saturated codes (`±127`, `-128`),
    /// - on length and block-count mismatches.
    #[test]
    fn fast_family_predicate_equals_the_magnitude_form() {
        let n_in = 256usize;
        let blocks = n_in / ACT_QK;
        let cases = [(GGML_TYPE_Q8_0, 128i128), (GGML_TYPE_Q5_0, 16i128)];
        for &(ggml_type, top) in &cases {
            let mut seed = 0x5eed_0f00_dbad_cafe_u64;
            let (mut yes, mut no) = (0usize, 0usize);
            // 1. Random columns across scale shifts.
            for shift in (20..=62).step_by(2) {
                for trial in 0..24 {
                    let x: Vec<i64> = (0..n_in)
                        .map(|j| {
                            let exp = (shift + i64::from(trial % 2 == 1 && j % 32 == 5)).min(62);
                            let mag = 1i64 << exp;
                            if j == 0 {
                                -mag
                            } else {
                                (next_rand(&mut seed).cast_signed() % (2 * (mag / 2) + 1)) - mag / 2
                            }
                        })
                        .collect();
                    let Some(q) = quant_act(&x) else { continue };
                    let old = q.len() == n_in
                        && q.scales().len() == blocks
                        && matches!(q.magnitude(), ActMagnitude::Exact(_))
                        && crate::engine::chart::q8act_fast_bound_ok(n_in, top, &q, true);
                    let new = fast_family(ggml_type, n_in, std::slice::from_ref(&q)).is_some();
                    assert_eq!(old, new, "type {ggml_type} shift {shift} trial {trial}");
                    yes += usize::from(new);
                    no += usize::from(!new);
                }
            }
            assert!(
                yes > 0 && no > 0,
                "type {ggml_type}: both outcomes must occur: {yes} / {no}"
            );

            // 2. Adversarial scales at the bound edge: s_max and s_max + 1.
            let s_max = (i64::MAX as i128 / (32 * top * 127)) as i64;
            for &s in &[s_max, s_max + 1, -s_max, -(s_max + 1)] {
                let q = QuantAct::from_parts(vec![s; blocks], vec![127i8; n_in]);
                let old = q.len() == n_in
                    && q.scales().len() == blocks
                    && matches!(q.magnitude(), ActMagnitude::Exact(_))
                    && crate::engine::chart::q8act_fast_bound_ok(n_in, top, &q, true);
                let new = fast_family(ggml_type, n_in, std::slice::from_ref(&q)).is_some();
                assert_eq!(old, new, "type {ggml_type} edge scale {s}");
                if s.abs() == s_max {
                    assert!(new, "type {ggml_type} scale {s} must be accepted");
                } else {
                    assert!(!new, "type {ggml_type} scale {s} must be refused");
                }
            }

            // 3. Zero scales and saturated codes.
            let zero_q = QuantAct::from_parts(vec![0i64; blocks], vec![0i8; n_in]);
            let old_zero = zero_q.len() == n_in
                && zero_q.scales().len() == blocks
                && matches!(zero_q.magnitude(), ActMagnitude::Exact(_))
                && crate::engine::chart::q8act_fast_bound_ok(n_in, top, &zero_q, true);
            let new_zero = fast_family(ggml_type, n_in, std::slice::from_ref(&zero_q)).is_some();
            assert_eq!(old_zero, new_zero, "type {ggml_type} zero scale");
            assert!(new_zero, "type {ggml_type} zero scale must be accepted");

            for &sat_code in &[127i8, -127i8, -128i8] {
                let sat_q = QuantAct::from_parts(vec![s_max; blocks], vec![sat_code; n_in]);
                let old_sat = sat_q.len() == n_in
                    && sat_q.scales().len() == blocks
                    && matches!(sat_q.magnitude(), ActMagnitude::Exact(_))
                    && crate::engine::chart::q8act_fast_bound_ok(n_in, top, &sat_q, true);
                let new_sat = fast_family(ggml_type, n_in, std::slice::from_ref(&sat_q)).is_some();
                assert_eq!(
                    old_sat, new_sat,
                    "type {ggml_type} saturated code {sat_code} at s_max"
                );
                assert!(
                    new_sat,
                    "type {ggml_type} saturated code {sat_code} at s_max must be accepted"
                );

                let sat_q_over =
                    QuantAct::from_parts(vec![s_max + 1; blocks], vec![sat_code; n_in]);
                let old_over = sat_q_over.len() == n_in
                    && sat_q_over.scales().len() == blocks
                    && matches!(sat_q_over.magnitude(), ActMagnitude::Exact(_))
                    && crate::engine::chart::q8act_fast_bound_ok(n_in, top, &sat_q_over, true);
                let new_over =
                    fast_family(ggml_type, n_in, std::slice::from_ref(&sat_q_over)).is_some();
                assert_eq!(
                    old_over, new_over,
                    "type {ggml_type} saturated code {sat_code} at s_max + 1"
                );
                assert!(
                    !new_over,
                    "type {ggml_type} saturated code {sat_code} at s_max + 1 must be refused"
                );
            }

            // 4. Length and block-count mismatches.
            let bad_len = QuantAct::from_parts(vec![1i64; blocks], vec![1i8; n_in + ACT_QK]);
            assert!(fast_family(ggml_type, n_in, std::slice::from_ref(&bad_len)).is_none());
            let bad_blocks = QuantAct::from_parts(vec![1i64; blocks + 1], vec![1i8; n_in]);
            assert!(fast_family(ggml_type, n_in, std::slice::from_ref(&bad_blocks)).is_none());
        }
    }

    /// Timing probe (L1 deliverable 4): `time(k)/time(1)` for the two fast families at the
    /// 4864×896 projection shape. Run with
    /// `cargo test --release --all-features --locked kern -- --ignored --nocapture`.
    ///
    /// The epic's falsifier is `time(8) > 2·time(1)`. `KS` are sampled round-robin so that a
    /// transient load spike lands on every `k` alike, and each `k`'s estimator is its minimum
    /// over the rounds (the least-contended window). This is still a shared-host wall clock, not
    /// an idle-host measurement: the ratio is indicative, and a `measured` claim needs a quiesced
    /// host.
    /// k-column cost when the weights stream from DRAM (the decode regime): 256 MB of `Q8_0`
    /// matrices of the Qwen3.5 FFN shape, cycled so no call finds its matrix in cache.
    /// `cargo test --release --all-features --locked kern::matmat -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    #[allow(clippy::cast_precision_loss, reason = "byte counts fit f64 exactly")]
    fn timing_k_scaling_dram() {
        let (n_out, n_in) = (3584usize, 1024usize);
        let row_bytes = n_in / 32 * Q8_0_BLOCK_BYTES;
        let copies = 256_000_000 / (n_out * row_bytes);
        let mats: Vec<Vec<u8>> = (0..copies)
            .map(|c| random_payload(GGML_TYPE_Q8_0, n_out, n_in, 0x2000, 0x5eed + c as u64))
            .collect();
        let pool = acts(n_in, 8, 0xabcd);
        let mut base = 0.0f64;
        for k in [1usize, 2, 3, 4, 6, 8] {
            let _ = matmat_q8act(GGML_TYPE_Q8_0, &mats[0], n_out, n_in, &pool[..k]).expect("warm");
            let reps = 2 * copies;
            let start = Instant::now();
            let mut i = 0usize;
            while i < reps {
                let out = matmat_q8act(GGML_TYPE_Q8_0, &mats[i % copies], n_out, n_in, &pool[..k])
                    .expect("timing");
                black_box(&out);
                i += 1;
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if k == 1 {
                base = us;
            }
            let gbs = (n_out * row_bytes) as f64 / us / 1e3;
            println!(
                "dram k={k}: {us:.1} µs/call, {:.2}× k=1, weights {gbs:.1} GB/s",
                us / base
            );
        }
    }

    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    fn timing_k_scaling() {
        const KS: [usize; 5] = [1, 2, 4, 8, 16];
        const ROUNDS: usize = 15;
        let n_out = 4864usize;
        let n_in = 896usize;
        println!("matmat_q8act timing: n_out={n_out} n_in={n_in} rounds={ROUNDS} (min per k)");
        println!("{:<6} {:>4} {:>14} {:>8}", "type", "k", "best_ms", "ratio");
        for &ggml_type in &[GGML_TYPE_Q8_0, GGML_TYPE_Q5_0] {
            let raw = random_payload(ggml_type, n_out, n_in, 0x3C00, 0x5eed);
            let pool = acts(n_in, 16, 0xabcd);
            let _ = matmat_q8act(ggml_type, &raw, n_out, n_in, &pool[..1]).expect("warmup");
            let mut best = [Duration::MAX; KS.len()];
            let mut round = 0usize;
            while round < ROUNDS {
                let mut i = 0usize;
                while i < KS.len() {
                    let start = Instant::now();
                    let out =
                        matmat_q8act(ggml_type, &raw, n_out, n_in, &pool[..KS[i]]).expect("timing");
                    let elapsed = start.elapsed();
                    black_box(&out);
                    if elapsed < best[i] {
                        best[i] = elapsed;
                    }
                    i += 1;
                }
                round += 1;
            }
            let base = best[0].as_secs_f64() * 1e3;
            let mut i = 0usize;
            while i < KS.len() {
                let ms = best[i].as_secs_f64() * 1e3;
                println!("{ggml_type:<6} {:>4} {ms:>14.3} {:>8.2}", KS[i], ms / base);
                i += 1;
            }
        }
    }

    /// The fit's `X`-update projection shape: `n_out = 1024`, `n_in = 3072`, `k = r ∈ {864, 432,
    /// 96, 32}` columns. `k = 864` is the reference rank at 1.18 b/w; `k ≤ 16` is the portable
    /// kernel's, so `32` is here as the control just above the `MAX_K` branch (`32 > MAX_K` takes
    /// the pooled route; `16` would not).
    ///
    /// Prints the wall and the rate in multiply-adds per second, with the two streams the route
    /// touches: the weight payload (`n_out·n_in` bytes, read once per call by the shared-decode
    /// kernel and once per column by the reference) and the activation codes (`k·n_in` bytes per
    /// row, read once per row and column by both). The last column is the payload traffic the
    /// old per-column route paid, so the routing row and the traffic model sit in one table.
    #[test]
    #[ignore = "measurement: release run, three timed passes per arm"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the counters are benchmark shapes below 2^53 and RUNS is 3"
    )]
    fn pooled_rate_at_fit_shape() {
        const KS: [usize; 4] = [864, 432, 96, 32];
        const RUNS: usize = 3;
        let (n_out, n_in) = (1024usize, 3072usize);
        let raw = random_payload(GGML_TYPE_Q8_0, n_out, n_in, 0x3800, 0x9e37_79b9);
        let qs = acts(n_in, KS[0], 0x5eed_0f00);
        println!(
            "pooled_rate n_out {n_out} n_in {n_in} runs {RUNS} avx2 {} avxvnni {}",
            avx2_available(),
            std::is_x86_feature_detected!("avxvnni")
        );
        println!(
            "{:<5} {:>10} {:>10} {:>10} {:>10} {:>12}",
            "k", "best_ms", "mean_ms", "G_MAC_s", "act_MB", "per_col_MB"
        );
        for &k in &KS {
            let _ = matmat_q8act(GGML_TYPE_Q8_0, &raw, n_out, n_in, &qs[..k]).expect("warmup");
            let mut wall = Vec::new();
            let mut i = 0usize;
            while i < RUNS {
                let start = Instant::now();
                let out =
                    matmat_q8act(GGML_TYPE_Q8_0, &raw, n_out, n_in, &qs[..k]).expect("timing");
                let dt = start.elapsed().as_secs_f64();
                black_box(&out);
                wall.push(dt);
                i += 1;
            }
            let mean = wall.iter().sum::<f64>() / RUNS as f64;
            let best = wall.iter().copied().fold(f64::MAX, f64::min);
            let macs = (n_out * n_in * k) as f64;
            let acts = (n_out * k * n_in) as f64;
            let per_col = (k * n_out * (n_in / ACT_QK) * Q8_0_BLOCK_BYTES) as f64;
            println!(
                "{k:<5} {:>10.3} {:>10.3} {:>10.3} {:>10.1} {:>12.1}",
                best * 1e3,
                mean * 1e3,
                macs / mean / 1e9,
                acts / (1024.0 * 1024.0),
                per_col / (1024.0 * 1024.0)
            );
        }
    }

    /// The shared `Q8_0` pooled loop against the packed VNNI entry on one payload and one set of
    /// columns: the equality the fold must keep and the throughput it must not lose.
    ///
    /// `pooled` is the row-kernel route the one-path fold targets; `gemm_q8_0` is the packed entry
    /// `matmat_q8act` takes today. Prints the best and mean wall of each arm and their ratio at
    /// `k ∈ {32, 128, 512}` on the 3584-wide hidden size, plus the payload traffic the best wall
    /// implies.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "measurement: release run, three interleaved passes per arm"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the counters are benchmark shapes below 2^53 and RUNS is 3"
    )]
    fn pooled_against_gemm_q8_0() {
        const KS: [usize; 3] = [32, 128, 512];
        const RUNS: usize = 3;
        let (n_out, n_in) = (3584usize, 3584usize);
        let raw = random_payload(GGML_TYPE_Q8_0, n_out, n_in, 0x3800, 0x5eed_1a2b);
        let qs = acts(n_in, KS[2], 0x0bad_f00d);
        let geom = Geom {
            blocks: n_in / ACT_QK,
            block_bytes: Q8_0_BLOCK_BYTES,
            row_bytes: (n_in / ACT_QK) * Q8_0_BLOCK_BYTES,
        };
        println!(
            "pooled_vs_gemm n_out {n_out} n_in {n_in} runs {RUNS} avxvnni {}",
            std::is_x86_feature_detected!("avxvnni")
        );
        if !std::is_x86_feature_detected!("avxvnni") {
            println!(
                "skip: gemm_q8_0 admits only AVX-VNNI; the shared loop is the only route here"
            );
            return;
        }
        println!(
            "{:>5} {:>12} {:>12} {:>8} {:>12} {:>12}",
            "k", "pooled_ms", "gemm_ms", "p/g", "pooled_GBs", "gemm_GBs"
        );
        for &k in &KS {
            let pool = pooled(Family::Q8_0, &raw, n_out, geom, &qs[..k]).expect("pooled admits");
            let Some(gq8) =
                crate::engine::kern::gemm::gemm_q8_0(GGML_TYPE_Q8_0, &raw, n_out, n_in, &qs[..k])
            else {
                println!("{k:>5} gemm refuses this payload; skipped");
                continue;
            };
            assert_eq!(pool, gq8, "k = {k}: the two routes differ");
            let mut p = Vec::with_capacity(RUNS);
            let mut g = Vec::with_capacity(RUNS);
            let mut round = 0usize;
            while round < RUNS {
                let start = Instant::now();
                let out = pooled(Family::Q8_0, &raw, n_out, geom, &qs[..k]).expect("pooled");
                p.push(start.elapsed().as_secs_f64());
                black_box(&out);
                let start = Instant::now();
                let out = crate::engine::kern::gemm::gemm_q8_0(
                    GGML_TYPE_Q8_0,
                    &raw,
                    n_out,
                    n_in,
                    &qs[..k],
                )
                .expect("gemm");
                g.push(start.elapsed().as_secs_f64());
                black_box(&out);
                round += 1;
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let best = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
            let payload = (n_out * (n_in / ACT_QK) * Q8_0_BLOCK_BYTES) as f64;
            println!(
                "{k:>5} {:>12.3} {:>12.3} {:>8.2} {:>12.2} {:>12.2}",
                best(&p) * 1e3,
                best(&g) * 1e3,
                best(&p) / best(&g),
                payload / best(&p) / 1e9,
                payload / best(&g) / 1e9
            );
            println!(
                "      mean {:.3} / {:.3} ms",
                mean(&p) * 1e3,
                mean(&g) * 1e3
            );
        }
    }

    /// The shared `Q4_K` pooled carrier against the per-column contract on small shapes: every
    /// column equal, including `k > MAX_K` and several 256-weight super-blocks.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pooled_q4_k_matches_per_column() {
        const SHAPES: [(usize, usize); 2] = [(3, 512), (2, 1024)];
        const KS: [usize; 4] = [2, 8, 33, 64];
        if !avx2_available() {
            return;
        }
        let seed = 0x51ce_9a77_2d10_4e01u64;
        for (si, &(rows, n_in)) in SHAPES.iter().enumerate() {
            let raw = random_payload(GGML_TYPE_Q4_K, rows, n_in, 0x3800, seed ^ si as u64);
            let blocks = n_in / QK_K;
            let geom = Geom {
                blocks,
                block_bytes: Q4_K_BLOCK_BYTES,
                row_bytes: blocks * Q4_K_BLOCK_BYTES,
            };
            for &k in &KS {
                let qs = acts(n_in, k, seed ^ ((k as u64) << 21));
                let got = pooled(Family::Q4_K, &raw, rows, geom, &qs).expect("Q4_K admits");
                for (c, q) in qs.iter().enumerate() {
                    let want = matvec_q8act(GGML_TYPE_Q4_K, &raw, rows, n_in, q)
                        .expect("the contract admits");
                    assert_eq!(
                        got[c], want,
                        "shape ({rows},{n_in}) column {c} k {k} diverged"
                    );
                }
            }
        }
    }

    /// `Q6_K` on the shared path: `matmat_q8act` and the `pooled` carrier each equal
    /// [`matvec_q8act`] column for column at `n_out ∈ {3, 2}`, `n_in ∈ {512, 1024}` and
    /// `k ∈ {2, 8, 33, 64}`, on random codes and scales. `Q6_K` has no offset term, so the
    /// activation needs no `xsum`; the lane's shift grid is the same shape as `Q4_K`'s.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pooled_q6_k_matches_per_column() {
        const SHAPES: [(usize, usize); 2] = [(3, 512), (2, 1024)];
        const KS: [usize; 4] = [2, 8, 33, 64];
        if !avx2_available() {
            return;
        }
        let seed = 0x6c6b_51ce_9a77_2d10u64;
        for (si, &(rows, n_in)) in SHAPES.iter().enumerate() {
            let raw = random_payload(GGML_TYPE_Q6_K, rows, n_in, 0x3800, seed ^ si as u64);
            let blocks = n_in / QK_K;
            let geom = Geom {
                blocks,
                block_bytes: Q6_K_BLOCK_BYTES,
                row_bytes: blocks * Q6_K_BLOCK_BYTES,
            };
            for &k in &KS {
                let qs = acts(n_in, k, seed ^ ((k as u64) << 21));
                let got = pooled(Family::Q6_K, &raw, rows, geom, &qs).expect("Q6_K admits");
                let entry =
                    matmat_q8act(GGML_TYPE_Q6_K, &raw, rows, n_in, &qs).expect("the entry admits");
                assert_eq!(
                    got, entry,
                    "shape ({rows},{n_in}) k {k}: pooled and entry diverge"
                );
                for (c, q) in qs.iter().enumerate() {
                    let want = matvec_q8act(GGML_TYPE_Q6_K, &raw, rows, n_in, q)
                        .expect("the contract admits");
                    assert_eq!(
                        got[c], want,
                        "shape ({rows},{n_in}) column {c} k {k} diverged"
                    );
                }
            }
        }
    }

    /// The `Q4_K` carrier's rate on the production shapes: the 3584-wide hidden projection at
    /// `k ∈ {32, 128, 512}` and the 7B FFN's `18944 × 256` up-projection at `k = 256`, which the
    /// direct-write band path now admits. Prints the wall and the payload rate.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "measurement: release run, three passes per shape"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the counters are benchmark shapes below 2^53 and RUNS is 3"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one measured shape loop per production geometry; the printed table is the artifact"
    )]
    fn q4_k_carrier_rate_at_production_shapes() {
        const RUNS: usize = 3;
        let (n_out, n_in) = (3584usize, 3584usize);
        let raw = random_payload(GGML_TYPE_Q4_K, n_out, n_in, 0x3800, 0x5eed_1a2b);
        let qs = acts(n_in, 512, 0x0bad_f00d);
        println!(
            "q4_k_carrier_rate n_out {n_out} n_in {n_in} runs {RUNS} avx2 {}",
            avx2_available()
        );
        if !avx2_available() {
            println!("skip: the Q4_K carrier needs AVX2");
            return;
        }
        println!(
            "{:>7} {:>7} {:>12} {:>12} {:>8} {:>12}",
            "n_out", "k", "pooled_ms", "matmat_ms", "m/p", "GB/s"
        );
        for &k in &[32usize, 128, 512] {
            let geom = Geom {
                blocks: n_in / QK_K,
                block_bytes: Q4_K_BLOCK_BYTES,
                row_bytes: (n_in / QK_K) * Q4_K_BLOCK_BYTES,
            };
            let mut p = Vec::with_capacity(RUNS);
            let mut m = Vec::with_capacity(RUNS);
            let mut round = 0usize;
            while round < RUNS {
                // Alternate the order: the second arm of a round runs on a hotter package, and at
                // these walls that is a larger effect than the entry's own overhead.
                let mat_first = round % 2 == 1;
                if !mat_first {
                    let start = Instant::now();
                    let out = pooled(Family::Q4_K, &raw, n_out, geom, &qs[..k]).expect("pooled");
                    p.push(start.elapsed().as_secs_f64());
                    black_box(&out);
                }
                let start = Instant::now();
                let out = matmat_q8act(GGML_TYPE_Q4_K, &raw, n_out, n_in, &qs[..k])
                    .expect("the carrier admits");
                m.push(start.elapsed().as_secs_f64());
                black_box(&out);
                if mat_first {
                    let start = Instant::now();
                    let out = pooled(Family::Q4_K, &raw, n_out, geom, &qs[..k]).expect("pooled");
                    p.push(start.elapsed().as_secs_f64());
                    black_box(&out);
                }
                round += 1;
            }
            let best = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
            let payload = (n_out * (n_in / QK_K) * Q4_K_BLOCK_BYTES) as f64;
            println!(
                "{n_out:>7} {k:>7} {:>12.3} {:>12.3} {:>8.2} {:>12.2}",
                best(&p) * 1e3,
                best(&m) * 1e3,
                best(&m) / best(&p),
                payload / best(&m) / 1e9
            );
        }
        // Decode is the same payload with one column: the per-column row kernel the engine's
        // `k = 1` route takes. The 7B at 20 tok/s needs 141G weights/s; this is the rate to beat.
        let mut wall = Vec::with_capacity(RUNS);
        let mut round = 0usize;
        while round < RUNS {
            let start = Instant::now();
            let out = matvec_q8act(GGML_TYPE_Q4_K, &raw, n_out, n_in, &qs[0])
                .expect("the decode contract admits");
            wall.push(start.elapsed().as_secs_f64());
            black_box(&out);
            round += 1;
        }
        let best = wall.iter().copied().fold(f64::MAX, f64::min);
        let weights = (n_out * n_in) as f64;
        println!(
            "decode k=1 n_out {n_out} n_in {n_in}: best {:.3} ms, {:.1} G weights/s",
            best * 1e3,
            weights / best / 1e9
        );

        // The 7B FFN up-projection: 18,944 rows over the 3,584-wide hidden size, 256 columns. Its
        // `n_out·k = 4.85M` output was refused by the old `n_out·k ≤ 2^22` staging cap; the band
        // path reads each row once.
        let (n_out, n_in, k) = (18_944usize, 3584usize, 256usize);
        let raw = random_payload(GGML_TYPE_Q4_K, n_out, n_in, 0x3800, 0x5eed_7b00);
        let qs = acts(n_in, k, 0x0bad_7b00);
        let geom = Geom {
            blocks: n_in / QK_K,
            block_bytes: Q4_K_BLOCK_BYTES,
            row_bytes: (n_in / QK_K) * Q4_K_BLOCK_BYTES,
        };
        let mut p = Vec::with_capacity(RUNS);
        let mut m = Vec::with_capacity(RUNS);
        let mut round = 0usize;
        while round < RUNS {
            let mat_first = round % 2 == 1;
            if !mat_first {
                let start = Instant::now();
                let out = pooled(Family::Q4_K, &raw, n_out, geom, &qs).expect("pooled");
                p.push(start.elapsed().as_secs_f64());
                black_box(&out);
            }
            let start = Instant::now();
            let out =
                matmat_q8act(GGML_TYPE_Q4_K, &raw, n_out, n_in, &qs).expect("the FFN shape admits");
            m.push(start.elapsed().as_secs_f64());
            black_box(&out);
            if mat_first {
                let start = Instant::now();
                let out = pooled(Family::Q4_K, &raw, n_out, geom, &qs).expect("pooled");
                p.push(start.elapsed().as_secs_f64());
                black_box(&out);
            }
            round += 1;
        }
        let best = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
        let payload = (n_out * (n_in / QK_K) * Q4_K_BLOCK_BYTES) as f64;
        println!(
            "{n_out:>7} {k:>7} {:>12.3} {:>12.3} {:>8.2} {:>12.2}",
            best(&p) * 1e3,
            best(&m) * 1e3,
            best(&m) / best(&p),
            payload / best(&m) / 1e9
        );
    }
}
