//! Row-parallel projections: [`matvec_fused`] over whole rows split across scoped threads.
//!
//! The decode step's resident path calls [`row_chunk_budget`] for a projection's row count and
//! [`matvec_rows_chunked`] to run the row chunks; the sequential test path calls [`matvec_fixed`]
//! unchunked. [`matvec_fused`] is value-identical to [`matvec_fixed`] (it falls back to it
//! whenever it cannot prove the same arithmetic), so the chunked and unchunked results agree.
//! Split out of `step.rs` to keep that file inside the owned-source line budget.

use crate::format::quant;

use super::fused::matvec_fused;
use crate::engine::MatvecError;

/// The fewest rows a projection chunk may carry, so scoped-thread overhead does not dominate.
#[cfg(not(target_arch = "wasm32"))]
const MIN_ROWS_PER_CHUNK: usize = 64;

/// The row-chunk count for a projection with `n_out` rows: at most the machine's available
/// parallelism and never more than `n_out / MIN_ROWS_PER_CHUNK` chunks, so every chunk keeps at
/// least [`MIN_ROWS_PER_CHUNK`] rows. Never zero.
///
/// # Complexity
///
/// `O(1)`; the CPU count is resolved once behind a process-wide `OnceLock`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn row_chunk_budget(n_out: usize) -> usize {
    chunk_budget(n_out, online_threads())
}

/// The wasm32 build has no scoped threads (`std::thread::scope` is not executed) and no worker
/// pool, so every projection runs as one chunk through the unchunked row kernel. Fixed here, at
/// compile time, rather than by a runtime flag.
#[cfg(target_arch = "wasm32")]
pub(crate) fn row_chunk_budget(_n_out: usize) -> usize {
    1
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn chunk_budget(n_out: usize, threads: usize) -> usize {
    let by_rows = n_out / MIN_ROWS_PER_CHUNK;
    if by_rows == 0 {
        1
    } else if by_rows < threads {
        by_rows
    } else {
        threads
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn online_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if let Some(t) = THREADS.get() {
        return *t;
    }
    let t = online_cpus();
    let _ = THREADS.set(t);
    THREADS.get().copied().unwrap_or(t)
}

/// The machine's online CPU count, independent of the calling thread's affinity.
///
/// `std::thread::available_parallelism` reports the calling thread's affinity mask, and the pool
/// pins its dispatching thread to one core on its first dispatch. Cached after such a dispatch, it
/// read 1 and serialised every chunked projection: Qwen3.5 decode fell from 69 to 19 tok/s
/// whenever a pooled quantization ran before the first step (the batched prefill's order).
#[cfg(not(target_arch = "wasm32"))]
fn online_cpus() -> usize {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sysconf` reads a system constant and has no preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if let Ok(n) = usize::try_from(n) {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

#[cfg(test)]
mod cpu_tests {
    /// A thread pinned to one core still sees every online CPU.
    #[test]
    #[cfg(target_os = "linux")]
    fn online_cpus_ignores_the_calling_threads_affinity() {
        let free = super::online_cpus();
        let pinned = std::thread::spawn(|| {
            // SAFETY: a zeroed `cpu_set_t` owned by this frame with one bit set, applied to the
            // calling thread only.
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(0, &mut set);
                let _ = libc::sched_setaffinity(
                    0,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &raw const set,
                );
            }
            let seen = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
            (seen, super::online_cpus())
        })
        .join()
        .expect("pinned thread");
        assert_eq!(pinned.0, 1, "the pinned thread's own view is one CPU");
        assert_eq!(pinned.1, free, "the online count ignores the pin");
    }
}

/// [`matvec_fused`] over `n_out` whole rows split into `chunks` contiguous row chunks, run on
/// scoped threads and concatenated in row order.
///
/// `raw` is the row-major `n_out × n_in` payload of ggml type `ggml_type`; each chunk is a whole
/// number of rows, so chunk `c` covers rows `[lo, hi)` and the byte range
/// `[lo·row_bytes, hi·row_bytes)`, with `row_bytes` from [`quant::block_geom`] — the same helper
/// and the same stride [`matvec_fused`] uses. The split is balanced: `n_out / chunks` rows per
/// chunk with the first `n_out % chunks` chunks one row longer, so any `chunks` covers every row
/// once even when it does not divide `n_out`. `chunks <= 1` or `chunks >= n_out` that reduces to a
/// single chunk is exactly one [`matvec_fused`] call over the whole payload.
///
/// # Errors
///
/// Returns the first chunk's refusal in row order: the chunks are joined in spawn order
/// (ascending `lo`) and the first `Err` is returned. A short payload, an unsupported quant family
/// and a shape mismatch are the same refusals [`matvec_fused`] makes.
///
/// # Complexity
///
/// `O(n_out·n_in)` time and `O(n_out)` output — the same work as one [`matvec_fused`] — plus
/// `O(chunks)` join handles and no copy of the weight payload.
pub(crate) fn matvec_rows_chunked(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
    chunks: usize,
) -> Result<Vec<i64>, MatvecError> {
    let (qk, bb) =
        quant::block_geom(ggml_type).ok_or(MatvecError::UnsupportedQuant { ggml_type })?;
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let row_bytes = n_in
        .div_ceil(qk)
        .checked_mul(bb)
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
    let upper = if n_out == 0 { 1 } else { n_out };
    let actual = chunks.clamp(1, upper);
    if actual == 1 {
        return matvec_fused(ggml_type, raw, n_out, n_in, x);
    }
    let base = n_out / actual;
    let extra = n_out % actual;
    let mut out: Vec<i64> = Vec::with_capacity(n_out);
    std::thread::scope(|scope| -> Result<(), MatvecError> {
        let mut handles = Vec::with_capacity(actual);
        let mut lo = 0usize;
        let mut c = 0usize;
        while c < actual {
            let rows = base + usize::from(c < extra);
            let hi = lo + rows;
            let bytes =
                raw.get(lo * row_bytes..hi * row_bytes)
                    .ok_or(MatvecError::ShortPayload {
                        need,
                        have: raw.len(),
                    })?;
            handles.push(scope.spawn(move || matvec_fused(ggml_type, bytes, rows, n_in, x)));
            lo = hi;
            c += 1;
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(v)) => out.extend_from_slice(&v),
                Ok(Err(e)) => return Err(e),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(())
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        reason = "test-only fixture/reference arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::{matvec_fixed, EngineError};

    /// A `Q8_0` payload of `rows` rows over `n_in = 32`; `bad` gives that row an `Inf` `f16` scale
    /// so its dequantization refuses with [`EngineError::NonFiniteScale`].
    fn q8_rows(rows: usize, bad: Option<usize>) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::with_capacity(rows * 34);
        let mut r = 0usize;
        while r < rows {
            let scale: u16 = if bad == Some(r) { 0x7c00 } else { 0x3c00 };
            b.extend_from_slice(&scale.to_le_bytes());
            let mut c = 0usize;
            while c < 32 {
                b.push(u8::try_from((r * 7 + c * 3) % 11).expect("small"));
                c += 1;
            }
            r += 1;
        }
        b
    }

    /// The chunked row projection equals the unchunked kernel for every chunk count, including
    /// counts that do not divide the row count (uneven splits).
    #[test]
    fn row_chunked_matvec_equals_the_unchunked_kernel() {
        const ROWS: usize = 10;
        const N_IN: usize = 32;
        let raw = q8_rows(ROWS, None);
        let mut x: Vec<i64> = Vec::with_capacity(N_IN);
        let mut i = 0usize;
        while i < N_IN {
            x.push((i64::try_from(i).expect("small") - 16) * 1_000_000);
            i += 1;
        }
        let want = matvec_fixed(quant::GGML_TYPE_Q8_0, &raw, ROWS, N_IN, &x).expect("unchunked");
        for chunks in [1usize, 2, 3, 4, 5, 7, 10, 64] {
            let got = matvec_rows_chunked(quant::GGML_TYPE_Q8_0, &raw, ROWS, N_IN, &x, chunks)
                .expect("chunked");
            assert_eq!(got, want, "chunks {chunks}");
        }
    }

    /// A defect in any row is surfaced, whichever chunk of a two-way split carries it.
    #[test]
    fn row_chunked_matvec_surfaces_a_defect_in_every_chunk() {
        const ROWS: usize = 10;
        const N_IN: usize = 32;
        let x = vec![1i64 << 24; N_IN];
        for bad in 0..ROWS {
            let raw = q8_rows(ROWS, Some(bad));
            let got = matvec_rows_chunked(quant::GGML_TYPE_Q8_0, &raw, ROWS, N_IN, &x, 2);
            assert!(got.is_err(), "a defect in row {bad} was not surfaced");
        }
    }

    /// With a refusal in each of two chunks, the row-order join returns the earlier chunk's.
    #[test]
    fn row_chunked_matvec_returns_the_first_chunk_error_in_row_order() {
        const ROWS: usize = 10;
        const N_IN: usize = 32;
        // Row 0 (chunk 0) carries an `Inf` scale, so its dequantization refuses; rows 5..10
        // (chunk 1) are well formed but, against `x = i64::MAX`, round past `i64` in the dot. A
        // two-chunk split must report the earlier chunk's refusal, never the later chunk's.
        let mut raw = q8_rows(ROWS, None);
        raw[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        let x = vec![i64::MAX; N_IN];
        match matvec_rows_chunked(quant::GGML_TYPE_Q8_0, &raw, ROWS, N_IN, &x, 2) {
            Err(MatvecError::Dequant(EngineError::NonFiniteScale { bits })) => {
                assert_eq!(bits, 0x7c00);
            }
            other => panic!("expected the first chunk's non-finite-scale refusal, got {other:?}"),
        }
    }
}
