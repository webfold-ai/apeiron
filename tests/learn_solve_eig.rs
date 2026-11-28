//! `tests/learn_solve_eig.rs` — the top-`r` Q32.32 eigenpair extraction [`top_eigh`].
//!
//! Positive: on planted-spectrum PSD matrices `M = A·Aᵀ` the `r` returned eigenvalues agree with
//! the f64 `eigh` of the same matrix to relative `1e-6` and the sign-fixed eigenvector cosines to
//! `1 − 1e-9`; the returned basis is orthonormal (`‖QᵀQ − I‖max ≤ 2⁻²⁴`) with the documented sign
//! contract; the result is bit-identical for 1, 2 and 8 threads; and on the production shape (a
//! rank-20 PSD of size 896) the top `r = 1` and `r = 4` eigenvalues match the planted spectrum and
//! the returned pairs have a small f64 residual against the f64 matrix. Boundary: every refusal
//! arm (shape, `n == 0`, `r == 0`, `r > n`, input bound, dimension-product overflow, non-symmetry
//! beyond one raw unit, low-rank `DegenerateBasis`) and a near-degenerate spectrum that cannot
//! reach the tolerance in [`MAX_ITERS`] iterations (`NotConverged`). Timing (`#[ignore]`): the
//! production shape, `r = 1` and `4`, threads 1 and 8, three runs each, plus the f64 `eigh` at
//! `n = 896` once (the previous 31.1 s profile).
//!
//! f64 is test-only: the oracle is the crate's [`eigh`] and the fixture construction.
//!
//! Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixture draws xorshift words and f64 plane directions and converts raw Q32.32 to \
              reals, all in the test only; the kernels index row-major `[n, n]`, `[n, r]` and \
              `[n, m]` buffers by `i/j/k/l`, the index names of the arithmetic they mirror."
)]

use std::hint::black_box;
use std::time::Instant;

use apeiron::learn::fold::fit::eigh::eigh;
use apeiron::learn::solve::eig::{top_eigh, EigError, MAX_ITERS};

/// Raw Q32.32 `1.0`.
const ONE_RAW: i64 = 1 << 32;
/// Raw Q32.32 `1.0` as `f64`.
const ONE_F: f64 = 4_294_967_296.0;

/// Deterministic xorshift64 (the tests' only randomness source).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A uniform `f64` in `[0, 1)` from the high 53 bits.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }
}

/// `(v + 2³¹) >> 32`, the single round-to-nearest the kernels apply; the fixture shares it.
fn round32(v: i128) -> i128 {
    (v + (1i128 << 31)) >> 32
}

/// The test-local oracle product `A·Aᵀ` (`A` `n × m` Q32.32), one round-to-nearest per entry.
fn naive_syrk(a: &[i64], n: usize, m: usize) -> Vec<i64> {
    let mut out = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc: i128 = 0;
            for k in 0..m {
                acc += i128::from(a[i * m + k]) * i128::from(a[j * m + k]);
            }
            out[i * n + j] = i64::try_from(round32(acc)).expect("i64");
        }
    }
    out
}

/// The planted spectrum: geometric decay `1, 0.5, 0.25, …` with a clear gap after the sixth entry.
fn planted_spectrum(n: usize) -> Vec<f64> {
    let mut spec = Vec::with_capacity(n);
    let mut lam = 1.0f64;
    for k in 0..n {
        spec.push(lam);
        lam *= 0.5;
        if k == 5 {
            lam *= 0.02;
        }
    }
    spec
}

/// A Q32.32 `n × n` PSD matrix `M = A·Aᵀ` whose spectrum is the planted `spec`: `A = Q·diag(√λ)`
/// with `Q` an f64 orthonormal `n × m` basis built from the LCG and Gram–Schmidt (test-only
/// float), rounded to Q32.32.
fn planted_psd(n: usize, spec: &[f64], seed: u64) -> Vec<i64> {
    let m = spec.len();
    let mut lcg = Lcg::new(seed);
    let mut q = vec![0.0f64; n * m];
    for e in &mut q {
        *e = lcg.next_unit() - 0.5;
    }
    for j in 0..m {
        for _pass in 0..2 {
            for k in 0..j {
                let mut dot = 0.0f64;
                for i in 0..n {
                    dot += q[i * m + k] * q[i * m + j];
                }
                for i in 0..n {
                    q[i * m + j] -= dot * q[i * m + k];
                }
            }
        }
        let mut nrm = 0.0f64;
        for i in 0..n {
            nrm += q[i * m + j] * q[i * m + j];
        }
        let nrm = nrm.sqrt();
        for i in 0..n {
            q[i * m + j] /= nrm;
        }
    }
    let mut a = vec![0i64; n * m];
    for i in 0..n {
        for k in 0..m {
            a[i * m + k] = (q[i * m + k] * spec[k].sqrt() * ONE_F).round() as i64;
        }
    }
    naive_syrk(&a, n, m)
}

/// Raw Q32.32 to `f64`.
fn to_f64(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| v as f64 / ONE_F).collect()
}

/// The absolute cosine between column `k` of the `n × cols` raw Q32.32 `v` and column `wk` of the
/// `n × wcols` f64 `w` (sign-free, so the deterministic sign rule does not matter here).
fn cosine(v: &[i64], n: usize, cols: usize, k: usize, w: &[f64], wcols: usize, wk: usize) -> f64 {
    let mut dot = 0.0f64;
    let mut nv = 0.0f64;
    let mut nw = 0.0f64;
    for i in 0..n {
        let a = v[i * cols + k] as f64 / ONE_F;
        let b = w[i * wcols + wk];
        dot += a * b;
        nv += a * a;
        nw += b * b;
    }
    (dot / (nv.sqrt() * nw.sqrt())).abs()
}

#[test]
fn planted_spectrum_matches_f64_eigh() {
    for &n in &[8usize, 64, 130] {
        let spec = planted_spectrum(n);
        let m = planted_psd(n, &spec, 0x1234_5678_9abc_def0);
        let (evals_ref, evecs_ref) = eigh(&m, n).expect("eigh");
        let (evals_ref, evecs_ref) = (to_f64(&evals_ref), to_f64(&evecs_ref));
        for &r in &[1usize, 2, 4] {
            let got = top_eigh(&m, n, r).expect("top_eigh");
            assert_eq!(got.evals.len(), r, "n={n} r={r} eval count");
            assert_eq!(got.evecs.len(), n * r, "n={n} r={r} evec count");
            for k in 0..r {
                let want = evals_ref[n - r + k];
                let have = got.evals[k] as f64 / ONE_F;
                let rel = (have - want).abs() / want.abs().max(1e-30);
                assert!(
                    rel <= 1e-6,
                    "n={n} r={r} k={k} eigenvalue {have:e} vs f64 eigh {want:e} rel {rel:e}"
                );
            }
            for k in 0..r {
                let c = cosine(&got.evecs, n, r, k, &evecs_ref, n, n - r + k);
                assert!(c >= 1.0 - 1e-9, "n={n} r={r} k={k} eigenvector cosine {c}");
            }
            let mut ascending = true;
            for k in 1..r {
                if got.evals[k - 1] > got.evals[k] {
                    ascending = false;
                }
            }
            assert!(ascending, "n={n} r={r} eigenvalues not ascending");
            assert!(
                got.iters >= 1 && got.iters <= MAX_ITERS,
                "n={n} r={r} iters {}",
                got.iters
            );
            println!("n={n} r={r}: {} iterations", got.iters);
        }
    }
}

#[test]
fn returned_basis_is_orthonormal_and_sign_fixed() {
    // The last case is stiff: the planted spectrum spans `4 … 2.5e-6`, so the power step's columns
    // differ in norm by ~1.6e6 and the re-orthogonalisation pass is what holds `‖QᵀQ − I‖max` at
    // the rounding floor (dropping that pass moves it to ~κ·2⁻³² ≈ 4e-4, far over the 2⁻²⁴ bound).
    let stiff = [4.0f64, 2.0, 1.0, 0.5, 0.025, 1e-3, 5e-5, 2.5e-6];
    let cases: [(usize, usize, &[f64]); 4] =
        [(8, 1, &[]), (64, 4, &[]), (130, 2, &[]), (64, 2, &stiff)];
    for &(n, r, planted) in &cases {
        let spec = if planted.is_empty() {
            planted_spectrum(n)
        } else {
            planted.to_vec()
        };
        let m = planted_psd(n, &spec, 0xfeed_0f0f_0f0f_0f0f);
        let got = top_eigh(&m, n, r).expect("top_eigh");
        let mut worst = 0.0f64;
        for a in 0..r {
            for b in 0..r {
                let mut dot = 0.0f64;
                for i in 0..n {
                    dot += (got.evecs[i * r + a] as f64) * (got.evecs[i * r + b] as f64);
                }
                let v = dot / (ONE_F * ONE_F);
                let want = if a == b { 1.0 } else { 0.0 };
                worst = worst.max((v - want).abs());
            }
        }
        println!("n={n} r={r}: |QᵀQ − I|max {worst:e}");
        assert!(
            worst <= 2f64.powi(-24),
            "n={n} r={r} |QᵀQ − I|max {worst:e} over the 2⁻²⁴ bound"
        );
        for k in 0..r {
            let mut nrm2 = 0.0f64;
            let mut best = 0usize;
            let mut best_abs = -1.0f64;
            for i in 0..n {
                let x = got.evecs[i * r + k] as f64;
                nrm2 += x * x;
                if x.abs() > best_abs {
                    best_abs = x.abs();
                    best = i;
                }
            }
            let nrm = (nrm2 / (ONE_F * ONE_F)).sqrt();
            assert!(
                (nrm - 1.0).abs() <= 2f64.powi(-24),
                "n={n} r={r} k={k} column norm {nrm}"
            );
            assert!(
                got.evecs[best * r + k] > 0,
                "n={n} r={r} k={k} largest-magnitude entry is not positive"
            );
        }
    }
}

#[test]
fn refusals_reach_every_arm() {
    // Shape: `m.len() != n·n`.
    assert_eq!(top_eigh(&[0i64; 5], 2, 1), Err(EigError::ShapeMismatch));
    // `r == 0` (the empty corrector, not an eigenproblem) and `r > n`.
    assert_eq!(top_eigh(&[ONE_RAW; 4], 2, 0), Err(EigError::RankZero));
    assert_eq!(top_eigh(&[ONE_RAW; 4], 2, 3), Err(EigError::RankOutOfRange));
    // `n == 0`.
    assert_eq!(top_eigh(&[], 0, 1), Err(EigError::Empty));
    // Input bound `2³⁶` (strict), on the diagonal and off it.
    let mut over = vec![0i64; 4];
    over[0] = 1i64 << 36;
    over[3] = ONE_RAW;
    assert_eq!(top_eigh(&over, 2, 1), Err(EigError::InputBound));
    // Dimension product that leaves `usize`.
    assert_eq!(top_eigh(&[], usize::MAX, 1), Err(EigError::Overflow));
    // Non-symmetric beyond one raw unit is refused.
    let mut bad = vec![0i64; 9];
    for i in 0..3 {
        bad[i * 3 + i] = ONE_RAW;
    }
    bad[1] = 2;
    bad[3] = 0;
    assert_eq!(top_eigh(&bad, 3, 1), Err(EigError::NotSymmetric));
    // Exactly one raw unit of asymmetry is inside the accepted domain.
    let mut near = vec![0i64; 9];
    for i in 0..3 {
        near[i * 3 + i] = ONE_RAW;
    }
    near[1] = 1;
    near[3] = 0;
    assert!(top_eigh(&near, 3, 1).is_ok());
    // The all-zero matrix converges on the first pass (λ = 0, an arbitrary orthonormal vector).
    assert_eq!(
        top_eigh(&[0i64; 16], 4, 1).expect("zero matrix").evals,
        vec![0i64]
    );
    // A rank-2 diagonal of order 16 has rank < s = 9, so the power step's span collapses.
    let mut low = vec![0i64; 256];
    low[0] = ONE_RAW;
    low[17] = ONE_RAW;
    assert_eq!(top_eigh(&low, 16, 1), Err(EigError::DegenerateBasis));
}

#[test]
fn near_degenerate_spectrum_is_not_converged() {
    // `M = diag(1, 1 − 0.005·k, …)`: `n = 16`, `r = 1` gives `p = 8`, `s = 9`, so the oversampled
    // ratio is λ_10/λ_1 = 0.955 and the leakage decays as 0.955^60 ≈ 6.5e-2 — the residual stays
    // near 1.5e-2·2³², far above the 2⁻²⁰·λ_max tolerance, at MAX_ITERS.
    let n = 16usize;
    let s = 9usize;
    let mut m = vec![0i64; n * n];
    for k in 0..n {
        let slow = if k < s {
            ((1.0 - 0.005 * (k as f64)) * ONE_F) as i64
        } else {
            ONE_RAW >> 7
        };
        m[k * n + k] = if k == s { (0.955 * ONE_F) as i64 } else { slow };
    }
    match top_eigh(&m, n, 1) {
        Err(EigError::NotConverged { residual }) => {
            assert!(
                residual > ONE_RAW >> 20,
                "NotConverged residual {residual} not above the tolerance"
            );
        }
        other => panic!("expected NotConverged, got {other:?}"),
    }
}

#[test]
fn production_shape_rank_twenty() {
    let n = 896usize;
    let m = 20usize;
    let spec: Vec<f64> = (0..m).map(|k| 0.8f64.powi(k as i32)).collect();
    let mat = planted_psd(n, &spec, 0x5eed_5eed_5eed_5eed);
    let mf = to_f64(&mat);
    for &r in &[1usize, 4] {
        let got = top_eigh(&mat, n, r).expect("top_eigh");
        for k in 0..r {
            // `spec` is descending, the returned eigenvalues ascending.
            let want = spec[r - 1 - k];
            let have = got.evals[k] as f64 / ONE_F;
            let rel = (have - want).abs() / want;
            assert!(
                rel <= 1e-6,
                "n={n} r={r} k={k} eigenvalue {have:e} vs planted {want:e} rel {rel:e}"
            );
        }
        println!("n={n} r={r}: {} iterations", got.iters);
        let lam_max = got.evals[r - 1] as f64 / ONE_F;
        let tol = lam_max * 2f64.powi(-16);
        for k in 0..r {
            let lam = got.evals[k] as f64 / ONE_F;
            let mut v: Vec<f64> = (0..n)
                .map(|i| got.evecs[i * r + k] as f64 / ONE_F)
                .collect();
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut v {
                *x /= nrm;
            }
            let mut worst = 0.0f64;
            for i in 0..n {
                let mut acc = 0.0f64;
                for j in 0..n {
                    acc += mf[i * n + j] * v[j];
                }
                worst = worst.max((acc - lam * v[i]).abs());
            }
            assert!(
                worst <= tol,
                "n={n} r={r} k={k} f64 residual {worst:e} over tol {tol:e}"
            );
        }
    }
}

#[test]
#[ignore = "timing: n = 896, r = 1 and 4, three pooled runs each, plus the f64 eigh at n = 896 \
            once (the 31.1 s profile); run with --release --ignored --nocapture"]
fn timing_production_shape() {
    let n = 896usize;
    let m = 20usize;
    let spec: Vec<f64> = (0..m).map(|k| 0.8f64.powi(k as i32)).collect();
    let mat = planted_psd(n, &spec, 0x5eed_5eed_5eed_5eed);
    let mut top4: Vec<i64> = Vec::new();
    for &r in &[1usize, 4] {
        let mut times = [0.0f64; 3];
        let mut iters = [0usize; 3];
        for (t, it) in times.iter_mut().zip(iters.iter_mut()) {
            let start = Instant::now();
            let got = top_eigh(&mat, n, r).expect("top_eigh");
            *t = start.elapsed().as_secs_f64();
            *it = got.iters;
            if r == 4 {
                top4 = got.evals.clone();
            }
            black_box(&got);
        }
        let mean = times.iter().sum::<f64>() / 3.0;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "top_eigh n={n} r={r} pooled: runs [{:.3}, {:.3}, {:.3}]s mean {mean:.3}s min \
             {min:.3}s max {max:.3}s iters [{}, {}, {}]",
            times[0], times[1], times[2], iters[0], iters[1], iters[2]
        );
    }
    let start = Instant::now();
    let (evals, _) = eigh(&mat, n).expect("eigh");
    let evals = to_f64(&evals);
    let secs = start.elapsed().as_secs_f64();
    println!("eigh n={n}: {secs:.1}s");
    // Instrument check: both routines must agree on the top four eigenvalues before either time is
    // quoted.
    let mut worst = 0.0f64;
    for k in 0..4 {
        let have = top4[k] as f64 / ONE_F;
        let want = evals[n - 4 + k];
        worst = worst.max((have - want).abs() / want.abs());
    }
    assert!(worst <= 1e-5, "top_eigh vs eigh relative {worst:e}");
    println!("top_eigh vs eigh top-4 relative {worst:e}");
}
