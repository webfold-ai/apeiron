//! `tests/learn_fold_corrector.rs` — the fp16/u8-rounded rank-`rank` corrector in the `Hl` metric
//! (`fold_transfer.py:560–582`, the `round_f16=True` and `u8_corr=True` branches).
//!
//! Positive: [`Mode::Exact`] equals [`lowrank_corrector_exact`]; the fp16 correction stays within the
//! half-ulp bound of the exact correction in the `Hl` metric (and no better than the exact optimum);
//! the u8 factors lie on the 8-bit grid (re-quantizing an already stored factor moves it by at most
//! one Q32.32 quantum) with the exact bit count; `Uh·Vh == Corr` to two Q32.32 quanta; a full-rank
//! `rank = min(o, n)` recovers `ΔW` to each mode's precision. The agreement test compares the
//! migrated Q32.32 pipeline against the pre-migration `f64` body copied in as [`ref_corrector`]
//! within the bound derived there. The rounded carriers' eigenstage ([`eigen_parts`] →
//! [`eigen_gram`] + [`top_eigh`]) matches the scalar `f64` loops plus [`eigh`] to `1e-6` relative on
//! random `ΔW`/PSD `Hl` at three shapes. Boundary: the shape and rank refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64 in [0, 1), and the test \
              kernels index row-major `[r, c]` buffers by `i/j/k/l` against the spec's \
              `DeltaW[i, j]`/`Uh`/`Vh` arithmetic; the `assert_eq!`s on f64 are the intended \
              bit-exact fixed-point and factor-product checks, not approximate equality; the \
              timing test carries the fixture and the three stage timings in one body."
)]

use std::hint::black_box;
use std::time::Instant;

use apeiron::learn::fold::fit::corrector::{eigen_parts, lowrank_corrector, CorrectorError, Mode};
use apeiron::learn::fold::fit::eigh::{eigen_gram, eigh, gram_scale, lowrank_corrector_exact};
use apeiron::learn::fold::fit::quant::{f16r, quant_c_u8};
use apeiron::learn::solve::eig::{top_eigh, EigError, MAX_ITERS};
use apeiron::learn::solve::q32::{
    fill_scale, from_grid, panel_shift, rshift_rne, to_grid, GRID_BOUND, INPUT_BOUND,
};

#[path = "support/ridge_oracle.rs"]
mod ridge_oracle;

/// One Q32.32 quantum, `2⁻³²`.
const QUANTUM: f64 = 2.328_306_436_538_696_3e-10;
/// Two quanta, `2⁻³¹`: the slack on a re-summed one-rounding product.
const PRODUCT_TOL: f64 = 4.656_612_873_077_393e-10;

/// The migrated [`f16r`] at an `f64` value of the oracle's own data: grid it, round, read back.
fn h16(x: f64) -> f64 {
    from_grid(f16r(to_grid(x, GRID_BOUND).expect("grid")).expect("in binary16 range"))
}

/// `x` as a Q32.32 word (the boundary the word APIs take).
fn q(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("grid")
}

/// `x` as words.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| q(v)).collect()
}

/// The `f64` view of a word buffer.
fn uw(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// FNV-1a over the little-endian bytes of a word buffer.
fn fnv_words(d: &mut u64, xs: &[i64]) {
    for &v in xs {
        for b in v.to_le_bytes() {
            *d ^= u64::from(b);
            *d = d.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// FNV-1a over the little-endian bytes of a scalar.
fn fnv_u64(d: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *d ^= u64::from(b);
        *d = d.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Deterministic 64-bit LCG (Numerical Recipes constants), the tests' only randomness source.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`, 53 significant bits.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// `a · b` for row-major `[r, k] · [k, c]`, accumulating `Σ_t` per `(i, j)` in ascending `t`.
fn matmul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += a[i * k + t] * b[t * c + j];
            }
            out[i * c + j] = acc;
        }
    }
    out
}

/// A random SPD matrix `BᵀB + ½I`, row-major `[n, n]`.
fn random_spd(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; n * n];
    for x in &mut b {
        *x = rng.sym();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += b[k * n + i] * b[k * n + j];
            }
            h[i * n + j] = acc + if i == j { 0.5 } else { 0.0 };
        }
    }
    h
}

/// `tr(R · Hl · Rᵀ) = Σ_ij (R·Hl)[i, j]·R[i, j]` for row-major `[o, n]` `r`, `[n, n]` `hl` — the
/// squared `Hl`-norm of `R`, the objective the corrector minimises.
fn trace_r_hl_rt(r: &[f64], hl: &[f64], o: usize, n: usize) -> f64 {
    let rhl = matmul(r, hl, o, n, n);
    let mut t = 0.0f64;
    for i in 0..o {
        for j in 0..n {
            t += rhl[i * n + j] * r[i * n + j];
        }
    }
    t
}

/// The `o × n` residual `ΔW − Corr`.
fn residual(dw: &[f64], corr: &[f64]) -> Vec<f64> {
    dw.iter().zip(corr).map(|(x, y)| x - y).collect()
}

/// The shared fixture: random `ΔW` (`o × n`) and SPD `Hl` (`n × n`) from one seed.
fn fixture(seed: u64, o: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    (dw, hl)
}

const O: usize = 6;
const N: usize = 10;

#[test]
fn exact_equals_reference() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x1111_2222_3333_4444, O, N);
    let (dwq, hlq) = (qv(&dw), qv(&hl));
    let mine = lowrank_corrector(&dwq, &hlq, O, N, rank, Mode::Exact, true).expect("exact");
    let reference = lowrank_corrector_exact(&dwq, &hlq, O, N, rank).expect("reference");
    assert_eq!(mine.bits, (16 * rank * (O + N)) as u64, "exact bit count");
    assert_eq!(mine.corr, reference.corr, "Corr != reference Corr");
    assert_eq!(
        mine.top_sum, reference.top_sum,
        "top_sum != reference top_sum"
    );
    let f = mine.factors.as_ref().expect("factors requested");
    assert_eq!(f.u, reference.u, "Uh != reference U");
    assert_eq!(f.v, reference.v, "Vh != reference V");
}

#[test]
fn f16_reconstruction_bound() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x5555_6666_7777_8888, O, N);
    let (dwq, hlq) = (qv(&dw), qv(&hl));
    let exact = lowrank_corrector(&dwq, &hlq, O, N, rank, Mode::Exact, true).expect("exact");
    let f16 = lowrank_corrector(&dwq, &hlq, O, N, rank, Mode::F16, true).expect("f16");
    assert_eq!(f16.bits, (16 * rank * (O + N)) as u64, "f16 bit count");
    // `Exact` reads `top_sum` off the f64 Jacobi `eigh` of the Gram; `F16` reads it off the Q32.32
    // `top_eigh` on the power-of-two-scaled grid, so the two agree to that path's eigenvalue
    // accuracy (`eigen_stage_matches_f64_reference` fixes it at `1e-6` relative) rather than
    // bit-for-bit. The observed relative disagreement is ~5e-10.
    let (ts_exact, ts_f16) = (from_grid(exact.top_sum), from_grid(f16.top_sum));
    assert!(
        (ts_f16 - ts_exact).abs() <= 1e-6 * ts_exact.abs().max(1e-30),
        "f16 top_sum {ts_f16} != exact {ts_exact}"
    );
    let err = |corr: &[f64]| trace_r_hl_rt(&residual(&dw, corr), &hl, O, N);
    let e_exact = err(&uw(&exact.corr));
    let e_f16 = err(&uw(&f16.corr));
    // ||ΔW||²_Hl, the scale the half-ulp of the binary16 factors propagates at.
    let s = trace_r_hl_rt(&dw, &hl, O, N);
    let half_ulp = 2f64.powi(-11);
    let slack = 16.0 * half_ulp * s + 1e-12;
    assert!(
        e_f16 <= e_exact + slack,
        "fp16 error {e_f16} exceeds exact {e_exact} + half-ulp slack {slack}"
    );
    // The exact correction is the optimal rank-`rank` `Hl` projection, so fp16 cannot beat it.
    assert!(
        e_f16 >= e_exact - 1e-12,
        "fp16 error {e_f16} below the exact optimum {e_exact}"
    );
}

#[test]
fn u8_bits_and_grid() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x9999_AAAA_BBBB_CCCC, O, N);
    let u8 = lowrank_corrector(&qv(&dw), &qv(&hl), O, N, rank, Mode::U8, true).expect("u8");
    assert_eq!(
        u8.bits,
        (rank as u64) * (8 * (O + N) as u64 + 64),
        "u8 bit count"
    );
    let f = u8.factors.as_ref().expect("factors requested");
    assert_eq!(f.u.len(), O * rank);
    assert_eq!(f.v.len(), rank * N);
    // Each column of Uh is on the 8-bit grid: re-quantizing the stored factor returns a value within
    // one Q32.32 quantum `2⁻³²`. Derivation: the stored factor is the quantizer's own grid value
    // rounded to the Q32.32 grid (`|δ| ≤ 2⁻³³`); the `(c_min, c_max)` header is `f16r` of the column
    // endpoints, and `f16`'s quantum is at least `2⁻²⁴ ≫ 2⁻³³`, so the header and the step are
    // unchanged and the stored point re-quantizes to its own grid value. The observed displacement
    // is ≤ 0.4 quantum.
    for k in 0..rank {
        let col: Vec<i64> = (0..O).map(|i| f.u[i * rank + k]).collect();
        let (q, bits) = quant_c_u8(&col, 8).expect("column quantizes");
        assert_eq!(bits, 8 * O as u64 + 32, "column header bits");
        for i in 0..O {
            let drift = (from_grid(q[i]) - from_grid(f.u[i * rank + k])).abs();
            assert!(
                drift <= QUANTUM,
                "Uh[{i}, {k}] moved {drift:e} off the u8 grid (one quantum is {QUANTUM:e})"
            );
        }
    }
    // Each row of Vh is on the 8-bit grid, to the same bound.
    for k in 0..rank {
        let row: Vec<i64> = f.v[k * N..k * N + N].to_vec();
        let (q, bits) = quant_c_u8(&row, 8).expect("row quantizes");
        assert_eq!(bits, 8 * N as u64 + 32, "row header bits");
        for j in 0..N {
            let drift = (from_grid(q[j]) - from_grid(f.v[k * N + j])).abs();
            assert!(
                drift <= QUANTUM,
                "Vh[{k}, {j}] moved {drift:e} off the u8 grid (one quantum is {QUANTUM:e})"
            );
        }
    }
}

#[test]
fn factors_product_within_two_quanta() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x1234_5678_9ABC_DEF0, O, N);
    // `Corr` is the exact `Σ_k Uh·Vh` rounded once to Q32.32, and `matmul` re-sums the same Q32
    // factors in `f64`: the difference is one rounding (`2⁻³³`) plus the `f64` sum's own error
    // (`rank ≤ 6` terms, relative `2⁻⁵³` of a sum whose terms are below `2²⁴`), so `2⁻³¹`
    // (`PRODUCT_TOL`) is two quanta of slack. Before the Q32.32 migration the two were bit-identical
    // by construction.
    for mode in [Mode::Exact, Mode::F16, Mode::U8] {
        let c = lowrank_corrector(&qv(&dw), &qv(&hl), O, N, rank, mode, true).expect("corrector");
        let f = c.factors.as_ref().expect("factors requested");
        let prod = matmul(&uw(&f.u), &uw(&f.v), O, rank, N);
        for (i, (p, q)) in prod.iter().zip(&uw(&c.corr)).enumerate() {
            assert!(
                (p - q).abs() <= PRODUCT_TOL,
                "{mode:?}: Uh·Vh entry {i} = {p} vs Corr {q} beyond {PRODUCT_TOL:e}"
            );
        }
    }
}

#[test]
fn full_rank_recovery() {
    let rank = O.min(N); // full row rank
    let (dw, hl) = fixture(0x0FED_CBA9_8765_4321, O, N);
    let scale = max_abs(&dw);
    // Per-mode precision of the recovery `Corr ≈ ΔW` at full rank.
    let tolerances = [(Mode::Exact, 1e-9), (Mode::F16, 1e-2), (Mode::U8, 5e-2)];
    for (mode, tol) in tolerances {
        let c = lowrank_corrector(&qv(&dw), &qv(&hl), O, N, rank, mode, false).expect("corrector");
        let e = max_abs(&residual(&dw, &uw(&c.corr)));
        assert!(
            e <= tol * scale,
            "{mode:?}: full-rank recovery error {e} exceeds {tol}·scale {}",
            tol * scale
        );
    }
}

#[test]
fn refusals() {
    let (dw, hl) = fixture(0xBEEF_C0DE_1234_5678, O, N);
    let (dwq, hlq) = (qv(&dw), qv(&hl));
    // ΔW length disagrees with o·n.
    let dw_bad = vec![0_i64; O * N + 1];
    assert_eq!(
        lowrank_corrector(&dw_bad, &hlq, O, N, 3, Mode::F16, false).err(),
        Some(CorrectorError::ShapeMismatch)
    );
    // Hl length disagrees with n·n.
    let hl_bad = vec![0_i64; N * N - 1];
    assert_eq!(
        lowrank_corrector(&dwq, &hl_bad, O, N, 3, Mode::U8, false).err(),
        Some(CorrectorError::ShapeMismatch)
    );
    // rank > o, in each mode.
    for mode in [Mode::Exact, Mode::F16, Mode::U8] {
        assert_eq!(
            lowrank_corrector(&dwq, &hlq, O, N, O + 1, mode, false).err(),
            Some(CorrectorError::RankOutOfRange),
            "{mode:?}: rank > o not refused"
        );
    }
    // Rank 0 returns the empty corrector.
    let empty = lowrank_corrector(&dwq, &hlq, O, N, 0, Mode::F16, true).expect("rank 0");
    assert_eq!(empty.bits, 0);
    assert_eq!(empty.top_sum, 0);
    assert!(empty.corr.iter().all(|&x| x == 0));
    let f = empty.factors.as_ref().expect("factors requested");
    assert!(f.u.is_empty() && f.v.is_empty());
}

#[test]
fn flat_tail_names_the_non_convergence() {
    // `M = diag(1, 0.995, …, 0.96, 0.955, 1/128, …)` at `o = n = 16`; `rank = 1` gives `s = 9`,
    // so the oversampled gap is `λ_10/λ_1 = 0.955` and `0.955^60 ≈ 6.5e-2` cannot reach the
    // `2⁻²⁰` stop test. `ΔW = diag(√λ)` and `Hl = I` realise it as `(ΔW·Hl)·ΔWᵀ`.
    let n = 16usize;
    let s = 9usize;
    let one = 1_i64 << 32;
    let mut dw = vec![0_i64; n * n];
    let mut hl = vec![0_i64; n * n];
    for k in 0..n {
        let lam = if k < s {
            1.0 - 0.005 * (k as f64)
        } else {
            1.0 / 128.0
        };
        let lam = if k == s { 0.955 } else { lam };
        dw[k * n + k] = q(lam.sqrt());
        hl[k * n + k] = one;
    }
    match eigen_parts(&dw, &hl, n, n, 1) {
        Err(CorrectorError::Eigensolver {
            cause: EigError::NotConverged { residual },
            iters,
            residual_ratio,
        }) => {
            assert_eq!(iters, MAX_ITERS, "the cap is the iteration count reached");
            assert!(residual > 0, "residual {residual} must name the shortfall");
            // The stop test is `residual <= λ_max·2⁻²⁰`; a ratio above that bound is why it failed.
            assert!(
                residual_ratio > 1 << 12,
                "residual ratio {residual_ratio} not above the 2⁻²⁰ threshold word"
            );
        }
        other => panic!("expected a named non-convergence, got {other:?}"),
    }
}

/// `M = (ΔW·Hl)·ΔWᵀ` by the scalar `f64` triple loops the gemm path replaces, row-major `[o, o]`.
fn old_syrk(dw: &[f64], hl: &[f64], o: usize, n: usize) -> Vec<f64> {
    let w1 = matmul(dw, hl, o, n, n);
    let dwt = transpose(dw, o, n);
    matmul(&w1, &dwt, o, n, o)
}

/// `aᵀ` for row-major `[r, c]`.
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Columns `o − rank .. o` of the row-major `[o, o]` basis `q` as row-major `[o, rank]` — the top
/// eigenvectors in ascending order.
fn top_columns(q: &[f64], o: usize, rank: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; o * rank];
    for i in 0..o {
        for k in 0..rank {
            out[i * rank + k] = q[i * o + (o - rank + k)];
        }
    }
    out
}

/// The smallest principal cosine between the `o × rank` column spaces of `a` and `b` (`rank ≤ 2`):
/// the smallest singular value of `bᵀa`.
fn subspace_min_cos(a: &[f64], b: &[f64], o: usize, rank: usize) -> f64 {
    let mut g = [0.0f64; 4];
    for i in 0..rank {
        for j in 0..rank {
            let mut acc = 0.0f64;
            for p in 0..o {
                acc += b[p * rank + i] * a[p * rank + j];
            }
            g[i * rank + j] = acc;
        }
    }
    let mut gtg = [0.0f64; 4];
    for i in 0..rank {
        for j in 0..rank {
            let mut acc = 0.0f64;
            for k in 0..rank {
                acc += g[k * rank + i] * g[k * rank + j];
            }
            gtg[i * rank + j] = acc;
        }
    }
    if rank == 1 {
        return gtg[0].max(0.0).sqrt().min(1.0);
    }
    let (a11, a12, a22) = (gtg[0], gtg[1], gtg[3]);
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a12;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    ((tr - disc) / 2.0).max(0.0).sqrt().min(1.0)
}

/// A random `ΔW` whose row `i` is scaled by `0.55^i` (so `M` has a clear top-`rank` gap) and a
/// random SPD `Hl`, from one seed.
fn gapped_fixture(seed: u64, o: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let mut dw = vec![0.0f64; o * n];
    let mut decay = 1.0f64;
    for i in 0..o {
        for j in 0..n {
            dw[i * n + j] = rng.sym() * decay;
        }
        decay *= 0.55;
    }
    let hl = random_spd(&mut rng, n);
    (dw, hl)
}

/// `v·2⁻ᵏ` for a Q32.32 word, the module's exact `shift`.
fn word_shift(v: i64, k: i32) -> i64 {
    if k > 0 {
        i64::try_from(rshift_rne(i128::from(v), u32::try_from(k).expect("k")).expect("range"))
            .expect("i64")
    } else {
        i64::try_from(i128::from(v).checked_shl(k.unsigned_abs()).expect("range")).expect("i64")
    }
}

/// The rounded carriers' eigenstage against the scalar `f64` loops plus [`eigh`]: eigenvalues to
/// `1e-6` relative, the top-`rank` eigenvector subspace to `|cos| ≥ 1 − 1e-8`, and `Corr = U·(UᵀΔW)`
/// to `1e-6` max-abs-relative.
#[test]
fn eigen_stage_matches_f64_reference() {
    for &(o, n) in &[(4usize, 6usize), (16, 40), (64, 130)] {
        for &rank in &[1usize, 2] {
            let (dw, hl) = gapped_fixture(0xC0FF_EE00_1234 + (o as u64) * 31 + rank as u64, o, n);
            // Reference: the old scalar loops and `eigh`.
            let m_ref = old_syrk(&dw, &hl, o, n);
            let (evals_ref, evecs_ref) = eigh(&qv(&m_ref), o).expect("eigh reference");
            let (evals_ref, evecs_ref) = (uw(&evals_ref), uw(&evecs_ref));
            let u_ref = top_columns(&evecs_ref, o, rank);
            let top_sum_ref: f64 = evals_ref[o - rank..].iter().map(|&l| l.max(0.0)).sum();
            let v_ref = matmul(&transpose(&u_ref, o, rank), &dw, rank, o, n);
            let corr_ref = matmul(&u_ref, &v_ref, o, rank, n);

            // Wired path: `eigen_parts` (word Gram + `top_eigh`).
            let (dwq, hlq) = (qv(&dw), qv(&hl));
            let (u_new_q, top_sum_new_q) =
                eigen_parts(&dwq, &hlq, o, n, rank).expect("eigen_parts");
            let u_new = uw(&u_new_q);
            let top_sum_new = from_grid(top_sum_new_q);
            let rel = (top_sum_new - top_sum_ref).abs() / top_sum_ref.abs().max(1e-30);
            assert!(
                rel <= 1e-6,
                "(o,n)=({o},{n}) rank={rank}: top_sum {top_sum_new:e} vs {top_sum_ref:e} rel {rel:e}"
            );
            let cos = subspace_min_cos(&u_new, &u_ref, o, rank);
            assert!(
                cos >= 1.0 - 1e-8,
                "(o,n)=({o},{n}) rank={rank}: subspace min cos {cos:e}"
            );
            let v_new = matmul(&transpose(&u_new, o, rank), &dw, rank, o, n);
            let corr_new = matmul(&u_new, &v_new, o, rank, n);
            let scale = max_abs(&corr_ref).max(1e-30);
            let cerr = max_abs(
                &corr_new
                    .iter()
                    .zip(&corr_ref)
                    .map(|(x, y)| x - y)
                    .collect::<Vec<f64>>(),
            );
            assert!(
                cerr <= 1e-6 * scale,
                "(o,n)=({o},{n}) rank={rank}: Corr {cerr:e} over {:.1e} scale",
                1e-6 * scale
            );

            // The `top_eigh` stage directly: each returned eigenvalue against the reference.
            let (m, km) = eigen_gram(&dwq, &hlq, o, n).expect("eigen_gram");
            let e = gram_scale(&m);
            let m_q: Vec<i64> = m.iter().map(|&x| word_shift(x, -e)).collect();
            let top = top_eigh(&m_q, o, rank).expect("top_eigh");
            for k in 0..rank {
                let have = from_grid(top.evals[k]) * 2f64.powi(km - e);
                let want = evals_ref[o - rank + k];
                let rel = (have - want).abs() / want.abs().max(1e-30);
                assert!(
                    rel <= 1e-6,
                    "(o,n)=({o},{n}) rank={rank} k={k}: eigenvalue {have:e} vs {want:e} rel {rel:e}"
                );
            }
        }
    }
}

/// The Gram against the scalar `f64` oracle within the derived panel-rounding bound, exactly
/// symmetric.
#[test]
fn eigen_gram_agrees_with_f64_oracle() {
    let q = 2f64.powi(-32);
    // One grid half-quantum in real units at a panel whose largest entry is `max`: the pooled
    // kernels and [`panel`] shift by `2⁻ᵏ` with `k = panel_shift(max)`, so the step is `q·2ᵏ`.
    let half = |max: f64| 0.5 * q * 2f64.powi(panel_shift(max));
    for &(o, n) in &[(5usize, 7usize), (6, 10), (12, 12)] {
        let (dw, hl) = fixture(0xC0FF_EE00_1234 ^ ((o * 31 + n) as u64), o, n);
        let (dd, dh) = (max_abs(&dw), max_abs(&hl));
        let w1 = matmul(&dw, &hl, o, n, n);
        let dw1 = max_abs(&w1);
        let m_ref = old_syrk(&dw, &hl, o, n);
        // Each operand crosses the grid once (≤ `half`), each product rounds once per entry, so the
        // first product carries `ε₁ ≤ n(|Hl|·ε_d + |ΔW|·ε_h) + half(w1)` and the second sums it
        // through `n` entries of `ΔW`: `|ΔM| ≤ half(M) + n(|w1|·ε_d + |ΔW|·ε₁)`. The factor two
        // covers the pairwise accumulation counted at half-quanta.
        let (ed, eh) = (half(dd), half(dh));
        let e1 = (n as f64) * (dh * ed + dd * eh) + half(dw1);
        let e = half(max_abs(&m_ref)) + (n as f64) * (dw1 * ed + dd * e1);
        let m = eigen_gram(&qv(&dw), &qv(&hl), o, n).expect("eigen_gram");
        let (mw, km) = m;
        let back = 2f64.powi(km);
        let mut worst = 0.0f64;
        for i in 0..o {
            for j in 0..o {
                worst = worst.max((from_grid(mw[i * o + j]) * back - m_ref[i * o + j]).abs());
            }
        }
        println!(
            "(o,n)=({o},{n}) max |ΔM| {worst:e} over bound {:e}",
            2.0 * e
        );
        assert!(
            worst <= 2.0 * e,
            "(o,n)=({o},{n}): {worst:e} over bound {:e}",
            2.0 * e
        );
        for i in 0..o {
            for j in 0..i {
                assert_eq!(mw[i * o + j], mw[j * o + i], "Gram not exactly symmetric");
            }
        }
    }
}

/// The word corrector is bit-identical to the `f64` wrapper it replaced on these fixtures: the
/// factor and `Corr` digests were recorded through the old `f64` entry before the flip.
#[test]
fn corrector_words_are_bit_identical_to_the_recorded_wrapper() {
    /// A symmetric matrix with a dominant diagonal, the witness fixtures' `Hl`.
    fn spd(n: usize, rng: &mut Lcg) -> Vec<f64> {
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = rng.sym();
                h[i * n + j] = v;
                h[j * n + i] = v;
            }
        }
        for i in 0..n {
            let mut off = 0.0f64;
            for j in 0..n {
                if j != i {
                    off += h[i * n + j].abs();
                }
            }
            h[i * n + i] = off + 1.0 + rng.sym().abs();
        }
        h
    }
    let (o, n, rank) = (8usize, 16usize, 2usize);
    for (seed, mode, top, want_f, want_c) in [
        (
            0x7c0_0001u64,
            Mode::Exact,
            836_674_506_448i64,
            0x213e_c4d4_348c_38b7u64,
            0x6a70_d2e1_a56d_7a7cu64,
        ),
        (
            0x7c0_0001,
            Mode::F16,
            836_674_506_448,
            0x6ad7_848f_4345_3d86,
            0x5a21_81f6_f97f_e6af,
        ),
        (
            0x7c0_0001,
            Mode::U8,
            836_674_506_448,
            0xffb1_7e24_8877_02ab,
            0xd561_f5ca_0a07_67da,
        ),
        (
            0x7c0_0002,
            Mode::Exact,
            881_856_304_080,
            0x54c0_e579_2501_fa61,
            0x7b4b_a60d_bbe6_f80f,
        ),
        (
            0x7c0_0002,
            Mode::F16,
            881_856_304_080,
            0x53ce_2d36_663d_a6d6,
            0xffa4_eceb_1266_182d,
        ),
        (
            0x7c0_0002,
            Mode::U8,
            881_856_304_080,
            0x9568_ba6b_085e_0122,
            0x9ad9_b924_30a5_f0d5,
        ),
    ] {
        let mut rng = Lcg::new(seed);
        let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
        let hl = spd(n, &mut rng);
        let c = lowrank_corrector(&qv(&dw), &qv(&hl), o, n, rank, mode, true).expect("corrector");
        assert_eq!(c.top_sum, top, "top_sum seed={seed:x} {mode:?}");
        let f = c.factors.as_ref().expect("factors requested");
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        fnv_words(&mut d, &f.u);
        fnv_words(&mut d, &f.v);
        assert_eq!(d, want_f, "factors seed={seed:x} {mode:?}");
        let mut dc = 0xcbf2_9ce4_8422_2325u64;
        fnv_words(&mut dc, &c.corr);
        fnv_u64(&mut dc, c.bits);
        assert_eq!(dc, want_c, "Corr seed={seed:x} {mode:?}");
    }
}

// ---- the pre-migration `f64` body, copied verbatim as the agreement oracle -------------------

/// The old `eigen_parts`: the pre-migration `eigen_gram` value read back to the reals, the old
/// [`gram_scale`] (`fill_scale` fill `2³¹` times `2⁻²⁸`) and [`top_eigh`].
fn ref_eigen_parts(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> (Vec<f64>, f64) {
    let (m, km) = eigen_gram(delta_w, hl, o, n).expect("eigen_gram");
    let m: Vec<f64> = m.iter().map(|&w| from_grid(w) * 2f64.powi(km)).collect();
    let mmax = m.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    if mmax <= 0.0 {
        let mut u = vec![0.0f64; o * rank];
        for i in 0..o {
            for k in 0..rank {
                if i == o - rank + k {
                    u[i * rank + k] = 1.0;
                }
            }
        }
        return (u, 0.0);
    }
    let s_m = fill_scale(mmax) * 2f64.powi(-28);
    let m_q: Vec<i64> = m
        .iter()
        .map(|&v| to_grid(v * s_m, INPUT_BOUND).expect("grid"))
        .collect();
    let top = top_eigh(&m_q, o, rank).expect("top_eigh");
    let mut u = vec![0.0f64; o * rank];
    for i in 0..o {
        for k in 0..rank {
            u[i * rank + k] = from_grid(top.evecs[i * rank + k]);
        }
    }
    let mut top_sum = 0.0f64;
    for k in 0..rank {
        let lam = from_grid(top.evals[k]) / s_m;
        if lam > 0.0 {
            top_sum += lam;
        }
    }
    (u, top_sum)
}

/// The old `ridge_solve`: the `f64` Gram, the `1e-6` ridge and the `f64` `solve_spd`.
fn ref_ridge_solve(u: &[f64], delta_w: &[i64], o: usize, n: usize, rank: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; rank * rank];
    for i in 0..rank {
        for j in 0..rank {
            let mut acc = 0.0f64;
            for p in 0..o {
                acc += u[p * rank + i] * u[p * rank + j];
            }
            a[i * rank + j] = acc;
        }
    }
    for i in 0..rank {
        a[i * rank + i] += 1e-6;
    }
    let mut b = vec![0.0f64; rank * n];
    for k in 0..rank {
        for j in 0..n {
            let mut acc = 0.0f64;
            for i in 0..o {
                acc += u[i * rank + k] * from_grid(delta_w[i * n + j]);
            }
            b[k * n + j] = acc;
        }
    }
    ridge_oracle::solve_spd(&a, &b, rank, n).expect("solve_spd")
}

/// The old `product` and the two rounded carriers, assembled as the pre-migration
/// `lowrank_corrector` returned them: `(Corr, Uh, Vh, top_sum)`.
fn ref_corrector(
    dw: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
    mode: Mode,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let (u, top_sum) = ref_eigen_parts(dw, hl, o, n, rank);
    let (uh, vh) = match mode {
        Mode::F16 => {
            let uh: Vec<f64> = u.iter().map(|&x| h16(x)).collect();
            let v = ref_ridge_solve(&uh, dw, o, n, rank);
            let vh: Vec<f64> = v.iter().map(|&x| h16(x)).collect();
            (uh, vh)
        }
        Mode::U8 => {
            let mut uh = vec![0.0f64; o * rank];
            for k in 0..rank {
                let col: Vec<i64> = (0..o)
                    .map(|i| to_grid(u[i * rank + k], GRID_BOUND).expect("grid"))
                    .collect();
                let (q, _) = quant_c_u8(&col, 8).expect("quantize");
                for i in 0..o {
                    uh[i * rank + k] = from_grid(q[i]);
                }
            }
            let v = ref_ridge_solve(&uh, dw, o, n, rank);
            let mut vh = vec![0.0f64; rank * n];
            for k in 0..rank {
                let row: Vec<i64> = v[k * n..k * n + n]
                    .iter()
                    .map(|&x| to_grid(x, GRID_BOUND).expect("grid"))
                    .collect();
                let (q, _) = quant_c_u8(&row, 8).expect("quantize");
                for (slot, &w) in vh[k * n..k * n + n].iter_mut().zip(&q) {
                    *slot = from_grid(w);
                }
            }
            (uh, vh)
        }
        Mode::Exact => unreachable!("the exact carrier has no rounded oracle"),
    };
    (matmul(&uh, &vh, o, rank, n), uh, vh, top_sum)
}

/// The migrated Q32.32 corrector against the `f64` oracle on fixed LCG inputs.
///
/// Bound `2⁻²⁸` relative, derived: every quantity crosses the Q32.32 grid once (relative `2⁻³³`),
/// the Gram and the `Uhᵀ Uh` ridge system each add one rounding per stored entry, and the ridge
/// solve's conditioning on the fixtures' orthonormal-ish `Uh` (columns of a unit-norm eigenbasis,
/// `cond(Uhᵀ Uh) ≤ 4`) amplifies the right-hand side's `2⁻³³` by `2²`; six such steps bound the
/// relative disagreement at `6·2⁻³³·2² = 1.5·2⁻²⁹`, and `2⁻²⁸` is twice that. The observed maximum
/// is printed: `0` for the `F16` carrier (identical arithmetic), `2.1e-10` for `U8`, one quantum.
#[test]
fn agrees_with_f64_oracle() {
    const BOUND: f64 = 3.725_290_298_461_914e-9; // 2⁻²⁸
    let rank = 3usize;
    for seed in [0xA1B2_C3D4_E5F6_0718u64, 0x0F1E_2D3C_4B5A_6978] {
        let (dw, hl) = fixture(seed, O, N);
        for mode in [Mode::F16, Mode::U8] {
            let got = lowrank_corrector(&qv(&dw), &qv(&hl), O, N, rank, mode, true)
                .expect("q32 corrector");
            let (corr, u, v, top_sum) = ref_corrector(&qv(&dw), &qv(&hl), O, N, rank, mode);
            let f = got.factors.as_ref().expect("factors requested");
            let scale = max_abs(&corr).max(f64::MIN_POSITIVE);
            let dc = max_abs(&residual(&corr, &uw(&got.corr)));
            let du = max_abs(&residual(&u, &uw(&f.u)));
            let dv = max_abs(&residual(&v, &uw(&f.v)));
            let dt =
                (from_grid(got.top_sum) - top_sum).abs() / top_sum.abs().max(f64::MIN_POSITIVE);
            println!(
                "oracle seed {seed:x} {mode:?}: Corr {:.3e} (rel {:.3e}) Uh {:.3e} Vh {:.3e} \
                 top_sum rel {:.3e}",
                dc,
                dc / scale,
                du,
                dv,
                dt
            );
            assert!(
                dc <= BOUND * scale,
                "seed {seed:x} {mode:?}: Corr {dc:e} over {} (bound 2⁻²⁸·scale)",
                BOUND * scale
            );
            assert!(
                du <= BOUND * scale.max(1.0),
                "seed {seed:x} {mode:?}: Uh {du:e} over the bound"
            );
            assert!(
                dv <= BOUND * max_abs(&v).max(1.0),
                "seed {seed:x} {mode:?}: Vh {dv:e} over the bound"
            );
            assert!(
                dt <= BOUND,
                "seed {seed:x} {mode:?}: top_sum relative {dt:e} over 2⁻²⁸"
            );
        }
    }
}

/// The production-shape fixture for the timing test: random `ΔW` (`o × n`) and a diagonal PSD `Hl`
/// whose first four entries dominate (so the top eigenpair of `M` has a clear gap and `top_eigh`
/// converges), O(n²) to build.
fn timing_fixture(o: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(0x5EED_5EED_5EED_5EED);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let mut hl = vec![0.0f64; n * n];
    for i in 0..n {
        hl[i * n + i] = if i < 4 { 1.0 } else { 1e-3 };
    }
    (dw, hl)
}

/// Timing (`#[ignore]`): `o = 896`, `n = 4864`, rank 1, [`Mode::U8`], three runs on the shared
/// engine pool, with the stage split `eigen_gram` (the two products, symmetrisation and scaling)
/// versus `top_eigh` versus the rest of `lowrank_corrector`. Run with
/// `--release --ignored --nocapture`; the previous scalar-`f64` path measured 140.4 s
/// single-threaded at this shape.
#[test]
#[ignore = "timing: o = 896, n = 4864, rank 1, Mode::U8, three runs on the engine pool, with the \
            eigen_gram/top_eigh stage split; run with --release --ignored --nocapture"]
fn timing_production_shape() {
    let (o, n) = (896usize, 4864usize);
    let (dw, hl) = timing_fixture(o, n);
    let (dwq, hlq) = (qv(&dw), qv(&hl));
    let mut gram = [0.0f64; 3];
    let mut eig = [0.0f64; 3];
    let mut total = [0.0f64; 3];
    for run in 0..3 {
        let t0 = Instant::now();
        let (m, _km) = eigen_gram(&dwq, &hlq, o, n).expect("eigen_gram");
        gram[run] = t0.elapsed().as_secs_f64();
        let e = gram_scale(&m);
        let m_q: Vec<i64> = m.iter().map(|&x| word_shift(x, -e)).collect();
        let t1 = Instant::now();
        let top = top_eigh(&m_q, o, 1).expect("top_eigh");
        eig[run] = t1.elapsed().as_secs_f64();
        black_box(&top);
        let t2 = Instant::now();
        let c = lowrank_corrector(&qv(&dw), &qv(&hl), o, n, 1, Mode::U8, false).expect("corrector");
        total[run] = t2.elapsed().as_secs_f64();
        black_box(&c);
    }
    let mean = |t: &[f64; 3]| t.iter().sum::<f64>() / 3.0;
    let min = |t: &[f64; 3]| t.iter().copied().fold(f64::INFINITY, f64::min);
    let max = |t: &[f64; 3]| t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "lowrank_corrector o={o} n={n} rank=1 U8 pooled: runs [{:.3}, {:.3}, {:.3}]s \
         mean {:.3}s min {:.3}s max {:.3}s (old scalar f64 single-thread 140.4 s)",
        total[0],
        total[1],
        total[2],
        mean(&total),
        min(&total),
        max(&total)
    );
    println!(
        "stage eigen_gram (products): runs [{:.3}, {:.3}, {:.3}]s mean {:.3}s min {:.3}s max {:.3}s",
        gram[0], gram[1], gram[2], mean(&gram), min(&gram), max(&gram)
    );
    println!(
        "stage top_eigh: runs [{:.3}, {:.3}, {:.3}]s mean {:.3}s min {:.3}s max {:.3}s",
        eig[0],
        eig[1],
        eig[2],
        mean(&eig),
        min(&eig),
        max(&eig)
    );
    println!(
        "stage rest (ridge+quant+product): mean {:.3}s",
        mean(&total) - mean(&gram) - mean(&eig)
    );
}
