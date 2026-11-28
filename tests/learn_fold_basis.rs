//! Gates for `src/learn/fold/stats/basis.rs`: the reverse water-filling closed form and its
//! side-information accounting, the basis-invariance of `Π d_j = det H`, the adapted butterfly's
//! orthogonality and its `π/4` identity with the Hadamard basis, the concentration advantage on a
//! planted subspace, and the recorded FNV-1a digests of every word entry point. Grade `tested`.
//!
//! The world-facing `f64` here is the LCG fixture and the report's printing only: every entry point
//! is exercised on Q32.32 words (`words` quantises a fixture once, as the deleted boundary did), so
//! a digest pins the arithmetic rather than a decimal view of it. The real-model table is
//! `#[ignore]`d and prints one row per tensor.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the synthetic fixtures map integer LCG outputs to f64, and the real-model report \
              prints a word through `from_grid`; the single-letter names are the model's matrix \
              symbols; the equality checks are the water-filling closed form at a chosen θ, which \
              is exact by construction"
)]

use std::fs;
use std::time::Instant;

use apeiron::engine::{Engine, KernelMode};
use apeiron::learn::fold::run::npy::read_npy_i4;
use apeiron::learn::fold::stats::basis::{
    apply_basis, block_metric, butterfly_basis, butterfly_basis_fixed, butterfly_side_bits,
    cholesky_pivots, column_energies, damped_gram, distortion_at, eigen_basis, hadamard_basis,
    identity_basis, rate_at, row_energies, solve_water_fill, sum_log_column_energies,
    uniform_distortion, BasisError, Coefficients, Density, GROUP,
};
use apeiron::learn::fold::stats::capture::{Normalisation, SyrkAcc};
use apeiron::learn::fold::stats::hocapture::engine_tape_mode;
use apeiron::learn::grad::model::{Config, Weights};
use apeiron::learn::solve::q32::{from_grid, round_q32_even, to_grid, GRID_BOUND};

#[path = "support/fwht_oracle.rs"]
mod fwht_oracle;

use fwht_oracle::fwht128;

/// Deterministic xorshift64 LCG shared by every synthetic fixture.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// Standard normal (sum of twelve uniforms minus six).
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// One grid quantum, `2⁻³²`, as a raw word.
const Q: i64 = 1;

/// A fixture slice as the Q32.32 words the entry points take: `round(x·2³²)`, the replaced boundary.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("fixture inside the grid"))
        .collect()
}

/// One scalar fixture as a word.
fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("fixture inside the grid")
}

/// The `f64` view of a word buffer, for the bounds stated in the model's own units.
fn uw(x: &[i64]) -> Vec<f64> {
    x.iter().copied().map(from_grid).collect()
}

/// The word the deleted `f64` boundary printed: `to_grid(from_grid(v))`, exact at or below `2⁵³`
/// and the recorded comparison for a field whose magnitude exceeds it.
fn view(v: i64) -> i64 {
    to_grid(from_grid(v), GRID_BOUND).expect("view")
}

/// The FNV-1a digest of a word buffer, eight little-endian bytes per word.
fn fnv(w: &[i64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in w {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// (1) The closed-form `D(θ)` and `R(θ)` of a geometric `λ`, and the side-information budget.
#[test]
fn a_water_filling_closed_form_and_side_bits() {
    // Diagonal variances 4, 1, 0.25, 0.0625 = 4·4^-k. At θ = 1 the closed forms are
    // D = 1 + 1 + 0.25 + 0.0625 and R = ½·log₂ 4 = 1.
    let lam = words(&[4.0, 1.0, 0.25, 0.0625]);
    let theta = word(1.0);
    let d = distortion_at(&lam, theta).expect("D");
    let r = rate_at(&lam, theta).expect("R");
    let want_d = word(2.3125);
    // `D` sums the exact grid minima and rounds the sum once; `R` adds four `log₂` words, each
    // within 2q of the series bound, halves, then rounds: ≤ 16q.
    assert!((d - want_d).abs() <= 2 * Q, "D = {d} != {want_d}");
    assert!((r - word(1.0)).abs() < 16 * Q, "R = {r} != 1");

    // Side information is charged against the rate budget: budget 1.18·4 = 4.72, side 1.0 bits
    // leaves 3.72 bits of coefficient rate. Dropping the subtraction would solve for 4.72.
    let budget = word(1.18 * 4.0);
    let side = word(1.0);
    let sol = solve_water_fill(&lam, budget, side);
    let target = budget - side;
    // The solve stops at `|R − target| ≤ 4q·N`; the four `log₂` words add ≤ 8q.
    assert!(
        (sol.rate - target).abs() < 16 * Q * lam.len() as i64,
        "rate {} != target {target}",
        sol.rate
    );
    assert!(
        (sol.distortion - distortion_at(&lam, sol.theta).expect("D(θ)")).abs() < 16 * Q,
        "D {} != D(θ)",
        sol.distortion
    );
    assert!(
        (sol.rate - word(1.18 * 4.0)).abs() > word(0.1),
        "the side bits did not move the rate"
    );

    // The fixed-rate baseline is the closed form Σ λ·2^{-2·(1.18 - side/N)}.
    let want_u: f64 = [4.0, 1.0, 0.25, 0.0625]
        .iter()
        .map(|l| l * 2.0f64.powf(-2.0 * (1.18 - 1.0 / 4.0)))
        .sum();
    let u = uniform_distortion(&lam, budget, side).expect("D_u");
    // `bits` rounds once, `2^x` adds ≤ 2q relative, and `Σ λ·factor` rounds once (≤ ½q·N).
    let bound = 16 * word(want_u) + 4 * lam.len() as i64;
    assert!((u - word(want_u)).abs() < bound, "D_u = {u} != {want_u}");
}

/// A deterministic dense symmetric positive-definite Gram.
fn spd(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut a = vec![0.0f64; n * n];
    for v in &mut a {
        *v = rng.gauss();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] * a[j * n + k];
            }
            h[i * n + j] = s;
        }
    }
    for i in 0..n {
        h[i * n + i] += n as f64;
    }
    h
}

/// (2) `Π d_j = det H` for every orthogonal block basis: the log-pivot sums agree across the
/// identity and Hadamard bases, and with the Cholesky pivots of `H` itself.
#[test]
fn b_pivot_product_is_basis_invariant() {
    let n = GROUP;
    let h = damped_gram(&words(&spd(n, 0x5eed)), n).expect("damped H");
    let qi = identity_basis(n).expect("identity");
    let qh = hadamard_basis(n).expect("hadamard");
    let di = cholesky_pivots(&block_metric(&qi, &h, n).expect("M_i"), n).expect("pivots i");
    let dh = cholesky_pivots(&block_metric(&qh, &h, n).expect("M_h"), n).expect("pivots h");
    let s_i: f64 = di.iter().map(|&d| from_grid(d).ln()).sum();
    let s_h: f64 = dh.iter().map(|&d| from_grid(d).ln()).sum();
    let d_ref = cholesky_pivots(&h, n).expect("pivots H");
    let s_ref: f64 = d_ref.iter().map(|&d| from_grid(d).ln()).sum();
    // Each `d_j` is one exact `i128` sum rounded once, but the `L_jk` it subtracts carry their own
    // ≤½q rounding, so a pivot's relative error is bounded by `j·q`; the log sums over `n` pivots
    // are within `n²·q/2` of the f64 reference, and three such sums are compared.
    let tol = 2.0 * (n * n) as f64 * from_grid(Q);
    assert!((s_i - s_h).abs() < tol, "log Πd_i {s_i} != log Πd_h {s_h}");
    assert!(
        (s_i - s_ref).abs() < tol,
        "log Πd_i {s_i} != log det H {s_ref}"
    );
}

/// Column `j` of `q`, length `n`.
fn column(q: &[f64], n: usize, j: usize) -> Vec<f64> {
    (0..n).map(|r| q[r * n + j]).collect()
}

/// (3) The adaptive butterfly is orthogonal, and with every angle `π/4` it is `fwht/√128` up to
/// column signs.
#[test]
fn c_butterfly_orthogonal_and_fixed_angle_is_hadamard() {
    let n = 2 * GROUP;
    let o = 24;
    let mut rng = Lcg::new(0x00b0_77e2);
    let wf: Vec<f64> = (0..o * n).map(|_| rng.gauss()).collect();
    let w = words(&wf);

    let q = uw(&butterfly_basis(&w, o, n).expect("butterfly"));
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for r in 0..n {
                s += q[r * n + i] * q[r * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    // `Q` is a product of at most 28 Givens rotations (one mandatory pass plus three sweeps), each
    // with `c² + s² − 1 ≤ 4q` from its two rounded words and the two rounded products: the
    // `sin`/`cos` kernel returns the rounded value (analytic truncation `4e-13 ≪ q`, dense-sample
    // maximum `2.5q`), and each block product rounds once. The measured maximum of
    // `‖QᵀQ − I‖_max` on this fixture is `19.3q`, asserted at `128q`.
    assert!(worst < 128.0 * from_grid(Q), "‖QᵀQ − I‖_max = {worst}");

    let qf =
        uw(&butterfly_basis_fixed(&w, o, n, word(std::f64::consts::FRAC_PI_4)).expect("fixed"));
    let qh = uw(&hadamard_basis(n).expect("hadamard"));
    // Seven stages of `cos θ = sin θ = 2^{-1/2}` rounded to grid words: ≤ 7·(2q) per entry.
    let tol = 64.0 * from_grid(Q);
    for j in 0..n {
        let a = column(&qf, n, j);
        let b = column(&qh, n, j);
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
        let mut diff = 0.0f64;
        for (x, y) in a.iter().zip(&b) {
            diff = diff.max((sign * x - y).abs());
        }
        assert!(diff < tol, "column {j} differs by {diff} beyond a sign");
    }

    // The seam is the WHT body: one fixed transform of a unit vector is the Hadamard column.
    let e = column(&uw(&identity_basis(GROUP).expect("I")), GROUP, 5);
    let f = fwht128(&e).expect("fwht");
    for i in 0..GROUP {
        assert!(
            (f[i] - qh[i * n + 5]).abs() < from_grid(Q),
            "fwht column mismatch at {i}"
        );
    }
}

/// `t` orthonormalised standard-normal vectors of length `n`.
fn orthonormal(rng: &mut Lcg, n: usize, t: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(t);
    for _ in 0..t {
        let mut v: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
        for u in &out {
            let dot: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            for (a, b) in v.iter_mut().zip(u) {
                *a -= dot * b;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for a in &mut v {
            *a /= norm;
        }
        out.push(v);
    }
    out
}

/// (4) On a synthetic `W` carrying a dominant two-dimensional column subspace, the adapted
/// butterfly concentrates the column energy and lowers the water-filling distortion below the
/// Hadamard basis.
#[test]
fn d_adapted_butterfly_concentrates_a_planted_subspace() {
    let n = GROUP;
    let o = 3 * GROUP / 2;
    let mut rng = Lcg::new(0x009a_17e5);
    let u = orthonormal(&mut rng, o, 2);
    let b = orthonormal(&mut rng, n, 2);
    let mut w = vec![0.0f64; o * n];
    for v in &mut w {
        *v = 0.01 * rng.gauss();
    }
    for i in 0..o {
        for j in 0..n {
            w[i * n + j] += 8.0 * u[0][i] * b[0][j] + 5.0 * u[1][i] * b[1][j];
        }
    }
    let wq = words(&w);

    let h = damped_gram(&identity_basis(n).expect("I"), n).expect("damped I");
    let qj = butterfly_basis(&wq, o, n).expect("butterfly");
    let qh = hadamard_basis(n).expect("hadamard");
    let cj = column_energies(&apply_basis(&wq, o, n, &qj).expect("B_j"), o, n).expect("c_j");
    let ch = column_energies(&apply_basis(&wq, o, n, &qh).expect("B_h"), o, n).expect("c_h");

    let cjf = uw(&cj);
    let mut top: Vec<f64> = cjf.clone();
    top.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let total: f64 = cjf.iter().sum();
    let top2 = (top[0] + top[1]) / total;
    assert!(
        top2 > 0.5,
        "adapted basis keeps only {top2} of the energy in its top two columns"
    );
    let sj = sum_log_column_energies(&cj);
    let sh = sum_log_column_energies(&ch);
    assert!(sj < sh, "Σ log c_j adapted {sj} >= hadamard {sh}");

    let budget = word(1.18 * o as f64 * n as f64);
    let aj = Coefficients::from_basis(&wq, o, n, &h, &qj)
        .expect("coeff j")
        .evaluate(Density::Separable, butterfly_side_bits(n), budget)
        .expect("arm j");
    let ah = Coefficients::from_basis(&wq, o, n, &h, &qh)
        .expect("coeff h")
        .evaluate(Density::Separable, 0, budget)
        .expect("arm h");
    assert!(
        aj.water_filled < ah.water_filled,
        "D_wf(J) {} >= D_wf(H) {}",
        aj.water_filled,
        ah.water_filled
    );
}

/// The recorded FNV-1a digests of every public word entry point on the two fixed fixtures, in the
/// order [`digests`] pushes them. Recorded from the pre-migration `f64` face as
/// `to_grid(public f64 output)` and reproduced by the words API: exactly for every field at or
/// below `2⁵³`, through the same one-rounding [`view`] for the three arm fields, whose distortion
/// words exceed it.
const EXPECTED: [(&str, u64, u64); 28] = [
    ("damped_gram", 0x42fe_1928_1d1e_a59b, 0x79f7_a349_9966_4faa),
    (
        "identity_basis",
        0xe810_d12a_24e8_1b25,
        0xe810_d12a_24e8_1b25,
    ),
    (
        "hadamard_basis",
        0x6702_9d44_6e1a_ce25,
        0x6702_9d44_6e1a_ce25,
    ),
    (
        "butterfly_basis",
        0xea6c_d147_a4e3_05cf,
        0x1d34_bf21_0b1d_a622,
    ),
    (
        "butterfly_basis_fixed",
        0xaa7b_595d_f8ea_ce25,
        0xaa7b_595d_f8ea_ce25,
    ),
    ("eigen_basis", 0xa675_ef67_9936_09c7, 0x530d_8080_e952_4325),
    ("apply_basis", 0x1bb0_2bca_a48f_7b67, 0x3429_5d7d_4ab7_b5c8),
    ("block_metric", 0x7613_2be4_78c7_98df, 0x8eb9_3f76_96be_fb67),
    (
        "cholesky_pivots",
        0xcef2_f0c1_c9d8_e006,
        0xce02_8e7e_ee2f_93f6,
    ),
    ("row_energies", 0x0374_8d07_3055_b956, 0x3d86_c963_d4b2_d040),
    (
        "column_energies",
        0x0e12_9d02_a30d_b311,
        0xadab_80a0_e413_c100,
    ),
    (
        "sum_log_column_energies",
        0x4bf6_d10b_4336_3782,
        0x8a91_d1d9_315a_54be,
    ),
    (
        "butterfly_side_bits",
        0x2f83_3e31_e36a_4c6b,
        0x2f83_3e31_e36a_4c6b,
    ),
    (
        "side_bits_separable",
        0x5d7c_a106_e113_fd42,
        0x5d7c_a106_e113_fd42,
    ),
    (
        "side_bits_tile",
        0xab71_f407_0d3b_f145,
        0xab71_f407_0d3b_f145,
    ),
    (
        "side_bits_oracle",
        0xa8c7_f832_281a_39c5,
        0xa8c7_f832_281a_39c5,
    ),
    (
        "sigma_separable",
        0x4758_fd37_fc4e_55f3,
        0x104d_ba2c_38ed_354c,
    ),
    ("sigma_tile", 0x6615_d64c_8b83_8e25, 0x3442_a382_55e4_1325),
    ("sigma_oracle", 0x066c_e5de_99ad_6878, 0x63e7_51e1_3429_84f3),
    ("water_fill", 0x35e8_e22e_5023_ee79, 0x3b66_5ba9_730f_d7cc),
    (
        "distortion_at",
        0x548b_af5b_f6e7_3011,
        0x548b_af5b_f6e7_3011,
    ),
    ("rate_at", 0xeecd_03ae_6848_79c3, 0x5712_dfb5_8d2a_5e82),
    (
        "uniform_distortion",
        0xedb5_182b_06a8_f856,
        0x3957_bbc6_bf9e_d2b9,
    ),
    ("coeff_b", 0x97f1_cfbf_8f36_98d2, 0x3ec9_4b6a_ce16_4928),
    ("coeff_d", 0xae2a_3080_e9c6_8d5b, 0xf8dc_f8a2_948c_6784),
    (
        "arm_separable",
        0x7f4d_5457_c90b_38cf,
        0x6e4f_8c1e_d900_f888,
    ),
    ("arm_tile", 0xbf0f_a368_f33d_e9f2, 0x54bd_7877_53b1_a296),
    ("arm_oracle", 0xbdb1_4701_8363_63e4, 0x61b2_7b20_6836_f7de),
];

/// The digests of one fixture seed, in [`EXPECTED`] order.
fn digests(seed: u64) -> Vec<(&'static str, u64)> {
    let mut rng = Lcg::new(seed);
    let n = GROUP;
    let o = 24usize;
    let w = words(&(0..o * n).map(|_| rng.gauss()).collect::<Vec<f64>>());
    let h = words(&spd(n, seed ^ 0x55aa));
    let budget = word(1.18 * o as f64 * n as f64);
    let mut out: Vec<(&str, u64)> = Vec::new();
    let hd = damped_gram(&h, n).expect("damped");
    out.push(("damped_gram", fnv(&hd)));
    out.push(("identity_basis", fnv(&identity_basis(n).expect("id"))));
    let qh = hadamard_basis(n).expect("had");
    out.push(("hadamard_basis", fnv(&qh)));
    let qb = butterfly_basis(&w, o, n).expect("but");
    out.push(("butterfly_basis", fnv(&qb)));
    let qf = butterfly_basis_fixed(&w, o, n, word(std::f64::consts::FRAC_PI_4)).expect("fix");
    out.push(("butterfly_basis_fixed", fnv(&qf)));
    let qe = eigen_basis(&w, o, n).expect("eig");
    out.push(("eigen_basis", fnv(&qe)));
    let b = apply_basis(&w, o, n, &qh).expect("apply");
    out.push(("apply_basis", fnv(&b)));
    let m = block_metric(&qh, &hd, n).expect("metric");
    out.push(("block_metric", fnv(&m)));
    let d = cholesky_pivots(&m, n).expect("piv");
    out.push(("cholesky_pivots", fnv(&d)));
    let r = row_energies(&b, o, n).expect("row");
    out.push(("row_energies", fnv(&r)));
    let c = column_energies(&b, o, n).expect("col");
    out.push(("column_energies", fnv(&c)));
    out.push((
        "sum_log_column_energies",
        fnv(&[sum_log_column_energies(&c)]),
    ));
    let sbb = butterfly_side_bits(n);
    out.push(("butterfly_side_bits", fnv(&[sbb])));
    out.push((
        "side_bits_separable",
        fnv(&[Density::Separable.side_bits(o, n)]),
    ));
    out.push(("side_bits_tile", fnv(&[Density::Tile.side_bits(o, n)])));
    out.push(("side_bits_oracle", fnv(&[Density::Oracle.side_bits(o, n)])));
    let ss = Density::Separable.sigma(&b, o, n).expect("s");
    out.push(("sigma_separable", fnv(&ss)));
    let st = Density::Tile.sigma(&b, o, n).expect("t");
    out.push(("sigma_tile", fnv(&st)));
    let so = Density::Oracle.sigma(&b, o, n).expect("o");
    out.push(("sigma_oracle", fnv(&so)));
    let sol = solve_water_fill(&c, budget, sbb);
    out.push(("water_fill", fnv(&[sol.theta, sol.distortion, sol.rate])));
    out.push((
        "distortion_at",
        fnv(&[distortion_at(&c, word(0.7)).expect("D")]),
    ));
    out.push(("rate_at", fnv(&[rate_at(&c, word(0.7)).expect("R")])));
    out.push((
        "uniform_distortion",
        fnv(&[uniform_distortion(&c, budget, sbb).expect("D_u")]),
    ));
    let co = Coefficients::from_basis(&w, o, n, &hd, &qb).expect("coeff");
    out.push(("coeff_b", fnv(&co.b)));
    out.push(("coeff_d", fnv(&co.d)));
    let a1 = co.evaluate(Density::Separable, sbb, budget).expect("a1");
    // The arm words can exceed `2⁵³` (a distortion sums `O·n` pivots); the recorded digest is the
    // pre-migration public `f64` value, so the comparison is through the same one-rounding view.
    out.push((
        "arm_separable",
        fnv(&[view(a1.uniform), view(a1.water_filled), view(a1.rate)]),
    ));
    let a2 = co.evaluate(Density::Tile, sbb, budget).expect("a2");
    out.push((
        "arm_tile",
        fnv(&[view(a2.uniform), view(a2.water_filled), view(a2.rate)]),
    ));
    let a3 = co.evaluate(Density::Oracle, sbb, budget).expect("a3");
    out.push((
        "arm_oracle",
        fnv(&[view(a3.uniform), view(a3.water_filled), view(a3.rate)]),
    ));
    out
}

/// (5) Every word entry point reproduces the digest recorded on the pre-migration `f64` face.
#[test]
fn e_words_api_reproduces_recorded_digests() {
    let seeds = [0x51ed_270b_9a3c_0011u64, 0x1a2b_3c4d_5e6f_7788u64];
    for (k, &seed) in seeds.iter().enumerate() {
        let got = digests(seed);
        assert_eq!(got.len(), EXPECTED.len(), "digest count");
        for (i, ((name, have), (want_name, a, b))) in got.iter().zip(EXPECTED.iter()).enumerate() {
            assert_eq!(name, want_name, "digest {i} name");
            let want = if k == 0 { *a } else { *b };
            assert_eq!(
                *have, want,
                "{name} seed {seed:#x}: {have:#018x} != {want:#018x}"
            );
        }
    }
}

/// The real-model gate's constants.
const MODEL: &str = "~/models/gguf/qwen05b-f16.gguf";
const TOKENS: &str = "~/lanes/w18/kd/c4data/train-c4.qwen.npy";
const LAYERS: [usize; 3] = [2, 11, 20];
const BITS: f64 = 1.18;
const PROJECT_SECONDS: f64 = 300.0;
const MIN_TOKENS: usize = 256;
const TAPE_CAP_BYTES: u128 = 6 * (1u128 << 30);

/// A conservative tape-size estimate at `t` tokens, from the tape's own buffer widths.
fn tape_bytes_at(cfg: &Config, t: usize) -> u128 {
    let e = cfg.n_embd as u128;
    let q = (cfg.n_head * cfg.head_dim) as u128;
    let kv = (cfg.n_head_kv * cfg.head_dim) as u128;
    let ff = cfg.n_ff as u128;
    let per_token = 4 * e + 2 * q + 2 * kv + 3 * ff;
    let per_layer = t as u128 * per_token * 8 + (t * t * cfg.n_head) as u128 * 8;
    cfg.n_layer as u128 * per_layer + t as u128 * (2 * e + cfg.vocab as u128) * 8
}

/// One tensor's seven arm values for one density model.
#[derive(Clone, Copy, Default)]
struct Row {
    du_h: f64,
    wf_h: f64,
    wf_i: f64,
    wf_j: f64,
    wf_v: f64,
    or_h: f64,
    or_j: f64,
}

impl std::ops::AddAssign for Row {
    fn add_assign(&mut self, other: Row) {
        self.du_h += other.du_h;
        self.wf_h += other.wf_h;
        self.wf_i += other.wf_i;
        self.wf_j += other.wf_j;
        self.wf_v += other.wf_v;
        self.or_h += other.or_h;
        self.or_j += other.or_j;
    }
}

/// Evaluate all four bases on one tensor, returning the `(separable, tile)` rows.
fn measure(w: &[i64], o: usize, n: usize, h: &[i64], budget: i64) -> (Row, Row) {
    let mut s = Row::default();
    let mut t = Row::default();

    let qh = hadamard_basis(n).expect("hadamard");
    let ch = Coefficients::from_basis(w, o, n, h, &qh).expect("coeff H");
    let hs = ch.evaluate(Density::Separable, 0, budget).expect("H s");
    let ht = ch.evaluate(Density::Tile, 0, budget).expect("H t");
    let oht = ch.evaluate(Density::Oracle, 0, budget).expect("H oracle t");
    s.du_h = from_grid(hs.uniform);
    s.wf_h = from_grid(hs.water_filled);
    s.or_h = from_grid(oht.water_filled);
    t.du_h = from_grid(ht.uniform);
    t.wf_h = from_grid(ht.water_filled);
    t.or_h = from_grid(oht.water_filled);

    let qi = identity_basis(n).expect("identity");
    let ci = Coefficients::from_basis(w, o, n, h, &qi).expect("coeff I");
    s.wf_i = from_grid(
        ci.evaluate(Density::Separable, 0, budget)
            .expect("I s")
            .water_filled,
    );
    t.wf_i = from_grid(
        ci.evaluate(Density::Tile, 0, budget)
            .expect("I t")
            .water_filled,
    );

    let angle_bits = butterfly_side_bits(n);
    let qj = butterfly_basis(w, o, n).expect("butterfly");
    let cj = Coefficients::from_basis(w, o, n, h, &qj).expect("coeff J");
    s.wf_j = from_grid(
        cj.evaluate(Density::Separable, angle_bits, budget)
            .expect("J s")
            .water_filled,
    );
    t.wf_j = from_grid(
        cj.evaluate(Density::Tile, angle_bits, budget)
            .expect("J t")
            .water_filled,
    );
    s.or_j = from_grid(
        cj.evaluate(Density::Oracle, angle_bits, budget)
            .expect("J oracle s")
            .water_filled,
    );
    t.or_j = from_grid(
        cj.evaluate(Density::Oracle, angle_bits, budget)
            .expect("J oracle t")
            .water_filled,
    );

    let qv = eigen_basis(w, o, n).expect("eigen");
    let cv = Coefficients::from_basis(w, o, n, h, &qv).expect("coeff V");
    s.wf_v = from_grid(
        cv.evaluate(Density::Separable, 0, budget)
            .expect("V s")
            .water_filled,
    );
    t.wf_v = from_grid(
        cv.evaluate(Density::Tile, 0, budget)
            .expect("V t")
            .water_filled,
    );

    (s, t)
}

/// One density section: header, per-tensor rows, sums and ratios.
fn report(label: &str, rows: &[(String, usize, usize, Row)]) {
    println!("\n== density {label} ==  (target {BITS} b/w, D in the metric tr(ΔW·H·ΔWᵀ))");
    println!(
        "{:>8} {:>5} {:>5} | {:>11} {:>11} | {:>11} | {:>11} | {:>11} | {:>11} {:>11}",
        "tensor", "O", "n", "D_u(H)", "D_wf(H)", "D_wf(I)", "D_wf(J)", "D_wf(V)", "or(H)", "or(J)"
    );
    let mut sum = Row::default();
    for (name, o, n, r) in rows {
        sum += *r;
        println!(
            "{name:>8} {o:>5} {n:>5} | {:>11.5e} {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} {:>11.5e}",
            r.du_h, r.wf_h, r.wf_i, r.wf_j, r.wf_v, r.or_h, r.or_j
        );
    }
    println!(
        "{:>8} {:>5} {:>5} | {:>11.5e} {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} {:>11.5e}",
        "SUM", "", "", sum.du_h, sum.wf_h, sum.wf_i, sum.wf_j, sum.wf_v, sum.or_h, sum.or_j
    );
    println!(
        "ratio D_u(H)/D_wf(J) = {:.4}   D_wf(H)/D_wf(J) = {:.4}   D_wf(V)/D_wf(J) = {:.4}",
        sum.du_h / sum.wf_j,
        sum.wf_h / sum.wf_j,
        sum.wf_v / sum.wf_j
    );
}

/// The exact `i128` reference for `row_energies`/`column_energies`: `round64(Σ v²)`, refusing with
/// the same variant when the rounded word leaves `i64`.
fn energy_ref(b: &[i64], o: usize, n: usize, axis: usize) -> Result<Vec<i64>, BasisError> {
    let (len, inner) = if axis == 0 { (o, n) } else { (n, o) };
    let mut out = Vec::with_capacity(len);
    for k in 0..len {
        let mut acc: i128 = 0;
        for j in 0..inner {
            let v = i128::from(if axis == 0 {
                b[k * n + j]
            } else {
                b[j * n + k]
            });
            acc += v * v;
        }
        out.push(round_q32_even(acc).map_err(|_| BasisError::OutOfRange)?);
    }
    Ok(out)
}

/// (9) The energy scan is on [`rowdot_q32`](apeiron::engine::kern) and bit-equal to the exact
/// `i128` reference on every shape, both axes, with words spanning `±2²⁰`; a near-`i64`-max word
/// refuses `OutOfRange` from both the kernel and the reference.
#[test]
fn e_energy_words_are_bit_exact_on_the_kernel() {
    let shapes = [(8usize, 8usize), (17, 40), (40, 17), (1, 33), (33, 1)];
    let mag = 1u64 << 20;
    for (idx, (o, n)) in shapes.into_iter().enumerate() {
        let mut rng = Lcg::new(0x0000_1e17 + idx as u64);
        let b: Vec<i64> = (0..o * n)
            .map(|_| (rng.next_u64() % (2 * mag + 1)) as i64 - mag as i64)
            .collect();
        let r = row_energies(&b, o, n).expect("rows");
        let c = column_energies(&b, o, n).expect("cols");
        assert_eq!(
            r,
            energy_ref(&b, o, n, 0).expect("ref rows"),
            "row energies {o}x{n}"
        );
        assert_eq!(
            c,
            energy_ref(&b, o, n, 1).expect("ref cols"),
            "column energies {o}x{n}"
        );
    }

    // `(2⁶³ − 1)²` fits the reference's `i128` but its rounded `Q32.32` word leaves `i64`; the
    // kernel's block exponents leave the scale-placement window. Both refuse, and the entry point
    // reports the same variant for either.
    let big = vec![i64::MAX; 8];
    assert_eq!(energy_ref(&big, 8, 1, 0), Err(BasisError::OutOfRange));
    assert_eq!(row_energies(&big, 8, 1), Err(BasisError::OutOfRange));
}

/// The real `Qwen2.5 0.5B` 1.18 b/w table over layers 2, 11 and 20 and the seven linears.
#[test]
#[allow(clippy::type_complexity)]
#[ignore = "loads ~1 GB and runs the engine's exact tape over the C4 calibration tokens; --release --ignored --nocapture"]
fn basis_rate_qwen05b() {
    let start = Instant::now();
    let bytes = fs::read(TOKENS).expect("read token npy");
    let ids = read_npy_i4(&bytes).expect("parse token npy");
    assert!(ids.len() >= MIN_TOKENS, "only {} tokens", ids.len());
    let engine = Engine::open(MODEL).expect("open gguf");
    let (weights, cfg) = Weights::from_engine(&engine).expect("load weights");
    eprintln!(
        "[basis] loaded model at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    let e = cfg.n_embd;
    let q_out = cfg.q_out().expect("q_out");
    let kv_out = cfg.kv_out().expect("kv_out");
    let n_ff = cfg.n_ff;

    // Probe: 32 tokens, exact tape plus the full Gram accumulation, then project the wall linearly.
    let probe: Vec<u32> = ids[..32]
        .iter()
        .map(|&v| u32::try_from(v).expect("token"))
        .collect();
    let t0 = Instant::now();
    let (_, tape32) =
        engine_tape_mode(&engine, &cfg, &probe, KernelMode::Exact).expect("tape probe");
    let mut syrks32 = SyrkAcc::new(&cfg).expect("grams");
    syrks32.accumulate(&tape32, &cfg).expect("accumulate probe");
    let wall32 = t0.elapsed().as_secs_f64();
    eprintln!(
        "[basis] 32-token probe wall {wall32:.2}s at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    let by_time = (PROJECT_SECONDS / wall32 * 32.0) as usize;

    let mut ntok = (by_time / GROUP) * GROUP;
    while ntok > MIN_TOKENS && tape_bytes_at(&cfg, ntok) > TAPE_CAP_BYTES {
        ntok -= GROUP;
    }
    ntok = ntok.max(MIN_TOKENS).min(ids.len());
    let tokens: Vec<u32> = ids[..ntok]
        .iter()
        .map(|&v| u32::try_from(v).expect("token"))
        .collect();

    let t1 = Instant::now();
    let (_, tape) = engine_tape_mode(&engine, &cfg, &tokens, KernelMode::Exact).expect("tape");
    let mut grams = SyrkAcc::new(&cfg).expect("grams");
    grams.accumulate(&tape, &cfg).expect("accumulate");
    let wall_capture = t1.elapsed().as_secs_f64();
    eprintln!(
        "[basis] {ntok}-token capture {wall_capture:.2}s at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    drop(tape);
    let lg = grams.finish(Normalisation::Mean).expect("finish grams");
    let t2 = Instant::now();

    let mut sep: Vec<(String, usize, usize, Row)> = Vec::new();
    let mut tile: Vec<(String, usize, usize, Row)> = Vec::new();
    for &l in &LAYERS {
        let g = &lg[l];
        let h_qkv = damped_gram(&words(&g.qkv), e).expect("damped qkv");
        let h_o = damped_gram(&words(&g.o), q_out).expect("damped o");
        let h_gu = damped_gram(&words(&g.gate_up), e).expect("damped gate_up");
        let h_dn = damped_gram(&words(&g.down), n_ff).expect("damped down");
        let lw = &weights.layers[l];
        let tensors: [(&str, &[i64], usize, usize, &[i64]); 7] = [
            ("q", &lw.wq, q_out, e, &h_qkv),
            ("k", &lw.wk, kv_out, e, &h_qkv),
            ("v", &lw.wv, kv_out, e, &h_qkv),
            ("o", &lw.wo, e, q_out, &h_o),
            ("gate", &lw.w_gate, n_ff, e, &h_gu),
            ("up", &lw.w_up, n_ff, e, &h_gu),
            ("down", &lw.w_down, e, n_ff, &h_dn),
        ];
        for (name, w, o, n, h) in tensors {
            let tm = Instant::now();
            let budget = word(BITS * o as f64 * n as f64);
            let (s, t) = measure(w, o, n, h, budget);
            eprintln!(
                "[basis] L{l}.{name} {o}x{n} {:.2}s at {:.1}s",
                tm.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
            sep.push((format!("L{l}.{name}"), o, n, s));
            tile.push((format!("L{l}.{name}"), o, n, t));
        }
    }
    report("s: separable r_i c_j / m", &sep);
    report("t: 16x16 tile mean", &tile);
    println!(
        "\nbasis_rate_qwen05b: tokens = {ntok} (32-token probe {wall32:.2}s, projected {by_time}), \
         capture {wall_capture:.2}s, analysis {:.2}s, total {:.2}s",
        t2.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );
}
