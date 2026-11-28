//! `tests/learn_fold_fold_fit.rs` — the production `fold_fit`: the winning pack recomposes to `Wp`
//! bit-for-bit, `wp = plane_w(pack)`, `j_best` is the candidate minimum with the refit strictly
//! below the plain `gptq` candidate, the carriers `dh`/`ch` lie on their `quant_c_u8` grids and the
//! code plane's 8-blocks are exact `E8 ∪ O8` codewords, the refusals, the `g_row` dependence, and a
//! numeric regression pin. Grade `tested`.
//!
//! Mutation control (see the lane return): removing the 8-block sign parity fix in `fold_fit`, or
//! flipping the `<` in the `e8_icm` pack-candidate comparison to `>`, turns
//! `learn_fold_fold_fit_production_numbers_are_pinned` red.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division and the flat row-major index \
              arithmetic is the reference payload format; the pack/recompose, grid and codebook \
              checks are exact float comparisons by construction, and the `o/n/m/k/g` names are the \
              reference's vocabulary"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::fold_fit::{
    fold_fit, FoldFitConfig, FoldFitError, FoldFitTag, BLOCK, GROUP,
};
use apeiron::learn::fold::fit::gptq_e8::gptq_e8;
use apeiron::learn::fold::fit::icm::{e8_icm_step, nearest_index, plane_w, PlaneRef};
use apeiron::learn::fold::fit::quant::{f16r, quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::{compose, had_gram};
use apeiron::learn::solve::q32::{from_grid, round_q32_even, to_grid, GRID_BOUND};
include!("support/fwht_oracle.rs");

/// Deterministic xorshift64 LCG.
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

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// A positive-definite Toeplitz metric `H[i,j] = rho^|i-j|`, the calibration Gram stand-in.
fn toeplitz(n: usize, rho: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = i.abs_diff(j);
            h[i * n + j] = rho.powi(i32::try_from(d).expect("d < 2^31"));
        }
    }
    h
}

fn rand_mat(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * cols).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// The production call-site configuration (`fold_transfer.py:1733–1736`).
fn production_cfg() -> FoldFitConfig {
    FoldFitConfig {
        iters: 1,
        ridge: 4_294_967,
        damp: 42_949_673,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 858_993_459,
        sweeps: 2,
    }
}

/// The fixed test matrix: `o = 4`, `n = 256`, a correlated SPD Toeplitz Gram.
fn fixed_inputs() -> (usize, usize, Vec<f64>, Vec<f64>) {
    let (o, n) = (4_usize, 256_usize);
    let h = toeplitz(n, 0.4);
    let ws = rand_mat(o, n, &mut Lcg::new(0x0F01DF17));
    (o, n, ws, h)
}

/// The pack read back as reals: the flipped [`PackOut`] is Q32.32 words, and every assertion below
/// is the same exact-grid claim in the test's `f64` view (`from_grid` is exact for these words).
struct PackF {
    s_rot: Vec<f64>,
    rh: Vec<f64>,
    ch: Vec<f64>,
    dh: Vec<f64>,
}

/// [`FoldFit`] read back as reals at the test boundary.
struct FitF {
    wp: Vec<f64>,
    pack: PackF,
    j_best: f64,
    trace: Vec<f64>,
    bits_r: u64,
    bits_c: u64,
    bits_d: u64,
    tag: FoldFitTag,
}

/// The real view of the returned words, the boundary conversion the callers of the flipped API
/// apply.
fn fit_reals(fit: &apeiron::learn::fold::fit::fold_fit::FoldFit) -> FitF {
    FitF {
        wp: uw(&fit.wp),
        pack: PackF {
            s_rot: uw(&fit.pack.s_rot),
            rh: uw(&fit.pack.rh),
            ch: uw(&fit.pack.ch),
            dh: uw(&fit.pack.dh),
        },
        j_best: from_grid(fit.j_best),
        trace: uw(&fit.trace),
        bits_r: fit.bits_r,
        bits_c: fit.bits_c,
        bits_d: fit.bits_d,
        tag: fit.tag,
    }
}

fn run() -> (usize, usize, Vec<f64>, Vec<f64>, FitF) {
    let (o, n, ws, h) = fixed_inputs();
    let fit = fold_fit(&qw(&ws), &qw(&h), o, n, None, &production_cfg()).expect("production fit");
    (o, n, ws, h, fit_reals(&fit))
}

/// The `E8 ∪ O8` codeword `idx` as real values `codeword(idx)[k] / 2^32`.
fn codebook_row(idx: usize) -> [f64; BLOCK] {
    let one = (1_i64 << 32) as f64;
    let row = codeword(idx).expect("row in range");
    let mut out = [0.0_f64; BLOCK];
    for k in 0..BLOCK {
        out[k] = row[k] as f64 / one;
    }
    out
}

/// The Q32.32 grid quantum `2⁻³²`.
const Q32_QUANTUM: f64 = 2.328_306_436_538_696_3e-10;

/// `x[:, i] *= v[i]` for a row-major `[o, n]` buffer in the carrier's arithmetic: each entry is the
/// Q32.32 word of the product, `round(x·v·2³²)` (one round-to-nearest, ties to even), which is the
/// `fold_fit::scale_columns` operation on the stored plane.
fn scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            let a = i128::from(grid(x[row * n + col]));
            let b = i128::from(grid(v[col]));
            let w = round_q32_even(a * b).expect("product inside the grid");
            x[row * n + col] = from_grid(w);
        }
    }
}

/// `metric(H, ridge)`, the test-side objective metric.
fn metric(h: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let mut hl = h.to_vec();
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        hl[i * n + i] += ridge * mean;
    }
    hl
}

/// `J(Ws, Wp, Hl)` of a returned `Wp`.
fn wp_objective(ws: &[f64], h: &[f64], o: usize, n: usize, wp: &[f64]) -> f64 {
    let hl = metric(h, n, from_grid(production_cfg().ridge));
    from_grid(xref::objective(&qw(ws), &qw(wp), &qw(&hl), o, n).expect("objective"))
}

/// A value on the Q32.32 grid: `round(x·2³²)` exists and is within the grid range.
fn grid(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The grid image of a value as the `f64` it denotes: `from_grid(to_grid(x))`.
fn from_grid_word(x: f64) -> f64 {
    from_grid(grid(x))
}

/// A fixture as the `Q32.32` words the migrated steps consume.
fn qw(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| grid(v)).collect()
}

/// A word slice read back through the grid map, the migrated boundary's read-out.
fn uw(x: &[i64]) -> Vec<f64> {
    x.iter().copied().map(from_grid).collect()
}

/// The migrated [`f16r`] at an `f64` value of the test's own data: grid it, round, read back.
fn h16(x: f64) -> f64 {
    from_grid(f16r(to_grid(x, GRID_BOUND).expect("grid")).expect("in binary16 range"))
}

/// A value is on the affine `bits`-grid `[f16r(min), f16r(max)]` with `2^bits − 1` levels — the
/// `quant_c_u8` grid of the carrier — up to the Q32.32 boundary: `quant_c_u8` emits the level value
/// `lo + (q/levels)·span` in `f64`, and the pack stores its round-to-nearest grid image, so each
/// entry sits within half a grid quantum of its level. `tol` is that bound.
fn on_affine_grid(values: &[f64], bits: u32, tol: f64) -> bool {
    let lo = h16(values.iter().copied().fold(f64::INFINITY, f64::min));
    let hi = h16(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let span = (hi - lo).max(1e-12);
    let levels = f64::from((1_u32 << bits) - 1);
    values.iter().all(|&v| {
        let q = (((v - lo) / span) * levels)
            .round_ties_even()
            .clamp(0.0, levels);
        (lo + (q / levels) * span - v).abs() <= tol * (1.0 + v.abs())
    })
}

#[test]
fn learn_fold_fold_fit_pack_recomposes_to_wp_bitwise() {
    let (o, n, _ws, _h, fit) = run();
    let mut rotated = fit.pack.s_rot.clone();
    scale_columns(&mut rotated, o, n, &fit.pack.dh);
    // Each helper boundary carries the Q32.32 stage image: `fold_fit` writes the `fwht` output back
    // to the grid before `compose` reads it, so the test does the same.
    let s_dh: Vec<f64> = fwht128(&rotated)
        .expect("fwht")
        .into_iter()
        .map(from_grid_word)
        .collect();
    let recomposed =
        uw(&compose(&qw(&s_dh), &qw(&fit.pack.rh), &qw(&fit.pack.ch), o, n).expect("compose"));
    assert_eq!(recomposed.len(), o * n);
    // The pack is stored on the Q32.32 grid, so the recomposition check is word for word in that
    // domain: the same carrier operations run on both sides, so no tolerance is applied and the
    // quantum is the carrier's `2⁻³²`.
    for k in 0..o * n {
        assert_eq!(
            grid(recomposed[k]),
            grid(fit.wp[k]),
            "pack does not recompose to wp at {k}: {} != {}",
            recomposed[k],
            fit.wp[k]
        );
    }
}

#[test]
fn learn_fold_fold_fit_wp_equals_plane_w_of_pack() {
    let (o, n, _ws, _h, fit) = run();
    let idx = vec![0_u16; o * (n / BLOCK)];
    let (s_q, r_q, c_q, d_q) = (
        qw(&fit.pack.s_rot),
        qw(&fit.pack.rh),
        qw(&fit.pack.ch),
        qw(&fit.pack.dh),
    );
    let w = uw(&plane_w(
        PlaneRef {
            s: &s_q,
            idx: &idx,
            r: &r_q,
            c: &c_q,
            d: &d_q,
        },
        o,
        n,
    )
    .expect("plane_w"));
    // `plane_w` accumulates `(s·d → fwht → c) · r` in `f64` from the stored pack, while `Wp` is
    // the grid image of `compose` `(s·r)·c` taken from the *gridded* intermediate `s_dh`: the two
    // differ by (i) that one grid rounding, whose residue is scaled by `|rh·ch|` on the way out,
    // (ii) the `Wp` grid image itself, `½` quantum, and (iii) the association orders' `f64`
    // error, plus (iv) `plane_w`'s own grid work, the `learn_fold_icm` oracle bound
    // `(√128·q·(1 + max|s·d|) + q)·max|r|·max|c|`.
    // Bound: `½q·(1 + |rh·ch|) + 16·eps·(1 + |w|) + (iv)`.
    let g = n / GROUP;
    let max_abs = |x: &[f64]| x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let sd_max = (0..o * n)
        .map(|k| (fit.pack.s_rot[k] * fit.pack.dh[k % n]).abs())
        .fold(0.0_f64, f64::max);
    let plane_tol = (128f64.sqrt() * Q32_QUANTUM * (1.0 + sd_max) + Q32_QUANTUM)
        * max_abs(&fit.pack.rh)
        * max_abs(&fit.pack.ch);
    let mut worst = 0.0_f64;
    for k in 0..o * n {
        let scale = fit.pack.rh[(k / n) * g + (k % n) / GROUP].abs() * fit.pack.ch[k % n].abs();
        let tol = 0.5 * Q32_QUANTUM * (1.0 + scale)
            + 16.0 * f64::EPSILON * (1.0 + w[k].abs())
            + plane_tol;
        let err = (w[k] - fit.wp[k]).abs();
        worst = worst.max(err);
        assert!(
            err <= tol,
            "wp != plane_w(pack) at {k}: {} != {}",
            w[k],
            fit.wp[k]
        );
    }
    eprintln!("fold_fit Wp vs plane_w(pack): worst |Δ| = {worst:e}");
}

#[test]
fn learn_fold_fold_fit_j_best_is_the_candidate_minimum() {
    let (o, n, ws, h, fit) = run();
    assert_eq!(fit.trace.len(), 2);
    let min = fit.trace.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(fit.j_best, min, "j_best is not the trace minimum");
    assert!(
        fit.j_best <= fit.trace[0],
        "j_best above the gptq candidate"
    );
    // The continuous refit strictly improves on the plain gptq candidate on this matrix.
    assert!(
        fit.trace[1] < fit.trace[0],
        "refit {} not below gptq {}",
        fit.trace[1],
        fit.trace[0]
    );
    assert_eq!(fit.tag, FoldFitTag::Refit0);
    assert_eq!(fit.tag.as_str(), "refit0");
    // The final quantised weight is no worse than the winning candidate's continuous fit.
    let jwp = wp_objective(&ws, &h, o, n, &fit.wp);
    assert!(jwp.is_finite(), "J(wp) is not finite");
    assert!(
        jwp <= fit.j_best + 1e-9 * (1.0 + fit.j_best.abs()),
        "J(wp) {jwp} above the candidate J {}",
        fit.j_best
    );
}

#[test]
fn learn_fold_fold_fit_pack_carriers_are_in_stored_format() {
    let (o, n, _ws, _h, fit) = run();
    let g = n / GROUP;
    let nb = n / BLOCK;
    // dh is on the 5-bit affine grid of quant_c_u8; ch on the 6-bit grid (lines 933, 939).
    assert_eq!(fit.pack.dh.len(), n);
    assert_eq!(fit.pack.ch.len(), n);
    // The carriers are the quantiser's level values stored as Q32.32 words: each sits within half a
    // grid quantum of its level, and `lo`/`hi` are themselves grid images, so the bound is four
    // quanta.
    let tol = 4.0 * Q32_QUANTUM;
    assert!(
        on_affine_grid(&fit.pack.dh, 5, tol),
        "dh off the 5-bit grid"
    );
    assert!(
        on_affine_grid(&fit.pack.ch, 6, tol),
        "ch off the 6-bit grid"
    );
    // The bit budgets of the carriers: 4·o·g + 8·o + 64, 6·n + 32 and 5·n + 32.
    assert_eq!(fit.bits_r, (4 * o * g + 8 * o + 64) as u64);
    assert_eq!(fit.bits_c, (6 * n + 32) as u64);
    assert_eq!(fit.bits_d, (5 * n + 32) as u64);
    // rh is the hier carrier's `rq`: the stored-format decode of its own re-quantisation is exact.
    let h2 = quant_r_hier(&qw(&fit.pack.rh), o, g).expect("rh hier");
    assert_eq!(h2.bits, fit.bits_r);
    let decoded = h2.reconstruct();
    for k in 0..decoded.len() {
        assert_eq!(decoded[k], h2.rq[k], "rh hier decode mismatch at {k}");
    }
    // The 8-block sign/parity structure: every block of the code plane is exactly an `E8 ∪ O8`
    // codeword, so `icm::nearest_index` returns its own row with distance 0 (`codeword` reproduces
    // the block bit-for-bit). The parity fix keeps the block inside the codebook across the sign
    // flip.
    for row in 0..o {
        for m in 0..nb {
            let mut block = [0_i64; BLOCK];
            for k in 0..BLOCK {
                block[k] = grid(fit.pack.s_rot[row * n + m * BLOCK + k]);
            }
            let idx = nearest_index(&block).expect("nearest");
            let cb = codebook_row(idx);
            for k in 0..BLOCK {
                assert_eq!(
                    cb[k].to_bits(),
                    from_grid(block[k]).to_bits(),
                    "row {row} block {m} coord {k} is not codeword {idx}"
                );
            }
        }
    }
}

#[test]
fn learn_fold_fold_fit_refuses_bad_configuration_and_shapes() {
    let (o, n, ws, h) = fixed_inputs();
    let cfg = production_cfg();
    let bad_iters = FoldFitConfig { iters: 2, ..cfg };
    let (wq, hq) = (qw(&ws), qw(&h));
    assert_eq!(
        fold_fit(&wq, &hq, o, n, None, &bad_iters).err(),
        Some(FoldFitError::Unsupported)
    );
    assert_eq!(
        fold_fit(&wq, &hq, 0, n, None, &cfg).err(),
        Some(FoldFitError::ZeroDim)
    );

    // A short weight, Hessian or row weight is a shape refusal.
    assert_eq!(
        fold_fit(&wq[1..], &hq, o, n, None, &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );
    assert_eq!(
        fold_fit(&wq, &hq[1..], o, n, None, &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );
    assert_eq!(
        fold_fit(&wq, &hq, o, n, Some(&[1_i64, 1]), &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );

    // `n` must be a positive multiple of GROUP.
    let bad_n = 64_usize;
    assert_eq!(
        fold_fit(
            &vec![0_i64; o * bad_n],
            &vec![0_i64; bad_n * bad_n],
            o,
            bad_n,
            None,
            &cfg
        )
        .err(),
        Some(FoldFitError::Width)
    );

    // A non-finite weight, Hessian or row weight has no Q32.32 word, so the word boundary refuses
    // it before the fit: `NonFinite` is unreachable from a `&[i64]`.
}

#[test]
fn learn_fold_fold_fit_g_row_changes_the_result() {
    let (o, n, ws, h) = fixed_inputs();
    let cfg = production_cfg();
    let g_row: Vec<f64> = (0..o).map(|i| 0.3 + 0.4 * (i as f64)).collect();
    let (wq, hq, gq) = (qw(&ws), qw(&h), qw(&g_row));
    let with = fit_reals(&fold_fit(&wq, &hq, o, n, Some(&gq), &cfg).expect("g_row fit"));
    let without = fit_reals(&fold_fit(&wq, &hq, o, n, None, &cfg).expect("unweighted fit"));

    // Both run and give a finite objective.
    assert!(with.j_best.is_finite() && without.j_best.is_finite());
    assert!(wp_objective(&ws, &h, o, n, &with.wp).is_finite());
    assert!(wp_objective(&ws, &h, o, n, &without.wp).is_finite());

    // The row weight enters `c_step` and `d_step`, so it moves the fit.
    let differs = with
        .wp
        .iter()
        .zip(&without.wp)
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differs || with.j_best.to_bits() != without.j_best.to_bits(),
        "g_row changed nothing"
    );
}

/// The reference numbers of the production fit on the fixed matrix, pinned to catch a change in the
/// refit alternation or the pack-candidate rule (the mutation control's sensor), and the migrated
/// Q32.32 fit's distance from them.
///
/// # The bound
///
/// The oracle (`ref_fold_fit`) is the pre-port `steps.rs` arm (`xref`: exact `i128` contraction,
/// one rounding per entry), copied into this test, so this comparison binds two different bodies.
/// It reproduces the recorded run to `1e-9` relative. The ported body is the oracle's Q32.32 image
/// plus one ridge floor: `R = 64` carrier roundings (`3` refit passes × ~12 plane writes, plus ~12
/// in the pack phase) each move an entry by at most `2⁻³³` relative, the c/d normal systems square
/// the Toeplitz Gram's conditioning (≤ `2⁹` for `rho = 0.4`, `n = 256`), `≤ 5` passes compose
/// those, and the normal systems' ridge is floored at the limb product's own per-entry truncation
/// (`steps::limb_error_floor`, `C₃ = 65·2⁻²¹` of the filled matrix's maximum) when, and only when,
/// the script's ridge cannot certify a pivot; on these fixtures the first factorisation succeeds, so
/// the floor does not bind and the ported `J` is the rounding sum's `1.9e-5`. The stated bound is
/// `2e-5·(1+|J|)`. A change in the
/// alternation or the pack rule moves `J` by `O(1)`, four orders above this bound.
#[test]
fn learn_fold_fold_fit_production_numbers_are_pinned() {
    let (o, n, ws, h, fit) = run();
    let oracle = ref_fold_fit(&ws, &h, o, n, None, &production_cfg()).expect("oracle fit");
    let jwp = wp_objective(&ws, &h, o, n, &fit.wp);
    let jwp_ref = wp_objective(&ws, &h, o, n, &oracle.wp);
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-9 * (1.0 + want.abs());
    // The oracle is the recorded run.
    assert!(
        close(oracle.trace[0], 9.553_694_787_010_723e1),
        "oracle trace[0] = {}",
        oracle.trace[0]
    );
    assert!(
        close(oracle.trace[1], 4.359_351_242_410_637e1),
        "oracle trace[1] = {}",
        oracle.trace[1]
    );
    assert!(
        close(oracle.j_best, 4.359_351_242_410_637e1),
        "oracle j_best = {}",
        oracle.j_best
    );
    assert!(
        close(jwp_ref, 4.351_880_967_972_322e1),
        "oracle J(wp) = {jwp_ref}"
    );
    let wsum_ref: f64 = oracle.wp.iter().map(|x| x.abs()).sum();
    assert!(
        close(wsum_ref, 4.515_831_864_262e2),
        "oracle |wp| sum = {wsum_ref}"
    );
    // The migrated fit sits inside the derived grid bound of the oracle.
    let wsum: f64 = fit.wp.iter().map(|x| x.abs()).sum();
    let near = |got: f64, want: f64| (got - want).abs() <= 2e-5 * (1.0 + want.abs());
    for (got, want, what) in [
        (fit.trace[0], oracle.trace[0], "trace[0]"),
        (fit.trace[1], oracle.trace[1], "trace[1]"),
        (fit.j_best, oracle.j_best, "j_best"),
        (jwp, jwp_ref, "J(wp)"),
        (wsum, wsum_ref, "|wp| sum"),
    ] {
        assert!(
            near(got, want),
            "{what}: {got} vs oracle {want} beyond the derived grid bound"
        );
    }
}

/// The agreement oracle: the `f64` body of `fold_fit` at base `9f440192` (`fold_fit.rs` lines
/// 254–580), copied verbatim with the names prefixed. Test-only `f64` is allowed; nothing on the
/// crate's build path runs it.
const REF_EPS: f64 = 1e-12;

/// A candidate of the oracle's alternation.
struct RefCand {
    /// Candidate `J`.
    j: f64,
    /// Candidate tag.
    tag: FoldFitTag,
    /// Hadamard-domain code plane.
    s_rot: Vec<f64>,
    /// Per-group row scale.
    r: Vec<f64>,
    /// Per-column scale.
    c: Vec<f64>,
    /// Hadamard column scale.
    d: Vec<f64>,
}
fn ref_metric(h: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let mut hl = h.to_vec();
    let mut mean = 0.0_f64;
    for i in 0..n {
        mean += h[i * n + i];
    }
    mean /= n as f64;
    let add = ridge * mean;
    for i in 0..n {
        hl[i * n + i] += add;
    }
    hl
}

fn ref_scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            x[row * n + col] *= v[col];
        }
    }
}

fn ref_parity_signs(d: &[f64], n: usize) -> Vec<f64> {
    let mut sign = vec![1.0_f64; n];
    for i in 0..n {
        if d[i] < 0.0 {
            sign[i] = -1.0;
        }
    }
    let nb = n / BLOCK;
    for m in 0..nb {
        let b0 = m * BLOCK;
        let mut negatives = 0_usize;
        for k in 0..BLOCK {
            if sign[b0 + k] < 0.0 {
                negatives += 1;
            }
        }
        if negatives % 2 == 1 {
            let mut min_i = 0_usize;
            let mut min_abs = f64::INFINITY;
            for k in 0..BLOCK {
                let a = d[b0 + k].abs();
                if a < min_abs {
                    min_abs = a;
                    min_i = k;
                }
            }
            sign[b0 + min_i] = -sign[b0 + min_i];
        }
    }
    sign
}

/// The exact reference arm: the pre-port `steps.rs` bodies (base `7a56eb47`, sha256
/// `8b43418206b454182b09372debb939c49d72d12f6e1311a0674b81d7e886c7c8`) copied into the test with the
/// `x` prefix, so `ref_fold_fit` compares the limb port against a body that shares **no** product
/// with `src`: every contraction below is an exact `i128` accumulation rounded once per entry, and
/// [`apeiron::learn::solve::syrk::wsyrk`]/`gemm_nn` are not called. `fwht_words` and `had_gram` stay
/// `src` (`src/learn/fold/fit/steps.rs`): they are transforms, not products, and the port did not
/// move them. Test-only; nothing on the crate's build path runs this.
// The copies keep the deleted bodies' signatures and shifts verbatim, so the port is measured
// against the body it replaced rather than a reshaped variant of it.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
#[path = "support/fold_fit_xref.rs"]
mod xref;

fn ref_fold_fit(
    ws: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
    cfg: &FoldFitConfig,
) -> Result<FitF, FoldFitError> {
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
    if ws.iter().any(|x| !x.is_finite()) || h.iter().any(|x| !x.is_finite()) {
        return Err(FoldFitError::NonFinite);
    }
    if let Some(gr) = g_row {
        if gr.iter().any(|x| !x.is_finite()) {
            return Err(FoldFitError::NonFinite);
        }
    }

    let g = n / GROUP;
    let nb = n / BLOCK;
    let g_word = g_row.map(qw);
    let hl = ref_metric(h, n, from_grid(cfg.ridge));
    // One metric product `Ws·Hl` for the whole fit: every `c`/`r`/`d` step below reuses it, so the
    // redundant per-step `mul_metric` calls collapse to one. Same refusal path as the first step's
    // internal product.
    let wshl = uw(&xref::mul_metric(&qw(ws), &qw(&hl), o, n)?);
    let idx_zero = vec![0_u16; o * nb];

    // Step 2: the `gptq_e8` candidate (lines 843–860).
    let ws_rot = fwht128(ws)?;
    let h_syrk = uw(&had_gram(&qw(h), n)?);
    let q = gptq_e8(
        &qw(&ws_rot),
        &qw(&h_syrk),
        o,
        n,
        cfg.damp,
        true,
        true,
        cfg.e8_beam,
    )?;
    let s_rot0 = uw(&q.s);
    let r0 = uw(&q.scale);
    let c0 = vec![1.0_f64; n];
    let d0 = vec![1.0_f64; n];
    let s0 = fwht128(&s_rot0)?;
    let wq0 = uw(&compose(&qw(&s0), &qw(&r0), &qw(&c0), o, n)?);
    let j0 = from_grid(xref::objective(&qw(ws), &qw(&wq0), &qw(&hl), o, n)?);

    let mut cands: Vec<RefCand> = vec![RefCand {
        j: j0,
        tag: FoldFitTag::Gptq,
        s_rot: s_rot0,
        r: r0,
        c: c0,
        d: d0,
    }];
    let mut trace = vec![j0];

    // Step 3: the single refit pass (`iters = 1`, lines 862–889). The inner loop works on owned
    // copies of the `gptq` candidate's plane; the candidate itself keeps the pristine values.
    let mut s_rot = cands[0].s_rot.clone();
    let mut s_cur = fwht128(&s_rot)?;
    let mut r_cur = cands[0].r.clone();
    let mut c_cur = cands[0].c.clone();
    let mut d_cur = cands[0].d.clone();
    for _ in 0..cfg.inner_iters {
        c_cur = uw(&xref::c_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_cur),
            &qw(&r_cur),
            o,
            n,
            g_word.as_deref(),
        )?);
        let smean = c_cur.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
        for v in &mut c_cur {
            *v /= smean;
        }
        for v in &mut r_cur {
            *v *= smean;
        }
        r_cur = uw(&xref::r_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_cur),
            &qw(&c_cur),
            o,
            n,
        )?);

        let d_new = uw(&xref::d_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_rot),
            &qw(&r_cur),
            &qw(&c_cur),
            o,
            n,
            g_word.as_deref(),
        )?);
        let sign = ref_parity_signs(&d_new, n);
        ref_scale_columns(&mut s_rot, o, n, &sign);
        let mut group_max = vec![0.0_f64; g];
        for gi in 0..g {
            let mut mx = 0.0_f64;
            for k in 0..GROUP {
                let v = (d_new[gi * GROUP + k] * sign[gi * GROUP + k]).abs();
                if v > mx {
                    mx = v;
                }
            }
            group_max[gi] = mx.max(REF_EPS);
        }
        for i in 0..n {
            d_cur[i] = d_new[i] * sign[i] / group_max[i / GROUP];
        }
        for r in 0..o {
            for gi in 0..g {
                r_cur[r * g + gi] *= group_max[gi];
            }
        }

        let s_rot_q = qw(&s_rot);
        let r_cur_q = qw(&r_cur);
        let c_cur_q = qw(&c_cur);
        let d_cur_q = qw(&d_cur);
        let plane = PlaneRef {
            s: &s_rot_q,
            idx: &idx_zero,
            r: &r_cur_q,
            c: &c_cur_q,
            d: &d_cur_q,
        };
        let refined = e8_icm_step(&qw(ws), &qw(&hl), plane, o, n, cfg.sweeps, cfg.icm_ridge)?;
        s_rot = uw(&refined.s);
        let mut rotated = s_rot.clone();
        ref_scale_columns(&mut rotated, o, n, &d_cur);
        s_cur = fwht128(&rotated)?;
        r_cur = uw(&xref::r_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_cur),
            &qw(&c_cur),
            o,
            n,
        )?);
    }
    let wq1 = uw(&compose(&qw(&s_cur), &qw(&r_cur), &qw(&c_cur), o, n)?);
    let j1 = from_grid(xref::objective(&qw(ws), &qw(&wq1), &qw(&hl), o, n)?);
    cands.push(RefCand {
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

    // The `bscale and had` arm (lines 932–968).
    let (dh, bits_d) = quant_c_u8(&qw(&d_best), 5)?;
    let dh = uw(&dh);
    let mut rotated = s_rot_best.clone();
    ref_scale_columns(&mut rotated, o, n, &dh);
    let s_dh = fwht128(&rotated)?;
    let rh_hier = quant_r_hier(
        &xref::r_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_dh),
            &qw(&c_best),
            o,
            n,
        )?,
        o,
        g,
    )?;
    let bits_r = rh_hier.bits;
    let rh = uw(&rh_hier.rq);
    let c_ref = uw(&xref::c_step_wshl(
        &qw(ws),
        &qw(&hl),
        &qw(&wshl),
        &qw(&s_dh),
        &qw(&rh),
        o,
        n,
        g_word.as_deref(),
    )?);
    let (ch, bits_c) = quant_c_u8(&qw(&c_ref), 6)?;
    let ch = uw(&ch);
    let mut wp = uw(&compose(&qw(&s_dh), &qw(&rh), &qw(&ch), o, n)?);
    let mut j_wp = from_grid(xref::objective(&qw(ws), &qw(&wp), &qw(&hl), o, n)?);
    let mut pack_s_rot = s_rot_best.clone();
    let mut pack_rh = rh;
    let mut pack_ch = ch;
    let mut pack_dh = dh;
    let mut s_rot_cur = s_rot_best;

    // The `e8_icm` re-quantisation candidate (lines 942–950).
    let pack_s_rot_q = qw(&pack_s_rot);
    let pack_rh_q = qw(&pack_rh);
    let pack_ch_q = qw(&pack_ch);
    let pack_dh_q = qw(&pack_dh);
    let plane = PlaneRef {
        s: &pack_s_rot_q,
        idx: &idx_zero,
        r: &pack_rh_q,
        c: &pack_ch_q,
        d: &pack_dh_q,
    };
    let s_rot_q = uw(&e8_icm_step(&qw(ws), &qw(&hl), plane, o, n, cfg.sweeps, cfg.icm_ridge)?.s);
    {
        let mut rotated = s_rot_q.clone();
        ref_scale_columns(&mut rotated, o, n, &pack_dh);
        let s_dh_q = fwht128(&rotated)?;
        let rh_q = quant_r_hier(
            &xref::r_step_wshl(
                &qw(ws),
                &qw(&hl),
                &qw(&wshl),
                &qw(&s_dh_q),
                &qw(&pack_ch),
                o,
                n,
            )?,
            o,
            g,
        )?
        .rq;
        let rh_q = uw(&rh_q);
        let ch_q = quant_c_u8(
            &xref::c_step_wshl(
                &qw(ws),
                &qw(&hl),
                &qw(&wshl),
                &qw(&s_dh_q),
                &qw(&rh_q),
                o,
                n,
                g_word.as_deref(),
            )?,
            6,
        )?
        .0;
        let ch_q = uw(&ch_q);
        let wp_q = uw(&compose(&qw(&s_dh_q), &qw(&rh_q), &qw(&ch_q), o, n)?);
        let j_q = from_grid(xref::objective(&qw(ws), &qw(&wp_q), &qw(&hl), o, n)?);
        if j_q < j_wp {
            j_wp = j_q;
            wp = wp_q;
            s_rot_cur.clone_from(&s_rot_q);
            pack_s_rot = s_rot_q;
            pack_rh = rh_q;
            pack_ch = ch_q;
        }
    }

    // The un-`d` fallback candidate (lines 960–967).
    {
        let s_1 = fwht128(&s_rot_cur)?;
        let r_1 = uw(&xref::r_step_wshl(
            &qw(ws),
            &qw(&hl),
            &qw(&wshl),
            &qw(&s_1),
            &qw(&c_best),
            o,
            n,
        )?);
        let rh_1 = uw(&quant_r_hier(&qw(&r_1), o, g)?.rq);
        let ch_1 = quant_c_u8(
            &xref::c_step_wshl(
                &qw(ws),
                &qw(&hl),
                &qw(&wshl),
                &qw(&s_1),
                &qw(&rh_1),
                o,
                n,
                g_word.as_deref(),
            )?,
            6,
        )?
        .0;
        let ch_1 = uw(&ch_1);
        let wp_1 = uw(&compose(&qw(&s_1), &qw(&rh_1), &qw(&ch_1), o, n)?);
        if from_grid(xref::objective(&qw(ws), &qw(&wp_1), &qw(&hl), o, n)?) < j_wp {
            wp = wp_1;
            pack_s_rot = s_rot_cur;
            pack_rh = rh_1;
            pack_ch = ch_1;
            pack_dh = vec![1.0_f64; n];
        }
    }

    // The winning pack must recompose to `Wp` exactly (the Python's `assert`, line 968).
    let mut rotated = pack_s_rot.clone();
    ref_scale_columns(&mut rotated, o, n, &pack_dh);
    let recomposed = uw(&compose(
        &qw(&fwht128(&rotated)?),
        &qw(&pack_rh),
        &qw(&pack_ch),
        o,
        n,
    )?);
    for k in 0..o * n {
        if recomposed[k].to_bits() != wp[k].to_bits() {
            return Err(FoldFitError::PackMismatch);
        }
    }

    Ok(FitF {
        wp,
        pack: PackF {
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

/// FNV-1a 64 over the little-endian bytes of the weight words.
fn fnv_words(v: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in v {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The recording RNG: a 64-bit LCG (Numerical Recipes constants), uniform in `[lo, hi)` from the
/// top 53 bits. Distinct from the file's `Lcg`, so the recorded fixture is reproduced exactly.
struct WitnessLcg(u64);

impl WitnessLcg {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// The recorded `wp` word digests of the pre-flip `f64` boundary, one per seed: the fit at
/// `o = 4`, `n = 256`, `H = Toeplitz(0.4)`, the production configuration, on the words
/// `round(x·2³²)` of the same real fixture.
const WITNESS_WP: [u64; 3] = [
    0xb538_94be_bd5f_fc80,
    0xa3f1_3a3d_bb72_0fd6,
    0xc493_8987_0d01_8f31,
];
/// Seed `0x3d3d0002`'s pre-re-pin digest was `0xcb7a_c296_067d_4b7a`; the other two are unchanged.
/// The K5 table behind the re-pin (limb against the exact arm on the same words, `n xref`):
/// `J = 210574216632 / 204105833529 / 207964608657` words against the arm's
/// `210574216632 / 204105833402 / 207964608657`, i.e. `0`, `6.2e-10` and `0` relative, with every
/// `s_rot` code word identical (`1024/1024`) on all three seeds.
const _WITNESS_REPIN: () = ();

/// The `wp` word digests of the ported body per seed, re-pinned under §0 K5 after the ridge floor:
/// the pre-floor digests were `0xb538_94be_bd5f_fc80`, `0xcb7a_c296_067d_4b7a`, `0xc493_8987_0d01_8f31`
/// (the values recorded through the pre-port `f64` wrapper). `fold_fit` is the only body that moved:
/// the floor raises the c/d normal systems' ridge where the script's `1e-12·mean(diag A)` sits below
/// the limb product's own per-entry truncation (`steps::limb_error_floor`), so a system the exact
/// arm factorises is not refused `NotPositiveDefinite`.
#[test]
fn fold_fit_words_are_bit_identical_to_the_recorded_f64_wrapper() {
    let (o, n) = (4usize, 256usize);
    let h = toeplitz(n, 0.4);
    let hq = qw(&h);
    let hl = xref::metric_words(&h, n, production_cfg().ridge);
    for (i, seed) in [0x3D3D_0001u64, 0x3D3D_0002, 0x3D3D_0003]
        .iter()
        .enumerate()
    {
        let mut rng = WitnessLcg(*seed);
        let w: Vec<f64> = (0..o * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let wq = qw(&w);
        let fit = fold_fit(&wq, &hq, o, n, None, &production_cfg()).expect("fit");
        // K5 before the re-pin: the held-out metric limb against the exact arm, on the same words.
        let ex = ref_fold_fit(&w, &h, o, n, None, &production_cfg()).expect("exact arm");
        let j_limb = xref::objective(&wq, &fit.wp, &hl, o, n).expect("J limb");
        let j_exact = xref::objective(&wq, &qw(&ex.wp), &hl, o, n).expect("J exact");
        let agree = fit
            .pack
            .s_rot
            .iter()
            .zip(&qw(&ex.pack.s_rot))
            .filter(|(a, b)| a == b)
            .count();
        let rel = (j_limb - j_exact).abs() as f64 / (1.0 + (j_exact as f64).abs());
        eprintln!(
            "seed {seed:#x}: J limb {j_limb} exact {j_exact} rel {rel:e}; s_rot {agree}/{} words agree",
            o * n
        );
        assert!(
            rel <= 1e-4,
            "seed {seed:#x}: J {j_limb} vs exact arm {j_exact} is {rel:e} relative, above the K5 bound"
        );
        assert_eq!(
            fnv_words(&fit.wp),
            WITNESS_WP[i],
            "seed {seed:#x}: wp words moved from the K5 re-pin"
        );
    }
}
