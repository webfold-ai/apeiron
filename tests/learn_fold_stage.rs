//! `tests/learn_fold_stage.rs` — the per-stage sink-gated statistics and the `fold_gqa` split
//! intra/inter ridge solve (`src/learn/fold/layer/stage.rs`).
//!
//! Every migrated function is compared with the `f64` oracle it replaced (`ref_*` below, the
//! reference loop without the grid) on fixed LCG inputs, inside a derived grid bound. The `Q32.32`
//! quantum is `q = 2⁻³²`; a Gram entry rounds its inputs (≤ q/2 each) and its `rows`-term
//! contraction once (≤ q/2), and the linear-weight pass adds one rounded product per entry, so the
//! first-order error is `3·(rows+1)·q`; the tests assert `8·(rows+1)·q` (a factor of ~2.7 of
//! headroom for the second-order term) and print the observed maximum. The solve propagates that
//! error through `Cholesky`, whose amplification is `κ(H) = λmax/λmin`; for the fixtures
//! (`rows ≥ 2n`, `|x| ≤ 1`, `w ∈ [0.45, 1.35]`) `κ(H) ≤ 4`, so the asserted bound is `64·q`.
//!
//! Positive: two slabs fold to the same accumulator as their concatenation (one word on the `i64`
//! fields); `Hm_w = Hm` under unit weights (exactly, `w = 1` is a grid identity); `Hm_w = (Xw)ᵀ(Xw)`
//! with the weight applied linearly; the even-count lower-median rule; the `> thresh·r_med` token
//! weight; a solve with `Cs = Cts = H_solve·W0ᵀ` leaves `W0` unchanged; the solve satisfies the ridge
//! normal equations; the sink clamp caps a row at `τ·‖W0_row‖`; `ridge_solve` agrees with the old
//! `f64` `solve_spd` path to relative `1e-7` on a wide diagonal range (`1e-3..1e3`) and under
//! `1e-6`/`1e10` scaling; `metric` and `bias_update` agree with their `f64` oracles.
//! Refusals: shape mismatch, non-SPD, non-finite, empty median and missing `Cts`. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64, and the test kernels index \
              row-major `[r, c]` buffers by `i/j/k/o` against the spec's arithmetic; the exact float \
              comparisons are on values the LCG and the lowering produce bit-for-bit."
)]

use apeiron::learn::fold::layer::stage::{
    bias_update, metric, ridge_solve, sink_token_weights, solve_stage, SolveParams, StageAccum,
    StageError,
};
use apeiron::learn::fold::stats::hocapture::Kind;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

#[path = "support/ridge_oracle.rs"]
mod ridge_oracle;

/// The `Q32.32` grid quantum, `2⁻³²`.
const Q: f64 = 1.0 / 4_294_967_296.0;

/// The `Q32.32` word of `1.0`.
const ONE: i64 = 1 << 32;

/// `x` as a `Q32.32` word (the boundary the word APIs take).
fn q(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("grid")
}

/// `x` as words.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| q(v)).collect()
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

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn rand_mat(lcg: &mut Lcg, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|_| lcg.sym()).collect()
}

/// Row-major `[r, k] · [k, c]` product.
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

fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// The `f64` view of a `Q32.32` word buffer.
fn words(a: &[i64]) -> Vec<f64> {
    a.iter().map(|&v| from_grid(v)).collect()
}

/// `max |from_grid(word) − oracle|` and the largest oracle magnitude, for a printed receipt.
fn grid_error(a: &[i64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut err = 0.0f64;
    let mut mag = 0.0f64;
    for (&w, &y) in a.iter().zip(b) {
        err = err.max((from_grid(w) - y).abs());
        mag = mag.max(y.abs());
    }
    (err, mag)
}

/// The largest `|entry|` of a slice.
fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}

// ---------------------------------------------------------------------------
// The f64 oracles: the reference arithmetic, without the Q32.32 grid
// ---------------------------------------------------------------------------

/// The reference per-stage accumulator: the `fold_transfer.py:1541-1648` loop in `f64`.
struct Ref {
    n: usize,
    outs: Vec<usize>,
    with_ts: Vec<bool>,
    hm: Vec<f64>,
    hm_w: Vec<f64>,
    cs: Vec<Vec<f64>>,
    cts: Vec<Vec<f64>>,
    x_sum: Vec<f64>,
    y_sum: Vec<Vec<f64>>,
    n_toks: f64,
}

impl Ref {
    fn new(n: usize, outs: &[usize], with_ts: &[bool]) -> Self {
        Self {
            n,
            outs: outs.to_vec(),
            with_ts: with_ts.to_vec(),
            hm: vec![0.0; n * n],
            hm_w: vec![0.0; n * n],
            cs: outs.iter().map(|&o| vec![0.0; n * o]).collect(),
            cts: outs
                .iter()
                .zip(with_ts)
                .map(|(&o, &ts)| if ts { vec![0.0; n * o] } else { Vec::new() })
                .collect(),
            x_sum: vec![0.0; n],
            y_sum: outs.iter().map(|&o| vec![0.0; o]).collect(),
            n_toks: 0.0,
        }
    }

    /// `Σ XᵀX`, `Σ (Xw)ᵀ(Xw)`, the weighted cross-SyrkAcc and the unweighted sums, in `f64`.
    fn add_slab(&mut self, x: &[f64], w: &[f64], ys: &[&[f64]], y_ts: &[Option<&[f64]>]) {
        let rows = w.len();
        let n = self.n;
        for i in 0..n {
            for j in 0..n {
                let mut a = 0.0f64;
                let mut b = 0.0f64;
                for t in 0..rows {
                    a += x[t * n + i] * x[t * n + j];
                    b += (x[t * n + i] * w[t]) * (x[t * n + j] * w[t]);
                }
                self.hm[i * n + j] += a;
                self.hm_w[i * n + j] += b;
            }
            self.x_sum[i] += (0..rows).map(|t| x[t * n + i]).sum::<f64>();
        }
        for j in 0..ys.len() {
            let out = self.outs[j];
            for i in 0..n {
                for o in 0..out {
                    let mut cs = 0.0f64;
                    let mut cts = 0.0f64;
                    for t in 0..rows {
                        let xw = x[t * n + i] * w[t];
                        cs += xw * (ys[j][t * out + o] * w[t]);
                        if let Some(ts) = y_ts[j] {
                            cts += xw * (ts[t * out + o] * w[t]);
                        }
                    }
                    self.cs[j][i * out + o] += cs;
                    if self.with_ts[j] {
                        self.cts[j][i * out + o] += cts;
                    }
                }
            }
            for o in 0..out {
                self.y_sum[j][o] += (0..rows).map(|t| ys[j][t * out + o]).sum::<f64>();
            }
        }
        self.n_toks += rows as f64;
    }
}

/// `Hr = H` with `diag += shift`, then `Hr·X = C`, returned as `Xᵀ` (`[out, n]`).
fn solve_with(h: &[f64], c: &[f64], n: usize, out: usize, shift: f64) -> Vec<f64> {
    let mut hr = h.to_vec();
    for i in 0..n {
        hr[i * n + i] += shift;
    }
    let x = ridge_oracle::solve_spd(&hr, c, n, out).unwrap();
    let mut out_t = vec![0.0f64; out * n];
    for i in 0..n {
        for o in 0..out {
            out_t[o * n + i] = x[i * out + o];
        }
    }
    out_t
}

/// The mean of the diagonal, the ridge shift's scale.
fn diag_mean(h: &[f64], n: usize) -> f64 {
    (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64
}

/// The reference `solve_stage`: the `fold_transfer.py:1652-1682` split, in `f64`.
fn ref_solve_stage(r: &Ref, w0: &[&[f64]], kinds: &[Kind], p: &SolveParams) -> Vec<Vec<f64>> {
    let n = r.n;
    let mut ws = Vec::new();
    for j in 0..r.outs.len() {
        let out = r.outs[j];
        let mut c0 = vec![0.0f64; n * out];
        for i in 0..n {
            for o in 0..out {
                c0[i * out + o] = (0..n).map(|t| r.hm_w[i * n + t] * w0[j][o * n + t]).sum();
            }
        }
        let scale = n as f64 / p.n_ref;
        let mean = diag_mean(&r.hm_w, n);
        let mut ws_j = vec![0.0f64; out * n];
        if matches!(kinds[j], Kind::O | Kind::Down) {
            let d1: Vec<f64> = (0..n * out).map(|k| r.cts[j][k] - c0[k]).collect();
            let d2: Vec<f64> = (0..n * out).map(|k| r.cs[j][k] - r.cts[j][k]).collect();
            let intra = solve_with(&r.hm_w, &d1, n, out, p.ridge_intra * scale * mean);
            let inter = solve_with(&r.hm_w, &d2, n, out, p.ridge_inter * scale * mean);
            let mut dw = vec![0.0f64; out * n];
            for o in 0..out {
                for i in 0..n {
                    dw[o * n + i] = intra[o * n + i] + p.inter_gamma * inter[o * n + i];
                }
                let rn = (0..n)
                    .map(|i| dw[o * n + i] * dw[o * n + i])
                    .sum::<f64>()
                    .sqrt();
                let w0rn = (0..n)
                    .map(|i| w0[j][o * n + i] * w0[j][o * n + i])
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-6);
                if rn > p.sink_tau * w0rn {
                    let s = p.sink_tau * w0rn / rn;
                    for i in 0..n {
                        dw[o * n + i] *= s;
                    }
                }
                for i in 0..n {
                    ws_j[o * n + i] = w0[j][o * n + i] + dw[o * n + i];
                }
            }
        } else {
            let direct: Vec<f64> = (0..n * out).map(|k| r.cs[j][k] - c0[k]).collect();
            let step = solve_with(&r.hm_w, &direct, n, out, p.ridge_w0_gqa * scale * mean);
            for o in 0..out {
                for i in 0..n {
                    ws_j[o * n + i] = w0[j][o * n + i] + p.gin * step[o * n + i];
                }
            }
        }
        ws.push(ws_j);
    }
    ws
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Two slabs of the same geometry: `(x, w, ys, y_ts)` per slab.
fn two_slab_fixture(
    seed: u64,
    n: usize,
    outs: &[usize],
    with_ts: &[bool],
    rows: [usize; 2],
    wscale: f64,
) -> Vec<Slab> {
    let mut lcg = Lcg::new(seed);
    rows.iter()
        .map(|&r| {
            let x = rand_mat(&mut lcg, r, n);
            let w: Vec<f64> = if wscale == 1.0 {
                vec![1.0; r]
            } else {
                (0..r).map(|_| wscale * (0.5 + lcg.sym().abs())).collect()
            };
            let ys: Vec<Vec<f64>> = outs.iter().map(|&o| rand_mat(&mut lcg, r, o)).collect();
            let y_ts: Vec<Option<Vec<f64>>> = outs
                .iter()
                .zip(with_ts)
                .map(|(&o, &ts)| {
                    if ts {
                        Some(rand_mat(&mut lcg, r, o))
                    } else {
                        None
                    }
                })
                .collect();
            (x, w, ys, y_ts)
        })
        .collect()
}

type Slab = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Option<Vec<f64>>>);

fn slab_refs(s: &Slab) -> (Vec<&[f64]>, Vec<Option<&[f64]>>) {
    (
        s.2.iter().map(Vec::as_slice).collect(),
        s.3.iter().map(|o| o.as_deref()).collect(),
    )
}

fn fold_accum(n: usize, outs: &[usize], with_ts: &[bool], slabs: &[Slab]) -> StageAccum {
    let mut acc = StageAccum::new(n, outs, with_ts).unwrap();
    for s in slabs {
        let (x, w) = (qv(&s.0), qv(&s.1));
        let ys: Vec<Vec<i64>> = s.2.iter().map(|v| qv(v)).collect();
        let y_ts: Vec<Option<Vec<i64>>> = s.3.iter().map(|o| o.as_ref().map(|v| qv(v))).collect();
        let ys_r: Vec<&[i64]> = ys.iter().map(Vec::as_slice).collect();
        let yts_r: Vec<Option<&[i64]>> = y_ts.iter().map(|o| o.as_deref()).collect();
        acc.add_slab(&x, &w, &ys_r, &yts_r).unwrap();
    }
    acc
}

fn fold_ref(n: usize, outs: &[usize], with_ts: &[bool], slabs: &[Slab]) -> Ref {
    let mut r = Ref::new(n, outs, with_ts);
    for s in slabs {
        let (ys, y_ts) = slab_refs(s);
        r.add_slab(&s.0, &s.1, &ys, &y_ts);
    }
    r
}

// ---------------------------------------------------------------------------
// Accumulator tests
// ---------------------------------------------------------------------------

/// (a) `add_slab` agrees with the `f64` oracle on every accumulated field.
#[test]
fn add_slab_matches_f64_oracle() {
    let (n, outs, with_ts) = (4usize, [3usize, 2], [true, false]);
    let slabs = two_slab_fixture(0x51ab_cd01, n, &outs, &with_ts, [5, 4], 0.9);
    let acc = fold_accum(n, &outs, &with_ts, &slabs);
    let r = fold_ref(n, &outs, &with_ts, &slabs);
    let rows = 9.0f64;
    // Contraction: `rows` terms of two rounded factors plus one final rounding, times the linear
    // weight pass's extra rounding; `8·(rows+1)·q` is the asserted first-order bound with headroom.
    let bound = 8.0 * (rows + 1.0) * Q;
    let e_hm = grid_error(&acc.hm, &r.hm).0;
    let (e_hmw, m_hmw) = grid_error(&acc.hm_w, &r.hm_w);
    let e_cs0 = grid_error(&acc.cs[0], &r.cs[0]).0;
    let e_cs1 = grid_error(&acc.cs[1], &r.cs[1]).0;
    let e_cts = grid_error(&acc.cts[0], &r.cts[0]).0;
    let e_xs = grid_error(&acc.x_sum, &r.x_sum).0;
    let e_ys0 = grid_error(&acc.y_sum[0], &r.y_sum[0]).0;
    let e_ys1 = grid_error(&acc.y_sum[1], &r.y_sum[1]).0;
    println!(
        "add_slab: hm={e_hm:.3e} hm_w={e_hmw:.3e} (|.|max {m_hmw:.3e}) cs=({e_cs0:.3e},{e_cs1:.3e}) \
         cts={e_cts:.3e} x_sum={e_xs:.3e} y_sum=({e_ys0:.3e},{e_ys1:.3e}) bound={bound:.3e}"
    );
    assert!(e_hm <= bound, "hm {e_hm:e}");
    assert!(e_hmw <= bound, "hm_w {e_hmw:e}");
    assert!(e_cs0 <= bound && e_cs1 <= bound, "cs {e_cs0:e} {e_cs1:e}");
    assert!(e_cts <= bound, "cts {e_cts:e}");
    // A column sum rounds only at the slab boundary: one quantum per slab.
    assert!(e_xs <= 2.0 * Q, "x_sum {e_xs:e}");
    assert!(
        e_ys0 <= 2.0 * Q && e_ys1 <= 2.0 * Q,
        "y_sum {e_ys0:e} {e_ys1:e}"
    );
    assert_eq!(acc.n_toks as f64, r.n_toks);
}

/// (b) Folding two slabs equals folding their concatenation: the `i64` accumulators differ by at
/// most one rounding step (each slab rounds its own partial sum), the `f64` views by two, and the
/// token count is exact.
#[test]
fn two_slabs_equal_concatenation() {
    let (n, outs, with_ts) = (4usize, [3usize, 2], [true, false]);
    let slabs = two_slab_fixture(0x51ab_cd01, n, &outs, &with_ts, [3, 2], 1.0);
    let split = fold_accum(n, &outs, &with_ts, &slabs);
    let mut x = Vec::new();
    let mut w = Vec::new();
    let mut ys = vec![Vec::new(); outs.len()];
    let mut y_ts: Vec<Option<Vec<f64>>> = (0..outs.len()).map(|_| None).collect();
    for s in &slabs {
        x.extend_from_slice(&s.0);
        w.extend_from_slice(&s.1);
        for j in 0..outs.len() {
            ys[j].extend_from_slice(&s.2[j]);
            if let Some(src) = s.3[j].as_ref() {
                y_ts[j].get_or_insert_with(Vec::new).extend_from_slice(src);
            }
        }
    }
    let one = fold_accum(n, &outs, &with_ts, &[(x, w, ys, y_ts)]);
    let word_gap = |a: &[i64], b: &[i64]| -> i128 {
        a.iter().zip(b).fold(0i128, |m, (&x, &y)| {
            m.max((i128::from(x) - i128::from(y)).abs())
        })
    };
    for (a, b) in [
        (&split.hm_w, &one.hm_w),
        (&split.cs[0], &one.cs[0]),
        (&split.cs[1], &one.cs[1]),
        (&split.cts[0], &one.cts[0]),
    ] {
        assert!(word_gap(a, b) <= 1, "word gap {}", word_gap(a, b));
    }
    assert!(word_gap(&split.hm, &one.hm) <= 2);
    assert!(word_gap(&split.x_sum, &one.x_sum) <= 2);
    assert!(word_gap(&split.y_sum[0], &one.y_sum[0]) <= 2);
    assert!(word_gap(&split.y_sum[1], &one.y_sum[1]) <= 2);
    assert_eq!(split.n_toks, one.n_toks);
}

/// (c) With every `w_tok == 1`, `Hm_w` equals the unweighted `Hm` exactly: `1.0` is a `Q32.32`
/// identity and both entries round the same `i128` sum once.
#[test]
fn hm_w_equals_hm_under_unit_weights() {
    let (n, outs, with_ts) = (5usize, [2usize], [false]);
    let slabs = two_slab_fixture(11, n, &outs, &with_ts, [6, 0], 1.0);
    let acc = fold_accum(n, &outs, &with_ts, &slabs[..1]);
    assert_eq!(acc.hm_w, acc.hm);
}

/// (c') `Hm_w = (Xw)ᵀ(Xw)` with `Xw = X·w` applied linearly (not `√w`) — the oracle comparison
/// plus a sensitivity guard that `√w` is far outside the bound.
#[test]
fn hm_w_applies_weights_linearly() {
    let (n, outs, with_ts) = (5usize, [2usize], [false]);
    let slabs = two_slab_fixture(23, n, &outs, &with_ts, [7, 0], 0.9);
    let (x, w) = (&slabs[0].0, &slabs[0].1);
    let rows = w.len();
    let acc = fold_accum(n, &outs, &with_ts, &slabs[..1]);
    let expect = {
        let mut r = Ref::new(n, &outs, &with_ts);
        r.add_slab(x, w, &[&slabs[0].2[0]], &[None]);
        r.hm_w
    };
    let (err, _) = grid_error(&acc.hm_w, &expect);
    let bound = 8.0 * (rows + 1) as f64 * Q;
    println!("hm_w linear: err={err:.3e} bound={bound:.3e}");
    assert!(err <= bound, "hm_w {err:e}");
    // Sensitivity: with √w the matrix differs by far more than the bound.
    let mut xs = x.clone();
    for t in 0..rows {
        for i in 0..n {
            xs[t * n + i] *= w[t].sqrt();
        }
    }
    let sqrt_syrk = matmul(&transpose(&xs, rows, n), &xs, n, rows, n);
    assert!(max_abs_diff(&words(&acc.hm_w), &sqrt_syrk) > 1e-6);
}

// ---------------------------------------------------------------------------
// Sink-gate tests
// ---------------------------------------------------------------------------

/// The reference token weights: the `fold_transfer.py:1592-1597` rule in `f64`, same median rule.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the ported signature so the oracle is called with the same arguments."
)]
fn ref_sink(
    h_t: &[f64],
    p_mid: &[f64],
    p_out: &[f64],
    h_s: &[f64],
    p_in_mid: Option<&[f64]>,
    seqs: usize,
    t: usize,
    width: usize,
    thresh: f64,
) -> Vec<f64> {
    let rows = seqs * t;
    let norm = |a: &[f64], r: usize| -> f64 {
        (0..width)
            .map(|i| a[r * width + i] * a[r * width + i])
            .sum::<f64>()
            .sqrt()
    };
    let r_t: Vec<f64> = (0..rows)
        .map(|r| norm(h_t, r).max(norm(p_mid, r)).max(norm(p_out, r)))
        .collect();
    let mut med: Vec<f64> = Vec::new();
    for s in 0..seqs {
        for p in 1..t {
            med.push(r_t[s * t + p]);
        }
    }
    med.sort_by(f64::total_cmp);
    let r_med = med[(med.len() - 1) / 2];
    (0..rows)
        .map(|r| {
            let rs = match p_in_mid {
                Some(p) => norm(h_s, r).max(norm(p, r)),
                None => norm(h_s, r),
            };
            let r_tok = r_t[r].max(rs).max(1e-6);
            if r_tok * 1.0 > thresh * r_med {
                r_med / r_tok
            } else {
                1.0
            }
        })
        .collect()
}

/// (c) The even-count median takes the lower middle element (sorted index `(k-1)/2`).
#[test]
fn lower_median_on_even_count() {
    let h_t = [0.0f64, 1.0, 2.0, 3.0, 4.0];
    let zeros = [0.0f64; 5];
    let w = sink_token_weights(
        &qv(&h_t),
        &qv(&zeros),
        &qv(&zeros),
        &qv(&zeros),
        None,
        1,
        5,
        1,
        q(0.5),
    )
    .unwrap();
    let expect = ref_sink(&h_t, &zeros, &zeros, &zeros, None, 1, 5, 1, 0.5);
    // `r_t` is exact on integers; the only rounding is the `2/3` quotient (one quantum).
    let err = max_abs_diff(&words(&w), &expect);
    println!("lower_median: err={err:.3e} bound={:.3e}", 2.0 * Q);
    assert!(err <= 2.0 * Q, "err {err:e}");
}

/// (d) A token with `r_tok > thresh·r_med` gets `r_med/r_tok`; the rest get `1`; `r_s` includes
/// `p_in_mid` when given.
#[test]
fn token_weight_rule_and_mid_input() {
    let h_t = [1.0f64, 2.0, 10.0];
    let p_mid = [0.0f64; 3];
    let p_out = [0.0f64; 3];
    let h_s = [3.0f64, 0.0, 0.0];
    let p_in_mid = [0.0f64, 5.0, 0.0];
    let w = sink_token_weights(
        &qv(&h_t),
        &qv(&p_mid),
        &qv(&p_out),
        &qv(&h_s),
        Some(&qv(&p_in_mid)),
        1,
        3,
        1,
        q(2.0),
    )
    .unwrap();
    let expect = ref_sink(&h_t, &p_mid, &p_out, &h_s, Some(&p_in_mid), 1, 3, 1, 2.0);
    let err = max_abs_diff(&words(&w), &expect);
    println!("token_rule: err={err:.3e} bound={:.3e}", 2.0 * Q);
    assert!(err <= 2.0 * Q, "err {err:e}");
    assert!((from_grid(w[0]) - 1.0).abs() <= 2.0 * Q);
}

/// (e) The sink weights agree with the oracle on an LCG slab with a real width.
#[test]
fn sink_weights_match_f64_oracle() {
    let (seqs, t, width) = (2usize, 4usize, 5usize);
    let rows = seqs * t;
    let mut lcg = Lcg::new(0x5e17);
    let h_t = rand_mat(&mut lcg, rows, width);
    let p_mid = rand_mat(&mut lcg, rows, width);
    let p_out = rand_mat(&mut lcg, rows, width);
    let h_s = rand_mat(&mut lcg, rows, width);
    let p_in_mid = rand_mat(&mut lcg, rows, width);
    let w = sink_token_weights(
        &qv(&h_t),
        &qv(&p_mid),
        &qv(&p_out),
        &qv(&h_s),
        Some(&qv(&p_in_mid)),
        seqs,
        t,
        width,
        ONE,
    )
    .unwrap();
    let expect = ref_sink(
        &h_t,
        &p_mid,
        &p_out,
        &h_s,
        Some(&p_in_mid),
        seqs,
        t,
        width,
        1.0,
    );
    // Each norm is `isqrt` of `width` rounded squares (≤ width·q/2 absolute in the squared sum,
    // ≤ q/2 in the norm); the quotient adds one quantum. Four norms and one division bound it by
    // `4·width·q`, asserted with a factor of two.
    let bound = 8.0 * width as f64 * Q;
    let err = max_abs_diff(&words(&w), &expect);
    println!("sink oracle: err={err:.3e} bound={bound:.3e}");
    assert!(err <= bound, "err {err:e}");
}

// ---------------------------------------------------------------------------
// Solve tests
// ---------------------------------------------------------------------------

/// The solve fixture: two outputs, one split (`O`, `ts`) and one direct (`Q`).
struct SolveCase {
    n: usize,
    outs: [usize; 2],
    with_ts: [bool; 2],
    slabs: Vec<Slab>,
    w0: [Vec<f64>; 2],
}

fn solve_fixture() -> SolveCase {
    let (n, outs, with_ts) = (6usize, [2usize, 1], [true, false]);
    let slabs = two_slab_fixture(0x5017, n, &outs, &with_ts, [10, 6], 0.9);
    let mut lcg = Lcg::new(0x77a0);
    let w0_o = rand_mat(&mut lcg, outs[0], n);
    let w0_q = rand_mat(&mut lcg, outs[1], n);
    SolveCase {
        n,
        outs,
        with_ts,
        slabs,
        w0: [w0_o, w0_q],
    }
}

/// (f) `solve_stage` agrees with the `f64` split solve within `64·q` (`κ(H) ≤ 4` on the fixture).
#[test]
fn solve_stage_matches_f64_oracle() {
    let SolveCase {
        n,
        outs,
        with_ts,
        slabs,
        w0,
    } = solve_fixture();
    let acc = fold_accum(n, &outs, &with_ts, &slabs);
    let r = fold_ref(n, &outs, &with_ts, &slabs);
    let kinds = [Kind::O, Kind::Q];
    let p = SolveParams::default();
    let w0q = [qv(&w0[0]), qv(&w0[1])];
    let got = solve_stage(&acc, &[w0q[0].as_slice(), w0q[1].as_slice()], &kinds, &p).unwrap();
    let want = ref_solve_stage(&r, &[w0[0].as_slice(), w0[1].as_slice()], &kinds, &p);
    let mut m = 0.0f64;
    for j in 0..2 {
        let e = max_abs_diff(&words(&got[j]), &want[j]);
        println!("solve[{j}]: err={e:.3e} |w|max={:.3e}", max_abs(&want[j]));
        m = m.max(e);
    }
    // `q` accumulates through the Gram (≤ 3(rows+1)q) and the Cholesky solve amplifies by κ(H);
    // on this fixture (`rows ≥ 2n`, |x|≤1, w∈[0.45,1.35]) λmax/λmin(H) ≤ 4.
    let bound = 64.0 * Q;
    println!("solve_stage: max err={m:.3e} bound={bound:.3e}");
    assert!(m <= bound, "solve err {m:e}");
}

/// (g) With `Cs = Cts = H_solve·W0ᵀ` the split residuals vanish and `Ws == W0` (up to the one
/// rounding of `W0` onto the grid).
#[test]
fn solve_with_anchor_residual_is_identity() {
    let n = 4;
    let outs = [1usize];
    let with_ts = [true];
    let w0 = [0.5f64, -0.25, 0.1, 0.3];
    let w0q = qv(&w0);
    let mut x = vec![0.0f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    let y = w0.to_vec();
    let w = vec![1.0f64; n];
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    let (xq, wq, yq) = (qv(&x), qv(&w), qv(&y));
    acc.add_slab(&xq, &wq, &[yq.as_slice()], &[Some(yq.as_slice())])
        .unwrap();
    let params = SolveParams::default();
    let ws = solve_stage(&acc, &[w0q.as_slice()], &[Kind::O], &params).unwrap();
    // `W0` crosses onto the grid once (≤ q/2 per entry, `0.1` is not representable) and the
    // residual solve adds at most one quantum of contention, so `8·q` bounds the difference.
    let err = max_abs_diff(&words(&ws[0]), &w0);
    println!("identity: err={err:.3e} bound={:.3e}", 8.0 * Q);
    assert!(err <= 8.0 * Q, "err {err:e}");
}

/// (h) The solved `dW` satisfies the ridge normal equations
/// `(H + ridge·mean(diag H)·I)·dWᵀ = C`.
#[test]
fn solve_satisfies_normal_equations() {
    let (n, out, rows) = (6usize, 2usize, 12usize);
    let mut lcg = Lcg::new(7);
    let x = rand_mat(&mut lcg, rows, n);
    let h = matmul(&transpose(&x, rows, n), &x, n, rows, n);
    let c = rand_mat(&mut lcg, n, out);
    let ridge = 0.045 * (n as f64 / 896.0);
    let (hw, cw, rw) = (qv(&h), qv(&c), q(ridge));
    let dw = words(&ridge_solve(&hw, &cw, n, out, rw).unwrap());
    let (hr, cr) = (words(&hw), words(&cw));
    let mean = diag_mean(&hr, n);
    let r = from_grid(rw);
    let mut residual = 0.0f64;
    for i in 0..n {
        for o in 0..out {
            let mut s = 0.0f64;
            for k in 0..n {
                let hv = hr[i * n + k] + if i == k { r * mean } else { 0.0 };
                s += hv * dw[o * n + k];
            }
            residual = residual.max((s - cr[i * out + o]).abs());
        }
    }
    println!("normal equations residual: {residual:.3e}");
    assert!(residual < 1e-9, "residual {residual}");
}

/// (i) The sink clamp caps a `dW` row at `τ·max(‖W0_row‖, 1e-6)`.
#[test]
fn sink_clamp_caps_row_ratio() {
    let n = 4;
    let outs = [1usize];
    let with_ts = [true];
    let w0 = [0.5f64, 0.0, 0.0, 0.0];
    let w0q = qv(&w0);
    let mut x = vec![0.0f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    let y = [10.5f64, 0.0, 0.0, 0.0];
    let w = vec![1.0f64; n];
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    let (xq, wq, yq) = (qv(&x), qv(&w), qv(&y));
    acc.add_slab(&xq, &wq, &[yq.as_slice()], &[Some(yq.as_slice())])
        .unwrap();
    let params = SolveParams::default();
    let ws = solve_stage(&acc, &[w0q.as_slice()], &[Kind::O], &params).unwrap();
    let ws = words(&ws[0]);
    let dw_row = [ws[0] - w0[0], ws[1], ws[2], ws[3]];
    let dw_rn = dw_row.iter().map(|v| v * v).sum::<f64>().sqrt();
    let w0_rn = w0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ratio = dw_rn / w0_rn;
    println!("clamp ratio: {ratio:.9} tau={}", params.sink_tau);
    // The cap is a quotient of two `isqrt` norms and one grid scale, so `τ` up to `8·q`.
    assert!((ratio - params.sink_tau).abs() < 1e-7, "ratio {ratio}");
}

// ---------------------------------------------------------------------------
// `metric`, `bias_update` and the `ridge_solve` oracle
// ---------------------------------------------------------------------------

/// (j) `metric` adds `ridge·mean(diag H)` to the diagonal, matching the `f64` expression.
#[test]
fn metric_matches_f64_oracle() {
    let (n, rows) = (5usize, 9usize);
    let mut lcg = Lcg::new(0x3e7);
    let x = rand_mat(&mut lcg, rows, n);
    let hm = matmul(&transpose(&x, rows, n), &x, n, rows, n);
    let ridge = 0.25;
    let got = metric(&qv(&hm), n, q(ridge)).unwrap();
    let mut want = hm.clone();
    let shift = ridge * diag_mean(&hm, n);
    for i in 0..n {
        want[i * n + i] += shift;
    }
    // `H` and the shift each cross the grid once (≤ q/2 per entry), then the diagonal sum rounds.
    let (err, _) = grid_error(&got, &want);
    println!("metric: err={err:.3e} bound={:.3e}", 4.0 * Q);
    assert!(err <= 4.0 * Q, "metric err {err:e}");
}

/// (k) `bias_update = (y_sum − Wp·x_sum)/n_toks` matches the `f64` contraction.
#[test]
fn bias_update_matches_f64_oracle() {
    let (n, out) = (7usize, 3usize);
    let mut lcg = Lcg::new(0xb1a5);
    let y_sum = rand_mat(&mut lcg, out, 1);
    let x_sum = rand_mat(&mut lcg, n, 1);
    let wp = rand_mat(&mut lcg, out, n);
    let n_toks = 137.0f64;
    let got = bias_update(&qv(&y_sum), &qv(&x_sum), &qv(&wp), q(n_toks)).unwrap();
    let want: Vec<f64> = (0..out)
        .map(|o| {
            let dot: f64 = (0..n).map(|i| wp[o * n + i] * x_sum[i]).sum();
            (y_sum[o] - dot) / n_toks
        })
        .collect();
    // `n` products of two rounded entries plus one rounding of the difference: `(n+1)·q`.
    let (err, _) = grid_error(&got, &want);
    let bound = 2.0 * (n + 1) as f64 * Q;
    println!("bias_update: err={err:.3e} bound={bound:.3e}");
    assert!(err <= bound, "bias err {err:e}");
}

/// A deterministic symmetric matrix with a dominant diagonal (positive definite).
fn rand_spd(n: usize, lcg: &mut Lcg) -> Vec<f64> {
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = lcg.sym();
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
        h[i * n + i] = off + 1.0 + lcg.sym().abs();
    }
    h
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

/// The word `metric` is bit-identical to the `f64` wrapper it replaced on these fixtures: the
/// digests were recorded through the old `f64` entry before the flip.
#[test]
fn metric_words_are_bit_identical_to_the_recorded_wrapper() {
    for (seed, n, mag, want) in [
        (0x51ab_0001u64, 16usize, 1.0f64, 0xb7a3_909a_ca2c_31b1u64),
        (0x5eed_0042, 32, 1.0, 0xa7eb_4107_ed33_23b6),
    ] {
        let mut lcg = Lcg::new(seed ^ ((n as u64) << 32));
        let hm: Vec<f64> = rand_spd(n, &mut lcg).iter().map(|&v| v * mag).collect();
        let got = metric(&qv(&hm), n, q(0.045)).unwrap();
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        fnv_words(&mut d, &got);
        assert_eq!(d, want, "metric seed={seed:x} n={n}");
    }
}

/// The word `bias_update` is bit-identical to the `f64` wrapper it replaced on this fixture.
#[test]
fn bias_update_words_are_bit_identical_to_the_recorded_wrapper() {
    let (n, out) = (7usize, 3usize);
    let mut lcg = Lcg::new(0xb1a5);
    let y_sum = rand_mat(&mut lcg, out, 1);
    let x_sum = rand_mat(&mut lcg, n, 1);
    let wp = rand_mat(&mut lcg, out, n);
    let got = bias_update(&qv(&y_sum), &qv(&x_sum), &qv(&wp), q(137.0)).unwrap();
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    fnv_words(&mut d, &got);
    assert_eq!(d, 0x9add_15fc_2267_fd60);
}

/// A deterministic strictly-diagonally-dominant SPD matrix with diagonals log-spread over
/// `[1e-3, 1e3]`. The off-diagonal jitter is `0.25·min(diag)/n`, so each row's off-diagonal
/// magnitude sum is at most `0.25·min(diag) < diag_i`: symmetric and positive-definite.
fn spd_wide_diag(lcg: &mut Lcg, n: usize) -> Vec<f64> {
    let mut diag = vec![0.0f64; n];
    for i in 0..n {
        let t = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        diag[i] = 1e-3 * 1e6f64.powf(t);
    }
    let dmin = diag.iter().copied().fold(f64::INFINITY, f64::min);
    let jitter = 0.25 * dmin / n as f64;
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = if i == j { diag[i] } else { jitter * lcg.sym() };
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
    }
    h
}

/// The old `f64` path: `Hr = H` with `diag += ridge·mean(diag H)`, then `ridge::solve_spd`,
/// returned as `Xᵀ` (`[out, n]`). The independent reference `ridge_solve` is compared against.
fn old_ridge_solve(h: &[f64], c: &[f64], n: usize, out: usize, ridge: f64) -> Vec<f64> {
    solve_with(h, c, n, out, ridge * diag_mean(h, n))
}

/// (l) The Q32.32 `ridge_solve` agrees with the old f64 `solve_spd` path to relative `1e-7` on a
/// random symmetric SPD `H` with a wide diagonal range (`1e-3..1e3`), for `out ∈ {1, 3}`,
/// `n ∈ {7, 64, 130}`.
#[test]
fn ridge_solve_matches_old_path_wide_diagonal() {
    for &out in &[1usize, 3] {
        for &n in &[7usize, 64, 130] {
            let mut lcg = Lcg::new(0x9e37_79b9 ^ ((n as u64) << 8) ^ out as u64);
            let h = spd_wide_diag(&mut lcg, n);
            let c = rand_mat(&mut lcg, n, out);
            let ridge = 1e-8;
            let (hw, cw, rw) = (qv(&h), qv(&c), q(ridge));
            let new = words(&ridge_solve(&hw, &cw, n, out, rw).unwrap());
            let old = old_ridge_solve(&words(&hw), &words(&cw), n, out, from_grid(rw));
            let denom = max_abs(&old).max(1e-300);
            let rel = max_abs_diff(&new, &old) / denom;
            assert!(rel < 1e-7, "n={n} out={out} rel={rel}");
        }
    }
}

/// (l') The boundary scaling is exercised both ways: identical solves for `H`/`C` words scaled by
/// an exact `2⁻¹⁰` and `2¹⁰`, where `fill_exp` scales the words up and down respectively. The
/// solution is invariant under a common factor on both sides.
#[test]
fn ridge_solve_matches_old_path_extreme_magnitudes() {
    let (n, out) = (17usize, 2usize);
    let mut lcg = Lcg::new(0x00c0_ffee);
    let h0 = spd_wide_diag(&mut lcg, n);
    let c0 = rand_mat(&mut lcg, n, out);
    let rw = q(1e-8);
    for &f in &[2f64.powi(-10), 2f64.powi(10)] {
        let h: Vec<f64> = h0.iter().map(|&v| v * f).collect();
        let c: Vec<f64> = c0.iter().map(|&v| v * f).collect();
        let (hw, cw) = (qv(&h), qv(&c));
        let new = words(&ridge_solve(&hw, &cw, n, out, rw).unwrap());
        let old = old_ridge_solve(&words(&hw), &words(&cw), n, out, from_grid(rw));
        let denom = max_abs(&old).max(1e-300);
        let rel = max_abs_diff(&new, &old) / denom;
        assert!(rel < 1e-7, "f={f} rel={rel}");
    }
}

/// (m) A non-SPD input and a shape mismatch both return the refusal, with no panic.
#[test]
fn ridge_solve_refusals() {
    assert_eq!(
        ridge_solve(&qv(&[1.0, 2.0, 2.0, 1.0]), &qv(&[1.0, 0.0]), 2, 1, 0),
        Err(StageError::NotPositiveDefinite)
    );
    assert_eq!(
        ridge_solve(&qv(&[1.0, 0.0, 0.0]), &qv(&[1.0, 1.0]), 2, 1, q(0.1)),
        Err(StageError::Shape)
    );
}

/// (n) Refusals: shape mismatch, empty median, non-SPD, non-finite and missing `Cts`.
#[test]
fn refusals() {
    let z = [0i64; 4];
    assert_eq!(
        sink_token_weights(&[ONE, 2 * ONE], &[0], &z, &z, None, 1, 2, 1, ONE),
        Err(StageError::Shape)
    );
    assert_eq!(
        sink_token_weights(&[ONE], &[0], &[0], &[0], None, 1, 1, 1, ONE),
        Err(StageError::Empty)
    );
    // A non-finite input is refused where the caller now converts it (`to_grid`).
    assert_eq!(to_grid(f64::NAN, GRID_BOUND), None);
    let mut acc = StageAccum::new(3, &[2], &[false]).unwrap();
    assert_eq!(
        acc.add_slab(&[ONE, 2 * ONE], &[ONE], &[&[ONE, 2 * ONE]], &[None]),
        Err(StageError::Shape)
    );
    assert_eq!(to_grid(f64::INFINITY, GRID_BOUND), None);
    let mut acc2 = StageAccum::new(3, &[2], &[false]).unwrap();
    acc2.add_slab(&[ONE, 0, 0], &[ONE], &[&[ONE, 2 * ONE]], &[None])
        .unwrap();
    let params = SolveParams::default();
    assert_eq!(
        solve_stage(&acc2, &[&[ONE, 2 * ONE]], &[Kind::Q], &params),
        Err(StageError::Shape)
    );
    assert_eq!(
        solve_stage(
            &acc2,
            &[&[ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE]],
            &[Kind::O],
            &params
        ),
        Err(StageError::MissingCts)
    );
    assert_eq!(
        ridge_solve(&[0i64; 9], &[ONE; 3], 3, 1, 0),
        Err(StageError::NotPositiveDefinite)
    );
    assert_eq!(metric(&[1_i64; 3], 3, q(0.1)), Err(StageError::Shape));
    assert_eq!(
        bias_update(&[1], &[1, 1, 1], &[1, 1, 1], 0),
        Err(StageError::Empty)
    );
}

/// (o) Timing receipt (release): the production down-stage size `n = 4864`, `out = 1`, three
/// timed `ridge_solve` runs (the primitive's `SOLVE_THREADS = 8`) against one old-path
/// `ridge::solve_spd` call. `#[ignore]`; run with `--release -- --ignored --nocapture`. Not a
/// correctness gate.
#[test]
#[ignore = "timing receipt; run with --release -- --ignored --nocapture"]
fn timing_ridge_solve_4864() {
    use std::time::Instant;
    let (n, out) = (4864usize, 1usize);
    let mut lcg = Lcg::new(0x5eed_4864);
    let jitter = 0.25 / n as f64;
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = if i == j { 2.0 } else { jitter * lcg.sym() };
        }
    }
    let c = rand_mat(&mut lcg, n, out);
    let ridge = 0.03;
    let (hw, cw, rw) = (qv(&h), qv(&c), q(ridge));
    let mut times = Vec::with_capacity(3);
    for _ in 0..3 {
        let t = Instant::now();
        let _ = ridge_solve(&hw, &cw, n, out, rw).unwrap();
        times.push(t.elapsed().as_secs_f64());
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(0.0f64, f64::max);
    let t = Instant::now();
    let _ = old_ridge_solve(&h, &c, n, out, ridge);
    let old = t.elapsed().as_secs_f64();
    println!(
        "ridge_solve n={n} out={out} threads=8: mean={mean:.3}s min={min:.3}s max={max:.3}s; \
         old solve_spd={old:.3}s"
    );
    assert!(
        mean < 580.0,
        "new path exceeded the wall-clock budget: {mean}"
    );
}

#[test]
fn ridge_solve_words_are_bit_identical_to_the_recorded_wrapper() {
    let want = [
        0x8172_2f8c_752d_6759_u64,
        0xc10f_d0e1_5170_58ea,
        0x9b22_494f_c071_24a5,
    ];
    for (i, (n, out)) in [(17usize, 2usize), (7, 1), (64, 3)].into_iter().enumerate() {
        let mut lcg = Lcg::new(0x1234_5678 ^ (n as u64) ^ (out as u64) << 8);
        let h = spd_wide_diag(&mut lcg, n);
        let c = rand_mat(&mut lcg, n, out);
        let (hw, cw) = (qv(&h), qv(&c));
        let got = ridge_solve(&hw, &cw, n, out, q(1e-8)).unwrap();
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        fnv_words(&mut d, &got);
        assert_eq!(d, want[i], "ridge_solve n={n} out={out} words");
    }
}

// ---------------------------------------------------------------------------
// `Dot` (the pooled contraction inside `add_slab`): output digests and unit cost
// ---------------------------------------------------------------------------

/// FNV-1a 64 over the little-endian bytes of a word buffer.
fn digest_words(xs: &[i64]) -> u64 {
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    fnv_words(&mut d, xs);
    d
}

/// `(x, w_tok, ys, y_ts)` of one seeded `Dot` slab.
type DotSlab = (Vec<i64>, Vec<i64>, Vec<Vec<i64>>, Vec<Vec<i64>>);

/// A seeded slab for the `Dot` digests: `x` row-major `[rows, n]`, unit token weights and
/// `ys`/`ts` row-major `[rows, out]`, every entry `q(4·u)` for `u ∈ [-1, 1)`.
fn dot_slab(lcg: &mut Lcg, rows: usize, n: usize, outs: &[usize]) -> DotSlab {
    let words4 = |lcg: &mut Lcg, len: usize| -> Vec<i64> {
        rand_mat(lcg, 1, len).iter().map(|&v| q(4.0 * v)).collect()
    };
    let x = words4(lcg, rows * n);
    let w = vec![ONE; rows];
    let ys = outs.iter().map(|&o| words4(lcg, rows * o)).collect();
    let ts = outs.iter().map(|&o| words4(lcg, rows * o)).collect();
    (x, w, ys, ts)
}

fn add_dot_slab(acc: &mut StageAccum, s: &DotSlab, with_ts: &[bool]) -> Result<(), StageError> {
    let ys: Vec<&[i64]> = s.2.iter().map(Vec::as_slice).collect();
    let yts: Vec<Option<&[i64]>> =
        s.3.iter()
            .zip(with_ts)
            .map(|(t, &m)| m.then_some(t.as_slice()))
            .collect();
    acc.add_slab(&s.0, &s.1, &ys, &yts)
}

/// The `Dot` kernel's output is bit-identical to the recorded pre-blocking one: `hm`/`hm_w`
/// (`replace = true`, `x = y` for `hm`; digests 0 and 1) and, after a second slab, the
/// accumulated `cs`/`cts` (`replace = false` onto a non-zero destination; digests 2–4).
/// Fixture 1 has `rows = 37, n = 13, outs = [5, 7]`, fixture 2 `rows = 5001, n = 17,
/// outs = [5, 9]`: neither `n` nor any `out` is a multiple of the tile sizes.
#[test]
fn dot_output_digests_reproduce() {
    type Case = (u64, usize, usize, [usize; 2], [u64; 5]);
    let cases: [Case; 2] = [
        (
            0x5eed_0001,
            37,
            13,
            [5, 7],
            [
                16_569_049_911_569_093_702,
                16_569_049_911_569_093_702,
                17_543_790_876_757_816_417,
                14_615_254_248_315_779_465,
                3_824_674_395_350_365_230,
            ],
        ),
        (
            0x5eed_0002,
            5_001,
            17,
            [5, 9],
            [
                18_445_603_161_551_762_225,
                18_445_603_161_551_762_225,
                10_001_645_956_510_728_616,
                10_547_155_953_807_213_401,
                13_477_187_751_660_977_786,
            ],
        ),
    ];
    let with_ts = [true, false];
    let mut found = [[0u64; 5]; 2];
    for (i, &(seed, rows, n, outs, _)) in cases.iter().enumerate() {
        let mut lcg = Lcg::new(seed);
        let s1 = dot_slab(&mut lcg, rows, n, &outs);
        let s2 = dot_slab(&mut lcg, rows, n, &outs);
        let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
        let t = std::time::Instant::now();
        add_dot_slab(&mut acc, &s1, &with_ts).unwrap();
        let first = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        add_dot_slab(&mut acc, &s2, &with_ts).unwrap();
        let second = t.elapsed().as_secs_f64();
        let got = [
            digest_words(&acc.hm),
            digest_words(&acc.hm_w),
            digest_words(&acc.cs[0]),
            digest_words(&acc.cs[1]),
            digest_words(&acc.cts[0]),
        ];
        println!(
            "dot seed={seed:#x} rows={rows} n={n} outs={outs:?} wall={first:.6}s/{second:.6}s \
             digest={got:?}"
        );
        let _ = (first, second);
        found[i] = got;
    }
    for (&(seed, _, _, _, want), got) in cases.iter().zip(&found) {
        assert_eq!(*got, want, "dot digest seed={seed:#x}");
    }
}

/// An `i128` accumulator that leaves range is reported as [`StageError::Overflow`] from the
/// pooled entry (`i64::MAX² + i64::MAX²` on the Gram's `(0, 0)` entry).
#[test]
fn dot_overflow_reports_stage_error() {
    let (rows, n) = (2usize, 2usize);
    let mut x = vec![0i64; rows * n];
    x[0] = i64::MAX;
    x[1] = i64::MAX;
    let w = vec![ONE; rows];
    let y = vec![0i64; rows];
    let mut acc = StageAccum::new(n, &[1], &[false]).unwrap();
    assert_eq!(
        acc.add_slab(&x, &w, &[y.as_slice()], &[None]),
        Err(StageError::Overflow)
    );
}

/// Unit cost of the pooled `Dot`: one `add_slab` at `rows = 8192, n = out = 1024` is two
/// `rows·n²` Gram contractions and one `rows·n·out` cross contraction (`3·8.59·10⁹`
/// multiply-adds). Three timed runs; `#[ignore]`, a receipt, not a gate.
#[test]
#[ignore = "unit-cost receipt; run with --release -- --ignored --nocapture"]
fn dot_unit_cost() {
    let (rows, n, out) = (8_192usize, 1_024usize, 1_024usize);
    let mut lcg = Lcg::new(0xd07_0001);
    let x = qv(&rand_mat(&mut lcg, rows, n));
    let y = qv(&rand_mat(&mut lcg, rows, out));
    let w = vec![ONE; rows];
    let mut acc = StageAccum::new(n, &[out], &[false]).unwrap();
    let madds = (2 * rows * n * n + rows * n * out) as f64;
    let mut runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t = std::time::Instant::now();
        acc.add_slab(&x, &w, &[y.as_slice()], &[None]).unwrap();
        runs.push(t.elapsed().as_secs_f64());
    }
    let mean = runs.iter().sum::<f64>() / runs.len() as f64;
    let (min, max) = runs.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    println!(
        "dot unit cost rows={rows} n={n} out={out}: runs={runs:?}s mean={mean:.3}s \
         ns/madd={:.3} spread={:.3}..{:.3} ns",
        mean * 1e9 / madds,
        min * 1e9 / madds,
        max * 1e9 / madds
    );
}
