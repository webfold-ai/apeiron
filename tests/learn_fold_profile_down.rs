//! `tests/learn_fold_profile_down.rs` — a MEASUREMENT of where the `down` tensor's
//! `fit_tensor(o = 896, n = 4864, Kind::Down)` wall-clock goes.
//!
//! Not a correctness gate: every test is `#[ignore]`d and prints `std::time::Instant` seconds (mean
//! of three runs with the min/max spread where the phase fits the budget), the by-hand flop count
//! and the achieved GFLOP/s. One phase group per test, so a phase that stalls cuts alone.
//!
//! Inputs (deterministic LCG, as the measurement brief names): `Ws` `[896, 4864]` with entries
//! `~ N(0, 1/sqrt(4864))`; `H = XᵀX / rows` (`rows = 2048`) from random activation rows with eight
//! columns scaled by 20, PSD with a realistic spread; `g_row` positive of mean 1 (the shape the
//! `down` fit is handed). Production `down` config: `rg_q = 0.005·(4864/896)`,
//! `damp = 0.02·(4864/896)`, `e8_beam` 4, `icm_ridge` 0.20, `inner_iters` 3, `sweeps` 2, `iters` 1.
//!
//! Run one phase group:
//!
//! ```text
//! systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
//!   env -u CARGO_TARGET_DIR timeout 580 \
//!   ~/bin/cargo test --release --locked --test learn_fold_profile_down t_a \
//!   -- --ignored --nocapture
//! ```
//!
//! Grade: the printed wall-clock is `measured`; every flop count and the peak are `model`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "a measurement harness: it maps a fixed LCG to f64 by division, indexes the flat \
              row-major buffers the reference kernels consume, names the tensors `ws/h/ws_rot/s/r/c/d` \
              and the counters `o/i/j/r` as the reference does, and casts dimension counts \
              `usize → f64` to form the flop counts"
)]

use std::time::Instant;

use apeiron::learn::fold::fit::corrector::{lowrank_corrector, Mode};
use apeiron::learn::fold::fit::eigh::eigh;
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig};
use apeiron::learn::fold::fit::gptq_e8::gptq_e8;
use apeiron::learn::fold::fit::icm::{e8_icm_step, PlaneRef};
use apeiron::learn::fold::fit::steps::{c_step, compose, d_step, had_gram, r_step};
use apeiron::learn::fold::layer::stage::metric;
use apeiron::learn::fold::layer::tensor::{fit_tensor, TensorKind, TensorParams};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

#[path = "support/fwht_oracle.rs"]
mod fwht_oracle;

use fwht_oracle::fwht128;

/// The `down` tensor's input width (`intermediate_size`).
const N: usize = 4864;
/// The `down` tensor's output width (`hidden_size`).
const O: usize = 896;
/// Activation rows in the Gram fixture.
const ROWS: usize = 2048;
/// The stage reference width (`SolveParams::n_ref`).
const N_REF: f64 = 896.0;
/// `a.quant_ridge`.
const QUANT_RIDGE: f64 = 0.005;
/// `a.quant_damp`.
const QUANT_DAMP: f64 = 0.02;
/// Columns of the activation fixture scaled by [`OUTLIER_SCALE`].
const OUTLIER_COLS: [usize; 8] = [7, 601, 1207, 1813, 2401, 3001, 3607, 4201];
/// The outlier column scale.
const OUTLIER_SCALE: f64 = 20.0;

/// Deterministic 64-bit LCG (Numerical Recipes constants).
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

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// A standard-normal sample (sum of twelve uniforms minus six).
fn gauss(rng: &mut Lcg) -> f64 {
    (0..12).map(|_| rng.uniform(0.0, 1.0)).sum::<f64>() - 6.0
}

/// The synthetic `down` fixture: `Ws`, the PSD Gram `H`, the positive row weight, and the
/// production scalar config.
struct Fx {
    ws: Vec<f64>,
    h: Vec<f64>,
    g_row: Vec<f64>,
    rg_q: i64,
    damp: i64,
    fit: FoldFitConfig,
}

/// The production `down` parameters as [`TensorParams`].
fn params(fx: &Fx) -> TensorParams {
    TensorParams {
        rg_q: fx.rg_q,
        qmlp_rows: 0,
        fit: fx.fit,
        nb_sweeps: 2,
        nb_rounds: 2,
    }
}

/// Build the fixture: `XᵀX/rows` with outlier columns, `Ws`, `g_row`.
///
/// # Complexity
/// Temporal `O(rows·n²)` for the Gram (the dominant term) plus `O(o·n)` samples; spatial `O(n²)`.
fn fixture() -> Fx {
    fixture_dims(O, N, ROWS, &OUTLIER_COLS)
}

/// [`fixture`] at an arbitrary geometry (`n` a multiple of 128): `o` output rows, `n` input
/// columns, `rows` activation rows, and the `cols` scaled by [`OUTLIER_SCALE`].
///
/// # Complexity
/// Temporal `O(rows·n²)` for the Gram (the dominant term) plus `O(o·n)` samples; spatial `O(n²)`.
fn fixture_dims(o: usize, n: usize, rows: usize, cols: &[usize]) -> Fx {
    let mut rng = Lcg::new(20_261_016);
    // X [rows, n] ~ N(0,1), the `cols` columns scaled by OUTLIER_SCALE.
    let mut x = vec![0.0_f64; rows * n];
    for v in &mut x {
        *v = gauss(&mut rng);
    }
    for &col in cols {
        for r in 0..rows {
            x[r * n + col] *= OUTLIER_SCALE;
        }
    }
    // H = XᵀX / rows, upper triangle first, then mirrored.
    let mut h = vec![0.0_f64; n * n];
    for r in 0..rows {
        let xr = &x[r * n..(r + 1) * n];
        for i in 0..n {
            let a = xr[i];
            let row = &mut h[i * n + i..i * n + n];
            for (jj, xv) in xr[i..].iter().enumerate() {
                row[jj] += a * xv;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            h[i * n + j] = h[j * n + i];
        }
    }
    let inv_rows = 1.0 / rows as f64;
    for v in &mut h {
        *v *= inv_rows;
    }
    drop(x);
    // Ws [o, n] ~ N(0, 1/sqrt(n)).
    let s = 1.0 / (n as f64).sqrt();
    let ws: Vec<f64> = (0..o * n).map(|_| s * gauss(&mut rng)).collect();
    // g_row: positive, mean 1.
    let mut g_row: Vec<f64> = (0..o).map(|_| (0.35 * gauss(&mut rng)).exp()).collect();
    let gmean = g_row.iter().sum::<f64>() / o as f64;
    for v in &mut g_row {
        *v /= gmean;
    }
    let rg_q = to_grid(QUANT_RIDGE * (n as f64 / N_REF), GRID_BOUND).expect("rg_q grid");
    let damp = to_grid(QUANT_DAMP * (n as f64 / N_REF), GRID_BOUND).expect("damp grid");
    let fit = FoldFitConfig {
        iters: 1,
        ridge: rg_q,
        damp,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 858_993_459,
        sweeps: 2,
    };
    Fx {
        ws,
        h,
        g_row,
        rg_q,
        damp,
        fit,
    }
}

/// Print the machine context the numbers bind to.
fn header() {
    let par = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    eprintln!("MEAS machine: available_parallelism={par}, single-threaded scalar f64 path");
    eprintln!(
        "MEAS geometry: o={O} n={N} rows={ROWS} rg_q={} damp={}",
        QUANT_RIDGE * (N as f64 / N_REF),
        QUANT_DAMP * (N as f64 / N_REF)
    );
}

/// Time `reps` runs of one phase, print the mean/min/max seconds, the flop count and GFLOP/s.
fn phase(label: &str, flops: f64, reps: usize, mut f: impl FnMut() -> f64) {
    let mut v = Vec::with_capacity(reps);
    let mut chk = 0.0_f64;
    for _ in 0..reps {
        let t = Instant::now();
        chk = f();
        v.push(t.elapsed().as_secs_f64());
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let mn = v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let gflops = flops / mean / 1e9;
    eprintln!(
        "MEAS {label}: {mean:.3} s mean (min {mn:.3} max {mx:.3} n={reps}) | flops {flops:.4e} | {gflops:.2} GFLOP/s | chk {chk:.6e}"
    );
}

/// `n³/3 + 8.3e9`: `feedback_u`'s two triangular sweeps plus the beam-search scan (see the module
/// doc; the beam term is the derived `groups·blocks·o·beam·256·8` scan at `beam = 4`).
fn gptq_flops(o: f64, n: f64) -> f64 {
    n * n * n / 3.0 + 8.3e9 * (o / 896.0)
}

/// One `e8_icm_step`: two `O(o·n²)` matmuls, the metrics, and `sweeps` blocks × rows × `8·n`.
fn icm_flops(o: f64, n: f64, sweeps: f64) -> f64 {
    2.0 * o * n * n + sweeps * (n / 8.0) * o * 8.0 * n * 2.0 + 8.0 * n * n
}

/// `x` as Q32.32 words, the carrier the fit steps take. `O(len)`.
fn q(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
}

/// Words read back as `f64`. `O(len)`.
fn unq(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// The sign plane of `ws` (`o × n`) as words, the cheap code plane the ICM/steps phases consume.
fn sign_plane(ws: &[f64]) -> Vec<i64> {
    q(&ws
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
        .collect::<Vec<f64>>())
}

/// `hl = metric(H, rg_q)` and the flat `ones` the closed-form steps hold, as words.
fn metric_and_ones(fx: &Fx) -> (Vec<i64>, Vec<i64>) {
    let hl = metric(&q(&fx.h), N, fx.rg_q).expect("metric");
    (hl, q(&vec![1.0_f64; N]))
}

/// Phase group (1)(2)(3): the transforms, `metric`, and `gptq_e8`.
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_a_transforms_metric_gptq() {
    let fx = fixture();
    header();
    let (o, n) = (O as f64, N as f64);
    phase("(1a) fwht128(Ws) [o=896,n=4864]", o * n * 7.0, 3, || {
        fwht128(&fx.ws).expect("fwht")[0]
    });
    let hq = q(&fx.h);
    phase(
        "(1b) had_gram(H) [n=4864]",
        2.0 * n * n * 7.0 + 2.0 * n * n,
        3,
        || from_grid(had_gram(&hq, N).expect("had_gram")[N]),
    );
    phase("(2) metric(H, rg_q) [n=4864]", n * n, 3, || {
        from_grid(metric(&q(&fx.h), N, fx.rg_q).expect("metric")[0])
    });
    let ws_rot = fwht128(&fx.ws).expect("fwht");
    let h_syrk = unq(&had_gram(&q(&fx.h), N).expect("had_gram"));
    phase(
        "(3) gptq_e8(ws_rot, h_syrk, beam=4)",
        gptq_flops(o, n),
        3,
        || {
            from_grid(
                gptq_e8(&q(&ws_rot), &q(&h_syrk), O, N, fx.damp, true, true, 4)
                    .expect("gptq")
                    .scale[0],
            )
        },
    );
}

/// Phase group (4): one `e8_icm_step` at the production `sweeps = 2`, `icm_ridge = 0.20`.
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_b_icm_step() {
    let fx = fixture();
    header();
    let (hl, c) = metric_and_ones(&fx);
    let s = sign_plane(&fx.ws);
    let idx = vec![0_u16; O * (N / 8)];
    let r = q(&vec![1.0_f64; O * (N / 128)]);
    let d = q(&vec![1.0_f64; N]);
    let wsq = q(&fx.ws);
    let p = PlaneRef {
        s: &s,
        idx: &idx,
        r: &r,
        c: &c,
        d: &d,
    };
    phase(
        "(4) e8_icm_step(o=896,n=4864,sweeps=2)",
        icm_flops(O as f64, N as f64, 2.0),
        3,
        || {
            from_grid(
                e8_icm_step(&wsq, &hl, p, O, N, 2, 858_993_459)
                    .expect("icm")
                    .s[0],
            )
        },
    );
}

/// Phase group (5): the continuous `c`/`r`/`d` steps `fold_fit` runs.
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_c_steps_crd() {
    let fx = fixture();
    header();
    let (hl, ones) = metric_and_ones(&fx);
    let s = sign_plane(&fx.ws);
    let r = q(&vec![1.0_f64; O * (N / 128)]);
    let (o, n) = (O as f64, N as f64);
    let (wsq, gq) = (q(&fx.ws), q(&fx.g_row));
    let g = Some(gq.as_slice());
    // c_step: form A = hl ∘ (Σ_o Sh·Sh_g) `O(o·n²)` + Ws·Hl `O(o·n²)` + Cholesky `n³/6`.
    phase(
        "(5a) c_step [O(o n^2) + n^3]",
        2.0 * o * n * n + n * n * n / 6.0,
        3,
        || from_grid(c_step(&wsq, &hl, &s, &r, O, N, g).expect("c_step")[0]),
    );
    // r_step: the per-row G×G systems, `O(o·n²)` total.
    phase("(5b) r_step [O(o n^2)]", o * n * n, 3, || {
        from_grid(r_step(&wsq, &hl, &s, &ones, O, N).expect("r_step")[0])
    });
    // d_step: Ws·Hl + A `O(o·n²)` + H_rot + Cholesky `n³/6`.
    phase(
        "(5c) d_step [O(o n^2) + n^3]",
        2.0 * o * n * n + n * n * n / 6.0,
        3,
        || from_grid(d_step(&wsq, &hl, &s, &r, &ones, O, N, g).expect("d_step")[0]),
    );
}

/// Phase group (6): `lowrank_corrector` at the `down` rank/mode (`rank = 1`, `Mode::U8`).
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_d_corrector() {
    let fx = fixture();
    header();
    let hl = metric(&q(&fx.h), N, fx.rg_q).expect("metric");
    let ws_rot = fwht128(&fx.ws).expect("fwht");
    let h_syrk = unq(&had_gram(&q(&fx.h), N).expect("had_gram"));
    let qe8 = gptq_e8(&q(&ws_rot), &q(&h_syrk), O, N, fx.damp, true, true, 4).expect("gptq");
    // The gptq candidate weight wq0 = compose(fwht128(s_rot), scale, ones); delta = ws − wq0.
    let wq0_s = fwht128(&unq(&qe8.s)).expect("fwht");
    let ones_c = vec![1.0_f64; N];
    let wq0 = unq(&compose(&q(&wq0_s), &q(&unq(&qe8.scale)), &q(&ones_c), O, N).expect("compose"));
    let delta: Vec<f64> = fx.ws.iter().zip(&wq0).map(|(a, b)| a - b).collect();
    drop(wq0);
    let (o, n) = (O as f64, N as f64);
    // ΔW·Hl `O(o·n²)` + M = w1·ΔWᵀ `O(o²·n)` + eigh(o) `O(o³·sweeps)` + ridge solve/product.
    let flops = o * n * n + o * o * n + 10.0 * o * o * o / 2.0 + o * n;
    phase("(6) lowrank_corrector(rank=1, U8)", flops, 3, || {
        from_grid(
            lowrank_corrector(&q(&delta), &hl, O, N, 1, Mode::U8, true)
                .expect("corrector")
                .corr[0],
        )
    });
}

/// Phase group (7): `eigh` at the size the corrector path calls it (`n = o = 896`).
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_e_eigh() {
    let fx = fixture();
    header();
    let hl = unq(&metric(&q(&fx.h), N, fx.rg_q).expect("metric"));
    let ws_rot = fwht128(&fx.ws).expect("fwht");
    let h_syrk = unq(&had_gram(&q(&fx.h), N).expect("had_gram"));
    let qe8 = gptq_e8(&q(&ws_rot), &q(&h_syrk), O, N, fx.damp, true, true, 4).expect("gptq");
    let wq0_s = fwht128(&unq(&qe8.s)).expect("fwht");
    let ones_c = vec![1.0_f64; N];
    let wq0 = unq(&compose(&q(&wq0_s), &q(&unq(&qe8.scale)), &q(&ones_c), O, N).expect("compose"));
    // w1 = ΔW·Hl (o×n), m = w1·ΔWᵀ (o×o), exactly as `eigen_parts` builds it.
    let mut w1 = vec![0.0_f64; O * N];
    for i in 0..O {
        for k in 0..N {
            let mut acc = 0.0_f64;
            for j in 0..N {
                acc += (fx.ws[i * N + j] - wq0[i * N + j]) * hl[j * N + k];
            }
            w1[i * N + k] = acc;
        }
    }
    drop(wq0);
    let mut m = vec![0.0_f64; O * O];
    for i in 0..O {
        for l in 0..O {
            let mut acc = 0.0_f64;
            for k in 0..N {
                acc += w1[i * N + k] * (fx.ws[l * N + k]);
            }
            m[i * O + l] = acc;
        }
    }
    eprintln!("MEAS (7) eigh called once per corrector, at n = o = {O}");
    phase(
        "(7) eigh(o=896)",
        10.0 * (O as f64).powi(3) / 2.0,
        3,
        || from_grid(eigh(&q(&m), O).expect("eigh").0[0]),
    );
}

/// Phase group (8): the WHOLE `fold_fit` call. One run only (the phase exceeds three-run budget).
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_f_fold_fit() {
    let fx = fixture();
    header();
    eprintln!(
        "MEAS (8) fold_fit call counts per call (read from fold_fit.rs): gptq_e8 1, \
         c_step 6, r_step 9, d_step 3, e8_icm_step 4, objective 6, had_gram 4"
    );
    let flops = fold_fit_flops();
    phase("(8) WHOLE fold_fit(down: o=896,n=4864)", flops, 1, || {
        let f =
            fold_fit(&q(&fx.ws), &q(&fx.h), O, N, Some(&q(&fx.g_row)), &fx.fit).expect("fold_fit");
        from_grid(f.wp[0])
    });
}

/// Geometry of the small gate fixture: a `fold_fit` call that finishes in seconds.
const GATE_O: usize = 64;
/// Gate fixture input width (four 128-column groups).
const GATE_N: usize = 512;
/// Gate fixture activation rows.
const GATE_ROWS: usize = 256;
/// Gate fixture outlier columns.
const GATE_COLS: [usize; 8] = [3, 77, 130, 261, 300, 411, 450, 509];

/// Recorded f64 reference `[J(gptq), J(refit0)]` of the gate fixture, from the source before the
/// integer-carrier conversion of `src/learn/fold/fit/*` (`measured` at `59370c1a`).
const GATE_REF_TRACE: [f64; 2] = [23.846_899_540_239_9, 11.221_175_253_556_261];

/// The integer-carrier conversion gate: `fold_fit` on the small fixture reaches an objective no
/// worse than the recorded f64 reference by more than `2^-20` relative, for both candidates and
/// the best, with the bit budgets unchanged. A conversion that degrades `gptq_e8`, the refit
/// steps, the ICM passes or the scale quantisers moves one of these and reddens this test.
#[test]
fn gate_small_fold_fit_objective() {
    let fx = fixture_dims(GATE_O, GATE_N, GATE_ROWS, &GATE_COLS);
    let f = fold_fit(
        &q(&fx.ws),
        &q(&fx.h),
        GATE_O,
        GATE_N,
        Some(&q(&fx.g_row)),
        &fx.fit,
    )
    .expect("fold_fit");
    let (trace, j_best) = (unq(&f.trace), from_grid(f.j_best));
    eprintln!("GATE trace={trace:?} j_best={j_best:e} tag={:?}", f.tag);
    let tol = 1.0 + 2.0_f64.powi(-20);
    assert_eq!(trace.len(), 2);
    for (j, r) in trace.iter().zip(GATE_REF_TRACE) {
        assert!(
            *j <= r * tol,
            "J {j:e} exceeds the f64 reference {r:e} by more than 2^-20"
        );
    }
    assert!(j_best <= GATE_REF_TRACE[0].min(GATE_REF_TRACE[1]) * tol);
    let g = GATE_N / 128;
    assert_eq!(f.bits_r, (4 * GATE_O * g + 8 * GATE_O + 64) as u64);
    assert_eq!(f.bits_c, (6 * GATE_N + 32) as u64);
    assert_eq!(f.bits_d, (5 * GATE_N + 32) as u64);
}

/// FNV-1a 64 over the little-endian bytes of `words`.
fn fnv_words(words: impl Iterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// FNV-1a 64 of an `f64` slice's bit patterns.
fn fnv_f64(xs: &[f64]) -> u64 {
    fnv_words(xs.iter().map(|v| v.to_bits()))
}

/// Reference receipt for the integer-carrier conversion of `fit/*`: the f64 `gptq_e8` output
/// (code-index and scale digests) and the `fold_fit` result (`J` per candidate, plane digests,
/// the metric diagonal range) on the fixture. Every `FIX` line is a `measured` value of the f64
/// source at the committed revision; the integer lanes compare against it.
///
/// Recorded at `861fc239` (release, `available_parallelism` 32, one run, deterministic fixture):
///
/// ```text
/// had_gram diag min=8.759741943956822e-1 max=4.499357199903327e0 digest=e9d82d939e9ca4ec
/// ws_rot digest=0fa51ff095e1551d
/// gptq_e8 idx[544768] digest=71693eed28516abf scale[34048] digest=859b6c83e538c179 s digest=16da41ec9411c1e5
/// fold_fit trace=[157.31321023010784, 148.82239282700925] j_best=1.4882239282700925e2 tag=Refit0
/// fold_fit s_rot=005c1a6accbbee25 rh=07b96452cb07d2fc ch=a219646401359cda dh=7b926390b236c1a1 wp=12a470da2ada406f
/// bits r=143424 c=29216 d=24352
/// ```
///
/// The metric diagonal spans a factor 5.1, so this fixture does not exercise the large-condition
/// outlier columns of a real `H`; the held-out KL arm does.
#[test]
#[ignore = "reference receipt: --release --ignored --nocapture"]
fn t_h_fixture_receipt() {
    let fx = fixture();
    let ws_rot = fwht128(&fx.ws).expect("fwht");
    let h_syrk = unq(&had_gram(&q(&fx.h), N).expect("had_gram"));
    let dmin = (0..N)
        .map(|i| h_syrk[i * N + i])
        .fold(f64::INFINITY, f64::min);
    let dmax = (0..N).map(|i| h_syrk[i * N + i]).fold(0.0_f64, f64::max);
    eprintln!(
        "FIX had_gram diag min={dmin:e} max={dmax:e} digest={:016x}",
        fnv_f64(&h_syrk)
    );
    eprintln!("FIX ws_rot digest={:016x}", fnv_f64(&ws_rot));
    let g = gptq_e8(&q(&ws_rot), &q(&h_syrk), O, N, fx.damp, true, true, 4).expect("gptq");
    eprintln!(
        "FIX gptq_e8 idx[{}] digest={:016x} scale[{}] digest={:016x} s digest={:016x}",
        g.idx.len(),
        fnv_words(g.idx.iter().map(|&v| u64::from(v))),
        g.scale.len(),
        fnv_f64(&unq(&g.scale)),
        fnv_f64(&unq(&g.s))
    );
    let f = fold_fit(&q(&fx.ws), &q(&fx.h), O, N, Some(&q(&fx.g_row)), &fx.fit).expect("fold_fit");
    let (trace, j_best) = (unq(&f.trace), from_grid(f.j_best));
    eprintln!(
        "FIX fold_fit trace={trace:?} j_best={j_best:e} tag={:?}",
        f.tag
    );
    eprintln!(
        "FIX fold_fit s_rot={:016x} rh={:016x} ch={:016x} dh={:016x} wp={:016x}",
        fnv_f64(&unq(&f.pack.s_rot)),
        fnv_f64(&unq(&f.pack.rh)),
        fnv_f64(&unq(&f.pack.ch)),
        fnv_f64(&unq(&f.pack.dh)),
        fnv_f64(&unq(&f.wp))
    );
    eprintln!("FIX bits r={} c={} d={}", f.bits_r, f.bits_c, f.bits_d);
}

/// Phase group (9): the WHOLE `fit_tensor` call for `Kind::Down`. One run only.
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn t_g_fit_tensor() {
    let fx = fixture();
    header();
    let p = params(&fx);
    let flops = fold_fit_flops() + (O as f64) * (N as f64) * (N as f64) + 29.0e9;
    phase("(9) WHOLE fit_tensor(Down: o=896,n=4864)", flops, 1, || {
        let t = fit_tensor(
            TensorKind::Down,
            &q(&fx.ws),
            &q(&fx.h),
            O,
            N,
            Some(&q(&fx.g_row)),
            &p,
            None,
        )
        .expect("fit_tensor");
        eprintln!(
            "MEAS (9) fit_tensor result: planes {} nsign {} ncorr {}",
            t.planes.len(),
            t.bits.nsign,
            t.bits.ncorr
        );
        from_grid(t.wp[0])
    });
}

/// The by-hand flop count of one production `fold_fit` (see the module doc for the terms).
fn fold_fit_flops() -> f64 {
    let (o, n) = (O as f64, N as f64);
    let gptq = gptq_flops(o, n);
    let c = 2.0 * o * n * n + n * n * n / 6.0;
    let r = o * n * n;
    let d = 2.0 * o * n * n + n * n * n / 6.0;
    let icm = icm_flops(o, n, 2.0);
    let obj = o * n * n;
    // 3 refit passes of (c, r, d, icm); j0/j1 objectives; pack phase (2 r/c pairs + 1 icm + 6 obj).
    gptq + 3.0 * (c + r + d + icm) + 2.0 * obj + 3.0 * (c + r) + icm + 6.0 * obj + 4.0 * n * n
}
