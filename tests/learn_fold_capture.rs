//! Integration tests for the calibration activation-Gram capture (`src/learn/fold/capture.rs`).
//!
//! Grade: `tested`. The positive assertions compare the accumulated Gram against an independent
//! `Σ_t x_t x_tᵀ` computed in the test, check symmetry, positive semi-definiteness (`Cholesky` of
//! `H + εI`), additivity over token blocks (two tapes equal one concatenated tape), the token
//! counter and the shape refusals. The tape comes from the test's own synthetic `LayerTape`
//! fixtures: the deleted `forward` is gone, and this file holds no `GGUF` image to load the
//! engine from, so its `forward_tape_has_capture_dims` test is deleted with it.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    reason = "test-only arithmetic on a small deterministic Q32.32 fixture and its f64 Gram; the \
              single-letter names are the geometry's symbols and the float comparisons assert the \
              exact mirror and the exact f64 division the accumulator performs"
)]

use apeiron::learn::fold::stats::capture::{CaptureError, LayerSyrk, Normalisation, SyrkAcc};
use apeiron::learn::grad::model::{Config, LayerTape, Tape};

#[path = "support/ridge_oracle.rs"]
mod ridge_oracle;

/// Residual width.
const E: usize = 4;
/// Query heads.
const NH: usize = 2;
/// Per-head width.
const HD: usize = 2;
/// `q_out`, chosen equal to `E` so a swapped Gram matrix keeps the same shape.
const QW: usize = NH * HD;
/// KV heads.
const NKV: usize = 1;
/// Feed-forward width.
const FF: usize = 3;
/// Vocabulary.
const V: usize = 8;
/// Trunk layers.
const L: usize = 2;
/// `ONE` in Q32.32.
const ONE: i64 = 65_536;

/// The tiny geometry every test uses.
fn cfg() -> Config {
    Config {
        n_layer: L,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: E,
        n_ff: FF,
        vocab: V,
        head_dim: HD,
        eps: 42_950,
        rope_theta: 10_000.0,
    }
}

/// Deterministic Q32.32 values in `[-3, 3]` from a fixed `LCG`, so the fixture is reproducible.
fn lcg(state: &mut u64) -> i64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let v = (*state >> 33) % 7;
    v as i64 - 3
}

/// `n` deterministic Q32.32 weights.
fn buf(n: usize, state: &mut u64) -> Vec<i64> {
    (0..n).map(|_| lcg(state) * ONE).collect()
}

/// A `LayerTape` carrying only the four buffers the capture reads.
fn layer_from(ln1_out: Vec<i64>, attn_out: Vec<i64>, ln2_out: Vec<i64>, m: Vec<i64>) -> LayerTape {
    LayerTape {
        ln1_in: Vec::new(),
        ln1_out,
        q: Vec::new(),
        k: Vec::new(),
        v: Vec::new(),
        p: Vec::new(),
        attn_out,
        h2: Vec::new(),
        ln2_out,
        gate: Vec::new(),
        up: Vec::new(),
        m,
    }
}

/// A `Tape` carrying only the layers and the token count.
fn tape_from(t: usize, layers: Vec<LayerTape>) -> Tape {
    Tape {
        t,
        tokens: Vec::new(),
        layers,
        final_hidden: Vec::new(),
        final_normed: Vec::new(),
    }
}

/// One `t`-token layer of deterministic Q32.32 inputs.
fn one_layer(state: &mut u64, t: usize) -> LayerTape {
    layer_from(
        buf(t * E, state),
        buf(t * QW, state),
        buf(t * E, state),
        buf(t * FF, state),
    )
}

/// A two-layer `t`-token tape.
fn two_layer_tape(state: &mut u64, t: usize) -> Tape {
    tape_from(t, (0..L).map(|_| one_layer(state, t)).collect())
}

/// Rows `r0..r1` of a row-major `[_, w]` buffer.
fn slice_rows(x: &[i64], w: usize, r0: usize, r1: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity((r1 - r0) * w);
    for r in r0..r1 {
        out.extend_from_slice(&x[r * w..r * w + w]);
    }
    out
}

/// Rows `r0..r1` of every capture buffer of `lt`.
fn slice_layer(lt: &LayerTape, r0: usize, r1: usize) -> LayerTape {
    layer_from(
        slice_rows(&lt.ln1_out, E, r0, r1),
        slice_rows(&lt.attn_out, QW, r0, r1),
        slice_rows(&lt.ln2_out, E, r0, r1),
        slice_rows(&lt.m, FF, r0, r1),
    )
}

/// The independent `Σ_t x_t x_tᵀ` over a row-major Q32.32 `t × w` buffer, token-outermost order.
/// The agreement oracle: the pre-migration `fold_rows_upper` body in `f64` — `Σ_t x_t x_tᵀ` over the
/// full matrix (`src/learn/fold/stats/capture.rs` at base `9f440192`), with each Q32.32 word scaled
/// by the exact power of two `2⁻³²`. Test-only `f64`.
///
/// Derived agreement bound, used by [`tol`]: our entry is `Σ_t round_q32_even(x_ti·x_tj)` words, so
/// each of the `t` terms rounds by at most half a grid quantum (`2⁻³³` real) and the integer sum adds
/// nothing; the oracle adds `t` exact `f64` products, each with relative error at most `2⁻⁵³`, so at
/// most `t·M·2⁻⁵³` with `M = max |x_ti·x_tj|`. The two differ by at most `t·(2⁻³³ + M·2⁻⁵³)`.
fn naive_syrk(x: &[i64], t: usize, w: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; w * w];
    for i in 0..w {
        for j in 0..w {
            let mut s = 0.0f64;
            for r in 0..t {
                let a = x[r * w + i] as f64 / 4_294_967_296.0;
                let b = x[r * w + j] as f64 / 4_294_967_296.0;
                s += a * b;
            }
            g[i * w + j] = s;
        }
    }
    g
}

/// The derived bound of [`naive_syrk`]: `t` terms, operands bounded by `m` in magnitude.
fn tol(t: usize, m: f64) -> f64 {
    t as f64 * (2f64.powi(-33) + m * 2f64.powi(-53))
}

/// Element-wise closeness of two equal-length f64 buffers.
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} != {y} (tol {tol})");
    }
}

/// All four SyrkAcc of two finished layer lists are equal.
fn layers_close(a: &[LayerSyrk], b: &[LayerSyrk], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        close(&x.qkv, &y.qkv, tol);
        close(&x.o, &y.o, tol);
        close(&x.gate_up, &y.gate_up, tol);
        close(&x.down, &y.down, tol);
        assert_eq!(x.tokens, y.tokens);
    }
}

#[test]
fn syrk_equals_independent_xtx() {
    let c = cfg();
    let mut s = 11u64;
    let tape = two_layer_tape(&mut s, 3);
    let mut g = SyrkAcc::new(&c).unwrap();
    g.accumulate(&tape, &c).unwrap();
    let fin = g.finish(Normalisation::Sum).unwrap();
    assert_eq!(fin.len(), L);
    for (fl, lt) in fin.iter().zip(&tape.layers) {
        close(&fl.qkv, &naive_syrk(&lt.ln1_out, 3, E), tol(3, 9.0));
        close(&fl.o, &naive_syrk(&lt.attn_out, 3, QW), tol(3, 9.0));
        close(&fl.gate_up, &naive_syrk(&lt.ln2_out, 3, E), tol(3, 9.0));
        close(&fl.down, &naive_syrk(&lt.m, 3, FF), tol(3, 9.0));
        assert_eq!(fl.tokens, 3);
    }
}

#[test]
fn syrk_is_symmetric_and_psd() {
    let c = cfg();
    let mut s = 23u64;
    let tape = two_layer_tape(&mut s, 3);
    let mut g = SyrkAcc::new(&c).unwrap();
    g.accumulate(&tape, &c).unwrap();
    let fin = g.finish(Normalisation::Sum).unwrap();
    for fl in &fin {
        for (h, d) in [(&fl.qkv, E), (&fl.o, QW), (&fl.gate_up, E), (&fl.down, FF)] {
            for i in 0..d {
                for j in 0..d {
                    assert_eq!(h[i * d + j], h[j * d + i], "asymmetric at ({i},{j})");
                }
            }
            // PSD: `H + 1e-3·I` is positive-definite, so `Cholesky` succeeds.
            let mut a = h.clone();
            for i in 0..d {
                a[i * d + i] += 1e-3;
            }
            assert!(
                ridge_oracle::cholesky(&a, d).is_ok(),
                "H + εI is not positive-definite at d={d}"
            );
        }
    }
}

#[test]
fn blocks_add_like_concatenated_rows() {
    let c = cfg();
    let mut s = 37u64;
    let whole = two_layer_tape(&mut s, 3);
    let first = tape_from(
        2,
        (0..L)
            .map(|l| slice_layer(&whole.layers[l], 0, 2))
            .collect(),
    );
    let second = tape_from(
        1,
        (0..L)
            .map(|l| slice_layer(&whole.layers[l], 2, 3))
            .collect(),
    );
    let mut split = SyrkAcc::new(&c).unwrap();
    split.accumulate(&first, &c).unwrap();
    split.accumulate(&second, &c).unwrap();
    let mut one = SyrkAcc::new(&c).unwrap();
    one.accumulate(&whole, &c).unwrap();
    let split_fin = split.finish(Normalisation::Sum).unwrap();
    let one_fin = one.finish(Normalisation::Sum).unwrap();
    // The carrier's terms are the same rounded words in both groupings and the `i64` sum is exact,
    // so additivity over token blocks is bit-exact, not merely close.
    layers_close(&split_fin, &one_fin, 0.0);
    for lg in &split_fin {
        assert_eq!(lg.tokens, 3);
    }
}

#[test]
fn mean_divides_by_tokens_and_mean_diagonal_is_one() {
    let c = cfg();
    let mut s = 53u64;
    let tape = two_layer_tape(&mut s, 3);
    let mut g = SyrkAcc::new(&c).unwrap();
    g.accumulate(&tape, &c).unwrap();
    let sum = g.finish(Normalisation::Sum).unwrap();
    let mean = g.finish(Normalisation::Mean).unwrap();
    let md = g
        .finish(Normalisation::MeanDiagonal { eps: 42_949_673 })
        .unwrap();
    for l in 0..L {
        for (a, b) in [(&sum[l].qkv, &mean[l].qkv), (&sum[l].down, &mean[l].down)] {
            let scaled: Vec<f64> = a.iter().map(|v| v / 3.0).collect();
            // The division by the token count is `round(H·1/3)` words: the sum's `tol(3, 9)` bound
            // divided by 3, plus at most half a quantum from the quotient's own rounding.
            close(&scaled, b, tol(3, 9.0) / 3.0 + 2f64.powi(-33));
        }
        for (h, d) in [
            (&md[l].qkv, E),
            (&md[l].o, QW),
            (&md[l].gate_up, E),
            (&md[l].down, FF),
        ] {
            let mut diag = 0.0f64;
            for i in 0..d {
                diag += h[i * d + i];
                for j in 0..d {
                    assert_eq!(h[i * d + j], h[j * d + i]);
                }
            }
            assert!(
                (diag / d as f64 - 1.01).abs() < 2.0 * tol(3, 9.0) / 3.0,
                "mean diagonal {}",
                diag / d as f64
            );
        }
    }
}

#[test]
fn shape_refusals() {
    let c = cfg();
    let mut s = 71u64;

    // A tape with fewer layers than the config.
    let mut g = SyrkAcc::new(&c).unwrap();
    let short = tape_from(2, vec![one_layer(&mut s, 2)]);
    assert!(matches!(
        g.accumulate(&short, &c),
        Err(CaptureError::Shape(_))
    ));

    // A stage buffer whose length is not `t · w` on the second layer.
    let mut bad = two_layer_tape(&mut s, 2);
    bad.layers[1].m.pop();
    assert!(matches!(
        g.accumulate(&bad, &c),
        Err(CaptureError::Shape(_))
    ));

    // A token count that disagrees with the buffers.
    let mut bad_t = two_layer_tape(&mut s, 2);
    bad_t.t = 3;
    assert!(matches!(
        g.accumulate(&bad_t, &c),
        Err(CaptureError::Shape(_))
    ));

    // A config with different geometry than the accumulator was built for.
    let good = two_layer_tape(&mut s, 2);
    let mut other = cfg();
    other.n_embd = E + 1;
    assert!(matches!(
        g.accumulate(&good, &other),
        Err(CaptureError::Shape(_))
    ));

    // The mean over no tokens is undefined.
    let empty = SyrkAcc::new(&c).unwrap();
    assert!(matches!(
        empty.finish(Normalisation::Mean),
        Err(CaptureError::Shape(_))
    ));

    // A `q_out` that leaves `usize`, and a zero model dimension.
    let mut big = cfg();
    big.n_head = usize::MAX;
    assert!(matches!(SyrkAcc::new(&big), Err(CaptureError::Overflow)));
    let mut zero = cfg();
    zero.n_embd = 0;
    assert!(matches!(SyrkAcc::new(&zero), Err(CaptureError::Shape(_))));
}
