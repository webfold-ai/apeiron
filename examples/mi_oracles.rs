//! `mi_oracles` — mechanistic-interpretability identities, tested on the exact-integer pipeline.
//!
//! ```text
//! cargo run --release --example mi_oracles -- <edit|leace|all>
//! ```
//!
//! Every experiment prints its PREDICTIONS first (derived on paper, thresholds fixed in this
//! file before the first run), then the measurement, then PASS or FAIL per prediction. The data
//! are **SYNTHETIC**: they are generated in-program from a seeded generator, so a result here is
//! a check that an identity holds on our pipeline, not a measurement on a real model.
//!
//! * `edit`  — a ROME/MEMIT-style weight edit is a weighted pseudo-observation added to the
//!   exact `(G, B)`: finite weight `w` gives `W' = W + w/(1+w·s)·(v − W k)(C⁻¹k)ᵀ` with the
//!   leverage `s = kᵀC⁻¹k`; ROME is the `w → ∞` limit; the edit is reversed bit for bit by
//!   integer subtraction. (Sources: arXiv:2202.05262 Eq. 2; arXiv:2210.07229 Eq. 14.)
//! * `fourier` — modular addition mod 113: the closed-form learner as a ground-truth oracle for the Fourier mechanism
//!   (arXiv:2301.05217), its data-fraction curve, controls, and circulant spectral discovery with `top_eigh`.
//! * `super` — the Toy Models of Superposition data process (arXiv:2209.10652) as ground truth: spectral atoms vs
//!   a ridge decoder on the true codes, a positive control, a flat-spectrum case, and a shuffled control.
//! * `assoc` — a ridge associative memory: write = add, read = ridge contraction, forget = exact subtraction,
//!   uncertainty = leverage; capacity, Hebbian crosstalk, key noise, and the lambda sweep at n = d.
//! * `leace` — closed-form linear concept erasure (arXiv:2306.03819 Thm 4.2) from the additive
//!   statistics `n, Σx, Σxxᵀ, Σxzᵀ` and one Cholesky solve; shard merge and subtract.
//!
//! Scope, as in `paper/CLAIMS.yaml` F2: exact subtraction is established for a frozen-feature
//! ridge head only. Nothing here extends it to a trained encoder.

#![allow(clippy::all, clippy::pedantic, reason = "a measurement binary")]

use apeiron::learn::gram::gram_fold_i8;
use apeiron::learn::solve::chol::chol_solve;
use apeiron::learn::solve::eig::top_eigh;
use std::process::ExitCode;

const Q: f64 = 4_294_967_296.0;

// ---------------------------------------------------------------------------------------------
// A seeded generator, no dependencies.
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximately standard normal (sum of twelve uniforms).
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

fn q8(v: f64) -> i8 {
    v.round().clamp(-127.0, 127.0) as i8
}

// ---------------------------------------------------------------------------------------------
// The exact statistics and the Q32.32 solve.
// ---------------------------------------------------------------------------------------------

/// `G = XᵀX` of `i8` rows, exact in `i64`.
fn gram(x: &[i8], rows: usize, d: usize) -> Vec<i64> {
    gram_fold_i8(vec![0i64; d * d], x, rows, d).expect("gram_fold_i8")
}

/// Solve `(G/S + λI) X = B/S` in Q32.32 and return the raw Q32.32 solution. `S` is fixed by the
/// caller, so two arms with the same integer `(G, B)` get the same input to `chol_solve`.
fn solve_q(g: &[i64], b: &[i64], d: usize, k: usize, s: f64, lam: f64) -> Vec<i64> {
    let mut a: Vec<i64> = g
        .iter()
        .map(|&v| ((v as f64 / s) * Q).round() as i64)
        .collect();
    let lq = (lam * Q).round() as i64;
    for i in 0..d {
        a[i * d + i] += lq;
    }
    let bq: Vec<i64> = b
        .iter()
        .map(|&v| ((v as f64 / s) * Q).round() as i64)
        .collect();
    chol_solve(&a, &bq, d, k).expect("chol_solve").x
}

/// `C⁻¹ v` for a real vector `v`, `C = G/S + λI`.
fn cinv_vec(g: &[i64], d: usize, s: f64, lam: f64, v: &[f64]) -> Vec<f64> {
    let mut a: Vec<i64> = g
        .iter()
        .map(|&x| ((x as f64 / s) * Q).round() as i64)
        .collect();
    let lq = (lam * Q).round() as i64;
    for i in 0..d {
        a[i * d + i] += lq;
    }
    let bq: Vec<i64> = v.iter().map(|&x| (x * Q).round() as i64).collect();
    chol_solve(&a, &bq, d, 1)
        .expect("chol_solve")
        .x
        .iter()
        .map(|&x| x as f64 / Q)
        .collect()
}

fn to_f(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| v as f64 / Q).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `ŷ = Xᵀ_solᵀ x̃` for one row `x̃` (`d`), solution `d × k` row-major.
fn predict(sol: &[f64], d: usize, k: usize, xt: &[f64]) -> Vec<f64> {
    (0..k)
        .map(|c| (0..d).map(|j| sol[j * k + c] * xt[j]).sum())
        .collect()
}

struct Check {
    name: &'static str,
    pass: bool,
    detail: String,
}

fn report(title: &str, checks: &[Check]) -> bool {
    println!("\n== {title} ==");
    let mut all = true;
    for c in checks {
        println!(
            "  [{}] {} — {}",
            if c.pass { "PASS" } else { "FAIL" },
            c.name,
            c.detail
        );
        all &= c.pass;
    }
    all
}

// ---------------------------------------------------------------------------------------------
// E-EDIT: a weight edit is a weighted pseudo-observation.
// ---------------------------------------------------------------------------------------------

fn edit() -> bool {
    println!("\n--- E-EDIT (SYNTHETIC data) ---");
    println!("Setup: d=32 i8 features, k=8 classes, n=4000 train rows, ridge lambda=1e-3 (normalised units).");
    println!("Normalisation: x~ = x/127, S = n*127^2, targets are 127 on the class entry, so C = G/S + lam*I and");
    println!("B/S = (1/n) sum x~ y~^T with y~ the one-hot. A pseudo-row of weight w adds w/n to the normalised sums.");
    println!("PREDICTIONS (fixed before the first run):");
    println!("  P1 integer: the Gram of the data with w copies of the key row appended equals G + w k k^T bit for bit.");
    println!("  P2 closed form: solved W' equals W + [w_n/(1+w_n s)] (v - W k)(C^-1 k)^T, max abs diff < 1e-6, w_n = w/n.");
    println!("  P3 efficacy: W'k - v = (v - W k)/(1 + w_n s) to < 1e-6.");
    println!(
        "  P4 specificity: (W' - W) x = coef (v - W k)(u.x) on held-out rows, max abs diff < 1e-6."
    );
    println!("  P5 ROME limit: the hard-constraint ROME update (coef = 1/s) differs from the solved W' by a quantity that");
    println!("     shrinks as w grows and is < 1e-3 at w = 1e8.");
    println!("  P6 reversal: subtracting w k k^T and w k v^T from the integer (G', B') restores (G, B) bit for bit, and the");
    println!("     re-solved Q32.32 solution equals the original solution bit for bit.");
    println!("  P7 composition: two edits applied in either order give identical integer state, and removing the first");
    println!("     leaves the state of the second alone, bit for bit.");

    let (d, k, n, nt) = (32usize, 8usize, 4000usize, 2000usize);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // A hidden linear teacher.
    let teacher: Vec<f64> = (0..k * d).map(|_| rng.gauss()).collect();
    let mut make = |rows: usize, rng: &mut Rng| {
        let mut x = vec![0i8; rows * d];
        let mut y = vec![0usize; rows];
        for r in 0..rows {
            let xf: Vec<f64> = (0..d).map(|_| rng.gauss() * 40.0).collect();
            for j in 0..d {
                x[r * d + j] = q8(xf[j]);
            }
            let mut best = (f64::MIN, 0usize);
            for c in 0..k {
                let sc: f64 =
                    (0..d).map(|j| teacher[c * d + j] * xf[j]).sum::<f64>() + rng.gauss() * 40.0;
                if sc > best.0 {
                    best = (sc, c);
                }
            }
            y[r] = best.1;
        }
        (x, y)
    };
    let (x, y) = make(n, &mut rng);
    let (xt, _yt) = make(nt, &mut rng);

    let s = (n as f64) * 127.0 * 127.0;
    let lam = 1e-3;
    let g = gram(&x, n, d);
    let mut b = vec![0i64; d * k];
    for r in 0..n {
        for j in 0..d {
            b[j * k + y[r]] += i64::from(x[r * d + j]) * 127;
        }
    }
    let sol0 = solve_q(&g, &b, d, k, s, lam);
    let w0 = to_f(&sol0);

    // The edit: a held-out row relabelled to a class it is not currently predicted as.
    let key: Vec<i8> = xt[0..d].to_vec();
    let kt: Vec<f64> = key.iter().map(|&v| f64::from(v) / 127.0).collect();
    let pred0 = predict(&w0, d, k, &kt);
    let cur = (0..k)
        .max_by(|&a, &b2| pred0[a].partial_cmp(&pred0[b2]).unwrap())
        .unwrap();
    let target = (cur + 3) % k;
    let mut v = vec![0.0; k];
    v[target] = 1.0;

    let u = cinv_vec(&g, d, s, lam, &kt);
    let sv: f64 = kt.iter().zip(&u).map(|(a, b2)| a * b2).sum();
    println!("\nMeasured: baseline predicted class {cur}, edit target {target}, leverage s = k~ᵀC⁻¹k~ = {sv:.4}");

    let mut checks = Vec::new();

    // P1: integer equality of the Gram, via the real kernel on a stacked dataset.
    for &w in &[1usize, 5, 40] {
        let mut stacked = x.clone();
        for _ in 0..w {
            stacked.extend_from_slice(&key);
        }
        let g_stack = gram(&stacked, n + w, d);
        let mut g_edit = g.clone();
        for i in 0..d {
            for j in 0..d {
                g_edit[i * d + j] += (w as i64) * i64::from(key[i]) * i64::from(key[j]);
            }
        }
        checks.push(Check {
            name: "P1 stacked-data Gram == G + w k kᵀ",
            pass: g_stack == g_edit,
            detail: format!(
                "w={w}: {}",
                if g_stack == g_edit {
                    "bit-identical"
                } else {
                    "DIFFERENT"
                }
            ),
        });
    }

    // P2-P5 across weights.
    let ws: [i64; 6] = [1, 10, 1_000, 100_000, 10_000_000, 100_000_000];
    let mut p2 = 0f64;
    let mut p3 = 0f64;
    let mut p4 = 0f64;
    let mut rome_gap = Vec::new();
    for &w in &ws {
        let mut g2 = g.clone();
        let mut b2 = b.clone();
        for i in 0..d {
            for j in 0..d {
                g2[i * d + j] += w * i64::from(key[i]) * i64::from(key[j]);
            }
            for c in 0..k {
                b2[i * k + c] += w * i64::from(key[i]) * if c == target { 127 } else { 0 };
            }
        }
        let solw = to_f(&solve_q(&g2, &b2, d, k, s, lam));
        let wn = (w as f64) / (n as f64);
        let coef = wn / (1.0 + wn * sv);
        let resid: Vec<f64> = (0..k).map(|c| v[c] - pred0[c]).collect();
        let mut closed = w0.clone();
        for j in 0..d {
            for c in 0..k {
                closed[j * k + c] += coef * u[j] * resid[c];
            }
        }
        p2 = p2.max(max_abs_diff(&solw, &closed));
        // P3: efficacy gap.
        let predw = predict(&solw, d, k, &kt);
        let gap_meas: Vec<f64> = (0..k).map(|c| predw[c] - v[c]).collect();
        let gap_pred: Vec<f64> = (0..k)
            .map(|c| (pred0[c] - v[c]) / (1.0 + wn * sv))
            .collect();
        p3 = p3.max(max_abs_diff(&gap_meas, &gap_pred));
        // P4: specificity law on held-out rows.
        for r in 1..200 {
            let xr: Vec<f64> = xt[r * d..(r + 1) * d]
                .iter()
                .map(|&q| f64::from(q) / 127.0)
                .collect();
            let dot: f64 = xr.iter().zip(&u).map(|(a, b2)| a * b2).sum();
            let before = predict(&w0, d, k, &xr);
            let after = predict(&solw, d, k, &xr);
            let delta_meas: Vec<f64> = (0..k).map(|c| after[c] - before[c]).collect();
            let delta_pred: Vec<f64> = (0..k).map(|c| coef * resid[c] * dot).collect();
            p4 = p4.max(max_abs_diff(&delta_meas, &delta_pred));
        }
        // P5: ROME hard-constraint form.
        let rome_coef = 1.0 / sv;
        let mut rome = w0.clone();
        for j in 0..d {
            for c in 0..k {
                rome[j * k + c] += rome_coef * u[j] * resid[c];
            }
        }
        rome_gap.push((w, max_abs_diff(&solw, &rome), (predw[target] - 1.0).abs()));
    }
    checks.push(Check {
        name: "P2 closed form W' = W + w/(1+ws)(v−Wk)(C⁻¹k)ᵀ",
        pass: p2 < 1e-6,
        detail: format!("max abs diff over w in 1..1e8 = {p2:.3e} (threshold 1e-6)"),
    });
    checks.push(Check {
        name: "P3 efficacy gap = (v−Wk)/(1+ws)",
        pass: p3 < 1e-6,
        detail: format!("max abs diff = {p3:.3e}"),
    });
    checks.push(Check {
        name: "P4 specificity law (W'−W)x = coef (v−Wk)(u·x)",
        pass: p4 < 1e-6,
        detail: format!("max abs diff over 199 held-out rows and 6 weights = {p4:.3e}"),
    });
    let last = rome_gap.last().unwrap();
    let monotone = rome_gap.windows(2).all(|p| p[1].1 <= p[0].1 + 1e-12);
    checks.push(Check {
        name: "P5 ROME is the w→∞ limit",
        pass: last.1 < 1e-3 && monotone,
        detail: format!(
            "gap to ROME by w: {}; monotone={monotone}",
            rome_gap
                .iter()
                .map(|(w, g2, e)| format!("w={w:e}: {g2:.2e} (efficacy err {e:.2e})"))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    });

    // P6: reversal, bit for bit.
    let w6 = 1000i64;
    let mut g2 = g.clone();
    let mut b2 = b.clone();
    for i in 0..d {
        for j in 0..d {
            g2[i * d + j] += w6 * i64::from(key[i]) * i64::from(key[j]);
        }
        for c in 0..k {
            b2[i * k + c] += w6 * i64::from(key[i]) * if c == target { 127 } else { 0 };
        }
    }
    let edited = solve_q(&g2, &b2, d, k, s, lam);
    for i in 0..d {
        for j in 0..d {
            g2[i * d + j] -= w6 * i64::from(key[i]) * i64::from(key[j]);
        }
        for c in 0..k {
            b2[i * k + c] -= w6 * i64::from(key[i]) * if c == target { 127 } else { 0 };
        }
    }
    let back = solve_q(&g2, &b2, d, k, s, lam);
    checks.push(Check {
        name: "P6 reversal restores (G,B) and the solution bit for bit",
        pass: g2 == g && b2 == b && back == sol0 && edited != sol0,
        detail: format!(
            "G restored: {}, B restored: {}, solution restored: {}, edit changed the solution: {}",
            g2 == g,
            b2 == b,
            back == sol0,
            edited != sol0
        ),
    });

    // P7: two edits commute; removing one leaves the other.
    let key2: Vec<i8> = xt[d * 7..d * 8].to_vec();
    let tgt2 = (target + 2) % k;
    let apply = |g: &mut Vec<i64>, b: &mut Vec<i64>, key: &[i8], t: usize, w: i64, sign: i64| {
        for i in 0..d {
            for j in 0..d {
                g[i * d + j] += sign * w * i64::from(key[i]) * i64::from(key[j]);
            }
            for c in 0..k {
                b[i * k + c] += sign * w * i64::from(key[i]) * if c == t { 127 } else { 0 };
            }
        }
    };
    let (mut ga, mut ba) = (g.clone(), b.clone());
    apply(&mut ga, &mut ba, &key, target, 500, 1);
    apply(&mut ga, &mut ba, &key2, tgt2, 300, 1);
    let (mut gb, mut bb) = (g.clone(), b.clone());
    apply(&mut gb, &mut bb, &key2, tgt2, 300, 1);
    apply(&mut gb, &mut bb, &key, target, 500, 1);
    let commute = ga == gb && ba == bb;
    let (mut g_only2, mut b_only2) = (g.clone(), b.clone());
    apply(&mut g_only2, &mut b_only2, &key2, tgt2, 300, 1);
    let (mut g_rm, mut b_rm) = (ga.clone(), ba.clone());
    apply(&mut g_rm, &mut b_rm, &key, target, 500, -1);
    checks.push(Check {
        name: "P7 edits commute; removing one leaves the other",
        pass: commute && g_rm == g_only2 && b_rm == b_only2,
        detail: format!(
            "commute: {commute}; remove-first == second-only: {}",
            g_rm == g_only2 && b_rm == b_only2
        ),
    });

    report("E-EDIT", &checks)
}

// ---------------------------------------------------------------------------------------------
// E-LEACE: closed-form linear concept erasure from additive statistics.
// ---------------------------------------------------------------------------------------------

/// The additive statistics of a binary concept over `i8` rows, all exact integers.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Stats {
    n: i64,
    sx: Vec<i64>,
    g: Vec<i64>,
    sxz: Vec<i64>, // Σ x·z with z in {0,1}
    nz: i64,
}

impl Stats {
    fn of(x: &[i8], z: &[u8], rows: usize, d: usize) -> Self {
        let g = gram(x, rows, d);
        let mut sx = vec![0i64; d];
        let mut sxz = vec![0i64; d];
        let mut nz = 0i64;
        for r in 0..rows {
            for j in 0..d {
                sx[j] += i64::from(x[r * d + j]);
                if z[r] == 1 {
                    sxz[j] += i64::from(x[r * d + j]);
                }
            }
            nz += i64::from(z[r]);
        }
        Stats {
            n: rows as i64,
            sx,
            g,
            sxz,
            nz,
        }
    }
    fn merge(&mut self, o: &Stats, sign: i64) {
        self.n += sign * o.n;
        self.nz += sign * o.nz;
        for (a, b) in self.sx.iter_mut().zip(&o.sx) {
            *a += sign * b;
        }
        for (a, b) in self.g.iter_mut().zip(&o.g) {
            *a += sign * b;
        }
        for (a, b) in self.sxz.iter_mut().zip(&o.sxz) {
            *a += sign * b;
        }
    }
}

/// The eraser `r(x) = x − m − ... ` as the pair (a, c): `r(x~) = x~ − a · (cᵀ (x~ − μ))`,
/// `a = Σ_xz / s`, `c = Σ⁻¹ Σ_xz`, `s = Σ_xzᵀ Σ⁻¹ Σ_xz`, in x~ = x/127 units. Single binary
/// concept, so the pseudo-inverse is a scalar reciprocal.
fn eraser(st: &Stats, d: usize, eps: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>) {
    let n = st.n as f64;
    let mu: Vec<f64> = st.sx.iter().map(|&v| v as f64 / n / 127.0).collect();
    let pz = st.nz as f64 / n;
    // Σ~ = G/(n·127²) − μμᵀ ; Σ~_xz = Σxz/(n·127) − μ·pz  (all from exact integers).
    let mut sigma_q = vec![0i64; d * d];
    for i in 0..d {
        for j in 0..d {
            let v = st.g[i * d + j] as f64 / (n * 127.0 * 127.0) - mu[i] * mu[j];
            sigma_q[i * d + j] = (v * Q).round() as i64;
        }
        sigma_q[i * d + i] += (eps * Q).round() as i64;
    }
    let sxz: Vec<f64> = (0..d)
        .map(|j| st.sxz[j] as f64 / (n * 127.0) - mu[j] * pz)
        .collect();
    let bq: Vec<i64> = sxz.iter().map(|&v| (v * Q).round() as i64).collect();
    let solved = chol_solve(&sigma_q, &bq, d, 1).expect("chol_solve");
    let c: Vec<f64> = to_f(&solved.x);
    let s: f64 = sxz.iter().zip(&c).map(|(a, b)| a * b).sum();
    let a: Vec<f64> = sxz.iter().map(|&v| v / s).collect();
    (a, c, mu, solved.x)
}

fn erase(a: &[f64], c: &[f64], mu: &[f64], x: &[f64]) -> Vec<f64> {
    let t: f64 = c
        .iter()
        .zip(x.iter().zip(mu))
        .map(|(ci, (xi, mi))| ci * (xi - mi))
        .sum();
    x.iter().zip(a).map(|(xi, ai)| xi - ai * t).collect()
}

fn leace() -> bool {
    println!("\n--- E-LEACE (SYNTHETIC data) ---");
    println!("Setup: d=24 i8 features with strongly CORRELATED noise (x = M g, M random mixing) plus a planted concept");
    println!("direction a_dir: x += 30*z*a_dir, z in {{0,1}} balanced. Train n=6000, test n=3000. Eraser from exact integer");
    println!("statistics (n, Σx, Σxxᵀ, Σxz) and one Cholesky solve, single binary concept (the pseudo-inverse is a scalar).");
    println!("PREDICTIONS (fixed before the first run):");
    println!("  L1 guardedness: the covariance between erased features and z is < 1e-6 of its pre-erasure norm.");
    println!("  L2 probe: a closed-form ridge probe fitted on erased TRAIN data predicts z on TEST at <= 0.55 (base rate 0.5);");
    println!("     the same probe on un-erased data is >= 0.90.");
    println!("  L3 non-isotropy matters: ORTHOGONAL projection along Σ_xz (the naive 'remove the mean-difference direction')");
    println!("     leaves the probe >= 0.70 because the optimal probe direction is Σ⁻¹Σ_xz, not Σ_xz, when Σ is not ∝ I.");
    println!("  L4 control: with z shuffled the eraser changes the data by < 5% of the LEACE change on true z.");
    println!("  L5 monoid: the eraser from 8 merged shards, and from the whole minus one shard, equals the eraser from a");
    println!("     refit on the same rows, with bit-identical integer statistics and bit-identical Q32.32 solve output.");

    let (d, n, nt) = (24usize, 6000usize, 3000usize);
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mix: Vec<f64> = (0..d * d).map(|_| rng.gauss() * 0.6).collect();
    let adir: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
    let mut gen = |rows: usize, rng: &mut Rng| {
        let mut x = vec![0i8; rows * d];
        let mut z = vec![0u8; rows];
        for r in 0..rows {
            let gv: Vec<f64> = (0..d).map(|_| rng.gauss() * 25.0).collect();
            let zz = u8::from(rng.unit() < 0.5);
            z[r] = zz;
            for i in 0..d {
                let m: f64 = (0..d).map(|j| mix[i * d + j] * gv[j]).sum();
                x[r * d + i] = q8(m + 30.0 * f64::from(zz) * adir[i]);
            }
        }
        (x, z)
    };
    let (x, z) = gen(n, &mut rng);
    let (xt, zt) = gen(nt, &mut rng);

    let st = Stats::of(&x, &z, n, d);
    let eps = 1e-6;
    let (a, c, mu, _raw) = eraser(&st, d, eps);

    let to_tilde = |row: &[i8]| -> Vec<f64> { row.iter().map(|&v| f64::from(v) / 127.0).collect() };
    let erased_train: Vec<Vec<f64>> = (0..n)
        .map(|r| erase(&a, &c, &mu, &to_tilde(&x[r * d..(r + 1) * d])))
        .collect();
    let erased_test: Vec<Vec<f64>> = (0..nt)
        .map(|r| erase(&a, &c, &mu, &to_tilde(&xt[r * d..(r + 1) * d])))
        .collect();

    // Covariance of erased features with z (f64, from the data).
    let cov_xz = |rows: &[Vec<f64>], zs: &[u8]| -> Vec<f64> {
        let nn = rows.len() as f64;
        let mx: Vec<f64> = (0..d)
            .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / nn)
            .collect();
        let mz = zs.iter().map(|&v| f64::from(v)).sum::<f64>() / nn;
        (0..d)
            .map(|j| {
                rows.iter()
                    .zip(zs)
                    .map(|(r, &zv)| (r[j] - mx[j]) * (f64::from(zv) - mz))
                    .sum::<f64>()
                    / nn
            })
            .collect()
    };
    let raw_train: Vec<Vec<f64>> = (0..n).map(|r| to_tilde(&x[r * d..(r + 1) * d])).collect();
    let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
    let before = norm(&cov_xz(&raw_train, &z));
    let after = norm(&cov_xz(&erased_train, &z));

    // A closed-form ridge probe for z, fitted on train features, evaluated on test.
    let probe = |tr: &[Vec<f64>], te: &[Vec<f64>]| -> f64 {
        // Build exact integer Gram of the quantised features? Probes here are f64 ridge for the evaluation only.
        let nn = tr.len() as f64;
        let mx: Vec<f64> = (0..d)
            .map(|j| tr.iter().map(|r| r[j]).sum::<f64>() / nn)
            .collect();
        let mut gm = vec![0f64; d * d];
        let mut bv = vec![0f64; d];
        for (r, &zv) in tr.iter().zip(&z) {
            for i in 0..d {
                let xi = r[i] - mx[i];
                bv[i] += xi * (f64::from(zv) - 0.5);
                for j in 0..d {
                    gm[i * d + j] += xi * (r[j] - mx[j]);
                }
            }
        }
        for i in 0..d {
            gm[i * d + i] += 1e-6 * nn;
        }
        // Gaussian elimination (d is small; this is the evaluation instrument, not the method under test).
        let mut m = gm.clone();
        let mut rhs = bv.clone();
        for col in 0..d {
            let piv = (col..d)
                .max_by(|&p, &q| {
                    m[p * d + col]
                        .abs()
                        .partial_cmp(&m[q * d + col].abs())
                        .unwrap()
                })
                .unwrap();
            for j in 0..d {
                m.swap(col * d + j, piv * d + j);
            }
            rhs.swap(col, piv);
            for r in col + 1..d {
                let f = m[r * d + col] / m[col * d + col];
                for j in col..d {
                    m[r * d + j] -= f * m[col * d + j];
                }
                rhs[r] -= f * rhs[col];
            }
        }
        let mut wv = vec![0f64; d];
        for r in (0..d).rev() {
            let t: f64 = (r + 1..d).map(|j| m[r * d + j] * wv[j]).sum();
            wv[r] = (rhs[r] - t) / m[r * d + r];
        }
        let correct = te
            .iter()
            .zip(&zt)
            .filter(|(r, &zv)| {
                let sc: f64 = (0..d).map(|j| wv[j] * (r[j] - mx[j])).sum();
                u8::from(sc > 0.0) == zv
            })
            .count();
        correct as f64 / te.len() as f64
    };
    let raw_test: Vec<Vec<f64>> = (0..nt).map(|r| to_tilde(&xt[r * d..(r + 1) * d])).collect();
    let acc_raw = probe(&raw_train, &raw_test);
    let acc_leace = probe(&erased_train, &erased_test);

    // Naive orthogonal projection along Σ_xz.
    let nzv = norm(&{
        let nf = st.n as f64;
        (0..d)
            .map(|j| st.sxz[j] as f64 / (nf * 127.0) - mu[j] * (st.nz as f64 / nf))
            .collect::<Vec<f64>>()
    });
    let sxz_dir: Vec<f64> = (0..d)
        .map(|j| {
            (st.sxz[j] as f64 / (st.n as f64 * 127.0) - mu[j] * (st.nz as f64 / st.n as f64)) / nzv
        })
        .collect();
    let ortho = |row: &[f64]| -> Vec<f64> {
        let t: f64 = row.iter().zip(&sxz_dir).map(|(a, b)| a * b).sum();
        row.iter().zip(&sxz_dir).map(|(a, b)| a - t * b).collect()
    };
    let ortho_train: Vec<Vec<f64>> = raw_train.iter().map(|r| ortho(r)).collect();
    let ortho_test: Vec<Vec<f64>> = raw_test.iter().map(|r| ortho(r)).collect();
    let acc_ortho = probe(&ortho_train, &ortho_test);

    // L4 control: shuffled z.
    let mut zs = z.clone();
    for i in (1..zs.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        zs.swap(i, j);
    }
    let st_shuf = Stats::of(&x, &zs, n, d);
    let (a2, c2, mu2, _) = eraser(&st_shuf, d, eps);
    let change = |a: &[f64], c: &[f64], mu: &[f64]| -> f64 {
        (0..n)
            .map(|r| {
                let xr = to_tilde(&x[r * d..(r + 1) * d]);
                let e = erase(a, c, mu, &xr);
                e.iter()
                    .zip(&xr)
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n as f64
    };
    let ch_true = change(&a, &c, &mu);
    let ch_shuf = change(&a2, &c2, &mu2);

    // L5: shards.
    let shard_rows = n / 8;
    let shards: Vec<Stats> = (0..8)
        .map(|i| {
            Stats::of(
                &x[i * shard_rows * d..(i + 1) * shard_rows * d],
                &z[i * shard_rows..(i + 1) * shard_rows],
                shard_rows,
                d,
            )
        })
        .collect();
    let mut merged = shards[0].clone();
    for s in &shards[1..] {
        merged.merge(s, 1);
    }
    let whole = Stats::of(
        &x[..8 * shard_rows * d],
        &z[..8 * shard_rows],
        8 * shard_rows,
        d,
    );
    let (_, _, _, raw_merged) = eraser(&merged, d, eps);
    let (_, _, _, raw_whole) = eraser(&whole, d, eps);
    let mut minus3 = merged.clone();
    minus3.merge(&shards[3], -1);
    let mut rest_x = Vec::new();
    let mut rest_z = Vec::new();
    for i in (0..8).filter(|&i| i != 3) {
        rest_x.extend_from_slice(&x[i * shard_rows * d..(i + 1) * shard_rows * d]);
        rest_z.extend_from_slice(&z[i * shard_rows..(i + 1) * shard_rows]);
    }
    let refit = Stats::of(&rest_x, &rest_z, 7 * shard_rows, d);
    let (_, _, _, raw_minus3) = eraser(&minus3, d, eps);
    let (_, _, _, raw_refit) = eraser(&refit, d, eps);

    let checks = vec![
        Check {
            name: "L1 guardedness",
            pass: after < 1e-6 * before.max(1e-12),
            detail: format!(
                "‖Cov(erased x, z)‖ = {after:.3e} vs {before:.3e} before (ratio {:.2e})",
                after / before
            ),
        },
        Check {
            name: "L2 probe at base rate after LEACE, high before",
            pass: acc_leace <= 0.55 && acc_raw >= 0.90,
            detail: format!(
                "probe accuracy: raw {acc_raw:.3}, LEACE {acc_leace:.3} (base rate 0.5)"
            ),
        },
        Check {
            name: "L3 orthogonal projection does NOT erase (non-isotropic Σ)",
            pass: acc_ortho >= 0.70,
            detail: format!("probe accuracy after orthogonal projection {acc_ortho:.3}"),
        },
        Check {
            name: "L4 shuffled-z control changes the data far less",
            pass: ch_shuf < 0.05 * ch_true,
            detail: format!(
                "mean squared change: true z {ch_true:.4e}, shuffled {ch_shuf:.4e} (ratio {:.3})",
                ch_shuf / ch_true
            ),
        },
        Check {
            name: "L5 merge of 8 shards == whole, bit for bit",
            pass: merged == whole && raw_merged == raw_whole,
            detail: format!(
                "integer stats equal: {}; eraser solve output equal: {}",
                merged == whole,
                raw_merged == raw_whole
            ),
        },
        Check {
            name: "L5 whole minus one shard == refit on the rest, bit for bit",
            pass: minus3 == refit && raw_minus3 == raw_refit,
            detail: format!(
                "integer stats equal: {}; eraser solve output equal: {}",
                minus3 == refit,
                raw_minus3 == raw_refit
            ),
        },
    ];
    // ---- POST-HOC (formed AFTER L3 and L4 failed; exploratory, not pre-registered) -----------
    // Derivation: LEACE removes a·cᵀ(x−μ) with a = Σ_xz/s, c = Σ⁻¹Σ_xz, s = Σ_xzᵀΣ⁻¹Σ_xz, so
    // E‖Δx‖² = ‖a‖²·cᵀΣc = ‖Σ_xz‖²/s. With binary z, Σ_xz = ‖Σ_xz‖·â, so this is 1/(âᵀΣ⁻¹â).
    // The centred orthogonal projection along â changes x by âᵀ(x−μ)·â: E‖Δx‖² = âᵀΣâ.
    // Cauchy–Schwarz: (âᵀΣâ)(âᵀΣ⁻¹â) ≥ 1, so LEACE's change ≤ the projection's, equal iff â is an eigenvector.
    let sigma_f: Vec<f64> = {
        let nf = st.n as f64;
        let mut m = vec![0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] = st.g[i * d + j] as f64 / (nf * 127.0 * 127.0) - mu[i] * mu[j];
            }
        }
        m
    };
    let sxz_f: Vec<f64> = {
        let nf = st.n as f64;
        (0..d)
            .map(|j| st.sxz[j] as f64 / (nf * 127.0) - mu[j] * (st.nz as f64 / nf))
            .collect()
    };
    let s_scalar: f64 = sxz_f.iter().zip(&c).map(|(p, q)| p * q).sum();
    let predicted_change = norm(&sxz_f).powi(2) / s_scalar;
    let ahat: Vec<f64> = sxz_f.iter().map(|v| v / norm(&sxz_f)).collect();
    let ortho_change_pred: f64 = (0..d)
        .map(|i| {
            (0..d)
                .map(|j| ahat[i] * sigma_f[i * d + j] * ahat[j])
                .sum::<f64>()
        })
        .sum();
    // measured: centred orthogonal projection.
    let ortho_c = |row: &[f64]| -> Vec<f64> {
        let t: f64 = row
            .iter()
            .zip(&mu)
            .zip(&ahat)
            .map(|((x, m), a)| (x - m) * a)
            .sum();
        row.iter().zip(&ahat).map(|(x, a)| x - t * a).collect()
    };
    let ortho_c_train: Vec<Vec<f64>> = raw_train.iter().map(|r| ortho_c(r)).collect();
    let ortho_c_test: Vec<Vec<f64>> = raw_test.iter().map(|r| ortho_c(r)).collect();
    let ortho_change_meas = raw_train
        .iter()
        .zip(&ortho_c_train)
        .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>())
        .sum::<f64>()
        / n as f64;
    let acc_ortho_c = probe(&ortho_c_train, &ortho_c_test);
    // shuffled: the scale-invariance of the distortion.
    let sxz_shuf: Vec<f64> = {
        let nf = st_shuf.n as f64;
        let mu_s: Vec<f64> = st_shuf.sx.iter().map(|&v| v as f64 / nf / 127.0).collect();
        (0..d)
            .map(|j| st_shuf.sxz[j] as f64 / (nf * 127.0) - mu_s[j] * (st_shuf.nz as f64 / nf))
            .collect()
    };
    let s_shuf: f64 = sxz_shuf.iter().zip(&c2).map(|(p, q)| p * q).sum();
    let pred_shuf = norm(&sxz_shuf).powi(2) / s_shuf;
    let mut checks = checks;
    checks.push(Check {
        name: "POST-HOC H1 LEACE distortion = ‖Σ_xz‖²/s (analytic)",
        pass: ((predicted_change - ch_true) / ch_true).abs() < 1e-3,
        detail: format!("predicted {predicted_change:.5e}, measured {ch_true:.5e}"),
    });
    checks.push(Check {
        name: "POST-HOC H2 shuffled-z distortion is the same formula, NOT near zero",
        pass: ((pred_shuf - ch_shuf) / ch_shuf).abs() < 1e-3 && ch_shuf > 0.1 * ch_true,
        detail: format!(
            "‖Σ_xz‖ shuffled {:.3e} vs true {:.3e} (ratio {:.1e}), yet distortion {ch_shuf:.4e} vs {ch_true:.4e}; formula predicts {pred_shuf:.4e}",
            norm(&sxz_shuf),
            norm(&sxz_f),
            norm(&sxz_shuf) / norm(&sxz_f)
        ),
    });
    checks.push(Check {
        name: "POST-HOC H3 centred orthogonal projection also erases (binary z) but distorts at least as much as LEACE",
        pass: acc_ortho_c <= 0.55
            && ((ortho_change_pred - ortho_change_meas) / ortho_change_meas).abs() < 1e-3
            && predicted_change <= ortho_change_pred + 1e-12,
        detail: format!(
            "probe {acc_ortho_c:.3}; distortion: projection predicted {ortho_change_pred:.5e} measured {ortho_change_meas:.5e}; LEACE {predicted_change:.5e}"
        ),
    });
    println!(
        "\nMeasured: probe accuracy raw {acc_raw:.3} | LEACE {acc_leace:.3} | orthogonal {acc_ortho:.3}; ‖Cov(x,z)‖ {before:.3e} -> {after:.3e}"
    );
    report("E-LEACE", &checks)
}

// ---------------------------------------------------------------------------------------------
// f64 instrument (used ONLY where the experiment isolates the mathematics from i8 quantisation).
// ---------------------------------------------------------------------------------------------

/// Solve `(G + λI) X = B` in f64, `G` is `d×d`, `B` is `d×k`, by dense Cholesky.
fn solve_f64(g: &[f64], b: &[f64], d: usize, k: usize, lam: f64) -> Vec<f64> {
    let mut l = g.to_vec();
    for i in 0..d {
        l[i * d + i] += lam;
    }
    for j in 0..d {
        let mut s = l[j * d + j];
        for p in 0..j {
            s -= l[j * d + p] * l[j * d + p];
        }
        let dj = s.max(1e-300).sqrt();
        l[j * d + j] = dj;
        for i in j + 1..d {
            let mut t = l[i * d + j];
            for p in 0..j {
                t -= l[i * d + p] * l[j * d + p];
            }
            l[i * d + j] = t / dj;
        }
    }
    let mut x = b.to_vec();
    for c in 0..k {
        for i in 0..d {
            let mut t = x[i * k + c];
            for p in 0..i {
                t -= l[i * d + p] * x[p * k + c];
            }
            x[i * k + c] = t / l[i * d + i];
        }
        for i in (0..d).rev() {
            let mut t = x[i * k + c];
            for p in i + 1..d {
                t -= l[p * d + i] * x[p * k + c];
            }
            x[i * k + c] = t / l[i * d + i];
        }
    }
    x
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

// ---------------------------------------------------------------------------------------------
// E-FOURIER: modular addition, the known mechanism as a ground-truth oracle.
// ---------------------------------------------------------------------------------------------

const P: usize = 113;

/// Per-cell features for the frequency list: four products per frequency, then a constant.
/// `phi[(a*P+b)*d + j]`, `d = 4*len + 1`.
fn fourier_lift(freqs: &[f64]) -> (Vec<f64>, usize) {
    let d = 4 * freqs.len() + 1;
    let mut phi = vec![0f64; P * P * d];
    for a in 0..P {
        for b in 0..P {
            let row = &mut phi[(a * P + b) * d..(a * P + b + 1) * d];
            for (i, &f) in freqs.iter().enumerate() {
                let w = 2.0 * std::f64::consts::PI * f / P as f64;
                let (ca, sa, cb, sb) = (
                    (w * a as f64).cos(),
                    (w * a as f64).sin(),
                    (w * b as f64).cos(),
                    (w * b as f64).sin(),
                );
                row[4 * i] = ca * cb;
                row[4 * i + 1] = sa * sb;
                row[4 * i + 2] = sa * cb;
                row[4 * i + 3] = ca * sb;
            }
            row[d - 1] = 1.0;
        }
    }
    (phi, d)
}

/// A random product lift: `φ_j(a,b) = g_j[a]·h_j[b]` with random tables, plus a constant.
fn random_product_lift(width: usize, rng: &mut Rng) -> (Vec<f64>, usize) {
    let d = width + 1;
    let g: Vec<f64> = (0..width * P).map(|_| rng.gauss()).collect();
    let h: Vec<f64> = (0..width * P).map(|_| rng.gauss()).collect();
    let mut phi = vec![0f64; P * P * d];
    for a in 0..P {
        for b in 0..P {
            let row = &mut phi[(a * P + b) * d..(a * P + b + 1) * d];
            for j in 0..width {
                row[j] = g[j * P + a] * h[j * P + b] / 2.0;
            }
            row[d - 1] = 1.0;
        }
    }
    (phi, d)
}

/// Fit on the listed training cells (labels may be permuted by the caller), score on the test cells.
fn fit_and_score(
    phi: &[f64],
    d: usize,
    train: &[usize],
    labels: &[usize],
    test: &[usize],
    lam: f64,
) -> (f64, Vec<f64>) {
    let mut g = vec![0f64; d * d];
    let mut b = vec![0f64; d * P];
    for (&cell, &lab) in train.iter().zip(labels) {
        let row = &phi[cell * d..(cell + 1) * d];
        for i in 0..d {
            let ri = row[i];
            b[i * P + lab] += ri;
            for j in 0..=i {
                g[i * d + j] += ri * row[j];
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            g[j * d + i] = g[i * d + j];
        }
    }
    let w = solve_f64(&g, &b, d, P, lam);
    let mut ok = 0usize;
    for &cell in test {
        let row = &phi[cell * d..(cell + 1) * d];
        let (a, bb) = (cell / P, cell % P);
        let truth = (a + bb) % P;
        let mut best = (f64::MIN, 0usize);
        for c in 0..P {
            let sc: f64 = (0..d).map(|j| row[j] * w[j * P + c]).sum();
            if sc > best.0 {
                best = (sc, c);
            }
        }
        if best.1 == truth {
            ok += 1;
        }
    }
    (ok as f64 / test.len() as f64, w)
}

fn fourier() -> bool {
    println!("\n--- E-FOURIER: modular addition (a+b) mod 113, the Nanda et al. task (arXiv:2301.05217) (SYNTHETIC: exact table) ---");
    println!("Derivation: 1[a+b=c] = (1/p)[1 + 2 Σ_k cos w_k(a+b−c)], and cos w(a+b−c) = cos w(a+b)·cos wc + sin w(a+b)·sin wc,");
    println!("with cos w(a+b) = cos wa cos wb − sin wa sin wb and sin w(a+b) = sin wa cos wb + cos wa sin wb. So the target is EXACTLY");
    println!("linear in the four product features per frequency, and the closed-form ridge reproduces the 'Fourier mechanism' by");
    println!(
        "construction. THIS MAKES THE CLOSED-FORM LEARNER A GROUND-TRUTH ORACLE, NOT A DISCOVERY."
    );
    println!(
        "PREDICTIONS (fixed before the first run; quantities from the derivation, not tuned):"
    );
    println!("  F1 full table, f64: accuracy is 100% with ALL 56 frequencies, with a SINGLE frequency (k=1), and with 5 random ones");
    println!("     (cos(2πk x/p) is 1 only at x=0, runner-up cos(2π/p)=0.998, so one frequency already has the right argmax).");
    println!("  F2 data fraction, all-frequency lift (225 features), f64, tiny lambda: held-out accuracy >= 0.99 once the training");
    println!("     fraction is >= 5% (638 cells, 2.8x the feature count); at 0.5% (64 cells < 225) it is far lower; the curve has no");
    println!("     jump between adjacent grid points larger than 0.60 once n >= 225 (a kink at n = d, not a cliff).");
    println!("  F3 controls at EVERY fraction up to 30%: a random product lift of equal width, and the same Fourier lift fitted on");
    println!(
        "     label-shuffled training cells, both stay <= 0.03 held-out accuracy (chance 0.0088)."
    );
    println!("  F4 off-grid frequencies (random real frequencies, not characters of Z_113): held-out accuracy <= 0.10 at 30%.");
    println!("  F5 the i8 pipeline (gram_fold_i8 + chol_solve, features quantised to 127) is within 0.03 of the f64 accuracy at 5% and 30%.");
    println!("  F6 circulant discovery: for the symmetric operator S = A + Aᵀ with A[a][a+b] += w_b accumulated from cells, the Fourier");
    println!("     modes are its eigenvectors (DFT diagonalises circulants). On the FULL table top_eigh recovers the predicted top-8");
    println!("     eigenspace to subspace error < 1e-3 and the eigenvalues to relative error < 1e-3; with c shuffled the error is >= 0.80");
    println!("     (chance for random 8-dim subspaces of R^113 is sqrt(1−8/113) = 0.964); the error is non-increasing in the observed fraction.");

    let mut rng = Rng(0xA5A5_1234_DEAD_BEEF);
    let all_freqs: Vec<f64> = (1..=56).map(f64::from).collect();
    let cells: Vec<usize> = (0..P * P).collect();
    let truth_of = |cell: usize| (cell / P + cell % P) % P;
    let mut checks = Vec::new();

    // ---- F1
    let (phi_all, d_all) = fourier_lift(&all_freqs);
    let labels_full: Vec<usize> = cells.iter().map(|&c| truth_of(c)).collect();
    let (acc_all, _) = fit_and_score(&phi_all, d_all, &cells, &labels_full, &cells, 1e-6);
    let (phi_1, d_1) = fourier_lift(&[1.0]);
    let (acc_1, _) = fit_and_score(&phi_1, d_1, &cells, &labels_full, &cells, 1e-6);
    let mut five: Vec<f64> = Vec::new();
    while five.len() < 5 {
        let f = (1 + rng.next() % 56) as f64;
        if !five.contains(&f) {
            five.push(f);
        }
    }
    let (phi_5, d_5) = fourier_lift(&five);
    let (acc_5, _) = fit_and_score(&phi_5, d_5, &cells, &labels_full, &cells, 1e-6);
    checks.push(Check {
        name: "F1 full table: 56 freqs, 1 freq, 5 random freqs all exact",
        pass: acc_all == 1.0 && acc_1 == 1.0 && acc_5 == 1.0,
        detail: format!("accuracy: 56 freqs {acc_all:.4}, k=1 only {acc_1:.4}, 5 random {acc_5:.4} (freqs {five:?})"),
    });

    // ---- F2, F3, F4
    let (phi_rp, d_rp) = random_product_lift(224, &mut rng);
    let off: Vec<f64> = (0..56).map(|_| 0.5 + rng.unit() * 55.0).collect();
    let (phi_og, d_og) = fourier_lift(&off);
    let fracs = [
        0.005, 0.01, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10, 0.30,
    ];
    let seeds = 4;
    println!("\n  fraction  n_train | Fourier  | rand-product | shuffled-labels | off-grid freqs   (held-out accuracy, mean of {seeds} seeds, 2000 test cells)");
    let mut curve = Vec::new();
    let mut ctrl_max = 0f64;
    let mut og_at_30 = 0f64;
    for &f in &fracs {
        let n_tr = ((f * (P * P) as f64).round() as usize).max(1);
        let (mut sa, mut sr, mut ss, mut so) = (0f64, 0f64, 0f64, 0f64);
        for _ in 0..seeds {
            let mut perm = cells.clone();
            shuffle(&mut perm, &mut rng);
            let train = &perm[..n_tr];
            let test = &perm[n_tr..n_tr + 2000];
            let labels: Vec<usize> = train.iter().map(|&c| truth_of(c)).collect();
            let mut shuf = labels.clone();
            shuffle(&mut shuf, &mut rng);
            sa += fit_and_score(&phi_all, d_all, train, &labels, test, 1e-6).0;
            sr += fit_and_score(&phi_rp, d_rp, train, &labels, test, 1e-6).0;
            ss += fit_and_score(&phi_all, d_all, train, &shuf, test, 1e-6).0;
            so += fit_and_score(&phi_og, d_og, train, &labels, test, 1e-6).0;
        }
        let k = seeds as f64;
        let (a, r, s2, o) = (sa / k, sr / k, ss / k, so / k);
        println!("  {f:>8.4}  {n_tr:>7} | {a:>8.4} | {r:>12.4} | {s2:>15.4} | {o:>8.4}");
        curve.push((f, n_tr, a));
        ctrl_max = ctrl_max.max(r).max(s2);
        if (f - 0.30).abs() < 1e-9 {
            og_at_30 = o;
        }
    }
    let acc_at = |f: f64| {
        curve
            .iter()
            .find(|c| (c.0 - f).abs() < 1e-9)
            .map(|c| c.2)
            .unwrap_or(f64::NAN)
    };
    let mut max_jump = 0f64;
    for w in curve.windows(2) {
        if w[0].1 >= d_all {
            max_jump = max_jump.max(w[1].2 - w[0].2);
        }
    }
    checks.push(Check {
        name: "F2 held-out accuracy >= 0.99 by 5%, low at 0.5%, no cliff after n >= d",
        pass: acc_at(0.05) >= 0.99 && acc_at(0.005) < 0.5 && max_jump <= 0.60,
        detail: format!(
            "acc(5%)={:.4}, acc(0.5%)={:.4}, acc(10%)={:.4}, acc(30%)={:.4}; largest adjacent jump once n>=225: {max_jump:.4}",
            acc_at(0.05),
            acc_at(0.005),
            acc_at(0.10),
            acc_at(0.30)
        ),
    });
    checks.push(Check {
        name: "F3 random-product lift and label-shuffled Fourier lift stay near chance",
        pass: ctrl_max <= 0.03,
        detail: format!(
            "max accuracy over all fractions and both controls: {ctrl_max:.4} (chance 0.0088)"
        ),
    });
    checks.push(Check {
        name: "F4 off-grid frequencies do not generalise",
        pass: og_at_30 <= 0.10,
        detail: format!("accuracy at 30% with 56 random real frequencies: {og_at_30:.4}"),
    });

    // ---- F5: the i8 pipeline.
    let mut f5 = Vec::new();
    for &f in &[0.05, 0.30] {
        let n_tr = ((f * (P * P) as f64).round() as usize).max(1);
        let mut perm = cells.clone();
        shuffle(&mut perm, &mut rng);
        let train = &perm[..n_tr];
        let test = &perm[n_tr..n_tr + 2000];
        let labels: Vec<usize> = train.iter().map(|&c| truth_of(c)).collect();
        let acc64 = fit_and_score(&phi_all, d_all, train, &labels, test, 1e-6).0;
        // i8 features
        let d = d_all;
        let mut x8 = vec![0i8; n_tr * d];
        for (r, &cell) in train.iter().enumerate() {
            for j in 0..d {
                x8[r * d + j] = q8(phi_all[cell * d + j] * 127.0);
            }
        }
        let g = gram(&x8, n_tr, d);
        let mut b = vec![0i64; d * P];
        for (r, &lab) in labels.iter().enumerate() {
            for j in 0..d {
                b[j * P + lab] += i64::from(x8[r * d + j]) * 127;
            }
        }
        let s = (n_tr as f64) * 127.0 * 127.0;
        let sol = to_f(&solve_q(&g, &b, d, P, s, 1e-9));
        let mut ok = 0usize;
        for &cell in test {
            let xt: Vec<f64> = (0..d)
                .map(|j| f64::from(q8(phi_all[cell * d + j] * 127.0)) / 127.0)
                .collect();
            let pr = predict(&sol, d, P, &xt);
            let arg = (0..P)
                .max_by(|&u, &v| pr[u].partial_cmp(&pr[v]).unwrap())
                .unwrap();
            if arg == truth_of(cell) {
                ok += 1;
            }
        }
        f5.push((f, acc64, ok as f64 / test.len() as f64));
    }
    checks.push(Check {
        name: "F5 i8 exact pipeline tracks f64",
        pass: f5.iter().all(|(_, a, b)| (a - b).abs() <= 0.03),
        detail: f5
            .iter()
            .map(|(f, a, b)| format!("f={f}: f64 {a:.4} vs i8 {b:.4}"))
            .collect::<Vec<_>>()
            .join("; "),
    });

    // ---- F6: circulant discovery with top_eigh.
    let wts: Vec<i64> = {
        let mut w: Vec<i64> = (0..P).map(|_| (rng.next() % 201) as i64 - 100).collect();
        let m = w.iter().sum::<i64>() / P as i64;
        for v in w.iter_mut() {
            *v -= m;
        }
        w
    };
    let sum_abs: f64 = wts.iter().map(|v| v.abs() as f64).sum();
    let sigma = 1.0 / (2.0 * sum_abs);
    // predicted eigenvalues of S: lambda_k = 2 Σ_b w_b cos(2π k b / p)
    let lam_k: Vec<f64> = (0..=56)
        .map(|k| {
            2.0 * (0..P)
                .map(|b| {
                    wts[b] as f64
                        * (2.0 * std::f64::consts::PI * k as f64 * b as f64 / P as f64).cos()
                })
                .sum::<f64>()
        })
        .collect();
    let mut order: Vec<usize> = (1..=56).collect();
    order.sort_by(|&u, &v| lam_k[v].partial_cmp(&lam_k[u]).unwrap());
    let top4: Vec<usize> = order[..4].to_vec();
    let gap = lam_k[order[3]] - lam_k[order[4]];
    // predicted projector onto cos/sin of the top-4 frequencies
    let mut p_pred = vec![0f64; P * P];
    for &k in &top4 {
        let w = 2.0 * std::f64::consts::PI * k as f64 / P as f64;
        let cv: Vec<f64> = (0..P).map(|a| (w * a as f64).cos()).collect();
        let sv: Vec<f64> = (0..P).map(|a| (w * a as f64).sin()).collect();
        let (nc, ns) = (
            cv.iter().map(|v| v * v).sum::<f64>(),
            sv.iter().map(|v| v * v).sum::<f64>(),
        );
        for i in 0..P {
            for j in 0..P {
                p_pred[i * P + j] += cv[i] * cv[j] / nc + sv[i] * sv[j] / ns;
            }
        }
    }
    let r = 8usize;
    let subspace_error = |frac: f64, shuffle_c: bool, full: bool, rng: &mut Rng| -> (f64, f64) {
        let mut cellv = cells.clone();
        shuffle(&mut cellv, rng);
        let n_obs = ((frac * (P * P) as f64).round() as usize).max(1);
        let mut a_mat = vec![0i64; P * P];
        let mut cs: Vec<usize> = cellv[..n_obs].iter().map(|&c| truth_of(c)).collect();
        if shuffle_c {
            shuffle(&mut cs, rng);
        }
        for (&cell, &c) in cellv[..n_obs].iter().zip(&cs) {
            let (a, bb) = (cell / P, cell % P);
            a_mat[a * P + c] += wts[bb];
        }
        // M = sigma*(A + Aᵀ) + I, exactly symmetric in Q32.32.
        let mut m = vec![0i64; P * P];
        for i in 0..P {
            for j in 0..=i {
                let v = sigma * (a_mat[i * P + j] + a_mat[j * P + i]) as f64
                    + if i == j { 1.0 } else { 0.0 };
                let q = (v * Q).round() as i64;
                m[i * P + j] = q;
                m[j * P + i] = q;
            }
        }
        let eig = match top_eigh(&m, P, if full { P } else { r }) {
            Ok(e) => e,
            Err(e) => {
                println!(
                    "    top_eigh(r={}) refused: {e:?}",
                    if full { P } else { r }
                );
                return (f64::NAN, f64::NAN);
            }
        };
        let mut eig = eig;
        if full {
            // keep the top r columns of the full decomposition (evals ascending, column k for evals[k])
            let mut ev = Vec::with_capacity(r);
            let mut vc = vec![0i64; P * r];
            for c in 0..r {
                let src = P - r + c;
                ev.push(eig.evals[src]);
                for i in 0..P {
                    vc[i * r + c] = eig.evecs[i * P + src];
                }
            }
            eig.evals = ev;
            eig.evecs = vc;
        }
        let u: Vec<f64> = eig.evecs.iter().map(|&v| v as f64 / Q).collect();
        let mut p_est = vec![0f64; P * P];
        for i in 0..P {
            for j in 0..P {
                p_est[i * P + j] = (0..r).map(|c| u[i * r + c] * u[j * r + c]).sum();
            }
        }
        let diff2: f64 = p_est
            .iter()
            .zip(&p_pred)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        // eigenvalue check against the predicted top-8 (each frequency has multiplicity 2)
        let mut pred_vals: Vec<f64> = top4.iter().flat_map(|&k| [lam_k[k], lam_k[k]]).collect();
        pred_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let est_vals: Vec<f64> = eig
            .evals
            .iter()
            .map(|&v| ((v as f64 / Q) - 1.0) / sigma)
            .collect();
        let ev_err = pred_vals
            .iter()
            .zip(&est_vals)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0, f64::max);
        ((diff2 / (2.0 * r as f64)).sqrt(), ev_err)
    };
    println!("\n  top-4 predicted frequencies {top4:?}, eigenvalue gap between 4th and 5th predicted: {gap:.1} (largest eigenvalue {:.1})", lam_k[order[0]]);
    // F6 as PRE-REGISTERED: top_eigh(r = 8), subspace iteration. Run once; recorded as it is.
    println!("  F6 as pre-registered (top_eigh with r = 8):");
    let (e_full, ev_full) = subspace_error(1.0, false, false, &mut rng);
    checks.push(Check {
        name: "F6 full table: top_eigh(r=8) recovers the predicted Fourier eigenspace and eigenvalues",
        pass: e_full < 1e-3 && ev_full < 1e-3,
        detail: format!("subspace error {e_full:.3e}, max relative eigenvalue error {ev_full:.3e} (NaN = the primitive refused)"),
    });
    // ---- POST-HOC (formed after F6 was refused): the full decomposition, then the top 8 selected.
    println!("  POST-HOC F6b (top_eigh with r = n = 113, then keep the top 8):");
    let (e_full_b, ev_full_b) = subspace_error(1.0, false, true, &mut rng);
    let mut errs = Vec::new();
    for &fr in &[0.5, 0.3, 0.1, 0.03] {
        let mean = (0..3)
            .map(|_| subspace_error(fr, false, true, &mut rng).0)
            .sum::<f64>()
            / 3.0;
        errs.push((fr, mean));
    }
    let e_shuf = (0..3)
        .map(|_| subspace_error(1.0, true, true, &mut rng).0)
        .sum::<f64>()
        / 3.0;
    println!("  F6b subspace error (0 = identical, 1 = orthogonal): full table {e_full_b:.3e} (eigenvalue err {ev_full_b:.2e}); partial: {}; c shuffled {e_shuf:.3}", errs.iter().map(|(f, e)| format!("{:.0}% -> {e:.3}", f * 100.0)).collect::<Vec<_>>().join(", "));
    let monotone = errs.windows(2).all(|w| w[1].1 + 0.02 >= w[0].1);
    checks.push(Check {
        name: "POST-HOC F6b full table, full decomposition: eigenspace and eigenvalues recovered",
        pass: e_full_b < 1e-3 && ev_full_b < 1e-3,
        detail: format!(
            "subspace error {e_full_b:.3e}, max relative eigenvalue error {ev_full_b:.3e}"
        ),
    });
    checks.push(Check {
        name: "POST-HOC F6b control: shuffled c destroys it; error non-increasing in the observed fraction",
        pass: e_shuf >= 0.80 && monotone,
        detail: format!("shuffled-c error {e_shuf:.3} (chance 0.964); partial-table errors {:?}; monotone: {monotone}", errs),
    });
    // ---- POST-HOC F4b: what carries the result, the characters or the sum structure?
    let (phi_lin, d_lin) = {
        let d = 4 * 56 + 1;
        let mut phi = vec![0f64; P * P * d];
        for a in 0..P {
            for b in 0..P {
                let row = &mut phi[(a * P + b) * d..(a * P + b + 1) * d];
                for k in 1..=56usize {
                    let w = 2.0 * std::f64::consts::PI * k as f64 / P as f64;
                    row[4 * (k - 1)] = (w * a as f64).cos();
                    row[4 * (k - 1) + 1] = (w * a as f64).sin();
                    row[4 * (k - 1) + 2] = (w * b as f64).cos();
                    row[4 * (k - 1) + 3] = (w * b as f64).sin();
                }
                row[d - 1] = 1.0;
            }
        }
        (phi, d)
    };
    let mut perm = cells.clone();
    shuffle(&mut perm, &mut rng);
    let n30 = (0.30 * (P * P) as f64).round() as usize;
    let tr = &perm[..n30];
    let te = &perm[n30..n30 + 2000];
    let lab: Vec<usize> = tr.iter().map(|&c| truth_of(c)).collect();
    let acc_lin = fit_and_score(&phi_lin, d_lin, tr, &lab, te, 1e-6).0;
    checks.push(Check {
        name: "POST-HOC F4b Fourier features of a and b SEPARATELY (no products) cannot do it",
        pass: acc_lin <= 0.10,
        detail: format!("accuracy at 30% with the same 56 on-grid frequencies but cos/sin of a and of b only: {acc_lin:.4}"),
    });
    report("E-FOURIER", &checks)
}

// ---------------------------------------------------------------------------------------------
// E-SUPER: superposition ground truth (Toy Models of Superposition, arXiv:2209.10652).
// ---------------------------------------------------------------------------------------------

/// Centred covariance of `rows × m` i8 data from the exact integer statistics, entries / 127².
fn centred_cov(x: &[i8], rows: usize, m: usize) -> (Vec<f64>, Vec<i64>) {
    let g = gram(x, rows, m);
    let mut sx = vec![0i64; m];
    for r in 0..rows {
        for j in 0..m {
            sx[j] += i64::from(x[r * m + j]);
        }
    }
    let n = rows as f64;
    let mut c = vec![0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            c[i * m + j] = g[i * m + j] as f64 / n / (127.0 * 127.0)
                - (sx[i] as f64 / n / 127.0) * (sx[j] as f64 / n / 127.0);
        }
    }
    (c, g)
}

fn top_atoms(cov: &[f64], m: usize, r: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0i64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let v = (cov[i * m + j] * Q).round() as i64;
            q[i * m + j] = v;
            q[j * m + i] = v;
        }
    }
    // The FULL decomposition (r = m), then the top r selected. `top_eigh(…, r < m)` is subspace
    // iteration, which does not converge on a near-flat spectrum (a Marchenko–Pastur bulk is what
    // superposition produces): it refuses with `NotConverged`, loudly. With r = m the
    // oversampled subspace is the whole space and one Rayleigh–Ritz step is exact.
    let e = top_eigh(&q, m, m).expect("top_eigh (full)");
    let ev = to_f(&e.evals); // ascending
    let vecs = to_f(&e.evecs); // m × m row-major, column k for evals[k]
    let mut out_v = vec![0f64; m * r];
    let mut out_e = vec![0f64; r];
    for c in 0..r {
        let src = m - 1 - c; // descending
        out_e[c] = ev[src];
        for i in 0..m {
            out_v[i * r + c] = vecs[i * m + src];
        }
    }
    (out_e, out_v)
}

fn max_cos(truth: &[Vec<f64>], atoms: &[f64], m: usize, r: usize) -> Vec<f64> {
    truth
        .iter()
        .map(|w| {
            let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            (0..r)
                .map(|c| {
                    let a: Vec<f64> = (0..m).map(|i| atoms[i * r + c]).collect();
                    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                    (w.iter().zip(&a).map(|(p, q)| p * q).sum::<f64>() / (nw * na)).abs()
                })
                .fold(0.0, f64::max)
        })
        .collect()
}

fn superposition() -> bool {
    println!("\n--- E-SUPER: superposition ground truth (SYNTHETIC; Toy Models data process) ---");
    println!("Process (arXiv:2209.10652, 'Experiment Setup'): x_i = 0 with probability S, else Uniform[0,1], independent; observation h = W x");
    println!("with unit-norm columns. Centred covariance of x is σ²I with σ² = (1−S)(1+3S)/12 (independent features), so Cov(h) = σ² W Wᵀ.");
    println!("PREDICTIONS (fixed before the first run):");
    println!("  S1 A (n=64 features in m=16 dims, S=0.95): the top-8 eigenvectors of Cov(h) recover individual feature directions no");
    println!("     better than random atoms: mean max-cosine <= (random-atom baseline + 0.05). The spectrum cannot unmix features.");
    println!("  S2 A+ (positive control: m=n=16, orthogonal W, DISTINCT feature scales): variance gaps make eigenvectors unique, so the");
    println!("     top-8 scale features are recovered with mean max-cosine >= 0.95.");
    println!("  S3 B-oracle: a ridge decoder fitted on the TRUE codes (the Gram path, one Cholesky solve) recovers W: mean cosine >= 0.99.");
    println!("  S4 flat spectrum: with h = x (m=n=64, S=0.9, equal scales) the CENTRED cone fraction at r=8 is within [0.10, 0.20] (flat is");
    println!("     r/n = 0.125) although every feature is perfectly well defined — contradicting the docstring that a small cone fraction is");
    println!("     'noise'. The UNCENTRED Gram (what XᵀX accumulates) has cone fraction >= 0.35 because of the mean spike.");
    println!("  S5 control: with each column of H independently permuted (marginals kept, W destroyed), A is at the random baseline and the");
    println!("     B-oracle's mean cosine is <= 0.5.");

    let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
    let unit_cols = |m: usize, n: usize, rng: &mut Rng| -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| {
                let v: Vec<f64> = (0..m).map(|_| rng.gauss()).collect();
                let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                v.iter().map(|a| a / nv).collect()
            })
            .collect()
    };
    let sample_x =
        |rows: usize, n: usize, s: f64, scales: &[f64], rng: &mut Rng| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|_| {
                    (0..n)
                        .map(|i| {
                            if rng.unit() < s {
                                0.0
                            } else {
                                rng.unit() * scales[i]
                            }
                        })
                        .collect()
                })
                .collect()
        };
    let quant_h = |xs: &[Vec<f64>], cols: &[Vec<f64>], m: usize, scale: f64| -> Vec<i8> {
        let mut out = vec![0i8; xs.len() * m];
        for (r, x) in xs.iter().enumerate() {
            for i in 0..m {
                let v: f64 = x.iter().zip(cols).map(|(xj, w)| xj * w[i]).sum();
                out[r * m + i] = q8(v * scale);
            }
        }
        out
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let rows = 40000usize;
    let mut checks = Vec::new();

    // random-atom baseline in R^16 with 8 atoms
    let random_baseline = |truth: &[Vec<f64>], m: usize, r: usize, rng: &mut Rng| -> f64 {
        // Random ORTHONORMAL atoms (Gram–Schmidt on Gaussian columns): eigenvectors are orthonormal,
        // so a Gaussian-atom baseline would understate what a random basis achieves.
        let mut acc = 0f64;
        for _ in 0..20 {
            let mut cols: Vec<Vec<f64>> = Vec::new();
            while cols.len() < r {
                let mut v: Vec<f64> = (0..m).map(|_| rng.gauss()).collect();
                for u in &cols {
                    let dot: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
                    for (a, b) in v.iter_mut().zip(u) {
                        *a -= dot * b;
                    }
                }
                let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                cols.push(v.iter().map(|a| a / nv).collect());
            }
            let mut atoms = vec![0f64; m * r];
            for c in 0..r {
                for i in 0..m {
                    atoms[i * r + c] = cols[c][i];
                }
            }
            acc += mean(&max_cos(truth, &atoms, m, r));
        }
        acc / 20.0
    };

    // ---- S1: A, superposed (n=64 > m=16)
    let (n1, m1, r1) = (64usize, 16usize, 8usize);
    let w1 = unit_cols(m1, n1, &mut rng);
    let ones1 = vec![1.0; n1];
    let xs1 = sample_x(rows, n1, 0.95, &ones1, &mut rng);
    let h1 = quant_h(&xs1, &w1, m1, 250.0);
    let (cov1, _) = centred_cov(&h1, rows, m1);
    let (_, atoms1) = top_atoms(&cov1, m1, r1);
    let spec_cos = mean(&max_cos(&w1, &atoms1, m1, r1));
    let base1 = random_baseline(&w1, m1, r1, &mut rng);
    checks.push(Check {
        name: "S1 spectral atoms do not unmix superposed features",
        pass: spec_cos <= base1 + 0.05,
        detail: format!("mean max-cosine to the 64 true directions: spectral {spec_cos:.3} vs random atoms {base1:.3}"),
    });

    // ---- S2: A+, distinct scales, orthogonal W (m=n=16)
    let (n2, m2) = (16usize, 16usize);
    let mut w2: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n2 {
        let mut v: Vec<f64> = (0..m2).map(|_| rng.gauss()).collect();
        for u in &w2 {
            let dot: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            for (a, b) in v.iter_mut().zip(u) {
                *a -= dot * b;
            }
        }
        let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        w2.push(v.iter().map(|a| a / nv).collect());
    }
    let scales2: Vec<f64> = (0..n2).map(|j| 1.0 - 0.055 * j as f64).collect();
    let xs2 = sample_x(rows, n2, 0.5, &scales2, &mut rng);
    let h2 = quant_h(&xs2, &w2, m2, 160.0);
    let (cov2, _) = centred_cov(&h2, rows, m2);
    let (_, atoms2) = top_atoms(&cov2, m2, 8);
    let top_truth: Vec<Vec<f64>> = w2[..8].to_vec();
    let pos_cos = mean(&max_cos(&top_truth, &atoms2, m2, 8));
    checks.push(Check {
        name: "S2 positive control: variance gaps make the spectrum identify features",
        pass: pos_cos >= 0.95,
        detail: format!("mean max-cosine for the 8 highest-variance features: {pos_cos:.3}"),
    });

    // ---- S3: B-oracle, ridge decoder on the true codes via the exact pipeline
    let to_codes = |xs: &[Vec<f64>], n: usize| -> Vec<i8> {
        let mut out = vec![0i8; xs.len() * n];
        for (r, x) in xs.iter().enumerate() {
            for j in 0..n {
                out[r * n + j] = q8(x[j] * 127.0);
            }
        }
        out
    };
    let decode_cos = |codes: &[i8], h: &[i8], n: usize, m: usize, truth: &[Vec<f64>]| -> f64 {
        let g = gram(codes, rows, n);
        let mut b = vec![0i64; n * m];
        for r in 0..rows {
            for j in 0..n {
                for i in 0..m {
                    b[j * m + i] += i64::from(codes[r * n + j]) * i64::from(h[r * m + i]);
                }
            }
        }
        let s = rows as f64 * 127.0 * 127.0;
        // B above is X^T H with H in i8 units; normalise so the solve is (G/S + λI) W = B/(S·scale_h)
        let sol = to_f(&solve_q(&g, &b, n, m, s * 127.0, 1e-6));
        let cos: Vec<f64> = (0..n)
            .map(|j| {
                let w: Vec<f64> = (0..m).map(|i| sol[j * m + i]).collect();
                let nw = w.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
                (w.iter().zip(&truth[j]).map(|(a, b2)| a * b2).sum::<f64>() / nw).abs()
            })
            .collect();
        mean(&cos)
    };
    let codes1 = to_codes(&xs1, n1);
    let dec_cos = decode_cos(&codes1, &h1, n1, m1, &w1);
    checks.push(Check {
        name: "S3 given the true codes, the ridge decoder (Gram + one solve) recovers W",
        pass: dec_cos >= 0.99,
        detail: format!("mean cosine between decoded and true columns: {dec_cos:.4}"),
    });

    // ---- S4: flat spectrum, centred vs uncentred
    let (n4, r4) = (64usize, 8usize);
    let ones4 = vec![1.0; n4];
    let xs4 = sample_x(rows, n4, 0.9, &ones4, &mut rng);
    let x4 = to_codes(&xs4, n4);
    let (cov4, g4) = centred_cov(&x4, rows, n4);
    let frob_c: f64 = cov4.iter().map(|v| v * v).sum();
    let (ev_c, _) = top_atoms(&cov4, n4, r4);
    let cone_c = ev_c.iter().map(|v| v * v).sum::<f64>() / frob_c;
    let unc: Vec<f64> = g4
        .iter()
        .map(|&v| v as f64 / rows as f64 / (127.0 * 127.0))
        .collect();
    let frob_u: f64 = unc.iter().map(|v| v * v).sum();
    let (ev_u, _) = top_atoms(&unc, n4, r4);
    let cone_u = ev_u.iter().map(|v| v * v).sum::<f64>() / frob_u;
    checks.push(Check {
        name: "S4 flat spectrum for perfectly defined features; centring matters",
        pass: (0.10..=0.20).contains(&cone_c) && cone_u >= 0.35,
        detail: format!(
            "cone fraction at r=8: centred {cone_c:.3} (flat = 0.125), uncentred Gram {cone_u:.3}"
        ),
    });

    // ---- S5: control, columns of H independently permuted
    let mut h1s = h1.clone();
    for i in 0..m1 {
        let mut col: Vec<i8> = (0..rows).map(|r| h1[r * m1 + i]).collect();
        shuffle(&mut col, &mut rng);
        for r in 0..rows {
            h1s[r * m1 + i] = col[r];
        }
    }
    let (cov1s, _) = centred_cov(&h1s, rows, m1);
    let (_, atoms1s) = top_atoms(&cov1s, m1, r1);
    let spec_cos_s = mean(&max_cos(&w1, &atoms1s, m1, r1));
    let dec_cos_s = decode_cos(&codes1, &h1s, n1, m1, &w1);
    checks.push(Check {
        name: "S5 control: destroying W leaves A at baseline and the decoder at <= 0.5",
        pass: spec_cos_s <= base1 + 0.05 && dec_cos_s <= 0.5,
        detail: format!(
            "spectral {spec_cos_s:.3} (baseline {base1:.3}); decoder mean cosine {dec_cos_s:.3}"
        ),
    });
    report("E-SUPER", &checks)
}

// ---------------------------------------------------------------------------------------------
// E-ASSOC: a ridge associative memory — write = add, read = ridge contraction, forget = subtract,
// uncertainty = leverage. The "prediction machine" claim, tested at the smallest scale.
// ---------------------------------------------------------------------------------------------

fn assoc() -> bool {
    println!("\n--- E-ASSOC: ridge associative memory (SYNTHETIC) ---");
    println!("Setup: d=64 i8 key features (Gaussian, quantised), V=16 value classes. Write n (key, value) pairs into the exact");
    println!("(G, B): G += k kᵀ, B += k v127ᵀ. Read modes: HEBB logits = Bᵀk (a monoid, parallel, crosstalk); RIDGE logits = Bᵀ(G/n+λI)⁻¹k;");
    println!("LOOKUP = value of the most similar stored key (the KV-cache baseline: exact, but one slot per pair).");
    println!("Uncertainty: leverage u(q) = q̃ᵀ(G/n+λI)⁻¹q̃ — the same Cholesky, no extra pass.");
    println!("PREDICTIONS (fixed before the first run):");
    println!("  A1 capacity: RIDGE recall of stored pairs is >= 0.99 for every n <= 0.875·d (n <= 56) and <= 0.80 at n = 2d (128).");
    println!("  A2 crosstalk: HEBB is strictly worse than RIDGE at n = 0.75·d (48) by >= 0.10, and HEBB at n = d is below RIDGE at n = d.");
    println!("  A3 LOOKUP is 1.000 at every n (it keeps one slot per pair) and survives key noise (noise 0.5σ) at >= 0.99.");
    println!("  A4 uncertainty: leverage separates stored from never-written keys, AUROC >= 0.99 at n = 48; it falls as n approaches d.");
    println!("  A5 reversibility: after writing 48 pairs and then SUBTRACTING 8 of them, (G, B) equal those of a memory that never");
    println!("     held them, bit for bit; the removed keys read at <= 0.20 (chance 0.0625) and the 40 retained keys stay >= 0.99.");
    println!("  A6 control: a memory fed shuffled keys→values reads at chance (<= 0.12) on the true pairing.");

    let (d, v) = (64usize, 16usize);
    let mut rng = Rng(0xA550_C1A7_1FE0_BEEF);
    let key = |rng: &mut Rng| -> Vec<i8> { (0..d).map(|_| q8(rng.gauss() * 40.0)).collect() };
    let n_max = 256usize;
    let keys: Vec<Vec<i8>> = (0..n_max).map(|_| key(&mut rng)).collect();
    let vals: Vec<usize> = (0..n_max)
        .map(|_| (rng.next() % v as u64) as usize)
        .collect();
    let unseen: Vec<Vec<i8>> = (0..400).map(|_| key(&mut rng)).collect();
    let lam = 1e-4;
    let tilde = |k: &[i8]| -> Vec<f64> { k.iter().map(|&x| f64::from(x) / 127.0).collect() };

    let build = |ids: &[usize]| -> (Vec<i64>, Vec<i64>) {
        let mut flat = Vec::with_capacity(ids.len() * d);
        for &i in ids {
            flat.extend_from_slice(&keys[i]);
        }
        let g = gram(&flat, ids.len(), d);
        let mut b = vec![0i64; d * v];
        for &i in ids {
            for j in 0..d {
                b[j * v + vals[i]] += i64::from(keys[i][j]) * 127;
            }
        }
        (g, b)
    };
    let argmax = |x: &[f64]| {
        (0..x.len())
            .max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap())
            .unwrap()
    };

    println!(
        "\n  n   n/d |  RIDGE  HEBB  LOOKUP  LOOKUP(noisy) | leverage AUROC (stored vs unseen)"
    );
    let ns = [8usize, 16, 32, 48, 56, 64, 80, 96, 128, 192, 256];
    let mut recall_r = std::collections::BTreeMap::new();
    let mut recall_h = std::collections::BTreeMap::new();
    let mut lookup_ok = true;
    let mut lookup_noisy_min = 1f64;
    let mut auroc_by_n = Vec::new();
    for &n in &ns {
        let ids: Vec<usize> = (0..n).collect();
        let (g, b) = build(&ids);
        let s = n as f64 * 127.0 * 127.0;
        let w = to_f(&solve_q(&g, &b, d, v, s, lam));
        let (mut r_ok, mut h_ok, mut l_ok, mut ln_ok, mut rn_ok, mut hn_ok) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for &i in &ids {
            let kt = tilde(&keys[i]);
            let ridge = predict(&w, d, v, &kt);
            if argmax(&ridge) == vals[i] {
                r_ok += 1;
            }
            let hebb: Vec<f64> = (0..v)
                .map(|c| (0..d).map(|j| b[j * v + c] as f64 / 127.0 * kt[j]).sum())
                .collect();
            if argmax(&hebb) == vals[i] {
                h_ok += 1;
            }
            let look = |q: &[f64]| -> usize {
                let mut best = (f64::MIN, 0usize);
                for &j in &ids {
                    let kj = tilde(&keys[j]);
                    let dot: f64 = kj.iter().zip(q).map(|(a, c)| a * c).sum();
                    let nk: f64 = kj.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let sc = dot / nk;
                    if sc > best.0 {
                        best = (sc, j);
                    }
                }
                vals[best.1]
            };
            if look(&kt) == vals[i] {
                l_ok += 1;
            }
            // key noise: 0.5 of the per-coordinate key standard deviation (40/127), on every read mode.
            let noisy: Vec<f64> = kt
                .iter()
                .map(|x| x + 0.5 * (40.0 / 127.0) * rng.gauss())
                .collect();
            if look(&noisy) == vals[i] {
                ln_ok += 1;
            }
            if argmax(&predict(&w, d, v, &noisy)) == vals[i] {
                rn_ok += 1;
            }
            let hebb_n: Vec<f64> = (0..v)
                .map(|c| (0..d).map(|j| b[j * v + c] as f64 / 127.0 * noisy[j]).sum())
                .collect();
            if argmax(&hebb_n) == vals[i] {
                hn_ok += 1;
            }
        }
        let nf = n as f64;
        let (rr, hh, ll, lln) = (
            r_ok as f64 / nf,
            h_ok as f64 / nf,
            l_ok as f64 / nf,
            ln_ok as f64 / nf,
        );
        let (rrn, hhn) = (rn_ok as f64 / nf, hn_ok as f64 / nf);
        recall_r.insert(n, rr);
        recall_h.insert(n, hh);
        lookup_ok &= ll == 1.0;
        lookup_noisy_min = lookup_noisy_min.min(lln);
        // leverage AUROC: stored keys (first min(n,400) ids) vs unseen keys
        let lev = |k: &[i8]| -> f64 {
            let kt = tilde(k);
            let u = cinv_vec(&g, d, s, lam, &kt);
            kt.iter().zip(&u).map(|(a, c)| a * c).sum()
        };
        let stored: Vec<f64> = ids.iter().take(200).map(|&i| lev(&keys[i])).collect();
        let fresh: Vec<f64> = unseen.iter().take(200).map(|k| lev(k)).collect();
        let mut wins = 0f64;
        for f in &fresh {
            for st in &stored {
                wins += if f > st {
                    1.0
                } else if f == st {
                    0.5
                } else {
                    0.0
                };
            }
        }
        let auroc = wins / (fresh.len() * stored.len()) as f64;
        auroc_by_n.push((n, auroc));
        println!("  {n:>3} {:>5.2} | {rr:>6.3} {hh:>5.3} {ll:>7.3} {lln:>13.3} | {auroc:.3}   (noisy keys, exploratory: ridge {rrn:.3} hebb {hhn:.3})", nf / d as f64);
    }

    // A5: reversibility, bit for bit.
    let ids48: Vec<usize> = (0..48).collect();
    let (g48, b48) = build(&ids48);
    let removed: Vec<usize> = (0..8).map(|i| i * 6).collect();
    let (mut g_sub, mut b_sub) = (g48.clone(), b48.clone());
    for &i in &removed {
        for a in 0..d {
            for c in 0..d {
                g_sub[a * d + c] -= i64::from(keys[i][a]) * i64::from(keys[i][c]);
            }
            b_sub[a * v + vals[i]] -= i64::from(keys[i][a]) * 127;
        }
    }
    let kept: Vec<usize> = ids48
        .iter()
        .copied()
        .filter(|i| !removed.contains(i))
        .collect();
    let (g_never, b_never) = build(&kept);
    let exact = g_sub == g_never && b_sub == b_never;
    let s40 = 48.0 * 127.0 * 127.0; // the same fixed scale for both reads, so only the content differs
    let w_before = to_f(&solve_q(&g48, &b48, d, v, s40, lam));
    let w_after = to_f(&solve_q(&g_sub, &b_sub, d, v, s40, lam));
    let rec = |w: &[f64], set: &[usize]| -> f64 {
        set.iter()
            .filter(|&&i| argmax(&predict(w, d, v, &tilde(&keys[i]))) == vals[i])
            .count() as f64
            / set.len() as f64
    };
    let (rm_before, rm_after, kp_after) = (
        rec(&w_before, &removed),
        rec(&w_after, &removed),
        rec(&w_after, &kept),
    );
    println!("\n  reversal: integer state after subtracting 8 of 48 pairs == a memory that never held them: {exact}");
    println!("  recall of the 8 removed keys: before {rm_before:.3} -> after {rm_after:.3} (chance {:.4}); the 40 retained: {kp_after:.3}", 1.0 / v as f64);

    // EXPLORATORY (formed after the noisy columns above showed a dip at n = d): the ridge read against λ.
    println!("\n  exploratory: ridge read at n = d = 64 against lambda (clean / noisy keys, noise 0.5 per-coordinate std):");
    {
        let ids64: Vec<usize> = (0..64).collect();
        let (g64, b64) = build(&ids64);
        let s64 = 64.0 * 127.0 * 127.0;
        for &lm in &[1e-4, 1e-3, 1e-2, 1e-1, 1.0] {
            let w = to_f(&solve_q(&g64, &b64, d, v, s64, lm));
            let (mut c_ok, mut n_ok) = (0usize, 0usize);
            for rep in 0..5 {
                let _ = rep;
                for &i in &ids64 {
                    let kt = tilde(&keys[i]);
                    let noisy: Vec<f64> = kt
                        .iter()
                        .map(|x| x + 0.5 * (40.0 / 127.0) * rng.gauss())
                        .collect();
                    if argmax(&predict(&w, d, v, &kt)) == vals[i] {
                        c_ok += 1;
                    }
                    if argmax(&predict(&w, d, v, &noisy)) == vals[i] {
                        n_ok += 1;
                    }
                }
            }
            println!(
                "    lambda {lm:>7.0e}: clean {:.3}   noisy {:.3}",
                c_ok as f64 / 320.0,
                n_ok as f64 / 320.0
            );
        }
    }

    // A6 control: shuffled pairing.
    let mut perm: Vec<usize> = (0..48).collect();
    shuffle(&mut perm, &mut rng);
    let mut flat = Vec::new();
    for &i in &ids48 {
        flat.extend_from_slice(&keys[i]);
    }
    let gsh = gram(&flat, 48, d);
    let mut bsh = vec![0i64; d * v];
    for (pos, &i) in ids48.iter().enumerate() {
        for j in 0..d {
            bsh[j * v + vals[perm[pos]]] += i64::from(keys[i][j]) * 127;
        }
    }
    let wsh = to_f(&solve_q(&gsh, &bsh, d, v, s40, lam));
    let shuf_recall = ids48
        .iter()
        .filter(|&&i| argmax(&predict(&wsh, d, v, &tilde(&keys[i]))) == vals[i])
        .count() as f64
        / 48.0;

    let get = |m: &std::collections::BTreeMap<usize, f64>, n: usize| *m.get(&n).unwrap();
    let a1 = ns
        .iter()
        .filter(|&&n| n <= 56)
        .all(|&n| get(&recall_r, n) >= 0.99)
        && get(&recall_r, 128) <= 0.80;
    let a2 =
        get(&recall_r, 48) - get(&recall_h, 48) >= 0.10 && get(&recall_h, 64) < get(&recall_r, 64);
    let auroc_at = |n: usize| auroc_by_n.iter().find(|a| a.0 == n).unwrap().1;
    let a4 = auroc_at(48) >= 0.99 && auroc_at(96) < auroc_at(48);
    let checks = vec![
        Check { name: "A1 ridge capacity: exact to ~0.875·d, collapses by 2d", pass: a1, detail: format!("recall n=56 {:.3}, n=64 {:.3}, n=96 {:.3}, n=128 {:.3}", get(&recall_r, 56), get(&recall_r, 64), get(&recall_r, 96), get(&recall_r, 128)) },
        Check { name: "A2 Hebbian crosstalk costs recall that the ridge read does not", pass: a2, detail: format!("n=48: ridge {:.3} vs hebb {:.3}; n=64: ridge {:.3} vs hebb {:.3}", get(&recall_r, 48), get(&recall_h, 48), get(&recall_r, 64), get(&recall_h, 64)) },
        Check { name: "A3 lookup is exact and survives key noise", pass: lookup_ok && lookup_noisy_min >= 0.99, detail: format!("exact at every n: {lookup_ok}; worst noisy recall {lookup_noisy_min:.3}") },
        Check { name: "A4 leverage separates stored from unseen keys, weaker near capacity", pass: a4, detail: format!("AUROC by n: {}", auroc_by_n.iter().map(|(n, a)| format!("{n}:{a:.3}")).collect::<Vec<_>>().join(", ")) },
        Check { name: "A5 forgetting is exact subtraction", pass: exact && rm_after <= 0.20 && kp_after >= 0.99, detail: format!("state bit-identical {exact}; removed {rm_before:.3} -> {rm_after:.3}; retained {kp_after:.3}") },
        Check { name: "A6 control: shuffled pairing reads at chance", pass: shuf_recall <= 0.12, detail: format!("recall on the true pairing {shuf_recall:.3} (chance {:.4})", 1.0 / v as f64) },
    ];
    report("E-ASSOC", &checks)
}

fn main() -> ExitCode {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let mut ok = true;
    if which == "edit" || which == "all" {
        ok &= edit();
    }
    if which == "leace" || which == "all" {
        ok &= leace();
    }
    if which == "fourier" || which == "all" {
        ok &= fourier();
    }
    if which == "super" || which == "all" {
        ok &= superposition();
    }
    if which == "assoc" || which == "all" {
        ok &= assoc();
    }
    println!(
        "\n{}",
        if ok {
            "ALL PREDICTIONS HELD"
        } else {
            "AT LEAST ONE PREDICTION FAILED"
        }
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
