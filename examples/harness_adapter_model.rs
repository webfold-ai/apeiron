//! `harness_adapter_model` — the fold-native harness adapter, fitted on a real checkpoint.
//!
//! ```text
//! harness_adapter_model --captures <dir> --out <dir> [--rank 32] [--lambda 1e-2] [--revoke <row>]
//! ```
//!
//! # What this is
//!
//! The upstream artefact (`harness-hub/Qwen3.8-27B-pi-adapted`) is a **harness
//! adapter**: a small trained artefact that makes a general model good at ONE agent
//! harness. Theirs is a rank-32 LoRA on seven projections, fitted by RL on the pi
//! harness's trajectories.
//!
//! Ours is the same *kind* of artefact, fitted by the **fold**: for the readout, take
//! the student's activations on our harness's own traffic (the contexts and the teacher
//! continuations read out of our session log), accumulate the exact-integer Gram, solve
//! the Q32.32 ridge in the residual form `RHS = XᵀY - G·W`, and then **project the
//! closed-form delta onto rank r** — the LoRA analogue, obtained without a gradient.
//!
//! # Why the residual form
//!
//! The target for the readout at a position is the *teacher's* next token. Fitting the
//! readout to a one-hot target means fitting `W + ΔW` so that `(W + ΔW)x ≈ onehot`, i.e.
//! `ΔW = (G + λI)⁻¹ (XᵀY - G·W)`. The second term is the same algebra as the checkpoint
//! round-trip's lift, so the same primitives do it.
//!
//! # The vocabulary subset
//!
//! The readout is adapted over the top-K tokens our traffic actually uses (K = 2,048 by
//! default), because the full 151,936-column vocabulary makes the Q32.32 solve overflow
//! — a measured boundary, not a guess. The artefact is therefore an adapter over K
//! tokens: `ΔW = B·A` with `B` of shape `(D, r)` and `A` of shape `(r, K)`.
//!
//! # The revocation
//!
//! `--revoke <row>` subtracts one trajectory's contribution from the Gram and the
//! cross-moment — an integer subtraction, not a refit — and re-solves. That is the
//! operation a LoRA cannot do.

#![allow(clippy::all, clippy::pedantic, reason = "a measurement binary")]

use apeiron::learn::gram::gram_fold_i8;
use apeiron::learn::solve::chol::chol_solve;
use apeiron::learn::solve::eig::top_eigh;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

const Q32_ONE: f64 = 4_294_967_296.0;

struct Args {
    captures: String,
    out: String,
    rank: usize,
    lambda: f64,
    revoke: Option<usize>,
    holdout_frac: f64,
    rows_per_segment: usize,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        captures: String::new(),
        out: String::new(),
        rank: 32,
        lambda: 1e-2,
        revoke: None,
        holdout_frac: 0.2,
        rows_per_segment: 64,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--captures" => a.captures = val()?,
            "--out" => a.out = val()?,
            "--rank" => a.rank = val()?.parse()?,
            "--lambda" => a.lambda = val()?.parse()?,
            "--revoke" => a.revoke = Some(val()?.parse()?),
            "--holdout-frac" => a.holdout_frac = val()?.parse()?,
            "--rows-per-segment" => a.rows_per_segment = val()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if a.captures.is_empty() || a.out.is_empty() {
        return Err("--captures and --out are required".into());
    }
    Ok(a)
}

/// Read a raw binary file.
fn read_bin(path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut v = Vec::new();
    File::open(path)?.read_to_end(&mut v)?;
    Ok(v)
}

/// Read `<dir>/meta.json` for the capture's shape.
fn meta_of(dir: &str) -> Result<(usize, usize, f64), Box<dyn Error>> {
    let text = std::fs::read_to_string(format!("{dir}/meta.json"))?;
    let num = |key: &str| -> Result<f64, Box<dyn Error>> {
        let needle = format!("\"{key}\":");
        let at = text
            .find(&needle)
            .ok_or_else(|| format!("meta.json has no {key}"))?
            + needle.len();
        let rest = text[at..].trim_start();
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(rest.len());
        Ok(rest[..end].parse::<f64>()?)
    };
    Ok((
        num("rows")? as usize,
        num("dim")? as usize,
        num("scale")?,
    ))
}

/// A safetensors writer: `<u64 header_len><json header><data>`.
struct SafeWriter {
    tensors: BTreeMap<String, (String, Vec<usize>, Vec<u8>)>,
}

impl SafeWriter {
    fn new() -> Self {
        SafeWriter {
            tensors: BTreeMap::new(),
        }
    }
    fn add_f32(&mut self, name: &str, shape: Vec<usize>, data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors
            .insert(name.to_string(), ("F32".to_string(), shape, bytes));
    }
    fn add_i32(&mut self, name: &str, shape: Vec<usize>, data: &[i32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors
            .insert(name.to_string(), ("I32".to_string(), shape, bytes));
    }
    fn write(&self, path: &str) -> Result<u64, Box<dyn Error>> {
        let mut header = String::from("{");
        let mut offset = 0usize;
        let mut body: Vec<u8> = Vec::new();
        let mut first = true;
        for (name, (dtype, shape, bytes)) in &self.tensors {
            if !first {
                header.push(',');
            }
            first = false;
            let end = offset + bytes.len();
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
                shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
            ));
            body.extend_from_slice(bytes);
            offset = end;
        }
        header.push('}');
        let mut f = File::create(path)?;
        let hb = header.as_bytes();
        f.write_all(&(hb.len() as u64).to_le_bytes())?;
        f.write_all(hb)?;
        f.write_all(&body)?;
        Ok((hb.len() + 8 + body.len()) as u64)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = parse_args()?;
    std::fs::create_dir_all(&a.out)?;
    let dir = &a.captures;

    let (rows, d, scale) = meta_of(dir)?;
    let acts = read_bin(&format!("{dir}/acts.i8"))?;
    let targets = read_bin(&format!("{dir}/targets.i32"))?;
    let vocab = read_bin(&format!("{dir}/vocab_top.i32"))?;
    let base_rows = read_bin(&format!("{dir}/base_rows.f32"))?;
    if acts.len() != rows * d || targets.len() != rows * 4 {
        return Err("capture shapes do not match meta.json".into());
    }
    let k = vocab.len() / 4;
    println!("captures: {rows} rows x {d} dims, {k} vocabulary columns, scale {scale}");

    // ---- the token id -> column map -------------------------------------------
    let mut col_of = vec![u32::MAX; 1 << 17];
    for (c, id) in vocab.chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes([id[0], id[1], id[2], id[3]]) as usize;
        if id < col_of.len() {
            col_of[id] = c as u32;
        }
    }

    // The fit uses the TRAINING rows only; the held-out segments are the measurement.
    let segments = rows / a.rows_per_segment.max(1);
    let train_segments = ((segments as f64) * (1.0 - a.holdout_frac)).floor() as usize;
    let train_rows = (train_segments * a.rows_per_segment.max(1)).min(rows);
    println!(
        "split: {segments} segments, {train_segments} for the fit, {} held out",
        segments - train_segments
    );

    // ---- the exact-integer Gram and the cross-moment ---------------------------
    let t0 = Instant::now();
    let mut x_i8: Vec<i8> = Vec::with_capacity(train_rows * d);
    for r in 0..train_rows {
        for j in 0..d {
            x_i8.push(acts[r * d + j] as i8);
        }
    }
    let g_i64 = gram_fold_i8(vec![0i64; d * d], &x_i8, train_rows, d)
        .map_err(|e| format!("apeiron::gram_fold_i8 failed: {e:?}"))?;
    // The cross-moment X^T Y over the vocabulary subset: the one-hot of the teacher token.
    let mut r_i64 = vec![0i64; d * k];
    let mut covered = 0usize;
    for row in 0..train_rows {
        let t = u32::from_le_bytes([
            targets[row * 4],
            targets[row * 4 + 1],
            targets[row * 4 + 2],
            targets[row * 4 + 3],
        ]) as usize;
        let col = if t < col_of.len() { col_of[t] } else { u32::MAX };
        if col == u32::MAX {
            continue;
        }
        covered += 1;
        let c = col as usize;
        for j in 0..d {
            r_i64[j * k + c] += i64::from(x_i8[row * d + j]);
        }
    }
    println!(
        "gram + cross-moment: {:.1} s, {covered}/{train_rows} rows inside the vocabulary subset",
        t0.elapsed().as_secs_f64()
    );

    // ---- the base readout rows for the subset ----------------------------------
    let mut w_base = vec![0f64; d * k];
    for c in 0..k {
        for j in 0..d {
            let o = (c * d + j) * 4;
            w_base[c * d + j] = f64::from(f32::from_le_bytes([
                base_rows[o],
                base_rows[o + 1],
                base_rows[o + 2],
                base_rows[o + 3],
            ]));
        }
    }

    // ---- the solve, in the residual form --------------------------------------
    let t1 = Instant::now();
    let (g_q32, r_q32, delta) = solve_delta(&g_i64, &r_i64, &w_base, d, k, a.lambda)?;
    let solve_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let mut max_err = 0f64;
    for v in &delta {
        if v.abs() > max_err {
            max_err = v.abs();
        }
    }
    println!("solve: {solve_ms:.1} ms, |delta| max {max_err:.3e}");

    // ---- the rank-r projection (the LoRA analogue) ----------------------------
    let t2 = Instant::now();
    let (b_fac, a_fac, energy) = rank_project(&delta, d, k, a.rank)?;
    let proj_ms = t2.elapsed().as_secs_f64() * 1000.0;
    println!(
        "rank-{} projection: {proj_ms:.1} ms, retained energy {energy:.6}",
        a.rank
    );

    // ---- the artefact ----------------------------------------------------------
    let mut w = SafeWriter::new();
    let mut ids: Vec<i32> = Vec::with_capacity(k);
    for c in 0..k {
        let o = c * 4;
        ids.push(i32::from_le_bytes([vocab[o], vocab[o + 1], vocab[o + 2], vocab[o + 3]]));
    }
    w.add_i32("token_ids", vec![k], &ids);
    w.add_f32("B", vec![d, a.rank], &b_fac);
    w.add_f32("A", vec![a.rank, k], &a_fac);
    let bytes = w.write(&format!("{}/lora.safetensors", a.out))?;

    // ---- the revocation --------------------------------------------------------
    let mut revoked = None;
    if let Some(victim) = a.revoke {
        if victim >= train_rows {
            return Err(format!(
                "row {victim} is outside the training rows ({train_rows} of {rows})"
            )
            .into());
        }
        let t3 = Instant::now();
        // The victim's own Gram contribution and its cross-moment contribution, subtracted.
        let mut g2 = g_i64.clone();
        let mut r2 = r_i64.clone();
        for i in 0..d {
            for j in 0..d {
                g2[i * d + j] -= i64::from(x_i8[victim * d + i]) * i64::from(x_i8[victim * d + j]);
            }
        }
        let t = u32::from_le_bytes([
            targets[victim * 4],
            targets[victim * 4 + 1],
            targets[victim * 4 + 2],
            targets[victim * 4 + 3],
        ]) as usize;
        if t < col_of.len() && col_of[t] != u32::MAX {
            let c = col_of[t] as usize;
            for j in 0..d {
                r2[j * k + c] -= i64::from(x_i8[victim * d + j]);
            }
        }
        let (_, _, delta_rev) = solve_delta(&g2, &r2, &w_base, d, k, a.lambda)?;
        let mut max_rev = 0f64;
        for v in &delta_rev {
            if v.abs() > max_rev {
                max_rev = v.abs();
            }
        }
        // The revoked artefact, so the verification can measure what the revocation did.
        let (b_rev, a_rev, energy_rev) = rank_project(&delta_rev, d, k, a.rank)?;
        let mut wr = SafeWriter::new();
        wr.add_i32("token_ids", vec![k], &ids);
        wr.add_f32("B", vec![d, a.rank], &b_rev);
        wr.add_f32("A", vec![a.rank, k], &a_rev);
        wr.write(&format!("{}/lora_revoked.safetensors", a.out))?;
        println!("wrote {}/lora_revoked.safetensors (retained energy {energy_rev:.6})", a.out);
        // The exactness control: the Gram after the subtraction must equal a refit over
        // the remaining rows, bit for bit.
        let mut x_rest: Vec<i8> = Vec::with_capacity((train_rows - 1) * d);
        for row in 0..train_rows {
            if row == victim {
                continue;
            }
            x_rest.extend_from_slice(&x_i8[row * d..(row + 1) * d]);
        }
        let g_rest = gram_fold_i8(vec![0i64; d * d], &x_rest, train_rows - 1, d)
            .map_err(|e| format!("apeiron::gram_fold_i8 failed: {e:?}"))?;
        let syrk_exact = g_rest == g2;
        revoked = Some((max_rev, syrk_exact, t3.elapsed().as_secs_f64() * 1000.0));
        println!(
            "revocation of row {victim}: |delta| max {max_rev:.3e} (was {max_err:.3e}), \
             gram exact {syrk_exact}, {:.1} ms",
            t3.elapsed().as_secs_f64() * 1000.0
        );
    }

    let mut report = BTreeMap::new();
    report.insert("rows".to_string(), rows.to_string());
    report.insert("train_rows".to_string(), train_rows.to_string());
    report.insert("held_out_segments".to_string(), (segments - train_segments).to_string());
    report.insert("dim".to_string(), d.to_string());
    report.insert("vocab_columns".to_string(), k.to_string());
    report.insert("rows_in_vocab".to_string(), covered.to_string());
    report.insert("lambda".to_string(), format!("{:.3e}", a.lambda));
    report.insert("rank".to_string(), a.rank.to_string());
    report.insert("delta_max_abs".to_string(), format!("{max_err:.6e}"));
    report.insert("rank_projection_retained_energy".to_string(), format!("{energy:.6}"));
    report.insert("solve_ms".to_string(), format!("{solve_ms:.3}"));
    report.insert("rank_projection_ms".to_string(), format!("{proj_ms:.3}"));
    report.insert("lora_bytes".to_string(), bytes.to_string());
    report.insert("chol_residual_max".to_string(), g_q32.to_string());
    if let Some((m, exact, ms)) = revoked {
        report.insert("revoked_delta_max_abs".to_string(), format!("{m:.6e}"));
        report.insert("revoked_syrk_exact".to_string(), exact.to_string());
        report.insert("revoke_ms".to_string(), format!("{ms:.3}"));
    }
    let json = format!(
        "{{\n{}\n}}\n",
        report
            .iter()
            .map(|(k, v)| format!("  \"{k}\": {v}"))
            .collect::<Vec<_>>()
            .join(",\n")
    );
    print!("\n{json}");
    std::fs::write(format!("{}/report.json", a.out), &json)?;
    println!("wrote {}/lora.safetensors and report.json", a.out);
    Ok(())
}

/// The Q32.32 ridge in the residual form: `delta = (G/N + λI)^-1 (X^T Y / N - G W / N)`.
///
/// Returns the normalised Gram (for the report), the right-hand side, and the delta.
fn solve_delta(
    g_i64: &[i64],
    r_i64: &[i64],
    w_base: &[f64],
    d: usize,
    k: usize,
    lambda: f64,
) -> Result<(i64, Vec<i64>, Vec<f64>), Box<dyn Error>> {
    let rows = (g_i64.iter().sum::<i64>() as f64 / (d as f64 * d as f64)).max(1.0);
    // The Gram is a sum of outer products of i8 values, so its total is rows * 127^2 * d.
    // Recover the row count from the trace: trace(G) = rows * E[x^2] ~ rows * 127^2 * d / 3.
    let trace: i64 = (0..d).map(|i| g_i64[i * d + i]).sum();
    let rows_est = (trace as f64 / ((127.0 * 127.0) * d as f64 / 3.0)).round().max(1.0);
    let _ = rows;
    let scale_norm = rows_est * (127.0 * 127.0);
    let mut g_q32: Vec<i64> = g_i64
        .iter()
        .map(|&v| ((v as f64 / scale_norm) * Q32_ONE).round() as i64)
        .collect();
    let lam_q32 = (lambda.max(1e-4) * Q32_ONE).round() as i64;
    for i in 0..d {
        g_q32[i * d + i] = g_q32[i * d + i].saturating_add(lam_q32);
    }
    // RHS = X^T Y / N - (G / N) W
    let mut b_q32 = vec![0i64; d * k];
    for i in 0..d {
        for c in 0..k {
            let mut acc = 0f64;
            for j in 0..d {
                acc += (g_q32[i * d + j] as f64 / Q32_ONE) * w_base[c * d + j];
            }
            let rhs = (r_i64[i * k + c] as f64 / scale_norm) - acc;
            b_q32[i * k + c] = (rhs * Q32_ONE).round() as i64;
        }
    }
    let solved = chol_solve(&g_q32, &b_q32, d, k)
        .map_err(|e| format!("apeiron::chol_solve failed: {e:?}"))?;
    let delta: Vec<f64> = solved
        .x
        .iter()
        .map(|&v| v as f64 / Q32_ONE)
        .collect();
    Ok((solved.residual_max, b_q32, delta))
}

/// The truncated SVD of `delta` (d x k) at rank `r`, via the Gram of `delta delta^T`.
///
/// Returns `B` (d x r), `A` (r x k) with `delta ~= B A`, and the retained energy.
fn rank_project(
    delta: &[f64],
    d: usize,
    k: usize,
    r: usize,
) -> Result<(Vec<f32>, Vec<f32>, f64), Box<dyn Error>> {
    // G = delta delta^T (d x d), symmetric.
    let mut g = vec![0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut acc = 0f64;
            for c in 0..k {
                acc += delta[i * k + c] * delta[j * k + c];
            }
            g[i * d + j] = acc;
            g[j * d + i] = acc;
        }
    }
    let mut g_q32: Vec<i64> = g
        .iter()
        .map(|&v| (v * Q32_ONE).round() as i64)
        .collect();
    let lam_q32 = (1e-9 * Q32_ONE).round() as i64;
    for i in 0..d {
        g_q32[i * d + i] = g_q32[i * d + i].saturating_add(lam_q32);
    }
    let eig = top_eigh(&g_q32, d, r).map_err(|e| format!("apeiron::top_eigh failed: {e:?}"))?;
    let r = eig.evals.len().min(r);
    // The eigenvalues of delta delta^T are the squared singular values.
    let mut u = vec![0f64; d * r];
    let mut s = vec![0f64; r];
    for (idx, &lam) in eig.evals.iter().take(r).enumerate() {
        s[idx] = (lam as f64 / Q32_ONE).max(0.0).sqrt();
    }
    for i in 0..d {
        for idx in 0..r {
            u[i * r + idx] = eig.evecs[i * r + idx] as f64 / Q32_ONE;
        }
    }
    // V = delta^T U / s  (r x k), and B = U diag(s) (d x r).
    let mut v = vec![0f64; r * k];
    for idx in 0..r {
        if s[idx] <= 0.0 {
            continue;
        }
        for c in 0..k {
            let mut acc = 0f64;
            for i in 0..d {
                acc += delta[i * k + c] * u[i * r + idx];
            }
            v[idx * k + c] = acc / s[idx];
        }
    }
    let mut b = vec![0f64; d * r];
    let mut a = vec![0f64; r * k];
    let mut retained = 0f64;
    for idx in 0..r {
        for i in 0..d {
            b[i * r + idx] = u[i * r + idx] * s[idx];
        }
        for c in 0..k {
            a[idx * k + c] = v[idx * k + c];
        }
    }
    let total: f64 = delta.iter().map(|v| v * v).sum();
    for i in 0..d {
        for c in 0..k {
            let mut acc = 0f64;
            for idx in 0..r {
                acc += b[i * r + idx] * a[idx * k + c];
            }
            retained += acc * acc;
        }
    }
    let energy = if total > 0.0 { retained / total } else { 0.0 };
    Ok((
        b.iter().map(|&v| v as f32).collect(),
        a.iter().map(|&v| v as f32).collect(),
        energy,
    ))
}
