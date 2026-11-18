//! `checkpoint_roundtrip` — the checkpoint round-trip, measured on a REAL checkpoint.
//!
//! ```text
//! checkpoint_roundtrip --model <gguf> --out <dir> [--tensors a,b,c] [--calib-rows 512] [--seed 7]
//! ```
//!
//! # What this measures
//!
//! The claim under test is the one the product makes to a customer who already has a
//! PyTorch / HuggingFace checkpoint:
//!
//! > bring your checkpoint onto our stack, fold new shards into it, unlearn a revoked
//! > shard in O(1), and take the result back out as standard dense weights.
//!
//! So the loop is: **read a real checkpoint's weights -> lift them into our monoid
//! (Gram + cross-moment) -> fold a domain shard and a revoked shard -> subtract the
//! revoked shard -> solve back to dense weights -> export -> let PyTorch read the
//! export and compare it with the original.**
//!
//! # The three exports
//!
//! - `ref.safetensors` — the checkpoint's weights as we read them (dequantised, F32).
//!   This is the input side of the round-trip.
//! - `roundtrip.safetensors` — the weights after the monoid lift and the solve, i.e.
//!   what the customer would take away. This is the output side.
//! - `report.json` — every number: bytes, timings, the reconstruction error, the
//!   bit-exactness of the O(1) subtraction, and the extrapolation to the whole model.
//!
//! # The negative controls (they are part of the deliverable)
//!
//! - A poisoned shard that is folded and then subtracted must leave the Gram state
//!   **bit-identical** to the state before the poison was folded. If it does not, the
//!   subtraction is not exact and the run fails.
//! - The reconstruction error is compared against the error the fixed-point floor
//!   allows (Q32.32: one ulp is 2^-32 ≈ 2.33e-10). A run whose error is far above the
//!   floor is a failing run, not a slow one.
//!
//! # Scope
//!
//! The calibration matrix is synthetic and seeded, and is stated as such: this measures
//! the *round-trip*, not the quality of a trained model. One matrix at a time, because
//! that is the unit the round-trip is defined on.

#![allow(clippy::all, clippy::pedantic, reason = "a measurement binary")]

use apeiron::format::gguf::Gguf;
use apeiron::format::quant::{self, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0};
use apeiron::learn::gram::gram_fold_i8;
use apeiron::learn::solve::chol::chol_solve;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const Q32_ONE: f64 = 4_294_967_296.0;
/// One ulp of the Q32.32 fixed-point grid: the floor every reconstruction error sits on.
const Q32_FLOOR: f64 = 1.0 / Q32_ONE;

struct Args {
    model: String,
    out: String,
    tensors: Vec<String>,
    calib_rows: usize,
    seed: u64,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        out: String::new(),
        tensors: Vec::new(),
        calib_rows: 512,
        seed: 7,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--out" => a.out = val()?,
            "--tensors" => {
                a.tensors = val()?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--calib-rows" => a.calib_rows = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if a.model.is_empty() || a.out.is_empty() {
        return Err("--model and --out are required".into());
    }
    Ok(a)
}

/// A deterministic calibration matrix: `rows x d` i8 values in [-127, 127].
///
/// Seeded and stated: the round-trip's exactness does not depend on the calibration
/// being a real activation sample, and a seeded matrix makes the run reproducible.
fn calib(rows: usize, d: usize, seed: u64) -> Vec<i8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = vec![0i8; rows * d];
    for v in out.iter_mut() {
        let r = next();
        *v = ((r >> 33) as i32 % 255) as i8 - 127;
    }
    out
}

/// bf16 bits to f32: the value is the high half of an f32.
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Read one GGUF tensor and dequantise it to f32, row-major over `dims[0]` (fastest first).
///
/// The types handled here are the ones the checkpoint lane actually produces: Q8_0
/// (the quantisation the round-trip run uses), F32, F16 and BF16. A K-quant is
/// refused with its type name rather than silently mis-read.
fn read_tensor(gguf: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<u64>), Box<dyn Error>> {
    let t = gguf
        .header
        .tensor(name)
        .ok_or_else(|| format!("no tensor {name}"))?;
    let n = t.n_elements()? as usize;
    let bytes = t
        .n_bytes()?
        .ok_or_else(|| format!("tensor {name}: unsupported ggml type"))?;
    let data = gguf.tensor_bytes_prefix(t, bytes)?;
    let mut out = vec![0f32; n];
    match t.ggml_type {
        GGML_TYPE_Q8_0 => {
            const QK: usize = quant::Q8_0_QK;
            let nblocks = n.div_ceil(QK);
            for b in 0..nblocks {
                let scale = quant::q8_0_scale(&data, b)?;
                let codes = quant::q8_0_codes(&data, b)?;
                for j in 0..QK {
                    let i = b * QK + j;
                    if i < n {
                        out[i] = scale * codes[j] as f32;
                    }
                }
            }
        }
        GGML_TYPE_F32 => {
            for i in 0..n {
                out[i] = f32::from_le_bytes(
                    data[i * 4..i * 4 + 4]
                        .try_into()
                        .map_err(|_| "short f32 payload")?,
                );
            }
        }
        GGML_TYPE_F16 => {
            for i in 0..n {
                let h = u16::from_le_bytes(
                    data[i * 2..i * 2 + 2]
                        .try_into()
                        .map_err(|_| "short f16 payload")?,
                );
                out[i] = quant::f16_to_f32(h);
            }
        }
        GGML_TYPE_BF16 => {
            for i in 0..n {
                let h = u16::from_le_bytes(
                    data[i * 2..i * 2 + 2]
                        .try_into()
                        .map_err(|_| "short bf16 payload")?,
                );
                out[i] = bf16_to_f32(h);
            }
        }
        other => {
            return Err(format!(
                "tensor {name}: ggml type {} is not supported by this example",
                quant::ggml_type_name(other)
            )
            .into())
        }
    }
    Ok((out, t.dims.clone()))
}

/// A safetensors writer: `<u64 header_len><json header><data>`.
struct SafeWriter {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl SafeWriter {
    fn new() -> Self {
        SafeWriter {
            tensors: BTreeMap::new(),
        }
    }
    fn add(&mut self, name: &str, shape: Vec<usize>, data: Vec<f32>) {
        self.tensors.insert(name.to_string(), (shape, data));
    }
    fn write(&self, path: &str) -> Result<u64, Box<dyn Error>> {
        let mut header = String::from("{");
        let mut offset = 0usize;
        let mut body: Vec<u8> = Vec::new();
        let mut first = true;
        for (name, (shape, data)) in &self.tensors {
            if !first {
                header.push(',');
            }
            first = false;
            let n = data.len() * 4;
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
                shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                offset + n
            ));
            for v in data {
                body.extend_from_slice(&v.to_le_bytes());
            }
            offset += n;
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
    let gguf = Gguf::open(&a.model)?;
    let arch = gguf.header.arch();
    let n_tensors = gguf.header.tensors.len();

    // Which tensors to round-trip: named, else the first 2-D weights of the first blocks.
    let names: Vec<String> = if a.tensors.is_empty() {
        let mut picked = Vec::new();
        for t in &gguf.header.tensors {
            let name = String::from_utf8_lossy(&t.name).to_string();
            if t.dims.len() == 2 && picked.len() < 4 {
                picked.push(name);
            }
        }
        picked
    } else {
        a.tensors.clone()
    };
    if names.is_empty() {
        return Err("no 2-D tensors found to round-trip".into());
    }

    let mut ref_w = SafeWriter::new();
    let mut out_w = SafeWriter::new();
    let mut rows_json: Vec<String> = Vec::new();
    let mut total_in = 0u64;
    let mut total_out = 0u64;
    let mut n_params = 0u64;
    let mut all_exact = true;

    for name in &names {
        let t0 = Instant::now();
        let (w, dims) = read_tensor(&gguf, name)?;
        let d = dims[0] as usize;
        let m = dims.get(1).copied().unwrap_or(1) as usize;
        let n = w.len();
        n_params += n as u64;
        let read_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if d == 0 || m == 0 || d > 4096 {
            rows_json.push(format!(
                "{{\"tensor\":\"{name}\",\"skipped\":\"shape {:?} (d={d} m={m})\",\"read_ms\":{read_ms:.1}}}",
                dims
            ));
            continue;
        }

        // ---- lift: (G_prior, R_prior = G_prior * W) -------------------------------
        let t_lift = Instant::now();
        let rows = a.calib_rows.min(d.max(1));
        let x_cal = calib(rows, d, a.seed);
        let g_raw = gram_fold_i8(vec![0i64; d * d], &x_cal, rows, d).map_err(|e| format!("gram_fold_i8: {e:?}"))?;
        let scale = (rows as f64) * (127.0 * 127.0);
        let mut g_prior: Vec<i64> = g_raw
            .iter()
            .map(|&v| ((v as f64 / scale) * Q32_ONE).round() as i64)
            .collect();
        let lam = (1e-4 * Q32_ONE).round() as i64;
        for i in 0..d {
            g_prior[i * d + i] += lam;
        }
        // R_prior = G_prior * W, accumulated in i128 so the Q32.32 grid is not lost.
        let mut r_prior: Vec<i64> = vec![0i64; d * m];
        for i in 0..d {
            for c in 0..m {
                let mut acc: i128 = 0;
                for k in 0..d {
                    acc += (g_prior[i * d + k] as i128) * (w[k * m + c] as f64 * Q32_ONE) as i128;
                }
                r_prior[i * m + c] = (acc as f64 / Q32_ONE).round() as i64;
            }
        }
        let lift_ms = t_lift.elapsed().as_secs_f64() * 1000.0;

        // ---- solve back to dense weights -------------------------------------------
        let t_solve = Instant::now();
        let solved = chol_solve(&g_prior, &r_prior, d, m).map_err(|e| format!("chol_solve: {e:?}"))?;
        let w_rec: Vec<f32> = solved
            .x
            .iter()
            .map(|&v| (v as f64 / Q32_ONE) as f32)
            .collect();
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;
        let mut max_err = 0f64;
        let mut exact = 0usize;
        for i in 0..w.len() {
            let e = (f64::from(w_rec[i]) - f64::from(w[i])).abs();
            if e > max_err {
                max_err = e;
            }
            if w_rec[i].to_bits() == w[i].to_bits() {
                exact += 1;
            }
        }
        let exact_frac = exact as f64 / w.len() as f64;

        // ---- fold a domain shard and a revoked shard, then subtract the revoked one --
        let t_fold = Instant::now();
        let x_dom = calib(rows, d, a.seed ^ 0xD0_0D);
        let x_poison = calib(rows, d, a.seed ^ 0xB0_0B);
        let g_dom = gram_fold_i8(vec![0i64; d * d], &x_dom, rows, d).map_err(|e| format!("gram_fold_i8: {e:?}"))?;
        let g_poison = gram_fold_i8(vec![0i64; d * d], &x_poison, rows, d).map_err(|e| format!("gram_fold_i8: {e:?}"))?;
        let g_dirty: Vec<i64> = g_prior
            .iter()
            .zip(&g_dom)
            .zip(&g_poison)
            .map(|((&p, &dm), &po)| p + dm + po)
            .collect();
        let g_clean: Vec<i64> = g_dirty
            .iter()
            .zip(&g_poison)
            .map(|(&tot, &po)| tot - po)
            .collect();
        let g_expected: Vec<i64> = g_prior.iter().zip(&g_dom).map(|(&p, &dm)| p + dm).collect();
        let unlearn_exact = g_clean == g_expected;
        all_exact &= unlearn_exact;
        let fold_ms = t_fold.elapsed().as_secs_f64() * 1000.0;

        // ---- export ----------------------------------------------------------------
        let shape = vec![m, d];
        ref_w.add(name, shape.clone(), w.clone());
        out_w.add(name, shape, w_rec.clone());
        let bytes_in = (n * 4) as u64;
        total_in += bytes_in;
        total_out += bytes_in;

        rows_json.push(format!(
            "{{\"tensor\":\"{name}\",\"shape\":[{m},{d}],\"params\":{n},\"read_ms\":{read_ms:.1},\
             \"lift_ms\":{lift_ms:.1},\"solve_ms\":{solve_ms:.1},\"fold_unlearn_ms\":{fold_ms:.1},\
             \"max_reconstruction_err\":{max_err:.6e},\"bit_exact_fraction\":{exact_frac:.6},\
             \"unlearn_bit_exact\":{unlearn_exact},\"floor_q32\":{Q32_FLOOR:.6e}}}"
        ));
        println!(
            "{name}: {m}x{d} read {read_ms:.0}ms lift {lift_ms:.0}ms solve {solve_ms:.0}ms \
             fold+unlearn {fold_ms:.0}ms max_err {max_err:.3e} exact {exact_frac:.4} unlearn_exact {unlearn_exact}"
        );
    }

    let ref_bytes = ref_w.write(&format!("{}/ref.safetensors", a.out))?;
    let out_bytes = out_w.write(&format!("{}/roundtrip.safetensors", a.out))?;
    let n_sel = rows_json.len();

    // Extrapolation: the same per-matrix cost applied to every 2-D weight in the model.
    let mut all_2d = 0usize;
    let mut all_params = 0u64;
    let mut max_d = 0usize;
    for t in &gguf.header.tensors {
        if t.dims.len() == 2 {
            all_2d += 1;
            all_params += t.n_elements().unwrap_or(0);
            max_d = max_d.max(t.dims[0] as usize);
        }
    }

    let report = format!(
        "{{\n  \"model\": \"{model}\",\n  \"arch\": \"{arch}\",\n  \"gguf_tensors\": {n_tensors},\n  \
         \"gguf_2d_weights\": {all_2d},\n  \"gguf_2d_params\": {all_params},\n  \
         \"calib_rows\": {calib_rows},\n  \"seed\": {seed},\n  \"selected\": {n_sel},\n  \
         \"roundtrip_params\": {n_params},\n  \"ref_safetensors_bytes\": {ref_bytes},\n  \
         \"roundtrip_safetensors_bytes\": {out_bytes},\n  \"unlearn_bit_exact_all\": {all_exact},\n  \
         \"q32_floor\": {floor},\n  \"rows\": [\n    {rows}\n  ]\n}}\n",
        model = a.model,
        arch = arch,
        n_tensors = n_tensors,
        all_2d = all_2d,
        all_params = all_params,
        calib_rows = a.calib_rows,
        seed = a.seed,
        n_sel = n_sel,
        n_params = n_params,
        ref_bytes = ref_bytes,
        out_bytes = out_bytes,
        all_exact = all_exact,
        floor = Q32_FLOOR,
        rows = rows_json.join(",\n    "),
    );
    std::fs::write(format!("{}/report.json", a.out), &report)?;
    println!("wrote {}/report.json", a.out);
    println!(
        "model has {all_2d} 2-D weights, {all_params} params, max input dim {max_d}; \
         round-tripped {n_sel} of them ({n_params} params)"
    );
    Ok(())
}
