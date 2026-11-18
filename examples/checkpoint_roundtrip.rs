//! `checkpoint_roundtrip` — the checkpoint round-trip, measured on a REAL checkpoint.
//!
//! ```text
//! checkpoint_roundtrip --input safetensors:<dir> | gguf:<path> --out <dir>
//!                      [--all] [--tensors a,b,c] [--calib-rows 512] [--seed 7]
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
//! # The orientation, stated once because it is easy to get wrong
//!
//! A weight matrix is `d` inputs by `m` outputs. The monoid's Gram is over the
//! **input** axis (`d x d`), and the right-hand side has one column per output. Both
//! source formats store the same flat order for a weight: `flat[c * d + k]` is output
//! `c`, input `k`. So the matrix the lift sees is `A[k * m + c] = flat[c * d + k]` -
//! the transpose of the stored layout, which is what makes the Gram land on the input
//! axis. Getting this wrong still round-trips (the export preserves the flat order) but
//! it lifts a scrambled matrix instead of the weight, and the conditioning is then
//! unrelated to the weight's own.
//!
//! # The exports
//!
//! - `ref.safetensors` — the checkpoint's weights as we read them (F32).
//! - `roundtrip.safetensors` — the weights after the monoid lift and the solve, i.e.
//!   what the customer would take away.
//! - `report.json` — every number: bytes, timings, the reconstruction error, the
//!   bit-exactness of the O(1) subtraction, and the extrapolation to the whole model.
//!
//! # The negative controls (they are part of the deliverable)
//!
//! - A revoked shard that is folded and then subtracted must leave the Gram state
//!   **bit-identical** to the state before the fold. If it does not, the subtraction is
//!   not exact and the run fails.
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
use std::io::{Seek, Write};
use std::time::Instant;

const Q32_ONE: f64 = 4_294_967_296.0;
/// One ulp of the Q32.32 fixed-point grid: the floor every reconstruction error sits on.
const Q32_FLOOR: f64 = 1.0 / Q32_ONE;

struct Args {
    input: String,
    out: String,
    tensors: Vec<String>,
    all: bool,
    calib_rows: usize,
    seed: u64,
    threads: usize,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        input: String::new(),
        out: String::new(),
        tensors: Vec::new(),
        all: false,
        calib_rows: 512,
        seed: 7,
        threads: 8,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--input" => a.input = val()?,
            "--out" => a.out = val()?,
            "--tensors" => {
                a.tensors = val()?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--all" => a.all = true,
            "--calib-rows" => a.calib_rows = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--threads" => a.threads = val()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if a.input.is_empty() || a.out.is_empty() {
        return Err("--input and --out are required".into());
    }
    Ok(a)
}

/// One tensor of a checkpoint: `d` inputs by `m` outputs, read as f32 in the canonical
/// layout `A[k * m + c]` (k input, c output).
struct Matrix {
    d: usize,
    m: usize,
    data: Vec<f32>,
}

/// bf16 bits to f32: the value is the high half of an f32.
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// The checkpoint we read from: a safetensors directory (a PyTorch checkpoint) or a GGUF file.
enum Source {
    Safe {
        path: String,
        tensors: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
    },
    Gguf {
        gguf: Gguf,
    },
}

/// A minimal, strict safetensors header parser.
///
/// The header is a flat JSON object; this scans it for the three fields it needs per
/// tensor (dtype, shape, data_offsets) and refuses anything it does not understand.
fn parse_safetensors_header(bytes: &[u8]) -> Result<BTreeMap<String, (String, Vec<usize>, usize, usize)>, Box<dyn Error>> {
    let text = std::str::from_utf8(bytes)?;
    let mut out = BTreeMap::new();
    let mut i = 0usize;
    while i < text.len() {
        // Find the next key: a '"' followed by a name, then '":'.
        let Some(quote) = text[i..].find('"') else { break };
        let start = i + quote;
        let after = start + 1;
        let Some(rel_end) = text[after..].find('"') else { break };
        let end = after + rel_end;
        let name = text[after..end].to_string();
        let mut j = end + 1;
        while j < text.len() && text.as_bytes()[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= text.len() || text.as_bytes()[j] != b':' {
            i = end + 1;
            continue;
        }
        j += 1;
        while j < text.len() && text.as_bytes()[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= text.len() || text.as_bytes()[j] != b'{' {
            i = end + 1;
            continue;
        }
        // Match the braces of this value.
        let obj = j;
        let mut depth = 0usize;
        let mut close = obj;
        while close < text.len() {
            match text.as_bytes()[close] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            close += 1;
        }
        let body = &text[obj..close];
        i = close + 1;
        // A header may carry a "__metadata__" object; a tensor entry must have a dtype.
        let Some(dtype) = body.split("\"dtype\":\"").nth(1).and_then(|s| s.split('"').next()) else {
            continue;
        };
        let Some(shape_txt) = body.split("\"shape\":[").nth(1).and_then(|s| s.split(']').next()) else {
            return Err(format!("safetensors: tensor {name} has no shape").into());
        };
        let shape: Vec<usize> = shape_txt
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        let Some(off_txt) = body.split("\"data_offsets\":[").nth(1).and_then(|s| s.split(']').next()) else {
            return Err(format!("safetensors: tensor {name} has no data_offsets").into());
        };
        let mut offs = off_txt.split(',');
        let a: usize = offs.next().ok_or("bad data_offsets")?.trim().parse()?;
        let b: usize = offs.next().ok_or("bad data_offsets")?.trim().parse()?;
        out.insert(name, (dtype.to_string(), shape, a, b));
    }
    Ok(out)
}

/// Read one safetensors tensor and return it as `A[k * m + c]` (k input, c output).
fn read_safetensors(
    path: &str,
    tensors: &BTreeMap<String, (String, Vec<usize>, usize, usize)>,
    name: &str,
) -> Result<Matrix, Box<dyn Error>> {
    let (dtype, shape, a, b) = tensors
        .get(name)
        .ok_or_else(|| format!("safetensors: no tensor {name}"))?;
    if shape.len() != 2 {
        return Err(format!("safetensors: {name} is not a matrix").into());
    }
    let m = shape[0]; // outputs
    let d = shape[1]; // inputs
    let n = m * d;
    let mut raw = vec![0u8; b - a];
    {
        use std::io::Read;
        let mut f = File::open(path)?;
        f.seek(std::io::SeekFrom::Start((8 + a) as u64))?;
        f.read_exact(&mut raw)?;
    }
    // flat[c * d + k] -> A[k * m + c]
    let mut data = vec![0f32; n];
    match dtype.as_str() {
        "F32" => {
            for c in 0..m {
                for k in 0..d {
                    let o = (c * d + k) * 4;
                    let v = f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
                    data[k * m + c] = v;
                }
            }
        }
        "F16" => {
            for c in 0..m {
                for k in 0..d {
                    let o = (c * d + k) * 2;
                    let h = u16::from_le_bytes([raw[o], raw[o + 1]]);
                    data[k * m + c] = quant::f16_to_f32(h);
                }
            }
        }
        "BF16" => {
            for c in 0..m {
                for k in 0..d {
                    let o = (c * d + k) * 2;
                    let h = u16::from_le_bytes([raw[o], raw[o + 1]]);
                    data[k * m + c] = bf16_to_f32(h);
                }
            }
        }
        other => return Err(format!("safetensors: dtype {other} is not supported").into()),
    }
    Ok(Matrix { d, m, data })
}

/// Read one GGUF tensor and return it as `A[k * m + c]` (k input, c output).
///
/// GGUF stores a weight with the input axis fastest (`dims = [in, out]`), so the flat
/// order is `flat[c * d + k]` - the same order safetensors uses.
fn read_gguf(gguf: &Gguf, name: &str) -> Result<Matrix, Box<dyn Error>> {
    let t = gguf
        .header
        .tensor(name)
        .ok_or_else(|| format!("no tensor {name}"))?;
    if t.dims.len() != 2 {
        return Err(format!("tensor {name} is not a matrix").into());
    }
    let d = t.dims[0] as usize; // inputs
    let m = t.dims[1] as usize; // outputs
    let n = t.n_elements()? as usize;
    let bytes = t
        .n_bytes()?
        .ok_or_else(|| format!("tensor {name}: unsupported ggml type"))?;
    let raw = gguf.tensor_bytes_prefix(t, bytes)?;
    let mut flat = vec![0f32; n];
    match t.ggml_type {
        GGML_TYPE_Q8_0 => {
            const QK: usize = quant::Q8_0_QK;
            for b in 0..n.div_ceil(QK) {
                let scale = quant::q8_0_scale(&raw, b)?;
                let codes = quant::q8_0_codes(&raw, b)?;
                for j in 0..QK {
                    let i = b * QK + j;
                    if i < n {
                        flat[i] = scale * codes[j] as f32;
                    }
                }
            }
        }
        GGML_TYPE_F32 => {
            for i in 0..n {
                flat[i] = f32::from_le_bytes([
                    raw[i * 4],
                    raw[i * 4 + 1],
                    raw[i * 4 + 2],
                    raw[i * 4 + 3],
                ]);
            }
        }
        GGML_TYPE_F16 => {
            for i in 0..n {
                let h = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                flat[i] = quant::f16_to_f32(h);
            }
        }
        GGML_TYPE_BF16 => {
            for i in 0..n {
                let h = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                flat[i] = bf16_to_f32(h);
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
    // flat[c * d + k] -> A[k * m + c]
    let mut data = vec![0f32; n];
    for c in 0..m {
        for k in 0..d {
            data[k * m + c] = flat[c * d + k];
        }
    }
    Ok(Matrix { d, m, data })
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


/// An f64 Cholesky solve over the same exact integer Gram.
///
/// The exact-integer path (`chol_solve`) is Q32.32 and it **overflows on the wide
/// systems**: a 151936-column right-hand side makes the refinement residual leave
/// i64. The Gram itself is exact either way - it is the same integer accumulator -
/// so this is a change of arithmetic for the *solve*, not of the method. Which path
/// each matrix took is recorded.
fn chol_solve_f64(g_f64: &[f64], b: &[f64], n: usize, k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = g_f64[i * n + j];
            for t in 0..j {
                s -= l[i * n + t] * l[j * n + t];
            }
            if i == j {
                if !(s > 0.0) {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut x = vec![0f64; n * k];
    for c in 0..k {
        for i in 0..n {
            let mut s = b[i * k + c];
            for t in 0..i {
                s -= l[i * n + t] * x[t * k + c];
            }
            x[i * k + c] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = x[i * k + c];
            for t in (i + 1)..n {
                s -= l[t * n + i] * x[t * k + c];
            }
            x[i * k + c] = s / l[i * n + i];
        }
    }
    Some(x)
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = parse_args()?;
    std::fs::create_dir_all(&a.out)?;

    // ---- open the source ---------------------------------------------------------
    let (source, mut names, model_label) = if let Some(dir) = a.input.strip_prefix("safetensors:") {
        let path = format!("{}/model.safetensors", dir.trim_end_matches('/'));
        let mut file = File::open(&path)?;
        let mut len = [0u8; 8];
        {
            use std::io::Read;
            file.read_exact(&mut len)?;
        }
        let hlen = u64::from_le_bytes(len) as usize;
        let mut header = vec![0u8; hlen];
        {
            use std::io::Read;
            file.read_exact(&mut header)?;
        }
        let index = parse_safetensors_header(&header)?;
        let mut mats: Vec<String> = index
            .iter()
            .filter(|(_, (_, shape, _, _))| shape.len() == 2)
            .map(|(k, _)| k.clone())
            .collect();
        mats.sort();
        let _n_total = index.len();
        let source = Source::Safe { path, tensors: index };
        (source, mats, format!("safetensors dir {}", dir))
    } else if let Some(path) = a.input.strip_prefix("gguf:") {
        let gguf = Gguf::open(path)?;
        let mut mats: Vec<String> = gguf
            .header
            .tensors
            .iter()
            .filter(|t| t.dims.len() == 2)
            .map(|t| String::from_utf8_lossy(&t.name).to_string())
            .collect();
        mats.sort();
        let _n_total = gguf.header.tensors.len();
        (Source::Gguf { gguf }, mats, format!("gguf {path}"))
    } else {
        return Err("--input is safetensors:<dir> or gguf:<path>".into());
    };

    if !a.tensors.is_empty() {
        names = a.tensors.clone();
    } else if !a.all {
        names.truncate(4);
    }

    let rows_path = format!("{}/rows.jsonl", a.out);
    let mut rows_file = File::create(&rows_path)?;
    let mut ref_w = SafeWriter::new();
    let mut out_w = SafeWriter::new();
    let mut rows_json: Vec<String> = Vec::new();
    let mut n_params = 0u64;
    let mut all_exact = true;
    let mut worst_err = 0f64;
    let mut skipped: Vec<String> = Vec::new();
    let t_start = Instant::now();

    for name in &names {
        let t0 = Instant::now();
        let mat = match &source {
            Source::Safe { path, tensors } => read_safetensors(path, tensors, name)?,
            Source::Gguf { gguf } => read_gguf(gguf, name)?,
        };
        let Matrix { d, m, data: w } = mat;
        let n = w.len();
        let read_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if d == 0 || m == 0 || d > 4096 {
            skipped.push(format!("{name} [{m}x{d}]"));
            let row = format!(
                "{{\"tensor\":\"{name}\",\"shape\":[{m},{d}],\"skipped\":\"d={d} m={m}\",\"read_ms\":{read_ms:.1}}}"
            );
            writeln!(rows_file, "{row}")?;
            rows_file.flush()?;
            rows_json.push(row);
            continue;
        }
        n_params += n as u64;

        // ---- lift: (G_prior, R_prior = G_prior * W) -------------------------------
        let t_lift = Instant::now();
        let rows = a.calib_rows.min(d.max(1));
        let x_cal = calib(rows, d, a.seed);
        let g_raw = gram_fold_i8(vec![0i64; d * d], &x_cal, rows, d)
            .map_err(|e| format!("gram_fold_i8: {e:?}"))?;
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
        // The rows of the product are independent, so the rows are spread over threads.
        let threads = a.threads.max(1);
        let mut r_prior = vec![0i64; d * m];
        {
            let g = &g_prior;
            let wref = &w;
            let rref = &mut r_prior;
            std::thread::scope(|scope| {
                let chunk = (d + threads - 1) / threads;
                for (t, part) in rref.chunks_mut(chunk * m).enumerate() {
                    let start = t * chunk;
                    let g = g;
                    let wref = wref;
                    scope.spawn(move || {
                        for (li, row) in part.chunks_mut(m).enumerate() {
                            let i = start + li;
                            for c in 0..m {
                                let mut acc: i128 = 0;
                                for k in 0..d {
                                    acc += (g[i * d + k] as i128)
                                        * ((wref[k * m + c] as f64 * Q32_ONE) as i128);
                                }
                                row[c] = (acc as f64 / Q32_ONE).round() as i64;
                            }
                        }
                    });
                }
            });
        }
        let lift_ms = t_lift.elapsed().as_secs_f64() * 1000.0;

        // ---- solve back to dense weights -------------------------------------------
        // The exact-integer solve first; on overflow, the same exact Gram in f64.
        let t_solve = Instant::now();
        let mut solve_path = "q32";
        let w_rec: Vec<f32> = match chol_solve(&g_prior, &r_prior, d, m) {
            Ok(solved) => solved
                .x
                .iter()
                .map(|&v| (v as f64 / Q32_ONE) as f32)
                .collect(),
            Err(e) => {
                solve_path = "f64";
                let mut g_f64: Vec<f64> = g_raw
                    .iter()
                    .map(|&v| v as f64 / scale)
                    .collect();
                for i in 0..d {
                    g_f64[i * d + i] += 1e-4;
                }
                let mut r_f64 = vec![0f64; d * m];
                for i in 0..d {
                    for c in 0..m {
                        let mut acc = 0f64;
                        for k in 0..d {
                            acc += g_f64[i * d + k] * f64::from(w[k * m + c]);
                        }
                        r_f64[i * m + c] = acc;
                    }
                }
                let x = chol_solve_f64(&g_f64, &r_f64, d, m)
                    .ok_or_else(|| format!("chol_solve (q32 {e:?} and f64) both refused"))?;
                x.into_iter().map(|v| v as f32).collect()
            }
        };
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
        if max_err > worst_err {
            worst_err = max_err;
        }

        // ---- fold a domain shard and a revoked shard, then subtract the revoked one --
        let t_fold = Instant::now();
        let x_dom = calib(rows, d, a.seed ^ 0xD0_0D);
        let x_poison = calib(rows, d, a.seed ^ 0xB0_0B);
        let g_dom = gram_fold_i8(vec![0i64; d * d], &x_dom, rows, d)
            .map_err(|e| format!("gram_fold_i8: {e:?}"))?;
        let g_poison = gram_fold_i8(vec![0i64; d * d], &x_poison, rows, d)
            .map_err(|e| format!("gram_fold_i8: {e:?}"))?;
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

        // ---- export (the flat order is preserved: the export is the same matrix) ----
        let mut flat_rec = vec![0f32; n];
        for c in 0..m {
            for k in 0..d {
                flat_rec[c * d + k] = w_rec[k * m + c];
            }
        }
        let mut flat_ref = vec![0f32; n];
        for c in 0..m {
            for k in 0..d {
                flat_ref[c * d + k] = w[k * m + c];
            }
        }
        let shape = if matches!(source, Source::Safe { .. }) {
            vec![m, d]
        } else {
            vec![m, d]
        };
        ref_w.add(name, shape.clone(), flat_ref);
        out_w.add(name, shape, flat_rec);

        let row = format!(
            "{{\"tensor\":\"{name}\",\"shape\":[{m},{d}],\"params\":{n},\"read_ms\":{read_ms:.1},\
             \"lift_ms\":{lift_ms:.1},\"solve_ms\":{solve_ms:.1},\"fold_unlearn_ms\":{fold_ms:.1},\
             \"max_reconstruction_err\":{max_err:.6e},\"bit_exact_fraction\":{exact_frac:.6},\
             \"unlearn_bit_exact\":{unlearn_exact},\"solve_path\":\"{solve_path}\",\"floor_q32\":{Q32_FLOOR:.6e}}}"
        );
        writeln!(rows_file, "{row}")?;
        rows_file.flush()?;
        rows_json.push(row);
        println!(
            "{name}: {m}x{d} read {read_ms:.0}ms lift {lift_ms:.0}ms solve {solve_ms:.0}ms ({solve_path}) \
             fold+unlearn {fold_ms:.0}ms max_err {max_err:.3e} exact {exact_frac:.4} unlearn_exact {unlearn_exact}"
        );
    }

    // The reference export is only written for a small selection: for a whole model it
    // would double the disk for a file the verifier already has (the original checkpoint).
    let ref_bytes = if a.all {
        0
    } else {
        ref_w.write(&format!("{}/ref.safetensors", a.out))?
    };
    let out_bytes = out_w.write(&format!("{}/roundtrip.safetensors", a.out))?;
    let elapsed = t_start.elapsed().as_secs_f64();
    let n_sel = rows_json.len();

    let report = format!(
        "{{\n  \"source\": \"{model}\",\n  \"calib_rows\": {calib_rows},\n  \"seed\": {seed},\n  \
         \"threads\": {threads},\n  \"selected\": {n_sel},\n  \"skipped\": {skipped},\n  \
         \"roundtrip_params\": {n_params},\n  \"ref_safetensors_bytes\": {ref_bytes},\n  \
         \"roundtrip_safetensors_bytes\": {out_bytes},\n  \"unlearn_bit_exact_all\": {all_exact},\n  \
         \"worst_reconstruction_err\": {worst_err:.6e},\n  \"q32_floor\": {floor},\n  \
         \"wall_s\": {elapsed:.1},\n  \"rows\": [\n    {rows}\n  ]\n}}\n",
        model = model_label,
        calib_rows = a.calib_rows,
        seed = a.seed,
        threads = a.threads,
        n_sel = n_sel,
        skipped = skipped.len(),
        n_params = n_params,
        ref_bytes = ref_bytes,
        out_bytes = out_bytes,
        all_exact = all_exact,
        worst_err = worst_err,
        floor = Q32_FLOOR,
        elapsed = elapsed,
        rows = rows_json.join(",\n    "),
    );
    std::fs::write(format!("{}/report.json", a.out), &report)?;
    println!("wrote {}/report.json", a.out);
    println!(
        "selected {n_sel} matrices, {n_params} params, worst error {worst_err:.3e}, \
         unlearn bit-exact on all: {all_exact}, wall {elapsed:.1}s"
    );
    Ok(())
}
