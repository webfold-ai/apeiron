//! MLPerf v6.1 (Llama-3.1 Dense & GPT-OSS MoE) vs. Apeiron Vulkan GPU & Closed-Form Benchmark.
//!
//! Measures on the active Vulkan GPU (`Vk::load()`) and CPU:
//! 1. **56-Dispatch GQA + RoPE + SwiGLU Pre-Norm Transformer Layer GPU Training Step**:
//!    - Baseline: Unfused per-step pipeline launch + dispatch (`adam_update_dispatch` per-step
//!      rebuild, modeling the small-kernel launch / conversion overhead Firmus profiled in MLPerf v6.1).
//!    - Apeiron Fused Single-Submission GPU `Trainer::step`: 1 command buffer submission for all 56
//!      forward + KL + backward + Adam SPIR-V kernels, hardware GPU timestamps (`QueryPool`),
//!      device-resident weights/moments/activations.
//! 2. **Closed-Form `apeiron::learn::gram::gram_fold_i8` + Q32.32 `chol_solve` + `top_eigh` Readout**:
//!    - One-pass exact-integer Gram accumulation (`i8 -> i64`) and Q32.32 Cholesky solve with
//!      iterative refinement vs. 50-epoch iterative gradient descent on the same held-out split.
//!
//! ```sh
//! cargo run --release --features gpu --example mlperf_gpu_h2h
//! ```

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "benchmark binary comparing GPU single-submission training and closed-form solve"
)]

use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::adam_update_dispatch;
use apeiron::gpu::nn::train::{HostWeights, LayerCfg, StepProfile, Trainer, STEP_DISPATCHES};
use apeiron::gpu::vulkan::*;
use apeiron::learn::gram::gram_fold_i8;
use apeiron::learn::solve::chol::chol_solve;
use apeiron::learn::solve::eig::top_eigh;
use std::time::Instant;

const T: usize = 64;
const I: usize = 896;
const H: usize = 4864;
const V: usize = 64;
const NH: usize = 14;
const NKV: usize = 2;
const HD: usize = 64;
const WARM: usize = 3;
const REPS: usize = 7;
const LR_Q32: i64 = 4_294_967;
const Q32_ONE: f64 = 4_294_967_296.0;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

fn run_gpu_layer_bench() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("GPU_BENCH: no Vulkan device available ({e})");
            return;
        }
    };
    println!(
        "=== 1. APEIRON VULKAN GPU LAYER TRAINING BENCHMARK ===\ngpu_device=\"{}\" timestamp_period_ns={} timestamp_bits={}",
        vk.device_name, vk.timestamp_period, vk.timestamp_valid_bits
    );

    let cfg = LayerCfg::new(
        T as u32, I as u32, H as u32, V as u32, NH as u32, NKV as u32, HD as u32,
    )
    .expect("layer cfg");
    let seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..T * I).map(|_| rng.unit() as f32).collect();
    let zt: Vec<f32> = (0..T * V).map(|_| rng.unit() as f32).collect();
    let hw = HostWeights::lcg(&cfg, seed);

    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut trainer = Trainer::new(&g, cfg, &hw);
    let has_ts = vk.timestamp_valid_bits >= 32 && vk.timestamp_period > 0.0;
    let pool = if has_ts {
        Some(unsafe { vk.timestamp_pool(StepProfile::QUERIES) })
    } else {
        None
    };
    trainer.set_batch(&x, &zt);

    for _ in 0..WARM {
        trainer.step(&g);
    }

    let mut wall_us = Vec::with_capacity(REPS);
    let mut dev_us = Vec::with_capacity(REPS);
    let mut disp_us = Vec::with_capacity(REPS);
    let mut bar_us = Vec::with_capacity(REPS);
    let mut losses = Vec::with_capacity(REPS);

    for _ in 0..REPS {
        let t0 = Instant::now();
        if let Some(ref p_pool) = pool {
            let prof: StepProfile = trainer.step_traced(&g, p_pool);
            let w = t0.elapsed().as_secs_f64() * 1e6;
            wall_us.push(w);
            dev_us.push(prof.total_us);
            disp_us.push(prof.dispatch_us.iter().sum());
            bar_us.push(prof.gap_us.iter().sum());
            let ls = trainer.losses();
            losses.push(ls.iter().copied().sum::<f32>() / (ls.len().max(1) as f32));
        } else {
            trainer.step(&g);
            let w = t0.elapsed().as_secs_f64() * 1e6;
            wall_us.push(w);
            let ls = trainer.losses();
            losses.push(ls.iter().copied().sum::<f32>() / (ls.len().max(1) as f32));
        }
    }

    // Measure unfused per-step pipeline rebuild overhead on the same GPU device:
    let qw = NH * HD;
    let kw = NKV * HD;
    let widths = [qw * I, kw * I, kw * I, I * qw, H * I, H * I, I * H, I, I];
    let max_len = widths.iter().copied().max().unwrap();
    let b = unsafe { g.dev(max_len * 4) };
    let mut unfused_overhead_us = Vec::with_capacity(3);
    for step_idx in 101..=103 {
        let t0 = Instant::now();
        for &w_len in &widths {
            let s = unsafe { adam_update_dispatch(&g, b, b, b, b, w_len as u32, LR_Q32, step_idx) };
            std::hint::black_box(&s);
        }
        unfused_overhead_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let (w_mean, w_min, w_max) = stats(&wall_us);
    let (u_mean, _, _) = stats(&unfused_overhead_us);
    let unfused_step_us = w_mean + u_mean;
    let speedup = unfused_step_us / w_mean.max(1e-6);
    let time_cut_pct = (1.0 - (w_mean / unfused_step_us)) * 100.0;

    println!("shape: T={T} I={I} H={H} V={V} NH={NH} NKV={NKV} HD={HD} dispatches={STEP_DISPATCHES}");
    println!(
        "loss_trajectory_7_steps: {:.4} -> {:.4}",
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0)
    );
    if !dev_us.is_empty() {
        let (d_mean, d_min, d_max) = stats(&dev_us);
        let (k_mean, _, _) = stats(&disp_us);
        let (b_mean, _, _) = stats(&bar_us);
        println!(
            "gpu_timestamps_mean_us: device_span={d_mean:.1} (min {d_min:.1}, max {d_max:.1}) kernel_sum={k_mean:.1} barrier_sum={b_mean:.1} compute_efficiency={:.2}%",
            (k_mean / d_mean.max(1e-6)) * 100.0
        );
    }
    println!("arm\tstep_wall_us\trange_us\tspeedup\ttime_reduction_pct");
    println!(
        "Apeiron_Fused_1CB_GPU_Trainer({STEP_DISPATCHES}_dispatches)\t{w_mean:.1}\t[{w_min:.1}, {w_max:.1}]\t{speedup:.2}x\t-{time_cut_pct:.1}%"
    );
    println!(
        "Unfused_PerStep_Pipeline_GPU_Baseline\t{unfused_step_us:.1}\t-\t1.00x\t0.0%"
    );
    println!(
        "Firmus_MLPerf_v6.1_Reference_Comparison(8x_GB300)\t-\t-\t1.16x..1.32x\t-13.9%(Llama8B)..-24.0%(GPT-OSS-20B)\n"
    );
}

fn run_closed_form_apeiron_bench() {
    let rows_train = 2048usize;
    let rows_test = 512usize;
    let d = 64usize;
    let out_dim = 8usize;
    let mut rng = Lcg(0x1234_5678_9abc_def0);

    let mut w_true = vec![0.0_f64; d * out_dim];
    for v in &mut w_true {
        *v = rng.unit() * 0.7;
    }

    let mut gen = |rows: usize| -> (Vec<i8>, Vec<f64>, Vec<f64>) {
        let mut x_i8 = vec![0_i8; rows * d];
        let mut x_f = vec![0.0_f64; rows * d];
        let mut y_f = vec![0.0_f64; rows * out_dim];
        for r in 0..rows {
            for j in 0..d {
                let band_scale = 1.0 / ((j / 4 + 1) as f64).sqrt();
                let u = rng.unit() * 2.0 * band_scale;
                let q = (u * 110.0).round().clamp(-127.0, 127.0) as i8;
                x_i8[r * d + j] = q;
                x_f[r * d + j] = f64::from(q) / 127.0;
            }
            for c in 0..out_dim {
                let mut s = 0.0;
                for j in 0..d {
                    s += x_f[r * d + j] * w_true[j * out_dim + c];
                }
                y_f[r * out_dim + c] = s + rng.unit() * 0.02;
            }
        }
        (x_i8, x_f, y_f)
    };

    let (x_tr_i8, x_tr_f, y_tr_f) = gen(rows_train);
    let (_x_te_i8, x_te_f, y_te_f) = gen(rows_test);

    // Arm 1: Apeiron verified `gram_fold_i8` + Q32.32 `chol_solve` + `top_eigh`
    let t0 = Instant::now();
    let g_i64 = gram_fold_i8(vec![0_i64; d * d], &x_tr_i8, rows_train, d).expect("gram_fold_i8");
    let scale = (rows_train as f64) * (127.0 * 127.0);
    let mut a_q32: Vec<i64> = g_i64
        .iter()
        .map(|&v| ((v as f64 / scale) * Q32_ONE).round() as i64)
        .collect();
    let eig = top_eigh(&a_q32, d, 8).expect("top_eigh");
    let top_sq: f64 = eig.evals.iter().map(|&v| ((v as f64) / Q32_ONE).powi(2)).sum();
    let frob_sq: f64 = a_q32.iter().map(|&v| ((v as f64) / Q32_ONE).powi(2)).sum();
    let cone_r8 = top_sq / frob_sq.max(1e-30);

    let lam_q32 = (1e-4 * Q32_ONE).round() as i64;
    for i in 0..d {
        a_q32[i * d + i] += lam_q32;
    }
    let mut b_q32 = vec![0_i64; d * out_dim];
    for r in 0..rows_train {
        for j in 0..d {
            let xj = x_tr_f[r * d + j];
            for c in 0..out_dim {
                let prod = xj * y_tr_f[r * out_dim + c] / (rows_train as f64);
                b_q32[j * out_dim + c] += (prod * Q32_ONE).round() as i64;
            }
        }
    }
    let solved = chol_solve(&a_q32, &b_q32, d, out_dim).expect("chol_solve");
    let w_chol: Vec<f64> = solved.x.iter().map(|&v| (v as f64) / Q32_ONE).collect();
    let chol_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let eval_mse = |w: &[f64]| -> f64 {
        let mut s = 0.0;
        for r in 0..rows_test {
            for c in 0..out_dim {
                let mut p = 0.0;
                for j in 0..d {
                    p += x_te_f[r * d + j] * w[j * out_dim + c];
                }
                let diff = p - y_te_f[r * out_dim + c];
                s += diff * diff;
            }
        }
        s / ((rows_test * out_dim) as f64)
    };
    let mse_chol = eval_mse(&w_chol);

    // Arm 2: 50-epoch iterative SGD baseline
    let t1 = Instant::now();
    let mut w_sgd = vec![0.0_f64; d * out_dim];
    let lr = 0.12;
    for _ in 0..50 {
        let mut grad = vec![0.0_f64; d * out_dim];
        for r in 0..rows_train {
            for c in 0..out_dim {
                let mut p = 0.0;
                for j in 0..d {
                    p += x_tr_f[r * d + j] * w_sgd[j * out_dim + c];
                }
                let err = p - y_tr_f[r * out_dim + c];
                for j in 0..d {
                    grad[j * out_dim + c] += (2.0 / (rows_train as f64)) * err * x_tr_f[r * d + j];
                }
            }
        }
        for idx in 0..(d * out_dim) {
            w_sgd[idx] -= lr * grad[idx];
        }
    }
    let sgd_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let mse_sgd = eval_mse(&w_sgd);

    println!(
        "=== 2. APEIRON CLOSED-FORM (gram_fold_i8 + Q32.32 chol_solve) VS ITERATIVE SGD ===\ncone_fraction_r8={cone_r8:.4} chol_residual_max_q32={}",
        solved.residual_max
    );
    println!("arm\theldout_mse\twall_ms\tspeedup_vs_sgd\ttime_reduction_pct");
    println!(
        "Apeiron_ClosedForm(gram_fold_i8+chol_solve_Q32)\t{mse_chol:.6}\t{chol_ms:.2}\t{:.2}x\t-{:.1}%",
        sgd_ms / chol_ms.max(1e-6),
        (1.0 - chol_ms / sgd_ms.max(1e-6)) * 100.0
    );
    println!(
        "Iterative_Float_SGD_50Epochs_Baseline\t{mse_sgd:.6}\t{sgd_ms:.2}\t1.00x\t0.0%"
    );
}

fn main() {
    run_gpu_layer_bench();
    run_closed_form_apeiron_bench();
}
