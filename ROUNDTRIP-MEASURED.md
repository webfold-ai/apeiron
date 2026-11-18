# The checkpoint round-trip, measured on a real checkpoint

**Operately:** `the-checkpoint-lane-seed-HgXTT3JL7innmb96OiI22g` (checkpoint lane; this is
the round-trip half of it).
**Code:** `examples/checkpoint_roundtrip.rs` (new), `bench-results/checkpoint_roundtrip_qwen2.5-0.5b.json`,
`bench-results/engine_profile_qwen2.5-0.5b.{json,md}`.
**Host:** 10.0.63.10 (Ryzen 7 5700X, 8 cores, 71 GB RAM), cargo 1.96.0, rustc 1.96.0.

## The claim under test

> Bring your PyTorch / HuggingFace checkpoint onto our stack, fold new shards into it,
> unlearn a revoked shard in O(1), and take the result back out as standard dense weights.

That has never been run on a real checkpoint. The bench that exercises it
(`mlperf_gpu_h2h.rs` §8, "ROUND-TRIP INTEROPERABILITY: SGD → MONOID → UNLEARN → SGD",
`webfold-ai/apeiron#13`) uses **synthetic** weights: `d = 64`, `out_dim = 8`, i.e. **512
parameters**, and reports `max_reconstruction_err = 9.30e-9`. This run is the same
algebra on a **real checkpoint's** weights.

## The loop that was run

```
HF safetensors (PyTorch)            Qwen/Qwen2.5-0.5B-Instruct, 988,097,824 B
        │  sha256 fdf756fa7fcbe740…
        ▼  llama.cpp's convert_hf_to_gguf.py (torch 2.13.0+cpu), --outtype q8_0
GGUF q8_0                            531,067,808 B, sha256 1d5e3ec337c3c42c…
        │  apeiron::format::gguf (our reader), dequantised to f32
        ▼  OUR STACK
   lift     (G_prior, R_prior = G_prior·W) in Q32.32, Gram from a seeded calibration
   fold     + a domain shard, + a revoked shard (integer Gram addition)
   unlearn  − the revoked shard (integer Gram subtraction)  → bit-exact
   solve    Q32.32 Cholesky → dense weights
        ▼  exported as safetensors (F32)
PyTorch reads BOTH exports and compares them with the ORIGINAL bf16 checkpoint
```

## The numbers

| tensor | shape | read | lift | solve | fold+unlearn | **round-trip max abs diff** | **q8_0 quantisation diff vs the original bf16** | unlearn bit-exact |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `token_embd.weight` | 151936×896 | 390 ms | 558,379 ms | 284,821 ms | 455 ms | **3.812e-6** | 7.486e-4 | **true** |
| `blk.0.ffn_gate.weight` | 4864×896 | 11 ms | 18,029 ms | 9,642 ms | 451 ms | **3.163e-6** | 1.862e-3 | **true** |
| `blk.0.ffn_up.weight` | 4864×896 | 5 ms | 17,917 ms | 9,510 ms | 440 ms | **3.089e-6** | 1.083e-3 | **true** |
| `blk.0.attn_k.weight` | 128×896 | 0 ms | 658 ms | 346 ms | 438 ms | **2.827e-6** | 3.906e-3 | **true** |

The model has **169 two-dimensional weights, 493,961,216 parameters**; four of them were
round-tripped (144,965,632 parameters, 29% of the model).

### What the numbers say

1. **The round-trip works, and PyTorch is the one saying so.** It read both exports and
   the original checkpoint. The largest disagreement between our exported weights and our
   own dequantised read is **3.8e-6 absolute** — which is **~200× smaller than the q8_0
   quantisation step the container already applied** (7.5e-4 … 3.9e-3). The round-trip's
   own error is therefore *below the quantisation noise floor of the format we used*.
2. **The O(1) unlearn is bit-exact at model scale.** Folding a revoked shard and
   subtracting it leaves the Gram state bit-identical to the state before the fold, on
   every matrix, including the 151936×896 embedding. That is the property the product
   sells, and it holds at real-checkpoint scale — not only at `d = 64`.
3. **The bench's 9.30e-9 does NOT transfer.** The real-checkpoint reconstruction error is
   **3.8e-6**, about **4,000× larger** than the bench figure, and **16,000× the Q32.32
   fixed-point floor** (2.33e-10). The bench number was a property of synthetic weights of
   magnitude ~0.65 with a well-conditioned random prior, not of the method. **The claim in
   the preprint and the business canvas — "max weight reconstruction error ≤ 9.30 × 10⁻⁹,
   bit-exact equivalence to retraining from scratch" — is not a claim we can make about a
   real checkpoint.**
4. **The cost is the honest part.** A 4864×896 matrix costs **18.0 s of lift + 9.6 s of
   solve** on 8 CPU cores; the embedding (151936×896) costs **558 s + 285 s**. The lift
   here is a naive single-threaded i128 triple loop at ~0.2 GFLOP/s — that is *this
   example's* implementation, not the method's arithmetic (3.7 GFLOP for that matrix), so
   a vectorised or parallel lift would be ~100× faster. The **solve** number is the
   library's own `chol_solve` and stands as measured. Per-matrix Gram state is `d²`:
   6.4 MB at d=896, **134 MB at d=4096** (an 8B-class model), which is the real scaling
   constraint, not the arithmetic.
5. **The calibration is synthetic and seeded** (`--calib-rows 512 --seed 7`). This measures
   the *round-trip*, not the quality of a trained model. A real activation sample would
   change the conditioning and therefore the error; that is a separate measurement.

## The inference side, same checkpoint, same host (and it is contaminated)

`examples/engine_profile.rs`, the interleaved harness, on the same box with the same GGUF,
CPU only on both sides, 3 rounds, pp 200 / tg 64:

| metric (tok/s) | ours (apeiron) | llama.cpp (best threads) | ratio |
|---|---:|---:|---:|
| prefill | 32.14 | 421.62 | **0.07** |
| decode | 1.42 | 24.88 | **0.06** |

**This run is flagged and must not be quoted as the answer.** The harness's quiet-host gate
(`busy < 0.05`) **timed out on at least one round** — other jobs were on the box — and our
own trace shows `pool.wait 97.8%` of the decode wall, i.e. we were starved on the task
pool while llama.cpp's plain threads were not. The same model on the same box, on a
**quiet** host earlier the same evening, did **60.52 tok/s** decode in `generate`
(prefill 43.77 tok/s). So the honest statement is: **the comparison is unresolved and
must be re-run on a quiet host**; the number above is an upper bound on the damage
contention does to our path, not our throughput.

Two smaller facts from the same run, both recorded because they will bite the next person:

- **Our engine refuses a stock llama.cpp LFM2.5-8B GGUF**: `missing metadata
  lfm2moe.attention.head_count_kv`. The GGUF stores that key as a **per-layer array**
  (`[0,0,8,0,0,0,8,…]` — 0 on the short-conv layers), and our loader requires a scalar.
  Supporting hybrid attention geometry is a real feature, not a metadata tweak.
- **`llama-cli` segfaults on this box** while `llama-bench` (which the harness drives)
  works: the incumbent's timing comes from `llama-bench`; the text-parity arm is
  unmeasured.

## Negative controls

- **The unlearn arm is the control and it is green:** folding a revoked shard and
  subtracting it must leave the Gram state bit-identical. It is, on all four matrices, and
  the check is in the run (`unlearn_bit_exact`), not asserted afterwards.
- **The error arm is a control too:** the reconstruction error is compared against the
  Q32.32 floor (2.33e-10). This run **failed that comparison** — 3.8e-6 is 16,000× the
  floor — and the report says so rather than reporting the bench's 9.30e-9.
- **PyTorch is the independent verifier:** the comparison is not our arithmetic grading
  itself; a different implementation reads the export and compares it with the original
  checkpoint.

## What is still open

- The **format half** of the checkpoint lane (`.seed` vs safetensors vs ONNX vs gzip:
  bytes, seal time, load time, random access) is not run here. This document is the
  **round-trip** half.
- The inference comparison needs a **quiet host**; the box was shared and busy.
- The round-trip at **8B scale** (d=4096) needs the Gram state and the solve cost
  measured, not extrapolated.
