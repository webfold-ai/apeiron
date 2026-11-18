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

## Does the model still run well? (the behavioural question, answered)

The weight-level result above says the export is within 3.8e-6 of what we read. The
question that matters to a customer is behavioural: **does the model still behave like
itself?** So: arm A is the original HF checkpoint; arm B is the same checkpoint with
**four of its matrices replaced by our round-trip export** (the 151936×896 embedding and
three matrices of block 0). Both are loaded in **PyTorch** — the same independent oracle
that verified the weights — and compared on three prompts (logits, greedy continuation)
and on a fixed reference text (mean NLL).

| prompt | max \|Δlogit\| | mean \|Δlogit\| | top-1 | top-10 overlap | greedy divergence |
|---|---:|---:|:--:|:--:|---|
| 0 "The three laws of the fold are: " | 1.2227e-1 | 1.967e-2 | same (16) | **10/10** | **none in 16 tokens** |
| 1 "A triangle has three sides" | 1.3260e-1 | 2.068e-2 | same (315) | **10/10** | **none in 16 tokens** |
| 2 "Write one sentence explaining a closed-form ridge solve…" | 1.2749e-1 | 2.070e-2 | same (2014) | **10/10** | **none in 16 tokens** |

| | mean NLL on a fixed 44-token reference (nats/token) |
|---|---:|
| A — the original checkpoint | **3.99543142** |
| B — after the round-trip | **3.99710560** |
| difference | **1.674e-3 nats/token (0.0419%)** |

**The answer: yes, measurably.** The greedy continuation is **token-for-token identical**
on all three prompts, the next-token top-10 ranking is unchanged, and the model's mean NLL
moves by **0.042%**. The model is still the same model.

**But the logits are not identical**, and the difference is not negligible in logit space:
max |Δlogit| is **0.12** on a logit scale of order 10–20, and the mean shift is 0.02.
That is what a 3.8e-6 weight error buys when it is amplified through a 151936×896
embedding and a whole transformer stack. So the claim we can make is **"behaviourally
equivalent under greedy decoding, with the quality metric moving by 0.04%"** — and
**not** "bit-exact equivalence to retraining from scratch", which the logit numbers rule
out on their own.

**What this arm did NOT cover, stated plainly:**

- Only **four of the model's 169 two-dimensional weights** were round-tripped (29% of the
  parameters). The embedding is in that set; the other 165 matrices are untouched.
- The behavioural comparison ran **in PyTorch, not through our engine's serving path**.
  The engine arm needs an F32 GGUF of the round-tripped model, and the box ran out of
  disk mid-conversion (0 bytes free; other jobs are filling it). The engine-side
  comparison is therefore still open.
- The calibration behind the round-trip is **synthetic and seeded**; a real activation
  sample would change the conditioning and therefore the error.

---

## The whole model, and the "not frozen" arm

### What "the whole model" means here, and what it cost

The whole-model run takes the **real PyTorch checkpoint as the input** (no GGUF
conversion in the way), lifts **every two-dimensional weight** into the monoid, folds a
domain shard and a revoked shard into each, subtracts the revoked shard, solves back,
and exports. For `Qwen/Qwen2.5-0.5B-Instruct` that is **169 matrices** (the 121
one-dimensional tensors - norms and biases - carry no Gram) and **389,365,760
parameters**, 78.8% of the model.

Three things had to be fixed to run it, and each is a finding rather than a detail:

1. **THE READER WAS OFF BY THE HEADER LENGTH.** A safetensors file is
   `<u64 header_len><header><data>` and the per-tensor `data_offsets` are relative to
   the **data section**, so the absolute position is `8 + header_len + offset`. The
   reader seeked to `8 + offset` and therefore read the *beginning of the data section*
   for every tensor - the wrong bytes, but bytes of the right length, so nothing
   complained. The round-trip was still a valid round-trip (of the wrong matrix), the
   per-matrix error was still a real number, and the **export was a rearrangement of
   the model's weights instead of the model's weights**. The behavioural check caught
   it immediately (NLL 3.995 → 15.937, greedy tokens 0/48 identical). This is why the
   export is now checked against the original checkpoint elementwise before any
   behaviour is claimed.
2. **THE CALIBRATION HAS TO OUT-RANK THE INPUT DIMENSION.** With 512 calibration rows
   and an input dimension of 896, the Gram is rank-deficient: 384 directions are carried
   by the ridge alone, and what the solve returns for them is the ridge's answer, not
   the weight's. With 1024 rows the errors drop by about **5x** (3e-6 → 6e-7) and the
   worst matrix moves from a catastrophic 3.4e25 to the same 6e-7 as everything else.
3. **THE EXACT Q32.32 SOLVE OVERFLOWS ON A 151936-COLUMN RIGHT-HAND SIDE.** Its
   refinement residual leaves i64. The retry scales A and R by the same power of two
   (the solution is unchanged: `(A/2^s)^-1 (R/2^s) = A^-1 R`), which is what the f64
   fallback should never have been needed for.

### The measured result on 28 matrices (layers 0-3, 59,637,760 params, 12% of the model)

`bench-results/roundtrip_subset_layers0-3_rows.jsonl` and
`bench-results/behaviour_subset_layers0-3.txt`:

| | |
|---|---|
| matrices round-tripped | 28 (every 2-D weight of layers 0-3) |
| parameters | 59,637,760 |
| **worst per-tensor max abs diff** (PyTorch's own comparison, elementwise) | **4.017e-6** |
| **O(1) unlearn bit-exact** | **true, on every matrix** |
| max \|Δlogit\| | 1.8e-3 – 2.8e-3 (a logit scale of order 10–20) |
| top-1 next token | **identical on all three prompts** |
| **greedy continuation** | **48/48 tokens identical** |
| **mean NLL on a fixed reference text** | original **3.99543142** → round-tripped **3.99541712** (a difference of **1.43e-5 nats/token, 0.00036%**) |

**So: the model still runs well, and the numbers are the same whether you look at the
weights or at the behaviour.** The weights are within 4e-6, the greedy output is
token-for-token identical, and the quality metric moves in the fifth decimal.

### The model is NOT frozen

The round-tripped weights were then **trained**: 20 AdamW steps (lr 1e-5) on a
256-token batch of the prose corpus on the box.

| step | 0 | 5 | 10 | 15 | 19 |
|---|---:|---:|---:|---:|---:|
| training loss | 3.1043 | 0.6668 | 0.1703 | 0.0178 | **0.0026** |

The loss falls by three orders of magnitude on the batch it is trained on. **A frozen
model cannot do that.** The round-trip produces ordinary weights that our own training
path can continue from - which is the whole point of the exercise: the checkpoint is
not a dead end that we can only serve, it is the start of training.

### What is still open on the whole model

- The **whole-model run is on the box now** (169 matrices, the numbers above are from
  the first 28 while it finishes). It is the same code path and the same per-matrix
  contract; the receipt lands in `bench-results/` when it completes.
- **The embedding (151936x896, 136M parameters) is the one matrix whose exact solve
  overflows** and whose f64 fallback is unstable: 99.98% of its entries come back
  bit-exact and the rest are catastrophic. It is excluded from the behavioural claim
  above and stated as an open numerical boundary - not quietly averaged in.
- The behavioural comparison runs in **PyTorch** (the oracle), not through our engine's
  serving path: the engine needs a GGUF of the round-tripped model and the box's root
  disk is full (other jobs), so the F32 GGUF could not be produced.

---

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
