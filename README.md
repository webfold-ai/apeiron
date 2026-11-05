# apeiron

A Rust library for the algebra `R[x]/(x² − σ)`, a growable hash table built on it, and an inference
engine for GGUF models that runs on the CPU and, through Vulkan, on the GPU. The algebra and the
word-level arithmetic are specified and checked in Lean 4.

- **`Cell<R>`**: a pair `(a, b)` over a ring `R`, with multiplication `(a₁a₂ + σb₁b₂, a₁b₂ + b₁a₂)`
  and norm form `a² − σb²`. `σ = −1, 0, 1` gives the elliptic, parabolic and hyperbolic faces.
- **`freq::Table`, `CoordMap`, `FreqMap`**: flat tables with a spill region. Lookup is a fixed number
  of probes; allocation failure and size overflow are refused and change nothing.
- **`engine`**: GGUF loading, Q32.32 fixed-point kernels, AVX2 with a portable fallback, and a
  `gpu` feature that loads Vulkan at run time.
- **`proof/`**: Lean 4 extraction of the Rust word bodies with Aeneas and Charon, and the
  refinement theorems that link them to the canonical definitions in `proof/canonical/`.

The default build has no dependencies; Linux adds `libc` for thread affinity. Rust 1.89 or later.

## Use

```sh
git clone https://github.com/webfold-ai/apeiron
cd apeiron
cargo build --lib
```

```rust
use apeiron::{Cell, Sigma};

let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);
let p = x.mul(Sigma::Hyperbolic, y);
assert_eq!((p.a, p.b), (3 + 2 * 5, 3 * 5 + 2 * 1));
```

```rust
use apeiron::freq::Table;

let mut counts = Table::<u64, i64>::new();
for word in [7u64, 9, 7, 7, 9, 4] {
    counts.insert_with(word, 1, |old, new| old + new);
}
assert_eq!(counts.get(&7), Some(&3));
assert_eq!(counts.len(), 3);
```

The [book](book/src/introduction.md) covers installation, the algebra, the tables, running a model
on the CPU and the GPU, and the verification. Build it with `mdbook build book`; its examples run
under `mdbook test book -L target/debug/deps`.

## Inference engine against llama.cpp

Current figures. `pp512` is 512 prompt tokens per second, `tg128` is 128 generated tokens per
second. Ratios are apeiron over llama.cpp. The commit is the apeiron revision the figure was
recorded at; the figures are copied from those recorded measurements, not re-run for this
release.

| metric | model / device | apeiron | llama.cpp | ratio | commit | receipt |
|---|---|---:|---:|---:|---|---|
| pp512 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 113,099 tok/s | 63,018.5 tok/s (CUDA) | **1.795×** | `5e476afb` | quotient of recorded medians |
| tg128 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 973.8 tok/s | 880.9 tok/s (CUDA) | **1.105×** | `ecd1fcf1` | quotient of recorded medians |
| pp512 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 917.82 tok/s | 416.41 tok/s (CPU) | **2.24×** (2.14–2.25) | `64cd46dd` | [`1792648053-qwen08b-q8-cpu`](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| tg128 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 92.13 tok/s | 51.42 tok/s (CPU) | **1.80×** (1.70–1.81) | `64cd46dd` | [`1792648053-qwen08b-q8-cpu`](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| pp512 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 658.10 tok/s | 632.67 tok/s (CPU) | 1.02× (1.00–1.10) | `5064c500` | [`1792653496-qwen08b-q4km-kqc-live`](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |
| tg128 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 84.19 tok/s | 60.39 tok/s (CPU) | **1.39×** (1.33–1.40) | `5064c500` | [`1792653496-qwen08b-q4km-kqc-live`](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |

- CPU rows: llama.cpp `60bc8866b` at its best of 16 and 32 threads, median of 5 paired rounds, range
  is the minimum to maximum round, every round passed the quiet-host gate (busy cores < 1).
- GPU rows: the CUDA comparator is `llama-bench -ngl 99`. The prefill comparator was measured on a
  contended host, and the ratios are quotients of recorded medians, not paired-round medians.
- The Q4_K_M prefill row is parity (1.02×); decode is ahead. First-token agreement has not been run
  on the Q4_K_M files.
- Q4_K_M receipts were recorded from a working tree with uncommitted paths; the receipt names the
  source sha256.

### Earlier CPU receipts

More models and prompt lengths, Q8_0, measured earlier at revision `644190d7771f` against llama.cpp
`0324696b8` at its best of 8, 16, 24 and 32 threads. `tg` here is 64 generated tokens. Every round
passed the quiet-host gate (busy cores < 0.05).

| weights | pp tokens | rounds | pp ratio | tg ratio | receipt |
|---|---:|---:|---|---|---|
| Qwen3.5-0.8B | 200 | 5 | 2.17 (2.11–2.28) | 1.76 (1.52–1.77) | [`1790795647-w6e-vs-llama-r3`](bench/cpu/1790795647-w6e-vs-llama-r3.md) |
| Qwen3.5-0.8B | 512 | 5 | 2.45 (2.30–2.79) | 1.68 (1.55–1.79) | [`1790798160-w7-q35r2-pp512`](bench/cpu/1790798160-w7-q35r2-pp512.md) |
| qwen05b | 200 | 5 | 1.57 (1.56–1.64) | 1.47 (1.35–1.54) | [`1790798706-w7-qwen05b-r2-pp200`](bench/cpu/1790798706-w7-qwen05b-r2-pp200.md) |
| qwen05b | 512 | 5 | 1.56 (1.37–1.58) | 1.55 (1.45–1.59) | [`1790797618-w7-qwen05b-pp512`](bench/cpu/1790797618-w7-qwen05b-pp512.md) |
| qwen15b | 200 | 5 | 1.77 (1.74–2.00) | 1.10 (1.08–1.14) | [`1790798386-w7-qwen15b-r5-pp200`](bench/cpu/1790798386-w7-qwen15b-r5-pp200.md) |
| qwen15b | 512 | 3 | 1.91 (1.89–1.96) | 1.19 (1.06–1.21) | [`1790797926-w7-qwen15b-pp512`](bench/cpu/1790797926-w7-qwen15b-pp512.md) |

All six were measured at revision `644190d7771f`, source sha256 `0218e7d911a3…`, with the default `f32`
carrier. Each receipt holds the model and prompt digests, the machine, the compiler and the
per-phase trace. Reproduce one with:

```sh
cargo run --release --locked --example engine_profile -- \
  --model <Qwen3.5-0.8B-Q8_0.gguf> --ids tests/fixtures/qwen35-prompt2.ids \
  --pp 200 --tg 64 --reps 5 \
  --llama <path/to/llama-bench> --llama-threads 8,16,24,32 --max-busy 0.05 --label repro --out <dir>
```

All rows are one machine (the i9-14900KF above); `measured` here means that host and no other. The
GGUF files are not distributed; each receipt names the model by sha256. Only the first row's prompt
(`tests/fixtures/qwen35-prompt2.ids`, sha256 `894a166a…`) is in the repository. The other five
name their prompt ids by sha256 in the receipt and are not reproducible from a clone.

Throughput is accepted only alongside agreement with the comparator on the same model (first
tokens and perplexity); the book's [Results](book/src/engine/results.md) chapter gives the protocol
and the GPU commands (`gpu_decode_bench`, `gpu_ppl`). The hash table comparison against
`hashbrown` is not in these tables: its ratios are not claimed here.

## Quantisation: the open gap

The goal is a 27B-class model in at most 4 GB of resident weights (about 1.18 bits per weight) that
stays within 10% of Q4_K_M: perplexity at most 1.15× and zero-shot task accuracy at least 0.90× of
Q4_K_M's, from the same run. That bar is not met yet.

| metric | model / setting | apeiron | Q4_K_M | ratio | bar |
|---|---|---:|---:|---:|---:|
| wikitext-2 perplexity | Qwen3-0.6B-Base, 1.18 b/w, 1024-step KD | 24.5751 | 13.7633 | 1.785× | 1.15× |
| mean 0-shot accuracy (boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge) | same | 0.4194 | 0.5556 | 0.755× | 0.90× |
| relative Hessian distortion, all 28 blocks (unreproduced, see below) | same, Fold E8 lattice codebook vs NanoQuant's sign-rank factorisation | 0.0149 | 0.0805 (NanoQuant) | 5.39× lower | n/a |

- NanoQuant as shipped, on the same model and harness: perplexity 24.3317 (1.768×), accuracy 0.4235
  (0.762×). At equal bits the port is 1.010× the reference's perplexity and 0.99× its accuracy, with
  5.39× lower layer distortion (unreproduced, below); both are short of the bar.
- The distortion row is a figure recorded on another machine. Its receipt, 112 per-linear fits
  (`q3-gram/l{0..27}.{down,o,up,gate}-nq-fold.i64`), is not in this repository, and the entry point
  it names, `learn_nanoquant::fold_fit_all`, is not in the source. The E8 codebook and its fit are
  here (`src/learn/code/e8.rs`, `src/learn/fold/fit/gptq_e8.rs`); the 0.0149 and 0.0805 are not
  re-derivable from a clone and carry no grade.
- Qwen3.8-27B (48 DeltaNet and 16 attention layers) runs end to end at 1.18 b/w with 3.753 GiB of
  resident weights, inside the 4 GB budget: prompt processing 586.4 tok/s and generation
  4.85–5.90 tok/s on an RTX 5080 with 38 layers offloaded plus CPU, prefix-cache hit 98.3%. It has
  no quality measurement yet.
- Figures are copied from the recorded measurements, not re-run for this release.

## What is proved, tested and open

| statement | grade | gate |
|---|---|---|
| `Cell<i64>` multiply and norm form refine the canonical ring operations for every `σ` and input | proved | `word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean` |
| the Lean axiom cone of every headline is within `propext`, `Classical.choice`, `Quot.sound` | proved | `proof/corpus/AxiomAudit.lean`, run by `lake build` |
| `Cell`, the tables, the engine and the SPIR-V fragment checker behave as documented on their boundary cases | tested | `cargo test --workspace --all-features --locked` |
| the SPIR-V fragment header check | proved | `Spirv.checkFragment_header` |
| the HashDoS bound on the tables | tested | `cargo test --test hashdos --locked` |
| speed against llama.cpp | measured | `cargo run --release --locked --example engine_profile`, receipts in `bench/cpu/` |
| the executed `freq::Table` body refines the canonical `Apeiron.Table` | open | |
| the whole SPIR-V fragment checker | open | |

Two extraction facts bound the proofs: the Lean files are generated from the Rust source by Aeneas
and Charon at pinned revisions (`proof/README.md`), and `proof/regen.sh --check` fails if the
committed extraction differs from a fresh one; `cd proof/corpus && lake build` checks the theorems.

## What this does not claim

- **Cryptography is unaudited.** `aes`, `chacha`, `sha256` and `crypto` are tested against
  known-answer vectors only and are not side-channel resistant (`aes` indexes its S-box by secret
  bytes). Do not use them to protect real data.
- **`unsafe` is mostly unproved.** Count it with
  `rg -n '\bunsafe\b' src --glob '*.rs' | rg -v ':\s*(//|\*)' | wc -l`; `SAFETY:` comments are
  counted with `rg SAFETY src --glob '*.rs' | wc -l`. Only the `src/table` window model is under a proof.
- **The GPU path is single-tenant.** Vulkan buffers run without `robustBufferAccess` and workgroup
  memory is not cleared between dispatches.
- **No claim below the extraction.** rustc and LLVM code generation, the kernel ABI, Vulkan drivers
  and GPU hardware are outside the evidence.
- **`i64` is not the real trichotomy.** `Z/2⁶⁴` carries the structural half of the algebra
  (commutativity, associativity, multiplicative norm) and not the classification; the book's
  [scalars](book/src/theory/scalars.md) chapter states where it stops.
- **Callbacks.** The tables assume `Eq`, `Clone` and merge callbacks that are total and stable; a
  callback that allocates, panics or diverges is outside the statements.

Security reports: [SECURITY.md](SECURITY.md). Licence: Apache-2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)).
