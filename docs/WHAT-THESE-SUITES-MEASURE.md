# What these suites actually measure

**Read this before quoting any row of any atlas example.** A row that names an
MLCommons workload is not automatically a measurement of that workload. This
file records, per suite, **what data the row was measured on** — with the
`file:line` that establishes it.

**The rule, in one line:** a table row must name its data source, and a row
measured on data generated inside the process is a benchmark for the named
**operator**, not a result for the named **workload**. Every table in
`examples/` now prints that source as a `# data_source=…` line immediately
above it (machine-greppable), and this file is the long form.

---

## 1. `examples/mlcommons_universal_atlas.rs` — **no dataset is read**

The binary's only file read in the whole program is the hostname:

```rust
// examples/mlcommons_universal_atlas.rs:1136
let host = std::fs::read_to_string("/etc/hostname")
```

There is no `File::open` and no other `fs::read` anywhere in the file. So
**every one of the twelve suites and Part II measures data generated inside
the process** — `Lcg` draws, a planted `w_true`, synthetic clusters.

| suite | function | data | what the number means |
|---|---|---|---|
| 1 AILuminate | `run_suite_1_ailuminate` (`:113`) | synthetic: 12 hazard classes as random cluster centres plus noise; the "prompts" are int8 vectors drawn around those centres (`gen_prompts`) | how well a ridge solve separates **generated** clusters — a gate benchmark, not AILuminate |
| 2 AlgoPerf | `run_suite_2_algoperf` (`:234`) | synthetic: planted `w_true`, `y = x·w_true + noise` | how closely `chol_solve` recovers a planted vector — not an AlgoPerf rule set |
| 3–9 inference spectrum | `run_suites_3_to_9_inference_spectrum` (`:376`) | synthetic: the "token generation" is an int8 matvec over **constant arrays** — `w_mla = vec![2i8; d_llm * r_mla]` (`:447`), `x_llm = vec![4i8; d_llm]` (`:448`) | a **real timing** of that loop on this hardware, with no model and no tokens behind it; not an LLM decode rate |
| 9 Tiny | same fn | synthetic: 4 Gaussian classes in 16 dimensions (`d_tiny = 16`, `:384`) | the accuracy and the 2,112 B footprint are real **for that generated problem**; not KWS/VWW/AD/IC |
| 10 Storage | `run_suite_10_storage` (`:512`) | synthetic streaming | the streaming/point-get timing is real for the harness used |
| 11 MLPerf Training v6.1 | `run_suite_11_mlperf_training_v6_1_new` (`:592`) | synthetic: planted weights; the DLRM events come from `gen_yambda_event` (random ids with a hand-written affinity rule) | not the DLRMv4-HSTU / Yambda-5B / FLUX.1 workloads |
| 12 MLPerf Training HPC | `run_suite_12_hpc` (`:849`) | synthetic | not CosmoFlow / DeepCAM / OpenCatalyst |
| Part II frontier lifecycle | `run_frontier_full_lifecycle_pipeline` (`:915`) | synthetic | not GLM-5.3 / DeepSeek-Flash-V4 / Qwen3.5-A17B |

**What survives:** every wall-clock, memory and energy number in this binary is
a real measurement **of the code that ran on the machine it ran on**. What does
not survive is the attribution: a row labelled `Suite_9_MLPerf_Inference_Tiny`
is not a KWS/VWW/AD/IC result, and a row labelled `Suite_6_MLPerf_Endpoints`
("365032× faster than a Python LLM endpoint") is a comparison between an
int8 matvec over constant arrays and a hard-coded baseline latency.

## 2. `examples/mlperf_gpu_h2h.rs` — one real arm, seven synthetic

| section | function | data |
|---|---|---|
| MLPerf v6.1 C4 language modelling | `run_mlperf_c4_workload` (`:898`) | **real** — `allenai/c4`: the token bins (`:90`), the pre-tokenised stream (`:230`) and the raw documents `c4_docs.bin` (`:268`) |
| GPU layer step | `run_gpu_layer_bench` (`:1170`) | synthetic (`Lcg`) |
| SSM / Mamba-2 | `run_ssm_mamba_bench` (`:1309`) | synthetic (`Lcg`, `gen_ssm_stream`) |
| DLRM-DCNv2 | `run_dlrm_dcnv2_bench` (`:1469`) | synthetic (`Lcg`, `gen_row`) |
| R-GAT GNN | `run_rgat_gnn_bench` (`:1594`) | synthetic (`Lcg`, planted `w_true`) |
| LoRA vs anchored ridge fold | `run_lora_vs_ridgefold_bench` (`:1765`) | synthetic (`Lcg`, `gen_split`) |
| monoid SGD round trip | `run_monoid_sgd_roundtrip_bench` (`:1929`) | synthetic (`Lcg`, `gen_data`) |
| power / energy | `run_power_resource_telemetry` (`:2200`) | hardware telemetry (`/proc/self/status`, GPU power env) |

The C4 arm is the one that matters for language modelling, and the one a
compression claim must be built on.

## 3. The other examples

- `examples/mlperf_a17b_fleet.rs` — **real**: pre-tokenised C4 (`:74`, `:450`).
- `examples/altar1_tasmania_showcase.rs` — **two real inputs**: the altar-1 KLD
  windows `kld_504B_w4a16.json` (`:51`) and the expert plan
  `plan_massmax192.json` (`:63`). But the per-layer activations it heals are
  generated, and the "healed KLD" rows are **computed by a formula with fixed
  multipliers** (`:214-217`, `× 4.2`), not measured. It also calls the live
  `peregrine-ng` FoldDecider over HTTP (`:367-370`).
- `examples/harness_adapter_model.rs` — **real**: fitted on the harness's own
  session traffic (`--captures <dir>`), i.e. real contexts and real teacher
  continuations, with the closed-form residual ridge and the rank-`r` projection.

## 4. Why this file exists

Because the pattern is right and the data was wrong, and the difference is
invisible in a results table: a solver benchmark and a workload result look
identical once they are printed as `suite | metric | measured | baseline |
advantage`. The consequence was a testing process that could report a
"365032× faster" advantage over a workload it never ran.

So, for every new table and every new suite:

1. print the data source with the table (`# data_source=…`);
2. if the data is generated in-process, say so in the row's own text — not only
   in a footnote;
3. a workload name is a **claim about the data**, and the data source is the
   evidence for it.

The atlas pattern itself (`docs/ATLAS-PATTERN.md`) is unaffected: the fold,
the spectrum, the continuous solve and the baseline discipline are all real
here — the *workloads* are what this file qualifies.
