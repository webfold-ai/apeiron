# The atlas pattern — and the three mistakes it exists to prevent

**Status:** normative for every new workload in this portfolio. Where a document
disagrees with the source, the source wins (apeiron `AGENTS.md`); where a
document disagrees with *this* one, fix this one.

**Why this file exists.** The pattern is not hard, and it has been got wrong
three times, in three different repos, in three different guises — OCR, SAM, and
the decision models. Each time the failure looked like "the model is not good
enough" and the repair that was attempted was another architecture. This file
states the pattern once, points at the code that implements it, and lists the
mistakes so the next agent does not have to rediscover them.

---

## 1. The pattern

Every workload in this portfolio — LLM decode, an optimizer, a graph GNN, an
audio model, a routing decision — is solved with the same four moves. This is
the atlas (`apeiron/examples/mlcommons_universal_atlas.rs:1-24`: twelve official
MLCommons suites plus a six-stage frontier pipeline), and it is not a style: it
is what the winning arms have in common.

### 1.1 Collapse the operator into a stationary sufficient statistic

The workload's operator becomes a **Gram** — a second-order, order-free,
shard-mergeable statistic of the design matrix. Nothing per-token, per-frame or
per-pixel survives; what survives is `G = XᵀX`.

- Primitive: `apeiron::learn::gram::gram_fold_i8` — `src/learn/gram.rs:123`
  (exact `i64` accumulation, no float rounding, order-free).
- Worked: `examples/mlperf_gpu_h2h.rs:1662`
  (`let g_i64 = gram_fold_i8(vec![0i64; d_feat * d_feat], &h2_i8, n_train, d_feat)`).
- The same shape is used by the routing decider in `peregrine-ng`
  (`src/serve/decider.rs`, step 2 of its own header at `:11-14`) and by the
  atlas's own suites.

### 1.2 Measure the spectrum **before** you solve

The spectrum of the *same* `G` is measured with the folded statistic's own
eigen-solver, and it is reported **next to** the result, never in a footnote.

- Primitive: `apeiron::learn::solve::eig::top_eigh` — `src/learn/solve/eig.rs:840`
- Worked: `examples/mlperf_gpu_h2h.rs:1665-1668` — `top_eigh` runs on `a_q32`,
  the very matrix the solve below consumes, and `cone_r8 = Σλ²/‖G‖_F²` is
  computed from it.
- Reported: `examples/mlperf_gpu_h2h.rs:1738-1740` (the shape line ends with
  `cone_fraction_r8=…`), `examples/mlcommons_universal_atlas.rs:211` and `:358`
  (`diagnostics: … cone_fraction_r8=…`), `:884` (same, for the HPC suite).
- `cone_fraction` is the *spectrum concentration* of the lift: `1.0` means the
  statistic lives in a rank-`r` subspace; a flat spectrum means the lift has
  nothing to read, and any solve on it is reading noise. `REG-3`.

**The order is the rule, not a threshold.** The atlas computes the spectrum
*before* the solve and prints it *with* the number. It does **not** refuse to
solve on a low cone: there is no floor anywhere in the examples — the only
`cone` comparison in `examples/` is the helper's own definition
(`examples/mlcommons_universal_atlas.rs:71`, `fn cone_r8_from_gram`). So **a new
workload must state its own floor and report against it** — the floors quoted
for these workloads (≈0.70–0.99, per workload) are the owner's stated bars, not
constants in the code — and the decider's own test uses `> 0.5`
(`peregrine-ng/src/serve/decider.rs:1120`). If your workload's cone is flat,
the fix is in the **lift**, not in the solver.

### 1.3 Solve directly in the continuous target basis

One SPD Cholesky, in fixed point, over the folded Gram. Never a lossy discrete
bottleneck on the way in, never a fitted second head on the way out.

- Primitive: `apeiron::learn::solve::chol::chol_solve` — `src/learn/solve/chol.rs:839`
  (Q32.32, same solver for one column or many).
- Worked: `examples/mlperf_gpu_h2h.rs:1682`; the routing decider solves
  `k = 2` columns in one call (`peregrine-ng/src/serve/decider.rs:372-378`).
- The atlas's own discipline, in one line: *"Solve directly in the continuous
  target basis (chol_q32 / Anchored RidgeFold + Wanda null-space pruning),
  never forcing a lossy discrete bottleneck."*

### 1.4 Report three columns against a baseline, on the same split

Every arm prints **quality | speed | the named incumbent**, on the *same*
held-out data, with a **negative control** somewhere in the suite.

- The atlas's table header: `examples/mlcommons_universal_atlas.rs:471`
  (`mlcommons_suite | scenario_metric | ptyktos_measured | incumbent_baseline(REG-1/REG-4) | advantage`).
- The GNN suite's header: `examples/mlperf_gpu_h2h.rs:1743`
  (`arm | heldout_node_acc | wall_ms | speedup_vs_sgd | baseline_sgd_acc`) — the
  SGD arm is the baseline, and it is in the table.
- `REG-4`: a new architecture must beat the **reservoir** baseline (random
  Fourier features + ridge) on the same split. The decider benchmarks against
  exactly that (`peregrine-ng/src/serve/decider.rs:26-28`); the atlas's safety
  gate benchmarks against an uncalibrated cosine guardrail
  (`examples/mlcommons_universal_atlas.rs:161`, `:222`), and its optimizer
  suite against an AdamW grid (`:365`).
- `REG-1`: three columns, always. A number without a baseline is not a result.

---

## 2. The three mistakes

### 2.1 OCR — a discrete codebook over a continuous field

**The mistake.** Mapping a continuous pixel field onto a discrete glyph
codebook / canonical character tokens, i.e. forcing an ill-posed discrete
classification boundary over a shift-covariant raw grid. Measured cost:
**~85% CER** (`omnidoc_ocr`, `glyph_ocr`, `ocr_evolve`; the experiment bins
live in `ptyktos-experiments/experiments/src/bin/`), and the same shape
produces 64.6% deletions.

**Why it is the same mistake as the LLM one.** It is trying to read a language
model without its continuous state: the raw grid's spectrum flat-lines
(`cone_fraction` collapses under translation), so the discrete boundary is
being fitted to noise.

**The repair, in the atlas's own terms.** Lift so the spectrum concentrates
(2-D autocorrelation / power spectrum + multi-scale `delta_scan`, or a frozen
patch embedder); then fold the lifted patches **onto the continuous latent
trajectories** of a frozen language model with an anchored ridge fold; then let
the existing n-gram/MLA decoder emit the token sequence. The language prior
resolves the ambiguity across kerning, fonts and CJK/Latin — which is exactly
what a discrete codebook throws away.

**The rule.** Never predict a discrete target from a raw field when a
continuous state is available. If the target must be discrete (glyphs, labels,
tokens), the *lift* must carry the continuity, and the decoder must own the
discretisation.

### 2.2 SAM — an external neural wrapper instead of the operator it is

**The mistake.** Treating segmentation as an external neural network to call
rather than as the algebra it is. `modelfold/src/vision/sam2.rs` is an
**1,818-line** checkpoint-backed SAM2.1 Hiera-T port with its own preprocessing
contract, its own attention, its own decoder — for a task the Gram already
affords: node classification on a pixel/superpixel affinity graph.

**The same operator is already proven.** `examples/mlperf_gpu_h2h.rs:1587-1592`
runs the R-GAT spectral graph coupling with the same `gram_fold_i8` /
`top_eigh` / `chol_solve` triple, on a 2,048-node graph with 8 classes.

**The repair.** A point or box prompt is literally a one-shot anchored shard
(`G_prompt`) added to the image's folded `G_img`; revoking a prompt is an exact
O(1) subtraction.

**The rule.** Before wrapping a model, ask what the operator *is*. If it is a
partition, a projection, a nearest-neighbour or a linear read over a fixed
feature space, the Gram + one Cholesky already do it — and do it in
microseconds, on CPU, with a spectrum diagnostic and a baseline to beat.

### 2.3 The decision models — a read that does not beat the majority class

**The mistake.** Treating a routing decision as "train a classifier and read
its argmax", instead of "read the criterion out of the folded statistic". The
symptom is a number **below the majority-class baseline** on real traffic.

**The measured evidence is already in the tree** — `peregrine-ng` PR #11
(`feat/systemone`), measured by `examples/systemone_measure.rs` on 3,396 real
harness requests (`examples/data/harness_requests.tsv`):

| task | accuracy | majority baseline | verdict |
|---|---:|---:|---|
| tier, 2-way (`is_large`) | 0.5557 | **0.5989** | below the baseline |
| model, 11-way (`model` id) | 0.1228 | **0.4452** | far below the baseline |

and the same head reads **a Rust request at 0.042** and **a trivia lookup at
0.420** — the criterion does not transfer.

**What that is, and is not.** It is *not* "the endpoint needs one more attempt":
the same PR records two attempts on that number (similarity only 0.5557; plus
head alignment 0.5557) and stops, because the third is the loop. And it is not
a missing-fitted-head problem either — a fitted support head over the criteria
was explicitly refused in that PR: *"Fitting one would have been a second
architecture."* The criteria the question names **are** the head.

**What it IS.** It is §1.2 failing silently: the lift's spectrum is not
concentrated on real harness traffic, so the solve reads noise and lands at the
majority class. The fix is in the **lift** (what features actually separate
these requests?) and in the **criterion** (is the axis a fitted head's axis?),
never in a third architecture.

**The rules.**

1. A read that does not beat the majority baseline on real traffic is a
   **lift/spectrum** finding. Report it with the cone, the baseline, and the
   lift. Do not retry the solve.
2. The model returns the **continuous** read; the cut belongs to the policy
   layer (telos-policy Cedar/ZEN), not inside the model. (`peregrine-ng`
   currently cuts at a fixed `0.5` inside `decide()` —
   `src/serve/decider.rs:532-533` — which is a policy decision living in the
   model.)
3. `cone_fraction` is measured *before* the solve and printed *with* the
   number. A workload whose cone is flat has a lift problem; say so.
4. When a question names criteria the fold does not span, the honest answer is
   the coverage measurement, not a fitted head: `noul = 1 − cone_fraction`
   restricted to the criteria (measured: related **0.2054**, unrelated
   **0.5130** — `peregrine-ng/src/serve/decider.rs:1225-1253`).

---

## 3. Checklist for a new workload

Run this **before** writing the solver, and again before believing the number.

1. **What is the operator's stationary sufficient statistic?** Name it. If it
   is not second-order and order-free, you have not found the operator yet.
2. **What is the lift, and what is its `cone_fraction`?** Measure it on the
   real held-out data, before the solve, and state the floor you are holding
   it to. Flat spectrum ⇒ go back to (2). No solver will save a flat lift.
3. **Where does the solve happen, and is it in the continuous target basis?**
   One Cholesky over the folded Gram, in fixed point. No discrete bottleneck
   on the way in; no fitted second head on the way out.
4. **What is the `REG-4` baseline, and does the arm beat it on the same
   split?** Random Fourier features + ridge is the floor. If the new
   architecture does not beat it, the new architecture is drag.
5. **Three columns** (REG-1): quality | speed | the named incumbent, on the
   same held-out data. A number without a baseline is not a result.
6. **A negative control** that fails when the behaviour is removed — a
   mutation, a reversed input, a seeded defect.
7. **If the number is bad, say which of the three knobs it was**: the cone
   (lift), the rank `r` (capacity), or the coupling order (what is folded
   against what). A bad number is a located finding, not a licence for another
   architecture.

---

## 4. Where the rules are machine-checked

| rule | where | what it means in code |
|---|---|---|
| `REG-1` | `~/projects/AGENTS.md` | every claim carries quality / speed / a named baseline |
| `REG-3` | `apeiron` `src/learn/solve/eig.rs:840` | the cone diagnostic: measure the spectrum before the solve |
| `REG-4` | the suite tables | the new arm must beat the reservoir/incumbent baseline on the same split |
| `FACE-2` / `SEED-2` | `src/learn/gram.rs:123` | the lift is fixed, the fold is gradient-free and exact |

---

## 5. The atlas, by example

| suite | file | what it shows |
|---|---|---|
| 12 official MLCommons suites + 6-stage pipeline | `examples/mlcommons_universal_atlas.rs` | the whole surface: five-column tables (`:471`), cone diagnostics (`:211`, `:358`, `:884`), REG-4 baselines (`:222`, `:365`) |
| MLPerf v6.1 study + Vulkan GPU dispatch | `examples/mlperf_gpu_h2h.rs` | the pattern in miniature: fold (`:1662`) → spectrum (`:1665-1668`) → solve (`:1682`) → table with the SGD baseline (`:1743`); the R-GAT graph suite (`:1587`) |
| A17B fleet | `examples/mlperf_a17b_fleet.rs` | the same skeleton at fleet scale |
| 6-system showcase (altar-1, webtrainer, iotadb, **peregrine-ng FoldDecider**, ai-gateway, guardfold) | `examples/altar1_tasmania_showcase.rs:1-7` | the pattern wired end to end; the live routing call is at `:367-370` (`POST /decide` against `PEREGRINE_ADDR`, default `127.0.0.1:38110`, `:382`) |

The showcase is the one to read first if you are new: it calls the *live*
decision model over HTTP and prints its answers next to the rest of the
system's numbers — which is exactly the discipline this file is about.
