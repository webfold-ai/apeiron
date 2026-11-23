# The five layers

apeiron is two things at once: an algebra, and the machine code that runs it. This chapter is the
map between them — five operations over one state, the law that licenses each, and the two faces
(the **learning** face and the **execution** face) they share.

Read it with [Evidence and grades](../verification/evidence.md) open. Every row here carries a
grade, and the grade is part of the claim.

---

## The distinction that makes the rest legible

Three things are routinely collapsed into the word "inference". They are not the same thing, and
keeping them apart is what makes the other four operations well-posed.

| | what it is | what changes |
|---|---|---|
| the **accumulated state** | what the system has retained | — |
| the **operation that changes it** | what an observation does to the state | the state |
| the **readout** | what a query gets back | nothing |

In explanatory notation — not an API, a way to ask the question:

```text
S' = S ⊕ C(x)          incorporate an observation into the state
y  = read(S, q)        answer a query against the state
```

The separation is what lets us ask **what persists**, instead of presuming that every query
permanently trains the system. It is also why removal is coherent: what you remove is a
*contribution to the state*, not a gradient step that "unlearns".

---

## The five operations

| # | operation | the law | grade |
|---|---|---|---|
| 1 | **accumulate** — fold a shard's contribution into the state | the Gram of stacked data is the sum of the Grams; the normal equations do not depend on order | `proved` |
| 2 | **solve** — read the answer out of the state in closed form | the drag is positive-definite, hence invertible; the head is *the* solution | `proved` |
| 3 | **read** — contract a query against the state | the read is one bulk ratio, and it streams in O(1) state | `proved` |
| 4 | **subtract** — remove exactly one contribution | subtracting a stage leaves exactly the state of the remaining stages | `proved` |
| 5 | **transform** — change the representation without changing the meaning | each family preserves its norm and carries an explicit inverse | `proved` |

### 1 · Accumulate

A shard contributes to the state; the state is the sum of the contributions. In the additive
face the shard carries `(G, B) = (XᵀX, XᵀY)`; in the multiplicative face it carries a cell.

- `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean` — `gram_stack`, `boundary_stack`,
  `normal_equations_order_free`, `normal_equations_stream`. Grade `proved`.
- `proof/corpus/Nous/Model/State.lean` — `tokenFold_append`: the fold over a concatenated list
  splits, `F(A ++ B) = F(A)·F(B)`. The shard boundary law in the multiplicative face. Grade
  `proved`.
- `proof/corpus/Nous/Model/Memory.lean` — `memory_step_model`: the table update at a key is
  `M k ↦ ringMul σ v (M k)`. Grade `proved`.
- The executing face: `simt::Op` and the reference `simt::op_ref` are the twelve ops and the one
  function the emitted code must match. Grade `proved` for the reference; see
  [The correspondence](#the-correspondence) for the link to the emitted code.

### 2 · Solve

```text
W = U⁻¹B ,   U = XᵀX + λ•1
```

- `TrainingCollapse.lean` — `drag_posDef`, `drag_isUnit`, `ridgeHead_solves`, `ridgeHead_unique`,
  `ridgeHead_minimizes`. Grade `proved`.
- The scaling law: `gram_replicate` and `boundary_replicate` — a stage folded with natural weight
  `w` equals that stage repeated `w` times, so a weighted merge is **exact upsampling**, licensed
  as exact rather than as a replay heuristic. Grade `proved`.
- The forward-only step: `newton_step_lands` — the damped Newton update lands the quadratic
  model's minimum exactly; gradient and curvature come from proven branch orbits, so there is no
  backward pass. Grade `proved`.
- `proof/canonical/Apeiron/Machine/Encoding.lean` — `guard_realises_divT` and
  `guard_realises_tmodT`: the zero-divisor guard makes the machine's partial division compute the
  *total* abstract op. What the silicon does is read from the vendor manual and stays an
  assumption. Grade `proved` for the guard equations; grade `model` for the silicon reading.

### 3 · Read

The read is a contraction against the state. In the corpus it is usually a **bulk ratio** — one
numerator over one drag — rather than a division per element.

- `proof/corpus/Substrate/Algebra/Neural/NullTransformer/Layers.lean` — `attnOut_is_bulk_ratio`:
  the attention output is one ratio, not `T` divisions. `attnOut_streams_in_O1_state`: the same
  read streams in **exact O(1) state**. `attnOut_budget_gap`: dropping attention mass has an error
  equal to the dropped-mass fraction, with no `T` in it. Grade `proved`.
- `proof/corpus/Nous/Model/Decomposition.lean` — `channelPair` is an equivalence of unital
  commutative algebras, and `channelPair_tokenFold`, `channelPair_memoryStep` and
  `channel_diagonal_fold` show that **both the token fold and the table memory update factor
  diagonally into independent scalar channels**. Grade `proved`.
- `proof/corpus/Nous/Model/Weighting.lean` — the channel projections compose multiplicatively
  across folds (`ringUpper_foldl`), but **the ratio is recomputed at each step and is not claimed to
  be multiplicative**; the zero-denominator case is a refusal
  (`channelWeight_denominator_eq_zero_iff`), and on the positive domain the ratio is the binary
  softmax of the log-channel pair. Grade `proved`.

### 4 · Subtract

The state is a monoid **with an inverse**, so removal is exact subtraction — not an approximate
"forget", and not an opposing training run.

- `TrainingCollapse.lean` — `stage_removal_exact`, which is `Collapse.streaming_exact_removal`:
  subtracting a stage's contribution leaves exactly the state of the remaining stages. Grade
  `proved`.
- `proof/corpus/Substrate/Collapse.lean` — `streaming_order_free` and `streaming_exact`. Grade
  `proved`.
- The executing face: `statfold` implements the same shape as `push` / `merge` / `unmerge` on every
  accumulator, with `Gram::unmerge` removing a merged part exactly. Grade `tested`.

**A distinction worth keeping.** The **cell product** used in the multiplicative fold and the
**additive ledger** used for subtraction are not interchangeable operations. The subtraction claim
belongs to the additive/group representation; do not read it as a statement about the cell
multiply.

### 5 · Transform

Data accumulation is not the only control surface. Transformations of the representation change how
the computation is expressed and executed, without changing what it means.

| family | established | source |
|---|---|---|
| hyperbolic Lorentz boost (`c² − s² = 1`) | volume-preserving, preserves the hyperbolic norm under its stated condition, explicit inverse | `proof/corpus/Nous/Combinator/CellTransform.lean` — `lorentzBoost_det_one`, `lorentzBoost_preserves_norm`, `lorentzBoost_inv` |
| elliptic / Givens rotation (`c² + s² = 1`) | volume-preserving, preserves the Euclidean norm, explicit inverse | same file — `givensRot_det_one`, `givensRot_preserves_norm`, `givensRot_inv` |
| parabolic shear (`det = 1`) | determinant one, explicit inverse | same file — `parabolicShear_inv` |
| spinor rotor | folds into the weight — pre-compose it so the runtime FWHT costs nothing — with the observable kept exact at unit norm | `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean` — `rotor_folds_into_weight`, `rotor_isometry` |

Grade `proved` for each. The generalisation behind them is change of coordinates:
`x' = Tx`, `W' = WT⁻¹` ⟹ `W'x' = Wx`.

Two further properties in `SpinorFold.lean` matter for a *fold* rather than for a model:
`spinorFold_schedule_free` — the one-pass two-wire reduce is unchanged under any reordering,
*derived* from the diagonalisation by permutation of a commutative product rather than assumed — and
`norm_of_reduction`, which is one closing multiply rather than a per-element square.

### 5.1 Fine-tuning without gradients — a second lane onto the same state

`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean` §6b — *"THE FINETUNE IS A
MASS-CONSERVING TRANSFER ONTO THE EVIDENCE"*. Domain adaptation without gradients: fold the domain
into a count table (the B/U accumulator), then refine the base distribution by one Kneser–Ney
absolute-discount lane.

```text
q_j = γ·p_j + (c_j − D c_j) / tot ,   γ = ΣD(c_i) / Σc_i
```

`kn_floor_refinement` proves the refined distribution **sums to one** (fine-tuning transfers mass,
never creates it), is **nonnegative**, and **keeps the table's evidence** regardless of the floor.
Grade `proved` for the theorem. The race figures recorded in the same section's commentary are
grade `model` here: they are a source reading, not a measurement this book decides.

---

## The two faces

| | the learning face | the execution face |
|---|---|---|
| carries | the state and its laws | the twelve ops and their reference |
| algebra | `R[x]/(x² − σ)` and the ring it instantiates | one trinity {throne, select, accum}, three core ops plus one doorway each — **twelve, closed** |
| where | `proof/corpus/Substrate/Algebra/Neural/`, `proof/corpus/Nous/` | `src/gpu/simt/` |
| rule | the state is a monoid with an inverse | one op table, one module wrapper, one reference |

The two meet at the op table. `src/gpu/simt/mod.rs` states the rule in its own header:

> The scalar substrates (x86-64 / ARM64 / RISC-V) are three *values* of one `Isa` structure — the
> target is a data parameter, never a hand-split backend. This module is the **fourth value**: the
> same twelve ops, lowered to SPIR-V words instead of instruction bytes.

One op table (`simt::encode_op`), one module wrapper, one reference (`simt::op_ref`) that the
emitted code must match. The same discipline holds on the learning side: `Apeiron.Machine.foldl_chunks`
licenses *"a worker pool or a SIMD lane split as an encoding schedule of the one abstract fold — the
instruction set does not grow to name the schedule."* Parallelism, tiling and split-K are
**schedules**, not new functionality.

**Where the counting stops being the argument.** The twelve is not a limit on what the system can
express; it is the size of the *verification unit*. A correspondence theorem for one target, plus a
composition theorem above it, carries the twelve to every composition — every fold, shard merge,
training step, training run and inference session — on every target, including kernels nobody has
written yet.

---

## The correspondence

This is the one obligation the book does **not** discharge, and it is stated here rather than left
implicit.

The profile in [The Song correspondence](../verification/song.md) fixes what a correspondence must
say, and it applies unchanged to the op table:

1. **Representation** — the emitted module denotes the canonical object, on a stated domain.
2. **State and effects** — the transition denotes the canonical transition.
3. **Admission, success and refusal** — the matched domain, each side's bounds, and what each does
   on unsupported input and on resource exhaustion.
4. **Resources** — allocation, time and memory, stated.

Applied to the twelve ops, the obligation is one theorem per op per target: the module the emitter
produces for op `o` computes `op_ref o`, under an admission predicate that admits only opcodes the
target's served set admits.

**Status: open.** The corpus proves the fragment's syntax and its semantics — the served opcode
sets, the decode obligations, the evaluation of the executing set, and the layout verdict that the
emitted packing realises the hyperbolic face. What it does not yet carry is the generic theorem
tying the op alphabet to its own encoder. The book's own sentence about the cross-implementation
case applies with equal force here: *"The cross-implementation correspondence is not established.
Each side has its own refinement to the canonical definition. The required result, one theorem
relating the two on a matched domain, is open."*

**The extraction route exists.** The extraction unit already reaches the GPU tree — the real
`src/gpu/spirv/check.rs` is an extraction root — and `src/gpu/mod.rs` declares the SIMT module
unconditionally, so `encode_op`, `op_ref` and the module wrapper are within reach of the same
Aeneas/Charon pipeline that carries the word kernels. The three functions are pure: no hash
collection in their closure, which is what the fail-closed audit rejects.

**Three named obstacles**, so the work is not open-ended: the SIMT module opens with inner
attributes in a non-crate-root module, which the extraction mirror does not currently tolerate; the
reference uses `u32` wrapping arithmetic and needs a `ZMod (2^32)` denotation beside the existing
`ZMod (2^64)` one; and the Lean op alphabet does not yet exist in this corpus and must be written
here, from the extracted reference.

---

## What this chapter does not claim

- That the correspondence is established. It is open, and the previous section says so.
- That a Lean proof of a law makes the corresponding Rust fast. The proof and the performance are
  different claims; see [Trust boundary](../verification/trust.md).
- That the driver's output is proved. The SPIR-V → cubin → SASS hop is a closed compiler: it is
  observed, asserted and differential-tested, not proved. See
  [SPIR-V and Vulkan](../engine/spirv.md).
- That boundedness and memory cost of the streaming fold are statements. They are not; the corpus
  says so where it belongs.

---

## Glossary

- **apeiron:** `simt::Op` — the twelve ops, one trinity of three faces.
- **apeiron:** `simt::op_ref` — the reference the emitted code must match.
- **apeiron:** `simt::encode_op` — the encoder; one op to its target words.
- **apeiron:** `simt::spv_module` — the module wrapper, the ELF-header analog.
- **apeiron:** `word_mul_denote` — the word multiply is the ring multiply in `Z/2^64`.
