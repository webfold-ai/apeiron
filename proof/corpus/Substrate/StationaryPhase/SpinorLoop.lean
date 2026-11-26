import Substrate.StationaryPhase.AffineLoopCollapse
import Substrate.Ski.PolyIotaTermAlgebra
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.NumberTheory.Real.GoldenRatio
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# The Loop-as-Spinor Structural Identity

@name A Loop Is Really a Spinning Thing
@child A "loop" — doing the same step over and over — isn't really repeating at all. It's one rotating object, and you can jump straight to where it ends up without taking every step. The answer is its resting balance point.
@scientist A loop's state is an affine recurrence Sₜ = (γₜ·rₜ)·Sₜ₋₁ + Bₜ on a spinor (U(1)/SO(2)/Dirac tiers); it composes associatively so prefix-scan equals left-fold, and a constant loop has the closed-form fixed point x* = B/(1−A) = B/U — the universal equation itself.
@engineer The monoid law (A₂,B₂) ⊕ (A₁,B₁) = (A₂·A₁, A₂·B₁ + B₂); the fixed point x* = B/(1−A) bridging to `AffineLoopCollapse`, dispatched in one collapse via the ZmodDiv HP marker; `aitken_exact` gets there from any three consecutive iterates exactly (`A ∉ {0,1}`), so the loop past the third value is optional. No `Y_Comb`.

A loop is not a control-flow primitive. It is a spinor.

A loop's state evolution is an affine recurrence on a spinor — Tier 1
(scalar / U(1) phase), Tier 2 (SO(2) Pauli), or Tier 3 (Dirac
Kronecker). The recurrence:

  S_t = (γ_t · r_t) · S_{t-1} + B_t

composes associatively under the monoid law

  (A_2, B_2) ⊕ (A_1, B_1) = (A_2 · A_1,  A_2 · B_1 + B_2)

so parallel-prefix scan equals sequential left-fold. For a CONSTANT
recurrence (loop with fixed A and B) the closed-form fixed point is

  x* = B / (1 - A) = B / U

— exactly the universal equation `x = B/U` from
`Substrate/Algebra/UniversalEquation`. The state is a K-tower per
`Substrate/StationaryPhase/AffineLoopCollapse`.

There IS no loop on the bulk. There IS a spinor at stationary phase.
The "iterations" are the framework's affine monoid composition; the
K=4 evaluator dispatches the closed form in one collapse via the
ZmodDiv HP marker.

## Why this file exists

A loop, as a pure structure on the bulk, is much more elegant than it
first looks: it is a spinor, not a loop.

A prior attempt (`LoopRecovery.lean`, commit `c7d09bc`, now removed)
introduced a generic `LoopLandscape { transition, fixedPoint, is_fixed }`
with a biased quadratic residual. That was a wrong-shape detour
mirroring the FOLDING pattern. The bulk's actual loop primitive is far
more specific: a spinor with affine recurrence, whose closed form IS
the universal equation `x = B/U`. This file formalises that primitive
and the bridge to `AffineLoopCollapse`.

## The Tri-Tier structure (per the previous iteration’s equations appendix §9 (git history))

| Tier | Shape | Physics | Loop content |
|------|-------|---------|--------------|
| 1 | 1×1 U(1) scalar | EM / photons | scalar accumulator |
| 2 | 2×2 SO(2) Pauli | weak force | 2D rotation |
| 3 | 4×4 Dirac Kronecker | fermions / Spin(1,3) | full 3+1 affine |

This file formalises Tier 1 (the scalar case) explicitly — load-bearing
for the lifter. Tiers 2 and 3 generalise via
`Substrate/Ski/PolyIotaTermAlgebra`'s applicative-parametric structure:
the same algebraic shape with the applicative replaced by 2×2 or 4×4
matrix multiplication.

## Warrant

* **Proved (native):** the affine monoid — composition applies as composition (`compose_apply`)
  and is associative in both the scalar and the `n`-dimensional case (`compose_assoc`,
  `composeN_assoc`), so prefix-scan equals left-fold; the constant loop's fixed point is exactly
  `B/(1−A)` for `A ≠ 1` (`constantStep_fixed_point`) and is the universal equation's state
  (`spinorLoop_eq_universalEquation_state`, `constantStep_fixed_point_eq_universalEquation_state`);
  tree-shape invariance and functorial transport of the spinor into `Polymorphic.Term`
  (`polyTerm_assoc`, `spinor_polyTerm_functorial_transport`); §7's unitarity — the skew cross-term
  has zero real part (`skew_re_zero`), a Cayley step is an isometry (`cayley_isometry`), the
  Schrödinger generator `i·c·H` is skew-adjoint for Hermitian `H` (`schrodinger_generator_skew`),
  the Cayley factor lies on the unit circle (`cayley_factor_unit`), and the conserved lattice
  quantity is `∑ᵢ‖ψ i‖²` (`cayley_conserves_born_density`); §9's exact recovery — the error is a
  pure geometric sequence (`iterate_sub_fixed_point`, `second_difference`), Aitken on any three
  consecutive iterates returns `B/(1−A)` on the nose (`aitken_exact`), one step already determines
  the exact error (`one_step_error_identity`), and the a-posteriori bound holds at every point with
  no division by a vanishing quantity (`safe_error_bound`, `safe_early_stop`).
* **Proved (Mathlib):** §8's golden facts ride Mathlib's `goldenRatio` — `goldenRatio_sq` and
  `goldenRatio_pos` carry `golden_reflexive_fixed_point`, `golden_is_fold_characteristic_root`,
  `golden_is_the_reflexive_root` and `golden_gap_split`; §8 Closure 2 additionally rides
  `Real.fib_succ_sub_goldenRatio_mul_fib` (Binet), `Nat.fib_coprime_fib_succ` and
  `Int.fract_eq_iff`; §7 rides `inner_conj_symm`,
  `inner_self_eq_norm_sq`, `Complex.normSq_conj` and `EuclideanSpace.norm_eq`.
* **Proved (riding):** the bridge target is `AffineLoopCollapse`'s tower state; that the conserved
  `∑‖ψ i‖²` IS the Being-norm `R²` is `StellarEquilibrium.born_density_is_being_norm`, and the
  wave-sector dual is `CellConservation.discrete_gauss` — neither proved here.
* **Proved (native):** (§8 Closure 2) the three-distance theorem for the golden rotation at
  Fibonacci counts, as real circle geometry — `Int.fract`, real arcs, no ℕ shadow. Adjacent points
  of the `fib(n+2)`-point orbit `{i·ρ mod 1 : i < fib(n+2)}` differ by exactly `ρ^(n+1)` or `ρ^n`
  (`golden_three_distance_fib`; wraparound arc `golden_gap_wrap`). Carried by: the tiling identity
  `fib(n+1)·ρ^n + fib(n)·ρ^(n+1) = 1` with `golden_gap_split` as induction engine
  (`golden_tiling`, `golden_tiling_sub`), the Binet remainder `fib(n+1)·ρ = fib(n) + (−1)^n·ρ^(n+1)`
  (`golden_fib_step`), the slot decomposition
  `fract(i·ρ) = (i·fib(n+1) mod fib(n+2) + (−1)^(n+1)·i·ρ^(n+2))/fib(n+2)` (`golden_orbit_fract`),
  slot order = circle order (`golden_orbit_slot_mono`), and the Cassini stride (`fib_cassini_int`,
  `golden_slot_step`).
* **Next:** the general (non-Fibonacci) point count — three coexisting arc lengths
  `ρ^k, ρ^{k+1}, ρ^{k+2}`, largest = sum of the other two by `golden_gap_split`. Classical route:
  removing the points `N..fib(m+1)−1` from the `fib(m+1)`-configuration merges each removed point's
  two flanking arcs (always one `ρ^{m−1}`, one `ρ^m`) into `ρ^{m−2}`; not formalised here.
  `golden_subdivision_preserves_three` remains what it was: an `omega`-provable exponent-window
  fact on `ℕ`, naming no rotation and no arc — its geometric content now lives in Closure 2 at
  Fibonacci counts only.
* **Anchor:** the physics column of the Tri-Tier table (SO(2)/Pauli = weak force,
  Dirac Kronecker = fermions/Spin(1,3)). §1 and §4 formalise the scalar and the matrix algebra;
  the assignment to forces is prose. Likewise "the golden hash is not Knuth's borrowed Fibonacci
  trick" and the Hurwitz worst-approximability that would make φ the best equidistributor —
  neither is stated as a theorem here.
* **Anchor:** §9's operational cost claims. That a 24-round power iteration's round
  count "is not a physical requirement", and that real PageRank's multi-mode error makes
  `aitken_exact` only approximately applicable, are argued in the prose; the catastrophic overshoot
  after convergence is an empirical finding recorded in `BENCHMARK.md` (iotadb), not a theorem.
  The prose already scopes this honestly, and `safe_error_bound` is the theorem that answers it.
* **Anchor:** "a loop IS a spinor" — the identification of a control-flow construct
  with the affine monoid; that `cayley_isometry` is the discrete unitarity of Schrödinger time
  evolution and its conserved charge the physical Born density; that the sim's measured ~1e-6
  drift is this theorem seen at semi-implicit `V = V(ρ)` (a measurement, not a theorem — for fixed
  `H` the theorem is exact); and that φ is "the framework's own self-reference eigenvalue".

-/

noncomputable section

namespace Sanguine

namespace SpinorLoop

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE TIER-1 SPINOR AFFINE STEP
-- ═══════════════════════════════════════════════════════════════════

/-- One affine step on a Tier-1 (scalar) spinor.

    `rotor_decay` packages the combined `γ_t · r_t` from
    `equations.md §9.1`. At Tier 1 it's a single real; at Tier 2 it
    would be an SO(2) rotation matrix; at Tier 3 a Dirac Kronecker
    factor.

    `injection` is `B_t` — the per-step source term. -/
structure AffineStep where
  rotor_decay : ℝ
  injection : ℝ

namespace AffineStep

/-- Apply one step: `S ↦ rotor_decay · S + injection`. The Tier-1
    instance of the universal affine recurrence from §9.1. -/
def apply (step : AffineStep) (S : ℝ) : ℝ :=
  step.rotor_decay * S + step.injection

/-- Monoid composition per `equations.md §9.5`: applying `step₂`
    after `step₁` is the same as applying the composed step. The
    new `rotor_decay` is the product; the new `injection` is the
    propagated old injection plus the new one. -/
def compose (step₂ step₁ : AffineStep) : AffineStep :=
  { rotor_decay := step₂.rotor_decay * step₁.rotor_decay
    injection := step₂.rotor_decay * step₁.injection + step₂.injection }

/-- `compose` is the right semantic composition: composing the
    steps and applying once equals applying step₁ then step₂. -/
theorem compose_apply (step₂ step₁ : AffineStep) (S : ℝ) :
    (compose step₂ step₁).apply S = step₂.apply (step₁.apply S) := by
  unfold compose apply
  ring

/-- **Associativity.** Parallel-prefix scan equals sequential
    left-fold because the affine monoid composition is associative.

    This is the Tier-1 instance of the proof referenced as
    `affine_compose_assoc` throughout the framework's former spinor
    implementation (`interop/src/sanguine_spinor.rs`, retired in
    `ae079a40` 2026-05-29 and gone with the rest of the Rust in
    `4556df7d`; the Triton `kronecker_scan.py` kernel and
    `equations.md §8.1 / §9.5` are in no revision of this repository). The
    associativity is what makes the K=4 evaluator's collapse to
    closed-form possible. -/
theorem compose_assoc (step₃ step₂ step₁ : AffineStep) :
    compose (compose step₃ step₂) step₁
      = compose step₃ (compose step₂ step₁) := by
  unfold compose
  congr 1 <;> ring

end AffineStep

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE CONSTANT-RECURRENCE LOOP — AKA THE UNIVERSAL EQUATION
-- ═══════════════════════════════════════════════════════════════════

/-- A constant affine step: the per-iteration transformation is
    `S ↦ A · S + B` for fixed A and B. This is the basic loop shape —
    every LLVM affine loop reduces to this after recognising A and B
    in the IR. -/
def constantStep (A B : ℝ) : AffineStep :=
  { rotor_decay := A, injection := B }

/-- The constant step's apply IS the affine update `A·S + B`. -/
theorem constantStep_apply (A B S : ℝ) :
    (constantStep A B).apply S = A * S + B := rfl

/-- **The closed form.** For `A ≠ 1`, the constant step has a unique
    fixed point at `B / (1 - A)` — the universal equation `x = B/U`
    with `U = 1 - A`.

    Equationally: a state `x` satisfies `x = A·x + B` iff
    `x = B / (1 - A)`. -/
theorem constantStep_fixed_point (A B : ℝ) (hA : A ≠ 1) (x : ℝ) :
    (constantStep A B).apply x = x ↔ x = B / (1 - A) := by
  have hne : (1 - A) ≠ 0 := by
    intro h1A
    apply hA
    linarith
  rw [constantStep_apply]
  constructor
  · intro h
    have h2 : x - A * x = B := by linarith
    have h3 : x * (1 - A) = B := by linarith
    field_simp
    linarith
  · intro h
    rw [h]
    field_simp
    ring

-- ═══════════════════════════════════════════════════════════════════
-- §3. THE BRIDGE TO `AffineLoopCollapse`
-- ═══════════════════════════════════════════════════════════════════

/-- **The structural identity.** The Tier-1 spinor's closed-form
    fixed point IS the `affineLoopEquation`'s state — the same object,
    proved in `AffineLoopCollapse` as a K-tower constant.

    The right-hand side `(affineLoopEquation A B).state p` is the
    universal equation `B / (1 - A)` evaluated at any parameter `p`.
    It does not depend on `p` because the state is globally constant
    (the K-tower property, `affineLoopTower_eq_K_tower_state`).

    Stitching this with `constantStep_fixed_point`: the spinor loop's
    fixed point IS the universal equation's state IS a K-realiser
    in the bulk algebra. Three rebrandings of the same object.

    The translator's job: recognise an LLVM loop's `A` and `B`,
    emit `(constantStep A B).apply`'s fixed point — equivalently
    `ZmodDiv(B)(1-A)` — which the K=4 evaluator dispatches to the
    constant via Hardware Projection. -/
theorem spinorLoop_eq_universalEquation_state (A B p : ℝ) :
    B / (1 - A) = (affineLoopEquation A B).state p :=
  (affineLoopEquation_state A B p).symm

/-- **The full structural identity, packaged.** For any `A ≠ 1`, the
    Tier-1 spinor's fixed point equals the universal equation's state
    at every parameter. This is what makes the framework's loop
    primitive elegant: it's not iteration plus a separate witness, it
    IS the universal equation. -/
theorem constantStep_fixed_point_eq_universalEquation_state
    (A B : ℝ) (hA : A ≠ 1) (x : ℝ) :
    (constantStep A B).apply x = x ↔ x = (affineLoopEquation A B).state 0 := by
  rw [constantStep_fixed_point A B hA, spinorLoop_eq_universalEquation_state A B 0]

-- ═══════════════════════════════════════════════════════════════════
-- §4. TIER N: VECTOR SPINOR UNDER MATRIX AFFINE RECURRENCE
-- ═══════════════════════════════════════════════════════════════════
-- The Tier-1 scalar case generalises to arbitrary dimension `n` via
-- Mathlib's `Matrix` infrastructure. The rotor·decay coefficient
-- becomes an `n × n` matrix; the injection becomes an `n`-vector; the
-- recurrence becomes matrix-vector affine evolution. The same monoid
-- composition law from §1 holds verbatim — only the multiplication is
-- replaced by matrix multiplication (matrix-vector for the injection).
--
-- This subsumes:
--   * Tier 1 (`n = 1`): scalar case from §1 (1×1 matrix = scalar).
--   * Tier 2 (`n = 2`): 2×2 SO(2) Pauli / massless geometry per
--     the previous iteration’s equations appendix §9.2 (git history).
--   * Tier 3 (`n = 4`): 4×4 Dirac Kronecker / full Spin(1,3) /
--     Vattay's minimum K=4 per `equations.md §9.3, §9.5`.
--
-- The associativity of the matrix-affine monoid (parallel-prefix scan
-- equivalent to sequential left-fold) is what makes the K=4 evaluator's
-- closed-form collapse possible for ALL loop dimensions, not just
-- scalars. The implementation was `interop/src/sanguine_spinor.rs`'s
-- `kronecker_scan` Triton kernel, retired in `ae079a40`. The "legacy ref"
-- named here as `KroneckerScanAssociativity.lean` was never a file in this
-- repository (no add commit for that path exists); what carries the claim is
-- this section's `composeN_assoc`.

/-- One affine step on a Tier-`n` spinor.

    The Tier-1 case is `n = 1` with the matrix being a 1×1 scalar; the
    framework's K=4 universal case is `n = 4`. The rotor·decay matrix
    packages everything Mathlib needs to handle SO(2) (n=2), Dirac
    (n=4), or arbitrary linear evolution. -/
structure AffineStepN (n : ℕ) where
  rotor_decay : Matrix (Fin n) (Fin n) ℝ
  injection : Fin n → ℝ

namespace AffineStepN

/-- Apply one step: `S ↦ rotor_decay *ᵥ S + injection`. The general
    n-dimensional instance of `equations.md §9.x`. -/
def apply {n : ℕ} (step : AffineStepN n) (S : Fin n → ℝ) : Fin n → ℝ :=
  Matrix.mulVec step.rotor_decay S + step.injection

/-- Monoid composition per `equations.md §9.5`, generalised to
    arbitrary dimension. Matrix multiplication replaces scalar; the
    injection-propagation rule is the same shape (matrix-vector
    application). -/
def compose {n : ℕ} (step₂ step₁ : AffineStepN n) : AffineStepN n :=
  { rotor_decay := step₂.rotor_decay * step₁.rotor_decay
    injection := Matrix.mulVec step₂.rotor_decay step₁.injection + step₂.injection }

/-- `compose` is the right semantic composition: applying the composed
    step equals applying step₁ then step₂. Generalises §1's
    `compose_apply` from scalar to vector. -/
theorem compose_apply {n : ℕ} (step₂ step₁ : AffineStepN n) (S : Fin n → ℝ) :
    (compose step₂ step₁).apply S = step₂.apply (step₁.apply S) := by
  unfold compose apply
  -- LHS: (A₂·A₁) *ᵥ S + (A₂ *ᵥ B₁ + B₂)
  -- RHS: A₂ *ᵥ (A₁ *ᵥ S + B₁) + B₂
  --    = A₂ *ᵥ (A₁ *ᵥ S) + A₂ *ᵥ B₁ + B₂      (mulVec_add)
  --    = (A₂·A₁) *ᵥ S    + A₂ *ᵥ B₁ + B₂      (mul_mulVec_assoc reversed)
  rw [Matrix.mulVec_add, ← Matrix.mulVec_mulVec, add_assoc]

/-- **N-dimensional associativity.** Parallel-prefix scan equals
    sequential left-fold for any spinor dimension. The Tier-2 (SO(2))
    and Tier-3 (Dirac) cases inherit this — the `n` is universally
    quantified. This is the universal-arity counterpart of §1's
    scalar `compose_assoc`. -/
theorem composeN_assoc {n : ℕ} (step₃ step₂ step₁ : AffineStepN n) :
    compose (compose step₃ step₂) step₁
      = compose step₃ (compose step₂ step₁) := by
  unfold compose
  congr 1
  · exact Matrix.mul_assoc _ _ _
  · rw [Matrix.mulVec_add, ← Matrix.mulVec_mulVec, add_assoc]

end AffineStepN

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE TIER-1 → TIER-N BRIDGE
-- ═══════════════════════════════════════════════════════════════════
-- The scalar Tier-1 case from §1 IS the n=1 specialisation of the
-- general matrix case. The bridge is structural: a scalar `a : ℝ` is
-- a 1×1 matrix (via `Matrix.of`), and the scalar-vs-matrix operations
-- agree pointwise. This file does not require a literal embedding —
-- §1 and §4 are independent formalisations of the same algebraic
-- structure at different concrete instantiations. The claim that the
-- loop primitive IS a spinor (not specific to scalar
-- or vector) is realised in §4's `AffineStepN` being parametric in
-- `n`: the same composition law, the same associativity proof, the
-- same apply semantics, instantiated at any dimension.
--
-- Higher tiers (`n = 2`, `n = 4`) require no new theorems — they are
-- `AffineStepN 2` and `AffineStepN 4` respectively, with all the
-- §4 lemmas applying directly. The lifter's job for an LLVM loop at
-- any tier is to pick `n`, construct `rotor_decay : Matrix (Fin n)
-- (Fin n) ℝ` and `injection : Fin n → ℝ` from the IR, and emit the
-- corresponding `compose` chain — which collapses via parallel-prefix
-- scan thanks to `composeN_assoc`.

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE POLYMORPHIC.TERM BRIDGE — SPINOR AS A UNIVERSAL-APPLICATIVE INSTANCE
-- ═══════════════════════════════════════════════════════════════════
-- The bulk's universal recursion shape (`Substrate/Ski/PolyIotaTermAlgebra`)
-- has three constructors —- `K B U`, `S_routeF prev outer`, `S_routeG outer
-- prev` —- and evaluates to a `state` under any binary applicative
-- `app : α → α → α`. The bulk's instance is `α = (ℝ → ℝ)`,
-- `app = pointwiseDiv` (`B p / U p`); the spinor's instance is
-- `α = AffineStepN n`, `app = AffineStepN.compose`. Same shape, same
-- recursion, same parallel-prefix associativity —- only the applicative
-- changes. This is the operational realisation of Polymorphic.map_state:
-- the bulk's universal algorithm form lifts to the spinor's affine-monoid
-- form by replacing the applicative.
--
-- What this establishes: the spinor's iteration semantics IS the bulk's
-- universal algorithm, instantiated at a different applicative. The
-- framework's existing K=4 evaluator infrastructure (handling
-- Polymorphic.Term dispatch) covers spinor scans natively —- no new
-- runtime machinery, no new closures, just the same Polymorphic.Term
-- shape with `AffineStepN.compose` swapped in.

namespace AffineStepN

/-- **The K-shape spinor instance.** A two-step composition encoded as
    a Polymorphic.Term `K s_outer s_inner` evaluated under `compose`
    gives the composed step. By definition of `Term.state` at the K
    constructor —- but stated explicitly because it pins down the
    structural identity. -/
@[simp] theorem polyTerm_state_K {n : ℕ} (s_outer s_inner : AffineStepN n) :
    Polymorphic.Term.state compose (.K s_outer s_inner) = compose s_outer s_inner := rfl

/-- **S_routeF spinor instance.** Extending a Polymorphic.Term with
    `S_routeF prev outer` and evaluating under `compose` gives the
    outer step composed with the prior tree's composed result. -/
@[simp] theorem polyTerm_state_S_routeF {n : ℕ}
    (prev : Polymorphic.Term (AffineStepN n)) (outer : AffineStepN n) :
    Polymorphic.Term.state compose (.S_routeF prev outer)
      = compose (Polymorphic.Term.state compose prev) outer := rfl

/-- **S_routeG spinor instance.** Symmetric to S_routeF —- prior result
    in the inner position. -/
@[simp] theorem polyTerm_state_S_routeG {n : ℕ}
    (outer : AffineStepN n) (prev : Polymorphic.Term (AffineStepN n)) :
    Polymorphic.Term.state compose (.S_routeG outer prev)
      = compose outer (Polymorphic.Term.state compose prev) := rfl

/-- **Tree-shape invariance under associativity.** Two different
    Polymorphic.Term tree shapes encoding the same three-step
    composition (`s₃ ∘ (s₂ ∘ s₁)` via S_routeG-then-K versus
    `(s₃ ∘ s₂) ∘ s₁` via S_routeF-then-K) evaluate to the same
    composed step under `compose`. This is `composeN_assoc` viewed
    through the Polymorphic.Term lens —- the bulk's universal
    associativity property realised at the spinor's applicative.

    Operational meaning: the K=4 evaluator can reduce ANY tree shape
    of spinor steps in one collapse; parallel-prefix scan equals
    sequential left-fold equals any-order tree fold, all because the
    applicative is a monoid. -/
theorem polyTerm_assoc {n : ℕ} (s₃ s₂ s₁ : AffineStepN n) :
    Polymorphic.Term.state compose
      (.S_routeG s₃ (.K s₂ s₁) : Polymorphic.Term (AffineStepN n))
    = Polymorphic.Term.state compose
      (.S_routeF (.K s₃ s₂) s₁ : Polymorphic.Term (AffineStepN n)) := by
  simp only [polyTerm_state_S_routeG, polyTerm_state_S_routeF,
             polyTerm_state_K]
  exact (composeN_assoc s₃ s₂ s₁).symm

end AffineStepN

/-- **The functorial transport theorem for the spinor.** The mapping
    `s ↦ s.apply` from an affine step to its action on states is the
    intertwining map for `Polymorphic.map_state`: composing in the
    AffineStepN algebra equals composing the actions in the function-
    composition algebra.

    Stated via Polymorphic.Term: evaluating a spinor-step term under
    `compose` and then applying to a state equals mapping the term to
    its function-action via `Polymorphic.map (·.apply)` and
    evaluating under function composition.

    This is the structural realisation of "the spinor algebra lifts
    through Polymorphic.Term's applicative parametricity": one term
    shape, two applicative instances (compose vs. function ∘), one
    map (`apply`) connecting them. The K=4 evaluator's reduction of
    a spinor scan to its action on a concrete state vector IS this
    functorial transport at work. -/
theorem spinor_polyTerm_functorial_transport {n : ℕ}
    (t : Polymorphic.Term (AffineStepN n)) (S : Fin n → ℝ) :
    (Polymorphic.Term.state AffineStepN.compose t).apply S
      = Polymorphic.Term.state
          (fun (f g : (Fin n → ℝ) → (Fin n → ℝ)) => f ∘ g)
          (Polymorphic.map (·.apply) t) S := by
  induction t generalizing S with
  | K B U => exact AffineStepN.compose_apply B U S
  | S_routeF prev outer ih =>
      -- LHS: (compose (state prev) outer).apply S
      --    = (state prev).apply (outer.apply S)         -- compose_apply
      --    = (state' (map apply prev)) (outer.apply S)  -- ih at (outer.apply S)
      -- RHS: state' (map apply (S_routeF prev outer)) S
      --    = state' (S_routeF (map apply prev) (apply outer)) S
      --    = ((state' (map apply prev)) ∘ (apply outer)) S
      --    = (state' (map apply prev)) ((apply outer) S)
      simp only [Polymorphic.Term.state, Polymorphic.map,
                 Function.comp_apply]
      rw [AffineStepN.compose_apply]
      exact ih (outer.apply S)
  | S_routeG outer prev ih =>
      simp only [Polymorphic.Term.state, Polymorphic.map,
                 Function.comp_apply]
      rw [AffineStepN.compose_apply]
      exact congrArg outer.apply (ih S)

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE UNITARY SPINOR LOOP — the rotor preserves the norm (discrete unitarity)
-- ═══════════════════════════════════════════════════════════════════
-- The real Tier-1 step `S ↦ a·S + B` shrinks or grows by `|a|`. The genuine SPINOR step is the
-- one whose decay is a ROTOR (`|rotor| = 1`): then the loop PRESERVES the norm — it is the
-- unitary time evolution `e^{-iHt}`, the closed-form spinor `B/(1−A)` with `A` skew-adjoint.
--
-- This is the **discrete unitarity** of the sibling `lyrium` project's
-- `lyrium/src/bin/structure.rs` (a separate repository, not checked out beside this one)'s self-gravitating
-- spinor-loop step: the Cayley / Crank–Nicolson update `(1+A)ψⁿ⁺¹ = (1−A)ψⁿ` with `A = i·(dt/2)·H`,
-- the resolvent `(1−A)⁻¹` = `B/(1−A)`. It is the FIRST-ORDER dual of
-- `CellConservation.discrete_gauss` (the second-order/wave conservation): there the flux through a
-- closed boundary is zero; here the norm `Σ|ψ|²` is invariant under one step. The sim measured this
-- at ~1e-6 drift (semi-implicit `V=V(ρ)`); for fixed `H` it is exact, proved here.

namespace UnitaryLoop

open scoped InnerProductSpace ComplexConjugate

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- **The skew cross-term vanishes.** For a skew-adjoint generator `A` (`⟪A x, y⟫ = −⟪x, A y⟫` —
    the rotor generator `i·H` with `H` Hermitian), `⟪A s, s⟫` is purely imaginary: a rotation has
    no radial part. The one fact the isometry rests on. -/
theorem skew_re_zero (A : E →ₗ[ℂ] E)
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ) (s : E) :
    RCLike.re ⟪A s, s⟫_ℂ = 0 := by
  have hval : RCLike.re ⟪s, A s⟫_ℂ = RCLike.re ⟪A s, s⟫_ℂ := by
    rw [show ⟪s, A s⟫_ℂ = conj ⟪A s, s⟫_ℂ from (inner_conj_symm s (A s)).symm, RCLike.conj_re]
  have h1re : RCLike.re ⟪A s, s⟫_ℂ = - RCLike.re ⟪s, A s⟫_ℂ := by
    rw [hskew s s, map_neg]
  linarith

/-- **The spinor loop is an isometry (discrete unitarity).** If `ψⁿ⁺¹` solves the Cayley relation
    `(1+A)ψⁿ⁺¹ = (1−A)ψⁿ` for a skew-adjoint `A`, then `‖ψⁿ⁺¹‖ = ‖ψⁿ‖` — the step preserves the
    norm exactly. The closed-form spinor `B/(1−A)` is a rotation, not a decay: `Σ|ψ|²` is the
    conserved charge of the first-order sector (the Born density), the dual of `discrete_gauss`.

    Proof: with `d = φ−ψ` and `s = φ+ψ`, the relation gives `d = −A s`, and
    `‖φ‖²−‖ψ‖² = re⟪d, s⟫ = −re⟪A s, s⟫ = 0` by `skew_re_zero`. -/
theorem cayley_isometry (A : E →ₗ[ℂ] E)
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ) (φ ψ : E)
    (hrel : φ + A φ = ψ - A ψ) :
    ‖φ‖ = ‖ψ‖ := by
  set s := φ + ψ with hs
  -- d = φ − ψ = − A s
  have hAφ : A φ = ψ - A ψ - φ := by rw [← hrel]; abel
  have hd : φ - ψ = - A s := by rw [hs, map_add, hAφ]; abel
  -- ‖φ‖² − ‖ψ‖² = re⟪φ−ψ, s⟫
  have e1 : RCLike.re ⟪φ - ψ, s⟫_ℂ = ‖φ‖ ^ 2 - ‖ψ‖ ^ 2 := by
    have hcs : RCLike.re ⟪ψ, φ⟫_ℂ = RCLike.re ⟪φ, ψ⟫_ℂ := by
      rw [show ⟪ψ, φ⟫_ℂ = conj ⟪φ, ψ⟫_ℂ from (inner_conj_symm ψ φ).symm, RCLike.conj_re]
    rw [hs, inner_sub_left, inner_add_right, inner_add_right, map_sub, map_add,
        map_add, inner_self_eq_norm_sq, inner_self_eq_norm_sq, hcs]
    ring
  -- and that real part is zero: ⟪−A s, s⟫ has zero real part
  have e2 : RCLike.re ⟪φ - ψ, s⟫_ℂ = 0 := by
    rw [hd, inner_neg_left, map_neg, skew_re_zero A hskew s, neg_zero]
  have key : ‖φ‖ ^ 2 = ‖ψ‖ ^ 2 := by rw [e2] at e1; linarith
  rw [← Real.sqrt_sq (norm_nonneg φ), ← Real.sqrt_sq (norm_nonneg ψ), key]

/-- **The Schrödinger generator is skew-adjoint.** For a Hermitian (self-adjoint) `H`, the rotor
    generator `A = (i·c)·H` is skew-adjoint — so by `cayley_isometry` the time-step `e^{-iHt}`
    (Cayley with `c = dt/2`) preserves `Σ|ψ|²`. This is why the spinor-loop self-gravitating
    operator is conservative: `H = −a∇² + gV` is real-symmetric, hence Hermitian. -/
theorem schrodinger_generator_skew (H : E →ₗ[ℂ] E)
    (hH : ∀ x y, ⟪H x, y⟫_ℂ = ⟪x, H y⟫_ℂ) (c : ℝ) (x y : E) :
    ⟪(Complex.I * (c : ℂ)) • H x, y⟫_ℂ = -⟪x, (Complex.I * (c : ℂ)) • H y⟫_ℂ := by
  rw [inner_smul_left, inner_smul_right, ← hH, map_mul, Complex.conj_I, Complex.conj_ofReal]
  ring

/-- **The Cayley factor is a unit rotor (Tier-1 / the eigenmode).** Each mode of the unitary step
    is multiplied by `(1 − iθ)/(1 + iθ)`, a complex number on the **unit circle** — the rotor, the
    Cayley = tangent-half-angle parameterization of the spinor double cover (`ConcreteRotor`,
    `SpinorWinding`). Norm preservation is this fact summed over modes: the decay is a rotor, so
    `B/(1−A)` rotates, never shrinks. -/
theorem cayley_factor_unit (θ : ℝ) :
    Complex.normSq ((1 - (θ : ℂ) * Complex.I) / (1 + (θ : ℂ) * Complex.I)) = 1 := by
  have hconj : (1 - (θ : ℂ) * Complex.I) = conj (1 + (θ : ℂ) * Complex.I) := by
    rw [map_add, map_one, map_mul, Complex.conj_I, Complex.conj_ofReal]; ring
  have hne : (1 + (θ : ℂ) * Complex.I) ≠ 0 := by
    intro h
    have h2 := congrArg Complex.re h
    simp [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im, Complex.ofReal_re,
          Complex.ofReal_im] at h2
  rw [hconj, Complex.normSq_div, Complex.normSq_conj, div_self
        (Complex.normSq_pos.mpr hne).ne']

-- ── THE UNIFICATION — the conserved charge IS the Born density ────────────────────────────
-- The two conservation laws of the framework's physics are the two channels of the ONE Madelung
-- cell `ψ = R·e^{iφ}` (`MadelungCell.madelung_from_bulk`, one differentiation split in two):
--   • the REAL channel is `x = B/U` (quantum Hamilton–Jacobi); its lattice conservation is the
--     wave-sector divergence theorem `CellConservation.discrete_gauss` (flux through a closed
--     boundary is zero);
--   • the IMAGINARY channel is continuity of the Born density `ρ = |ψ|² = R²`
--     (`MadelungCell.continuity_is_current_divergence`, `R·Im(cellCurv) = ∇·(R²∇φ)`).
-- `cayley_isometry` is the EXACT global form of that second channel: the unitary spinor loop's
-- conserved charge is exactly `Σ ρ = Σ |ψ|²` — the total Born density, which `StellarEquilibrium`
-- proves is the total **Being-norm** (`born_density_is_being_norm`, `|ψ|² = R²`). So the matter-sector
-- conservation and the wave-sector `discrete_gauss` are the two channels of one cell: the same
-- "closed region ⟹ conserved through the boundary" wearing the two faces of the trinity break.

open scoped BigOperators

/-- **The conserved charge of the unitary spinor loop is the total Born density.** On the lattice
    `EuclideanSpace ℂ (Fin n)`, `cayley_isometry` says the step preserves `‖ψ‖`; since
    `‖ψ‖² = ∑ᵢ ‖ψ i‖²` and each `‖ψ i‖² = ρ i` is the cell's Born density (the Being-norm `R²`,
    `StellarEquilibrium.born_density_is_being_norm`), the **total Born density `∑ᵢ ρ i` is invariant**
    under one step. This names the conserved charge: it is the matter (`R²`) the whole physics cluster
    is about — the imaginary (continuity) channel of the cell, the exact dual of the wave channel's
    `discrete_gauss`. -/
theorem cayley_conserves_born_density {n : ℕ}
    (A : EuclideanSpace ℂ (Fin n) →ₗ[ℂ] EuclideanSpace ℂ (Fin n))
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ)
    (φ ψ : EuclideanSpace ℂ (Fin n)) (hrel : φ + A φ = ψ - A ψ) :
    ∑ i, ‖φ i‖ ^ 2 = ∑ i, ‖ψ i‖ ^ 2 := by
  have hnorm : ‖φ‖ = ‖ψ‖ := cayley_isometry A hskew φ ψ hrel
  have key : ∀ x : EuclideanSpace ℂ (Fin n), ‖x‖ ^ 2 = ∑ i, ‖x i‖ ^ 2 := fun x => by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun i _ => sq_nonneg _)]
  rw [← key φ, ← key ψ, hnorm]

end UnitaryLoop

-- ═══════════════════════════════════════════════════════════════════
-- §8. THE GOLDEN REFLEXIVE FIXED POINT — φ IS x = B/U CLOSED ON ITSELF
-- ═══════════════════════════════════════════════════════════════════
-- §2 gave the AFFINE loop's rest point `x* = B/(1−A) = B/U` — the fixed point of `x ↦ A·x + B`.
-- §8 closes the loop on ITSELF: the reflexive fourth (`SanguinePath.self`, the `/∘/`), the map
-- that divides and re-adds the monad, `g(x) = 1 + 1/x = (x+1)/x`. This is `x = B/U` where the
-- numerator and denominator are BOTH the state — `B = x+1` (the successor), `U = x` (self). Its
-- rest point is the golden ratio `φ`: `φ = 1 + 1/φ`, i.e. `φ² = φ + 1`.
--
-- φ is EQUALLY the eigenvalue of the ONE fold. The Fibonacci recursor `Fₙ₊₁ = Fₙ + Fₙ₋₁` is the
-- Tier-2 spinor rotor `!![1,1;1,0]` (an `AffineStepN 2` with zero injection); its characteristic
-- polynomial is `X² − X − 1`, whose positive root is φ, with eigenvector `![φ,1]`. These are one
-- object at two levels: the fold as a LINEAR map on `(Fₙ₊₁, Fₙ)`, and its action on the RATIO
-- `Fₙ₊₁/Fₙ` as the reflexive Möbius `1 + 1/x` — because `Fₙ₊₂/Fₙ₊₁ = 1 + Fₙ/Fₙ₊₁`. The affine loop
-- rests at `B/(1−A)`; the reflexive loop rests at φ; the fold GROWS at rate φ.
--
-- Why this is the root of the golden-hash program: `hashtrinity`'s fold multiplier is
-- `round(2^W/φ)`, and `RadixDescent` uses only that it is an ODD UNIT (any invertible multiplier
-- gives injectivity/balance). The golden-SPECIFIC properties — optimal equidistribution, the
-- three-gap, the deterministic peeling — are absent because φ is treated as an arbitrary constant.
-- §8 names what φ IS: the framework's own self-reference eigenvalue. Its continued fraction is the
-- monad repeated forever, `[1;1,1,…]` (each step is `1 + 1/·`, nothing but the unit), which by
-- Hurwitz makes it the worst-approximable real ⇒ the best equidistributor. The golden hash is not
-- Knuth's borrowed Fibonacci trick; it is `x = B/U` closed on itself, discretized.

namespace Golden

open Real

/-- **φ is the reflexive fixed point of `x = B/U` (the root).** The golden ratio solves
    `x = 1 + 1/x` — the monad plus the reciprocal, `x = B/U` closed on itself (the `/∘/` reflexive
    fourth). Where `SpinorLoop.constantStep`'s AFFINE loop rests at `B/(1−A)`, the reflexive loop
    rests at φ. Equivalently `φ = (φ+1)/φ`: `B = φ+1` (the successor), `U = φ` (self). -/
theorem golden_reflexive_fixed_point : goldenRatio = 1 + 1 / goldenRatio := by
  have h0 : goldenRatio ≠ 0 := ne_of_gt goldenRatio_pos
  have hmul : goldenRatio * goldenRatio = goldenRatio + 1 := by
    rw [← sq]; exact goldenRatio_sq
  have hrhs : (1 : ℝ) + 1 / goldenRatio = (goldenRatio + 1) / goldenRatio := by
    rw [add_div, div_self h0]
  rw [hrhs, ← hmul, mul_div_assoc, div_self h0, mul_one]

/-- **φ solves the fold's characteristic equation — the eigenvalue of the one recursor.**
    `φ² − φ − 1 = 0`: φ is a root of `X² − X − 1`, the characteristic polynomial of the Fibonacci
    fold matrix `!![1,1;1,0]` (trace 1, det −1), the Tier-2 spinor rotor for `Fₙ₊₁ = Fₙ + Fₙ₋₁`
    with eigenvector `![φ,1]` (component 0 is exactly `φ² = φ+1`). φ is the rate the one fold
    grows by; the reflexive Möbius `1 + 1/x` above is this same recursor acting on the ratio
    `Fₙ₊₁/Fₙ` (since `Fₙ₊₂/Fₙ₊₁ = 1 + Fₙ/Fₙ₊₁`). -/
theorem golden_is_fold_characteristic_root : goldenRatio ^ 2 - goldenRatio - 1 = 0 := by
  have hsq : goldenRatio ^ 2 = goldenRatio + 1 := goldenRatio_sq
  linarith

/-- **The golden root, bundled.** φ is the reflexive fixed point of `x = B/U` closed on itself
    (`= 1 + 1/φ`) and the root of the one fold's characteristic polynomial (`φ² − φ − 1 = 0`, the
    Fibonacci recursor's eigenvalue). One object at two levels — the reflexive division map and the
    linear recursor — and it is what `hashtrinity`'s fold multiplier `round(2^W/φ)` is: the
    framework's self-reference eigenvalue, not an arbitrary odd unit. The root the three-gap,
    peeling, and dynamic-seed legs of the golden-hash program stand on. -/
theorem golden_is_the_reflexive_root :
    goldenRatio = 1 + 1 / goldenRatio ∧ goldenRatio ^ 2 - goldenRatio - 1 = 0 :=
  ⟨golden_reflexive_fixed_point, golden_is_fold_characteristic_root⟩

-- ── The three-gap MECHANISM, derived from the root (Area 1 of the golden-hash program) ──────────
-- The three-distance theorem: the golden rotation's N-point orbit partitions the SO(2) circle into
-- arcs of at most THREE lengths. Its inductive ENGINE is the gap-splitting identity below, and that
-- identity is φ's SPECIFIC gift: with ρ = 1/φ (the fractional part, ρ = φ−1), each gap is the sum of
-- the next two, `ρ^k = ρ^{k+1} + ρ^{k+2}`. Only the all-monads continued fraction [1;1,1,…]
-- subdivides by a single power like this — the reason φ, not an arbitrary unit, is the optimal
-- equidistributor. The full global ≤3-count is the combinatorial theorem this identity drives; here
-- we land the mechanism, straight from `golden_reflexive_fixed_point`.

/-- **The golden gap-splitting identity — the three-gap mechanism, from the root.** With `ρ = 1/φ`
    (the fractional part of φ, `ρ = φ − 1`), each power splits into the next two:
    `ρ^k = ρ^{k+1} + ρ^{k+2}`. In the golden rotation the coexisting gap sizes are consecutive powers
    of ρ, and inserting a point splits the largest gap `ρ^{k} = ρ^{k+1} + ρ^{k+2}` into the next two
    — so at most THREE distinct gap lengths ever coexist (a value `ρ^k` and its two children
    `ρ^{k+1}, ρ^{k+2}`). The "three" is the trinity, and `log₂3` bits name which gap a key falls in —
    the trit residual (`NullWeb.trit_residual`) read on the gap structure. This is φ's specific gift:
    the identity is exactly `1 + ρ = φ` (the reflexive fixed point) times `ρ^{k+1}`. The global
    ≤3-count over an arbitrary N-point orbit is the combinatorial theorem this identity is the
    inductive engine of. -/
theorem golden_gap_split (k : ℕ) :
    goldenRatio⁻¹ ^ k = goldenRatio⁻¹ ^ (k + 1) + goldenRatio⁻¹ ^ (k + 2) := by
  have h0 : goldenRatio ≠ 0 := ne_of_gt goldenRatio_pos
  have hφinv : goldenRatio⁻¹ * goldenRatio = 1 := inv_mul_cancel₀ h0
  have h1ρ : 1 + goldenRatio⁻¹ = goldenRatio := by
    rw [← one_div]; exact golden_reflexive_fixed_point.symm
  have hfactor : goldenRatio⁻¹ ^ (k + 1) + goldenRatio⁻¹ ^ (k + 2)
      = goldenRatio⁻¹ ^ (k + 1) * (1 + goldenRatio⁻¹) := by ring
  rw [hfactor, h1ρ, pow_succ, mul_assoc, hφinv, mul_one]

-- ── Closure 1: the three-gap INVARIANT — golden subdivision keeps ≤3 distinct gaps ─────────────
-- `golden_gap_split` gives the value-content of one split (`ρ^m = ρ^{m+1} + ρ^{m+2}`). The
-- three-distance theorem's global claim is that ≤3 distinct gap LENGTHS ever coexist. Its inductive
-- core is: if the present gap sizes are within three consecutive powers and you split the largest
-- (smallest exponent `m`) into its two children, you stay within three consecutive powers. That
-- span-preservation is the closure below; the geometric fact that inserting the N-th rotation point
-- splits a largest arc (so the abstract subdivision matches the real orbit) is the classical
-- remainder.

/-- The three-gap invariant: at most three distinct gap SIZES coexist — every two present exponents
    differ by ≤ 2, so the present exponents lie in a window of three consecutive integers
    (`ρ^k, ρ^{k+1}, ρ^{k+2}`). -/
def GapsWithin3 (S : Finset ℕ) : Prop := ∀ a ∈ S, ∀ b ∈ S, b ≤ a + 2

/-- **The golden subdivision preserves ≤3 gap sizes (Closure 1 core).** Inserting golden-rotation
    points splits each largest gap (smallest exponent `m`, size `ρ^m`) into its two children
    `ρ^{m+1}, ρ^{m+2}` (`golden_gap_split`: `ρ^m = ρ^{m+1} + ρ^{m+2}`). If the current gap sizes lie
    within three consecutive powers, with `m` the largest gap present (a lower bound on the
    exponents, upper bound `m+2`), then after splitting some `m`-gaps the sizes are STILL within
    three consecutive powers. So the golden rotation never shows more than three distinct gap
    lengths — the three-distance theorem's combinatorial invariant, driven by the root's gap-split.
    (That point-insertion realizes exactly this split of a largest arc is the classical remainder.) -/
theorem golden_subdivision_preserves_three (S : Finset ℕ) (m : ℕ)
    (hlb : ∀ x ∈ S, m ≤ x) (hub : ∀ x ∈ S, x ≤ m + 2) :
    GapsWithin3 (insert (m + 1) (insert (m + 2) S)) := by
  have hmem : ∀ y ∈ insert (m + 1) (insert (m + 2) S), m ≤ y ∧ y ≤ m + 2 := by
    intro y hy
    simp only [Finset.mem_insert] at hy
    rcases hy with rfl | rfl | hyS
    · omega
    · omega
    · exact ⟨hlb y hyS, hub y hyS⟩
  intro a ha b hb
  obtain ⟨ham, _⟩ := hmem a ha
  obtain ⟨_, hbM⟩ := hmem b hb
  omega

-- ── Closure 2: the three-distance theorem at Fibonacci counts — real arcs of the real orbit ─────
-- The N = fib(n+2) points {iρ mod 1 : i < N} of the golden rotation partition the circle into arcs
-- of exactly two lengths, ρ^{n+1} and ρ^n. This is the two-length case of the classical
-- three-distance theorem (Steinhaus; Sós 1958, Świerczkowski 1959): for α = ρ = [0;1,1,1,…] the
-- convergent denominators are q_k = fib(k+1), the return distances are η_k = (−1)^k(q_k·α − p_k)
-- = ρ^{k+1}, and at n = fib(m)−1 the third length's count q_k−(r+1) is 0 (Alessandri–Berthé,
-- "Three distance theorems and combinatorics on words", Enseign. Math. 44 (1998), §3).
-- Mechanism, proved below: fract(i·ρ) = (slot + (−1)^{n+1}·i·ρ^{n+2})/fib(n+2) with
-- slot = i·fib(n+1) mod fib(n+2) — the orbit is the rational lattice j/fib(n+2) perturbed by less
-- than one lattice cell (the Binet remainder fib(n+2)·ρ = fib(n+1) + (−1)^{n+1}·ρ^{n+2}), so slot
-- order is circle order; adjacent slots differ in index by the Cassini stride, forcing the gap to
-- be ρ^{n+1} or ρ^n. The tiling identity fib(n+1)·ρ^n + fib(n)·ρ^{n+1} = 1 (induction engine:
-- golden_gap_split) is the census the classical statement asserts — fib(n+1) arcs of ρ^n and
-- fib(n) arcs of ρ^{n+1} fill the circle; the multiplicities themselves are not tracked here.
-- The general (non-Fibonacci) N — three coexisting lengths, largest = sum of the other two via
-- golden_gap_split — is not proved here.

/-- **The circle tiles exactly.** `fib(n+1)·ρ^n + fib(n)·ρ^(n+1) = 1`: the two golden gap lengths,
    with Fibonacci multiplicities, sum to the full circle. Induction step is `golden_gap_split`. -/
theorem golden_tiling (n : ℕ) :
    (Nat.fib (n + 1) : ℝ) * goldenRatio⁻¹ ^ n
      + (Nat.fib n : ℝ) * goldenRatio⁻¹ ^ (n + 1) = 1 := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hsplit := golden_gap_split n
    have hfib : (Nat.fib (n + 2) : ℝ) = Nat.fib n + Nat.fib (n + 1) := by
      rw [Nat.fib_add_two]; push_cast; ring
    rw [hfib]
    linear_combination ih - (Nat.fib (n + 1) : ℝ) * hsplit

/-- Companion tiling: `fib(n+2)·ρ^n − fib(n)·ρ^(n+2) = 1` — the same census read across one
    `golden_gap_split`. -/
theorem golden_tiling_sub (n : ℕ) :
    (Nat.fib (n + 2) : ℝ) * goldenRatio⁻¹ ^ n
      - (Nat.fib n : ℝ) * goldenRatio⁻¹ ^ (n + 2) = 1 := by
  have h := golden_tiling (n + 1)
  have hsplit := golden_gap_split n
  have hfib : (Nat.fib (n + 2) : ℝ) = Nat.fib n + Nat.fib (n + 1) := by
    rw [Nat.fib_add_two]; push_cast; ring
  rw [hfib] at h ⊢
  linear_combination h + ((Nat.fib n : ℝ) + (Nat.fib (n + 1) : ℝ)) * hsplit

/-- **The Binet remainder.** `fib(n+1)·ρ = fib(n) + (−1)^n·ρ^(n+1)`: the orbit of a Fibonacci
    multiple sits at distance `ρ^(n+1)` from the integer `fib(n)`, alternating sides. Rides
    Mathlib's `Real.fib_succ_sub_goldenRatio_mul_fib` (Binet) via `ψ = −ρ`. -/
theorem golden_fib_step (n : ℕ) :
    (Nat.fib (n + 1) : ℝ) * goldenRatio⁻¹
      = Nat.fib n + (-1) ^ n * goldenRatio⁻¹ ^ (n + 1) := by
  have h := fib_succ_sub_goldenRatio_mul_fib n
  rw [show goldenConj = -goldenRatio⁻¹ by linarith [inv_goldenRatio], neg_pow] at h
  have hone : goldenRatio * goldenRatio⁻¹ = 1 := mul_inv_cancel₀ goldenRatio_ne_zero
  linear_combination goldenRatio⁻¹ * h + (Nat.fib n : ℝ) * hone

/-- The perturbation is under one lattice cell: `fib(n)·ρ^n < 1`. -/
theorem golden_fib_pow_lt_one (n : ℕ) : (Nat.fib n : ℝ) * goldenRatio⁻¹ ^ n < 1 := by
  cases n with
  | zero => simp
  | succ k =>
    have hρ0 : (0 : ℝ) < goldenRatio⁻¹ := inv_pos.2 goldenRatio_pos
    have hρ1 : goldenRatio⁻¹ < 1 := inv_lt_one_of_one_lt₀ one_lt_goldenRatio
    have ht := golden_tiling k
    have hfibpos : (0 : ℝ) < (Nat.fib (k + 1) : ℝ) := by
      exact_mod_cast Nat.fib_pos.2 k.succ_pos
    have hpow : goldenRatio⁻¹ ^ (k + 1) < goldenRatio⁻¹ ^ k := by
      rw [pow_succ]
      nlinarith [pow_pos hρ0 k]
    have h2 : (0 : ℝ) ≤ (Nat.fib k : ℝ) * goldenRatio⁻¹ ^ (k + 1) := by positivity
    nlinarith [mul_lt_mul_of_pos_left hpow hfibpos]

/-- Cassini's identity over `ℤ`: `fib(n+1)² − fib(n)·fib(n+2) = (−1)^n`. -/
theorem fib_cassini_int (n : ℕ) :
    ((Nat.fib (n + 1) : ℤ)) ^ 2 - (Nat.fib n : ℤ) * (Nat.fib (n + 2) : ℤ) = (-1) ^ n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have h1 : (Nat.fib (k + 2) : ℤ) = Nat.fib k + Nat.fib (k + 1) := by
      rw [Nat.fib_add_two]; push_cast; ring
    have h2 : (Nat.fib (k + 3) : ℤ) = Nat.fib (k + 1) + Nat.fib (k + 2) := by
      rw [show k + 3 = (k + 1) + 2 from rfl, Nat.fib_add_two]; push_cast; ring
    rw [pow_succ]
    linear_combination (-1 : ℤ) * ih + (Nat.fib (k + 2) : ℤ) * h1 - (Nat.fib (k + 1) : ℤ) * h2

/-- Perturbation bounds: for `i < fib(n+2)`, `0 ≤ i·ρ^(n+2) < 1`. -/
theorem golden_orbit_perturb (n : ℕ) {i : ℕ} (hi : i < Nat.fib (n + 2)) :
    0 ≤ (i : ℝ) * goldenRatio⁻¹ ^ (n + 2) ∧ (i : ℝ) * goldenRatio⁻¹ ^ (n + 2) < 1 := by
  refine ⟨by positivity, ?_⟩
  have hb := golden_fib_pow_lt_one (n + 2)
  have hiN : (i : ℝ) < (Nat.fib (n + 2) : ℝ) := by exact_mod_cast hi
  have hpow0 : (0 : ℝ) < goldenRatio⁻¹ ^ (n + 2) := pow_pos (inv_pos.2 goldenRatio_pos) _
  have := mul_lt_mul_of_pos_right hiN hpow0
  linarith

/-- The slot map is injective below the modulus: `i·fib(n+1) mod fib(n+2)` determines `i`. -/
theorem golden_slot_inj (n : ℕ) {i j : ℕ}
    (hi : i < Nat.fib (n + 2)) (hj : j < Nat.fib (n + 2))
    (hs : i * Nat.fib (n + 1) % Nat.fib (n + 2) = j * Nat.fib (n + 1) % Nat.fib (n + 2)) :
    i = j := by
  have hcop : Nat.Coprime (Nat.fib (n + 2)) (Nat.fib (n + 1)) :=
    (Nat.fib_coprime_fib_succ (n + 1)).symm
  have h := Nat.ModEq.cancel_right_of_coprime hcop hs
  unfold Nat.ModEq at h
  rwa [Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj] at h

/-- The slot map is surjective onto `range (fib(n+2))`: every lattice slot is hit by an orbit
    index. -/
theorem golden_slot_surj (n : ℕ) {k : ℕ} (hk : k < Nat.fib (n + 2)) :
    ∃ i, i < Nat.fib (n + 2) ∧ i * Nat.fib (n + 1) % Nat.fib (n + 2) = k := by
  classical
  have hN0 : 0 < Nat.fib (n + 2) := Nat.fib_pos.2 (by omega)
  have himg : (Finset.range (Nat.fib (n + 2))).image
      (fun i => i * Nat.fib (n + 1) % Nat.fib (n + 2)) = Finset.range (Nat.fib (n + 2)) := by
    apply Finset.eq_of_subset_of_card_le
    · intro x hx
      obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hx
      exact Finset.mem_range.2 (Nat.mod_lt _ hN0)
    · rw [Finset.card_image_of_injOn (fun a ha b hb hab =>
        golden_slot_inj n (Finset.mem_range.1 ha) (Finset.mem_range.1 hb) hab)]
  obtain ⟨i, hiR, hik⟩ := Finset.mem_image.1 (himg ▸ Finset.mem_range.2 hk)
  exact ⟨i, Finset.mem_range.1 hiR, hik⟩

/-- **The slot decomposition of the golden orbit.** For `i < fib(n+2)`,
    `fract(i·ρ) = (i·fib(n+1) mod fib(n+2) + (−1)^(n+1)·i·ρ^(n+2)) / fib(n+2)` — each orbit point
    is its rational-lattice slot plus a same-signed perturbation smaller than one cell. -/
theorem golden_orbit_fract (n : ℕ) {i : ℕ} (hi : i < Nat.fib (n + 2)) :
    Int.fract ((i : ℝ) * goldenRatio⁻¹)
      = (((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
          + (-1) ^ (n + 1) * (i : ℝ) * goldenRatio⁻¹ ^ (n + 2)) / (Nat.fib (n + 2) : ℝ) := by
  have hN0 : 0 < Nat.fib (n + 2) := Nat.fib_pos.2 (by omega)
  have hNR : (0 : ℝ) < (Nat.fib (n + 2) : ℝ) := by exact_mod_cast hN0
  have hstep := golden_fib_step (n + 1)
  have hmod := Nat.div_add_mod (i * Nat.fib (n + 1)) (Nat.fib (n + 2))
  have hsN : i * Nat.fib (n + 1) % Nat.fib (n + 2) < Nat.fib (n + 2) := Nat.mod_lt _ hN0
  obtain ⟨hper0, hper1⟩ := golden_orbit_perturb n hi
  have hs1 : i = 0 ∨ 1 ≤ i * Nat.fib (n + 1) % Nat.fib (n + 2) := by
    rcases Nat.eq_zero_or_pos i with h0 | hpos
    · exact Or.inl h0
    · refine Or.inr ?_
      rcases Nat.eq_zero_or_pos (i * Nat.fib (n + 1) % Nat.fib (n + 2)) with h0s | h1s
      · exfalso
        have hdvd : Nat.fib (n + 2) ∣ i * Nat.fib (n + 1) := Nat.dvd_of_mod_eq_zero h0s
        have hcop : Nat.Coprime (Nat.fib (n + 2)) (Nat.fib (n + 1)) :=
          (Nat.fib_coprime_fib_succ (n + 1)).symm
        have := Nat.le_of_dvd hpos (hcop.dvd_of_dvd_mul_right hdvd)
        omega
      · exact h1s
  have hmodR : (Nat.fib (n + 2) : ℝ) * (i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ)
      + ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) = (i : ℝ) * (Nat.fib (n + 1) : ℝ) := by
    exact_mod_cast hmod
  rw [Int.fract_eq_iff]
  refine ⟨?_, ?_, ⟨(i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ), ?_⟩⟩
  · -- 0 ≤ slot + sign·perturbation, divided by N
    apply div_nonneg _ hNR.le
    rcases Nat.even_or_odd n with he | ho
    · rw [(he.add_one).neg_one_pow]
      rcases hs1 with rfl | h1s
      · simp
      · have : (1 : ℝ) ≤ ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) := by
          exact_mod_cast h1s
        nlinarith
    · rw [(ho.add_one).neg_one_pow]
      have : (0 : ℝ) ≤ ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) := by positivity
      nlinarith
  · -- < 1
    rw [div_lt_one hNR]
    have hsR : ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) + 1 ≤ (Nat.fib (n + 2) : ℝ) := by
      exact_mod_cast hsN
    rcases Nat.even_or_odd n with he | ho
    · rw [(he.add_one).neg_one_pow]
      nlinarith
    · rw [(ho.add_one).neg_one_pow]
      nlinarith
  · -- the integer witness: i·ρ − value = i·fib(n+1) / fib(n+2) (Nat division)
    have key : (i : ℝ) * goldenRatio⁻¹ * (Nat.fib (n + 2) : ℝ)
        = ((i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ) : ℝ) * (Nat.fib (n + 2) : ℝ)
          + (((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
             + (-1) ^ (n + 1) * (i : ℝ) * goldenRatio⁻¹ ^ (n + 2)) := by
      linear_combination (i : ℝ) * hstep - hmodR
    have hdivval : (((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
        + (-1) ^ (n + 1) * (i : ℝ) * goldenRatio⁻¹ ^ (n + 2)) / (Nat.fib (n + 2) : ℝ)
        = (i : ℝ) * goldenRatio⁻¹ - ((i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ) : ℝ) := by
      rw [div_eq_iff hNR.ne']
      linear_combination (-1 : ℝ) * key
    rw [Int.cast_natCast]
    linarith [hdivval]

/-- **Slot order is circle order.** A smaller slot means a smaller orbit point: the perturbations
    share one sign and never reach one lattice cell. -/
theorem golden_orbit_slot_mono (n : ℕ) {i j : ℕ}
    (hi : i < Nat.fib (n + 2)) (hj : j < Nat.fib (n + 2))
    (hs : i * Nat.fib (n + 1) % Nat.fib (n + 2) < j * Nat.fib (n + 1) % Nat.fib (n + 2)) :
    Int.fract ((i : ℝ) * goldenRatio⁻¹) < Int.fract ((j : ℝ) * goldenRatio⁻¹) := by
  have hNR : (0 : ℝ) < (Nat.fib (n + 2) : ℝ) := by
    exact_mod_cast Nat.fib_pos.2 (show 0 < n + 2 by omega)
  rw [golden_orbit_fract n hi, golden_orbit_fract n hj]
  obtain ⟨hip0, hip1⟩ := golden_orbit_perturb n hi
  obtain ⟨hjp0, hjp1⟩ := golden_orbit_perturb n hj
  have hsR : ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) + 1
      ≤ ((j * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) := by
    exact_mod_cast hs
  have key : ∀ a b : ℝ, a < b →
      a / (Nat.fib (n + 2) : ℝ) < b / (Nat.fib (n + 2) : ℝ) := fun a b hab => by
    have h := mul_lt_mul_of_pos_right hab (inv_pos.2 hNR)
    simpa [div_eq_mul_inv] using h
  apply key
  rcases Nat.even_or_odd n with he | ho
  · rw [(he.add_one).neg_one_pow]; nlinarith
  · rw [(ho.add_one).neg_one_pow]; nlinarith

/-- **The Cassini stride.** If `d·fib(n+1) ≡ 1 (mod fib(n+2))` with `|d| < fib(n+2)`, then the
    signed stride `(−1)^(n+1)·d` is `−fib(n+1)` or `fib(n)`. Consecutive slots differ in orbit
    index by exactly one of two strides — the two gap lengths, before any real number appears. -/
theorem golden_slot_step (n : ℕ) {d : ℤ}
    (hlo : -(Nat.fib (n + 2) : ℤ) < d) (hhi : d < (Nat.fib (n + 2) : ℤ))
    (hdvd : (Nat.fib (n + 2) : ℤ) ∣ d * (Nat.fib (n + 1) : ℤ) - 1) :
    (-1 : ℤ) ^ (n + 1) * d = -(Nat.fib (n + 1) : ℤ)
    ∨ (-1 : ℤ) ^ (n + 1) * d = (Nat.fib n : ℤ) := by
  have hcassini := fib_cassini_int n
  have hNF : (Nat.fib n : ℤ) + (Nat.fib (n + 1) : ℤ) = (Nat.fib (n + 2) : ℤ) := by
    rw [Nat.fib_add_two]; push_cast; ring
  have hF1 : (1 : ℤ) ≤ (Nat.fib (n + 1) : ℤ) := by
    exact_mod_cast Nat.fib_pos.2 n.succ_pos
  have hFn0 : (0 : ℤ) ≤ (Nat.fib n : ℤ) := by positivity
  have hsq : ((-1 : ℤ)) ^ n * ((-1 : ℤ)) ^ n = 1 := by
    rcases Nat.even_or_odd n with h | h
    · rw [h.neg_one_pow]; ring
    · rw [h.neg_one_pow]; ring
  have h2 : (Nat.fib (n + 2) : ℤ) ∣ (d - (-1) ^ n * (Nat.fib (n + 1) : ℤ)) * (Nat.fib (n + 1) : ℤ) := by
    obtain ⟨c, hc⟩ := hdvd
    refine ⟨c - (-1) ^ n * (Nat.fib n : ℤ), ?_⟩
    linear_combination hc - (-1 : ℤ) ^ n * hcassini - hsq
  have hcop : IsCoprime ((Nat.fib (n + 2) : ℤ)) ((Nat.fib (n + 1) : ℤ)) := by
    rw [Int.isCoprime_iff_gcd_eq_one, Int.gcd_natCast_natCast]
    exact (Nat.fib_coprime_fib_succ (n + 1)).symm
  obtain ⟨c, hc⟩ := hcop.dvd_of_dvd_mul_right h2
  have hN0 : (0 : ℤ) < (Nat.fib (n + 2) : ℤ) := by linarith
  rcases Nat.even_or_odd n with he | ho
  · rw [he.neg_one_pow] at hc
    rw [(he.add_one).neg_one_pow]
    -- d − fib(n+1) = N·c with −N < d < N and 1 ≤ fib(n+1) ≤ N: c ∈ {0, −1}
    have hcb : c = 0 ∨ c = -1 := by
      have hub : (Nat.fib (n + 2) : ℤ) * c < (Nat.fib (n + 2) : ℤ) * 1 := by nlinarith
      have hlb : (Nat.fib (n + 2) : ℤ) * (-2) < (Nat.fib (n + 2) : ℤ) * c := by nlinarith
      have h1 : c < 1 := lt_of_mul_lt_mul_left hub hN0.le
      have h2' : (-2 : ℤ) < c := lt_of_mul_lt_mul_left hlb hN0.le
      omega
    rcases hcb with rfl | rfl
    · left; rw [mul_zero] at hc; linarith
    · right; rw [mul_neg_one] at hc; linarith
  · rw [ho.neg_one_pow] at hc
    rw [(ho.add_one).neg_one_pow]
    -- d + fib(n+1) = N·c with −N < d < N and 1 ≤ fib(n+1) ≤ N: c ∈ {0, 1}
    have hcb : c = 0 ∨ c = 1 := by
      have hub : (Nat.fib (n + 2) : ℤ) * c < (Nat.fib (n + 2) : ℤ) * 2 := by nlinarith
      have hlb : (Nat.fib (n + 2) : ℤ) * (-1) < (Nat.fib (n + 2) : ℤ) * c := by nlinarith
      have h1 : c < 2 := lt_of_mul_lt_mul_left hub hN0.le
      have h2' : (-1 : ℤ) < c := lt_of_mul_lt_mul_left hlb hN0.le
      omega
    rcases hcb with rfl | rfl
    · left; rw [mul_zero] at hc; linarith
    · right; rw [mul_one] at hc; linarith

/-- **The interior gap.** Orbit points in consecutive slots differ by exactly `ρ^(n+1)` or
    `ρ^n` — the two arc lengths of the `fib(n+2)`-point golden configuration. -/
theorem golden_gap_succ (n : ℕ) {i j : ℕ}
    (hi : i < Nat.fib (n + 2)) (hj : j < Nat.fib (n + 2))
    (hs : j * Nat.fib (n + 1) % Nat.fib (n + 2) = i * Nat.fib (n + 1) % Nat.fib (n + 2) + 1) :
    Int.fract ((j : ℝ) * goldenRatio⁻¹) - Int.fract ((i : ℝ) * goldenRatio⁻¹)
      = goldenRatio⁻¹ ^ (n + 1)
    ∨ Int.fract ((j : ℝ) * goldenRatio⁻¹) - Int.fract ((i : ℝ) * goldenRatio⁻¹)
      = goldenRatio⁻¹ ^ n := by
  have hN0 : 0 < Nat.fib (n + 2) := Nat.fib_pos.2 (by omega)
  have hNR : (0 : ℝ) < (Nat.fib (n + 2) : ℝ) := by exact_mod_cast hN0
  -- the index difference d = j − i satisfies d·fib(n+1) ≡ 1 (mod fib(n+2))
  have hjm : (Nat.fib (n + 2) : ℤ) * (j * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ)
      + ((j * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ) = (j : ℤ) * (Nat.fib (n + 1) : ℤ) := by
    exact_mod_cast Nat.div_add_mod (j * Nat.fib (n + 1)) (Nat.fib (n + 2))
  have him : (Nat.fib (n + 2) : ℤ) * (i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ)
      + ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ) = (i : ℤ) * (Nat.fib (n + 1) : ℤ) := by
    exact_mod_cast Nat.div_add_mod (i * Nat.fib (n + 1)) (Nat.fib (n + 2))
  have hsZ : ((j * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ)
      = ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ) + 1 := by exact_mod_cast hs
  have hdvd : (Nat.fib (n + 2) : ℤ) ∣ ((j : ℤ) - (i : ℤ)) * (Nat.fib (n + 1) : ℤ) - 1 :=
    ⟨(j * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ) - (i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ),
      by linear_combination him - hjm + hsZ⟩
  have hlo : -(Nat.fib (n + 2) : ℤ) < (j : ℤ) - (i : ℤ) := by
    have : (i : ℤ) < (Nat.fib (n + 2) : ℤ) := by exact_mod_cast hi
    have : (0 : ℤ) ≤ (j : ℤ) := Int.natCast_nonneg j
    omega
  have hhi' : (j : ℤ) - (i : ℤ) < (Nat.fib (n + 2) : ℤ) := by
    have : (j : ℤ) < (Nat.fib (n + 2) : ℤ) := by exact_mod_cast hj
    have : (0 : ℤ) ≤ (i : ℤ) := Int.natCast_nonneg i
    omega
  have hstep := golden_slot_step n hlo hhi' hdvd
  rw [golden_orbit_fract n hi, golden_orbit_fract n hj, div_sub_div_same]
  have hsR : ((j * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
      = ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) + 1 := by exact_mod_cast hs
  rcases hstep with hd | hd
  · left
    have hdR : (-1 : ℝ) ^ (n + 1) * ((j : ℝ) - (i : ℝ)) = -(Nat.fib (n + 1) : ℝ) := by
      have := congrArg (Int.cast : ℤ → ℝ) hd
      push_cast at this
      linarith
    have ht := golden_tiling (n + 1)
    rw [div_eq_iff hNR.ne']
    linear_combination hsR + goldenRatio⁻¹ ^ (n + 2) * hdR - ht
  · right
    have hdR : (-1 : ℝ) ^ (n + 1) * ((j : ℝ) - (i : ℝ)) = (Nat.fib n : ℝ) := by
      have := congrArg (Int.cast : ℤ → ℝ) hd
      push_cast at this
      linarith
    have ht := golden_tiling_sub n
    rw [div_eq_iff hNR.ne']
    linear_combination hsR + goldenRatio⁻¹ ^ (n + 2) * hdR - ht

/-- **The wraparound gap.** The arc from the top orbit point (slot `fib(n+2)−1`) back to `0` is
    also `ρ^(n+1)` or `ρ^n`: the circle closes on the same two lengths. -/
theorem golden_gap_wrap (n : ℕ) {i : ℕ} (hi : i < Nat.fib (n + 2))
    (hs : i * Nat.fib (n + 1) % Nat.fib (n + 2) + 1 = Nat.fib (n + 2)) :
    1 - Int.fract ((i : ℝ) * goldenRatio⁻¹) = goldenRatio⁻¹ ^ (n + 1)
    ∨ 1 - Int.fract ((i : ℝ) * goldenRatio⁻¹) = goldenRatio⁻¹ ^ n := by
  have hN0 : 0 < Nat.fib (n + 2) := Nat.fib_pos.2 (by omega)
  have hNR : (0 : ℝ) < (Nat.fib (n + 2) : ℝ) := by exact_mod_cast hN0
  have him : (Nat.fib (n + 2) : ℤ) * (i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ)
      + ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ) = (i : ℤ) * (Nat.fib (n + 1) : ℤ) := by
    exact_mod_cast Nat.div_add_mod (i * Nat.fib (n + 1)) (Nat.fib (n + 2))
  have hsZ : ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℤ) + 1 = (Nat.fib (n + 2) : ℤ) := by
    exact_mod_cast hs
  have hdvd : (Nat.fib (n + 2) : ℤ) ∣ (-(i : ℤ)) * (Nat.fib (n + 1) : ℤ) - 1 :=
    ⟨-(i * Nat.fib (n + 1) / Nat.fib (n + 2) : ℕ) - 1, by linear_combination him - hsZ⟩
  have hlo : -(Nat.fib (n + 2) : ℤ) < -(i : ℤ) := by
    have : (i : ℤ) < (Nat.fib (n + 2) : ℤ) := by exact_mod_cast hi
    omega
  have hhi' : -(i : ℤ) < (Nat.fib (n + 2) : ℤ) := by
    have : (0 : ℤ) ≤ (i : ℤ) := Int.natCast_nonneg i
    omega
  have hstep := golden_slot_step n hlo hhi' hdvd
  rw [golden_orbit_fract n hi]
  have hsR : ((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ) + 1 = (Nat.fib (n + 2) : ℝ) := by
    exact_mod_cast hs
  rcases hstep with hd | hd
  · left
    have hdR : (-1 : ℝ) ^ (n + 1) * (-(i : ℝ)) = -(Nat.fib (n + 1) : ℝ) := by
      have := congrArg (Int.cast : ℤ → ℝ) hd
      push_cast at this
      linarith
    have ht := golden_tiling (n + 1)
    have hval : (((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
        + (-1) ^ (n + 1) * (i : ℝ) * goldenRatio⁻¹ ^ (n + 2)) / (Nat.fib (n + 2) : ℝ)
        = 1 - goldenRatio⁻¹ ^ (n + 1) := by
      rw [div_eq_iff hNR.ne']
      linear_combination hsR - goldenRatio⁻¹ ^ (n + 2) * hdR + ht
    linarith [hval]
  · right
    have hdR : (-1 : ℝ) ^ (n + 1) * (-(i : ℝ)) = (Nat.fib n : ℝ) := by
      have := congrArg (Int.cast : ℤ → ℝ) hd
      push_cast at this
      linarith
    have ht := golden_tiling_sub n
    have hval : (((i * Nat.fib (n + 1) % Nat.fib (n + 2) : ℕ) : ℝ)
        + (-1) ^ (n + 1) * (i : ℝ) * goldenRatio⁻¹ ^ (n + 2)) / (Nat.fib (n + 2) : ℝ)
        = 1 - goldenRatio⁻¹ ^ n := by
      rw [div_eq_iff hNR.ne']
      linear_combination hsR - goldenRatio⁻¹ ^ (n + 2) * hdR + ht
    linarith [hval]

/-- **The three-distance theorem at Fibonacci counts (two-length case), for the golden rotation.**
    Adjacent points of the `fib(n+2)`-point golden orbit `{i·ρ mod 1 : i < fib(n+2)}` — no orbit
    point strictly between — differ by exactly `ρ^(n+1)` or `ρ^n`. With `golden_gap_split`
    (`ρ^n = ρ^(n+1) + ρ^(n+2)`) these are the parent and one child of the three-distance triple:
    at Fibonacci counts the third length's multiplicity is zero (Sós 1958; Świerczkowski 1959;
    Alessandri–Berthé 1998 §3, specialised to α = ρ, q_k = fib(k+1), η_k = ρ^(k+1)). The general
    non-Fibonacci count — three coexisting lengths — is not proved here. -/
theorem golden_three_distance_fib (n : ℕ) {x y : ℝ}
    (hx : x ∈ (Finset.range (Nat.fib (n + 2))).image
      (fun i : ℕ => Int.fract ((i : ℝ) * goldenRatio⁻¹)))
    (hy : y ∈ (Finset.range (Nat.fib (n + 2))).image
      (fun i : ℕ => Int.fract ((i : ℝ) * goldenRatio⁻¹)))
    (hxy : x < y)
    (hadj : ∀ z ∈ (Finset.range (Nat.fib (n + 2))).image
      (fun i : ℕ => Int.fract ((i : ℝ) * goldenRatio⁻¹)), z ≤ x ∨ y ≤ z) :
    y - x = goldenRatio⁻¹ ^ (n + 1) ∨ y - x = goldenRatio⁻¹ ^ n := by
  obtain ⟨i, hiR, rfl⟩ := Finset.mem_image.1 hx
  obtain ⟨j, hjR, rfl⟩ := Finset.mem_image.1 hy
  rw [Finset.mem_range] at hiR hjR
  have hN0 : 0 < Nat.fib (n + 2) := Nat.fib_pos.2 (by omega)
  have hslot : i * Nat.fib (n + 1) % Nat.fib (n + 2) < j * Nat.fib (n + 1) % Nat.fib (n + 2) := by
    rcases lt_trichotomy (i * Nat.fib (n + 1) % Nat.fib (n + 2))
        (j * Nat.fib (n + 1) % Nat.fib (n + 2)) with h | h | h
    · exact h
    · exfalso
      obtain rfl : i = j := golden_slot_inj n hiR hjR h
      exact lt_irrefl _ hxy
    · exfalso
      have := golden_orbit_slot_mono n hjR hiR h
      linarith
  rcases eq_or_lt_of_le (Nat.succ_le_of_lt hslot) with hsucc | hgap
  · exact golden_gap_succ n hiR hjR hsucc.symm
  · -- a slot strictly between would put an orbit point strictly between x and y
    exfalso
    have hsjN : j * Nat.fib (n + 1) % Nat.fib (n + 2) < Nat.fib (n + 2) := Nat.mod_lt _ hN0
    obtain ⟨k, hkN, hk⟩ := golden_slot_surj n
      (show i * Nat.fib (n + 1) % Nat.fib (n + 2) + 1 < Nat.fib (n + 2) by omega)
    have h1 : Int.fract ((i : ℝ) * goldenRatio⁻¹) < Int.fract ((k : ℝ) * goldenRatio⁻¹) :=
      golden_orbit_slot_mono n hiR hkN (by omega)
    have h2 : Int.fract ((k : ℝ) * goldenRatio⁻¹) < Int.fract ((j : ℝ) * goldenRatio⁻¹) :=
      golden_orbit_slot_mono n hkN hjR (by omega)
    have hkmem : Int.fract ((k : ℝ) * goldenRatio⁻¹)
        ∈ (Finset.range (Nat.fib (n + 2))).image
          (fun i : ℕ => Int.fract ((i : ℝ) * goldenRatio⁻¹)) :=
      Finset.mem_image.2 ⟨k, Finset.mem_range.2 hkN, rfl⟩
    rcases hadj _ hkmem with h | h <;> linarith

end Golden

-- ═══════════════════════════════════════════════════════════════════
-- §9. THE AITKEN EXACT RECOVERY — three iterates determine the fixed point, no A, no B, no loop
-- ═══════════════════════════════════════════════════════════════════
-- §1-§2 gave the closed form `x* = B/(1-A)` — but only as a CHARACTERIZATION: `constantStep_fixed_
-- point` says "x IS the fixed point IFF x = B/(1-A)", which requires already knowing A and B to
-- evaluate. This section proves the OPERATIONAL fact `SpinorLoop`'s own thesis demands: you never
-- need A or B, and you never need to run the loop to convergence, because the constant loop's error
-- is a PURE geometric sequence (`iterate_sub_fixed_point`) — and three points determine a geometric
-- sequence completely. Aitken's Δ² extrapolation, applied to ANY three consecutive iterates, recovers
-- x* EXACTLY (not asymptotically — an identity). This is the concrete answer to "does a 24-round
-- power-iteration loop (PageRank's `authority`) need more cores, or does the loop itself not need to
-- run": for a CONSTANT affine recurrence (same transition every round — exactly PageRank's shape,
-- `AffineStepN`'s matrix case with fixed damping and floor), the round count is not a physical
-- requirement, it is an unexamined default. Real PageRank sits ON this fact only approximately (its
-- error is a SUM of several geometric modes, one per eigenvalue of M, not a pure single mode) — this
-- section proves the exact case (§1's scalar Tier-1, and by the true single-dominant-mode limit,
-- what acceleration methods for PageRank are already known to exploit in practice), not the general
-- multi-mode case, honestly scoped to what is actually proved.

namespace AffineStep

/-- The forward iterate of a constant affine step from a seed `S₀` — literally running the loop `n`
    times. The baseline every acceleration is measured against. -/
def iterate (step : AffineStep) (S0 : ℝ) : ℕ → ℝ
  | 0 => S0
  | n + 1 => step.apply (iterate step S0 n)

/-- **The error is a PURE geometric sequence.** `iterate n - x* = A^n · (S₀ - x*)`, the same `A` at
    every step because the step is constant. This is the one structural fact underneath everything
    below — a constant affine loop does not merely converge, its distance to the fixed point is
    exactly `A` raised to the step count, times the initial displacement. -/
theorem iterate_sub_fixed_point (step : AffineStep) (hA : step.rotor_decay ≠ 1) (S0 : ℝ) (n : ℕ) :
    iterate step S0 n - step.injection / (1 - step.rotor_decay)
      = step.rotor_decay ^ n * (S0 - step.injection / (1 - step.rotor_decay)) := by
  have hne : (1 - step.rotor_decay) ≠ 0 := fun h => hA (by linarith)
  induction n with
  | zero => simp [iterate]
  | succ n ih =>
    have hxstar : step.rotor_decay * (step.injection / (1 - step.rotor_decay)) + step.injection
        = step.injection / (1 - step.rotor_decay) := by field_simp; ring
    simp only [iterate, apply]
    calc step.rotor_decay * iterate step S0 n + step.injection
            - step.injection / (1 - step.rotor_decay)
        = step.rotor_decay * (iterate step S0 n - step.injection / (1 - step.rotor_decay))
            + (step.rotor_decay * (step.injection / (1 - step.rotor_decay)) + step.injection
               - step.injection / (1 - step.rotor_decay)) := by ring
      _ = step.rotor_decay * (step.rotor_decay ^ n
              * (S0 - step.injection / (1 - step.rotor_decay))) := by rw [ih, hxstar]; ring
      _ = step.rotor_decay ^ (n + 1) * (S0 - step.injection / (1 - step.rotor_decay)) := by
            rw [pow_succ]; ring

/-- **The second difference is the error, rescaled by `(A-1)²` — an identity, no hypothesis beyond
    the fixed-point one.** `iterate(n+2) - 2·iterate(n+1) + iterate n = (A-1)² · A^n · (S₀ - x*)`.
    This is the quantity Aitken's Δ² divides by; it names exactly when that division is safe. -/
theorem second_difference (step : AffineStep) (hA : step.rotor_decay ≠ 1) (S0 : ℝ) (n : ℕ) :
    iterate step S0 (n + 2) - 2 * iterate step S0 (n + 1) + iterate step S0 n
      = (step.rotor_decay - 1) ^ 2 * step.rotor_decay ^ n
          * (S0 - step.injection / (1 - step.rotor_decay)) := by
  have h0 := iterate_sub_fixed_point step hA S0 n
  have h1 := iterate_sub_fixed_point step hA S0 (n + 1)
  have h2 := iterate_sub_fixed_point step hA S0 (n + 2)
  have e0 : iterate step S0 n
      = step.rotor_decay ^ n * (S0 - step.injection / (1 - step.rotor_decay))
          + step.injection / (1 - step.rotor_decay) := by linarith
  have e1 : iterate step S0 (n + 1)
      = step.rotor_decay ^ (n + 1) * (S0 - step.injection / (1 - step.rotor_decay))
          + step.injection / (1 - step.rotor_decay) := by linarith
  have e2 : iterate step S0 (n + 2)
      = step.rotor_decay ^ (n + 2) * (S0 - step.injection / (1 - step.rotor_decay))
          + step.injection / (1 - step.rotor_decay) := by linarith
  rw [e0, e1, e2, pow_succ, pow_succ]
  ring

/-- **Aitken's Δ² extrapolation** of three consecutive values — the classical acceleration formula,
    stated with no reference to any recurrence: purely a function of three numbers. -/
noncomputable def aitken (x0 x1 x2 : ℝ) : ℝ := x0 - (x1 - x0) ^ 2 / (x2 - 2 * x1 + x0)

/-- **THE HEADLINE: three consecutive iterates determine the fixed point EXACTLY — the loop is
    never actually necessary past the third value.** For a constant affine loop with `A ∉ {0,1}`
    that has not already reached its fixed point (`S₀ ≠ x*`), `aitken` applied to ANY three
    consecutive iterates equals `B/(1-A)` on the nose, for every `n` — not an approximation, an
    identity. `A ≠ 0` excludes the one-step-collapse degeneracy (`S ↦ B` outright, nothing to
    extrapolate from past the first value); `A ≠ 1` excludes the no-fixed-point degeneracy already
    named in §2. This is the operational meaning of "there is no loop on the bulk, there is a spinor
    at stationary phase": running a constant affine recurrence for `R` rounds to converge is an
    unexamined default, not a requirement — three rounds and one extrapolation reach the same
    answer exactly. Real power-iteration methods (PageRank's `authority` among them) sit on this
    fact only approximately, because their error is a SUM of geometric modes (one per eigenvalue of
    the transition, not the single mode this theorem assumes) — the exact case proved here is the
    reason Aitken/Anderson-style acceleration works so well on them in practice, not a claim that it
    replaces power iteration outright for a general (non-constant-single-mode) operator. -/
theorem aitken_exact (step : AffineStep) (hA0 : step.rotor_decay ≠ 0) (hA1 : step.rotor_decay ≠ 1)
    (S0 : ℝ) (n : ℕ) (hne : S0 ≠ step.injection / (1 - step.rotor_decay)) :
    aitken (iterate step S0 n) (iterate step S0 (n + 1)) (iterate step S0 (n + 2))
      = step.injection / (1 - step.rotor_decay) := by
  set xstar := step.injection / (1 - step.rotor_decay) with hxstar
  set A := step.rotor_decay with hA
  set e0 := S0 - xstar with he0
  have he0ne : e0 ≠ 0 := fun h => hne (by rw [he0] at h; linarith)
  have h0 := iterate_sub_fixed_point step hA1 S0 n
  have h1 := iterate_sub_fixed_point step hA1 S0 (n + 1)
  have h2 := iterate_sub_fixed_point step hA1 S0 (n + 2)
  have hd2 := second_difference step hA1 S0 n
  rw [← hxstar, ← he0] at h0 h1 h2 hd2
  rw [← hA] at h0 h1 h2 hd2
  -- Δx_n = iterate(n+1) - iterate(n) = A^n·(A-1)·e0
  have hΔ : iterate step S0 (n + 1) - iterate step S0 n = A ^ n * (A - 1) * e0 := by
    have := h1; have := h0
    have e1 : iterate step S0 (n + 1) = A ^ (n + 1) * e0 + xstar := by linarith
    have e0' : iterate step S0 n = A ^ n * e0 + xstar := by linarith
    rw [e1, e0', pow_succ]; ring
  have hd2ne : iterate step S0 (n + 2) - 2 * iterate step S0 (n + 1) + iterate step S0 n ≠ 0 := by
    rw [hd2]
    have h1 : (A - 1) ^ 2 ≠ 0 := pow_ne_zero 2 (sub_ne_zero.mpr hA1)
    have hAn : A ^ n ≠ 0 := pow_ne_zero n hA0
    exact mul_ne_zero (mul_ne_zero h1 hAn) he0ne
  unfold aitken
  rw [hΔ, hd2]
  have hAnne : A ^ n ≠ 0 := pow_ne_zero n hA0
  have hA1ne : (A - 1) ≠ 0 := sub_ne_zero.mpr hA1
  field_simp
  linarith [h0]

-- ── THE SAFE ALTERNATIVE: an a-posteriori bound with no division-by-near-zero risk ────────────────
-- `aitken_exact` recovers x* exactly from three iterates, but the empirical test against real
-- PageRank data (BENCHMARK.md, iotadb) found it can catastrophically overshoot once the sequence
-- has actually converged — the second difference it divides by is then near/exactly zero. This is
-- the alternative a real `authority`-style early-stop should actually use: a bound on the TRUE
-- error from ONE observed step and a KNOWN constant (PageRank's damping factor `d` bounds the
-- transition's contraction rate by standard Perron-Frobenius theory) — never dividing by anything
-- that can vanish, so it degrades gracefully instead of blowing up.

/-- **One step already determines the exact error.** `x - x* = (x - apply x)/(1-A)`, for ANY point
    `x`, not just an iterate — the operational form of `constantStep_fixed_point`, not a new fact. -/
theorem one_step_error_identity (step : AffineStep) (hA : step.rotor_decay ≠ 1) (x : ℝ) :
    x - step.injection / (1 - step.rotor_decay)
      = (x - step.apply x) / (1 - step.rotor_decay) := by
  have hne : (1 - step.rotor_decay) ≠ 0 := fun h => hA (by linarith)
  simp only [apply]
  field_simp
  ring

/-- **The safe error bound — computable from ONE observed step and a KNOWN contraction rate, never
    a measured one.** If `|A| ≤ ρ` for a KNOWN `ρ < 1` (PageRank's damping factor bounds the
    transition's contraction rate exactly this way), the TRUE distance to the fixed point is
    bounded by the observed one-step change, scaled by the FIXED constant `1/(1-ρ)` — never
    dividing by a quantity that can be near zero. This is the rigorous form of "stop once the
    change is small": unlike `aitken_exact`, which needs the sequence to be a pure single geometric
    mode and fails exactly when it has already converged, this bound holds at EVERY point,
    including ones already at the fixed point (both sides are then `0`, no division blow-up). -/
theorem safe_error_bound (step : AffineStep) (hA : step.rotor_decay ≠ 1) {ρ : ℝ} (hρ1 : ρ < 1)
    (hAρ : |step.rotor_decay| ≤ ρ) (x : ℝ) :
    |x - step.injection / (1 - step.rotor_decay)| ≤ |x - step.apply x| / (1 - ρ) := by
  rw [one_step_error_identity step hA x, abs_div]
  have hrev : 1 - |step.rotor_decay| ≤ |1 - step.rotor_decay| := by
    have h := abs_sub_abs_le_abs_sub (1 : ℝ) step.rotor_decay
    simpa using h
  have hρpos : (0 : ℝ) < 1 - ρ := by linarith
  have hAabs : (0 : ℝ) < 1 - |step.rotor_decay| := by linarith [abs_nonneg step.rotor_decay]
  have hle : 1 - ρ ≤ |1 - step.rotor_decay| := by linarith
  exact div_le_div_of_nonneg_left (abs_nonneg _) hρpos hle

/-- **The safe bound, packaged for the iterate sequence.** At any step `n`, the true distance to
    the fixed point is bounded by the one observable quantity a running loop actually has —
    `|iterate n - iterate (n+1)|` — scaled by the fixed constant `1/(1-ρ)`. This is what an
    `authority`-style early-stop should check: halt once this observed gap drops below tolerance,
    and the TRUE error is then provably below `tolerance/(1-ρ)`. No estimate of `A`, no risk of
    dividing by a vanishing quantity. -/
theorem safe_early_stop (step : AffineStep) (hA : step.rotor_decay ≠ 1) {ρ : ℝ} (hρ1 : ρ < 1)
    (hAρ : |step.rotor_decay| ≤ ρ) (S0 : ℝ) (n : ℕ) :
    |iterate step S0 n - step.injection / (1 - step.rotor_decay)|
      ≤ |iterate step S0 n - iterate step S0 (n + 1)| / (1 - ρ) := by
  have := safe_error_bound step hA hρ1 hAρ (iterate step S0 n)
  simpa [iterate] using this

end AffineStep

end SpinorLoop

end Sanguine
