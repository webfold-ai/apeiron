import Lean

/-!
# The citation registry

One entry per source, keyed by the string used in `@cite`. `Apeiron.Linter.Citation` checks
during elaboration that every `@cite` key resolves here, so a key naming nothing fails the
build instead of reading as a citation.

A `@cite` line goes in a module docstring or in any declaration's docstring. Put it on the
declaration when a source supports that one result, and on the module when it is the subject
of the file.

The linter catches a key that resolves to nothing. It cannot catch an entry crediting the
wrong person, which is a reading question and stays with whoever adds the entry.
-/

namespace Apeiron.Linter

/-- One source. -/
structure Source where
  /-- The key written after `@cite`. -/
  key : String
  /-- Author and year. -/
  who : String
  /-- The work. -/
  work : String
  /-- Journal, volume, or a URL. -/
  where? : String
  /-- What this corpus takes from it. -/
  contribution : String
  deriving Repr, Inhabited

/-- Every source the corpus may cite. -/
def sources : List Source := [
  { key := "euclid-elements-ii-5"
    who := "Euclid -300"
    work := "Elements, Book II, Proposition 5"
    where? := ""
    contribution :=
      "The difference of two squares as a geometric identity: a² − b² = "
      ++ "(a+b)(a−b). This factorization IS the null cone of the σ=1 cell: "
      ++ "the two factors are the two null directions, and their vanishing "
      ++ "is the zero-divisor pair." },
  { key := "pythagoras-sum-of-squares"
    who := "Pythagoreans -500"
    work := "the Pythagorean theorem"
    where? := ""
    contribution :=
      "a² + b² as the definite form. The elliptic (σ<0) normForm; the sign "
      ++ "that makes the algebra a field. The trichotomy is the one- "
      ++ "parameter family joining this to Euclid II.5." },
  { key := "bombelli-1572"
    who := "Rafael Bombelli 1572"
    work := "L'Algebra"
    where? := ""
    contribution :=
      "First systematic arithmetic of √−1: the rules for a + b√−1 as a "
      ++ "closed system. The σ = −1 cell before it had a name." },
  { key := "euler-1748"
    who := "Leonhard Euler 1748"
    work := "Introductio in analysin infinitorum"
    where? := ""
    contribution :=
      "e^{iθ} = cos θ + i sin θ. The elliptic cell's unit circle as the "
      ++ "exponential of its generator: the rotor. Split-complex e^{jφ} = "
      ++ "cosh φ + j sinh φ is the same construction at σ = +1, giving the "
      ++ "boost instead of the rotation." },
  { key := "argand-1806"
    who := "Jean-Robert Argand 1806"
    work := "Essai sur une manière de représenter les quantités imaginaires"
    where? := ""
    contribution :=
      "The complex plane: ℝ[i]/(i²+1) as a 2-dimensional real carrier ℝ × "
      ++ "ℝ." },
  { key := "gauss-1831"
    who := "Carl Friedrich Gauss 1831"
    work := "Theoria residuorum biquadraticorum, Commentatio secunda"
    where? := ""
    contribution :=
      "The complex numbers as ordered pairs of reals with a defined "
      ++ "multiplication: the carrier-plus-product presentation the "
      ++ "trichotomy generalizes over σ." },
  { key := "cockle-1849"
    who := "James Cockle 1849"
    work := "On Certain Functions Resembling Quaternions, and on a New Imaginary in Algebra"
    where? := "Philosophical Magazine 33, 435–439"
    contribution :=
      "The tessarines: the first description of a new imaginary with j² = "
      ++ "+1 rather than −1. The σ = +1 cell. Cockle notes explicitly that "
      ++ "division fails, exhibiting the zero divisors that the trichotomy "
      ++ "classifies." },
  { key := "clifford-1873"
    who := "William Kingdon Clifford 1873"
    work := "Preliminary Sketch of Biquaternions"
    where? := "Proceedings of the London Mathematical Society 4, 381–395"
    contribution :=
      "Split/hyperbolic numbers within geometric algebra; the systematic "
      ++ "treatment of generators squaring to +1 alongside those squaring to "
      ++ "−1." },
  { key := "minkowski-1908"
    who := "Hermann Minkowski 1908"
    work := "Raum und Zeit"
    where? := ""
    contribution :=
      "The indefinite quadratic form as the metric of spacetime: "
      ++ "signature (1,1) in two dimensions IS the σ>0 normForm a² − b², and the "
      ++ "light cone IS the null cone where division fails." },
  { key := "clifford-1873-dual"
    who := "William Kingdon Clifford 1873"
    work := "Preliminary Sketch of Biquaternions"
    where? := ""
    contribution :=
      "Dual numbers: a generator ε with ε² = 0. The σ = 0 cell." },
  { key := "study-1903"
    who := "Eduard Study 1903"
    work := "Geometrie der Dynamen"
    where? := ""
    contribution :=
      "Systematic development of the dual numbers and their use in "
      ++ "kinematics; the nilpotent generator as the degenerate limit "
      ++ "between the definite and indefinite cases." },
  { key := "weierstrass-1884"
    who := "Karl Weierstrass 1884"
    work := "Zur Theorie der aus n Haupteinheiten gebildeten complexen Grössen"
    where? := ""
    contribution :=
      "Finite-dimensional commutative associative algebras over ℝ "
      ++ "decompose into fields and nilpotent parts: the structural reason "
      ++ "the 2-dimensional case has exactly three species." },
  { key := "wedderburn-1908"
    who := "Joseph Wedderburn 1908"
    work := "On Hypercomplex Numbers"
    where? := "Proceedings of the London Mathematical Society 6, 77–118"
    contribution :=
      "The structure theory of finite-dimensional algebras. The "
      ++ "2-dimensional real commutative case is the smallest nontrivial "
      ++ "instance; `classification` proves it natively by completing the "
      ++ "square rather than by citing the general theorem." },
  { key := "boole-1847"
    who := "George Boole 1847"
    work := "The Mathematical Analysis of Logic"
    where? := ""
    contribution :=
      "x² = x as the defining law of logical classes. Idempotence as an "
      ++ "algebraic equation: the equation whose nontrivial solutions exist "
      ++ "only when σ > 0." },
  { key := "stone-1936"
    who := "Marshall Stone 1936"
    work := "The Theory of Representations for Boolean Algebras"
    where? := "Transactions of the AMS 40, 37–111"
    contribution :=
      "Boolean algebras are exactly Boolean rings (every element "
      ++ "idempotent), and their representation is by clopen sets. The "
      ++ "bridge from idempotents to logic that Claim 1 runs in the other "
      ++ "direction: which algebras admit them at all." },
  { key := "peirce-1870"
    who := "Benjamin Peirce 1870"
    work := "Linear Associative Algebra"
    where? := ""
    contribution :=
      "The Peirce decomposition: an idempotent e splits an algebra into "
      ++ "eAe ⊕ eA(1−e) ⊕ … At σ>0 the two orthogonal idempotents e± split "
      ++ "the cell into ℝ × ℝ: two independent channels." },
  { key := "barker-2001"
    who := "Chris Barker 2001"
    work := "Iota and Jot: the simplest languages?"
    where? :=
      "https://web.archive.org/web/20160823182917/http://semarch.linguistics.fas.nyu.edu/barker/Iota/"
    contribution :=
      "The combinator iota, written λf.fSK: one constant whose single "
      ++ "rule hands its argument S and K, from which both are recovered. "
      ++ "The one-point basis this corpus proves is the two-slot pair "
      ++ "holding S and K (`iota_eq_pack`)." },
  { key := "goldberg-2004"
    who := "Mayer Goldberg 2004"
    work := "A construction of one-point bases in extended lambda calculi"
    where? := "Information Processing Letters 89"
    contribution :=
      "One-point bases constructed systematically rather than exhibited. "
      ++ "Cited alongside Barker for the iota chain; song's prover carries "
      ++ "the same pairing as `Barker-Goldberg iota`." },
  { key := "schonfinkel-1924"
    who := "Moses Schönfinkel 1924"
    work := "Über die Bausteine der mathematischen Logik, Mathematische Annalen 92"
    where? := ""
    contribution :=
      "Application as the single binary operation, and the constants S "
      ++ "and K from which every closed term is built. The duplicate-and- "
      ++ "route content of S and the discard content of K are the two "
      ++ "behaviours the corpus derives from the algebra." },
  { key := "curry-1930"
    who := "Haskell B. Curry 1930"
    work := "Grundlagen der kombinatorischen Logik, American Journal of Mathematics 52"
    where? := ""
    contribution :=
      "Combinatory logic as a formal system: the reduction rules for S "
      ++ "and K, and bracket abstraction, by which S and K suffice for every "
      ++ "computable function. That sufficiency is cited from here and not "
      ++ "formalised in this corpus." },
  { key := "barendregt-1984"
    who := "H. P. Barendregt 1984"
    work := "The Lambda Calculus: Its Syntax and Semantics, North-Holland"
    where? := ""
    contribution :=
      "The standard reference for combinatory logic and lambda calculus, "
      ++ "including behavioural rather than syntactic equality of "
      ++ "combinators, the notion under which a one-constant basis generates "
      ++ "S and K." },
  { key := "kaucher-1973"
    who := "Edgar Kaucher 1973"
    work := "Über Eigenschaften und Anwendungsmöglichkeiten der erweiterten Intervallrechnung"
    where? := "PhD thesis, Karlsruhe, under Ulrich Kulisch; see also arXiv:2202.03058"
    contribution :=
      "The algebraic completion of interval arithmetic. Improper intervals, "
      ++ "where the lower endpoint has overtaken the upper one, become "
      ++ "elements rather than a failure state, every element gains an "
      ++ "additive inverse, and the completed arithmetic carries zero "
      ++ "divisors. Its inclusion-meet is max on the lower endpoint and min "
      ++ "on the upper, which is the propagation step." },
  { key := "shary-2022"
    who := "Sergey P. Shary 2022"
    work := "Non-traditional intervals and their use. Which ones really make sense?"
    where? := "arXiv:2202.03058"
    contribution :=
      "The reading of Kaucher arithmetic used here: closed intervals as "
      ++ "compacts and complete lattices, the case for improper intervals, "
      ++ "the zero divisor [−1,1]·[2,−3] = 0, and the sup-inf definition of "
      ++ "the product." },
  { key := "maldacena-1997"
    who := "Juan M. Maldacena 1997"
    work := "The Large N Limit of Superconformal Field Theories and Supergravity"
    where? := "Adv. Theor. Math. Phys. 2, 231-252, arXiv:hep-th/9711200"
    contribution :=
      "The correspondence itself: a gravitational theory in a bulk is dual "
      ++ "to a conformal field theory on its boundary." },
  { key := "lohmiller-slotine-2024"
    who := "Winfried Lohmiller and Jean-Jacques Slotine 2024"
    work := "On computing quantum waves exactly from classical and relativistic action"
    where? := "Proc. R. Soc. A 482, 20250413, arXiv:2405.06328"
    contribution :=
      "The claim this corpus was started in response to: that the "
      ++ "Schrodinger equation can be solved exactly from classical least "
      ++ "action alone, with no quasi-classical approximation. Refuted by "
      ++ "vattay-2026-comment and vattay-2026-reply. What survives the "
      ++ "refutation is the exactness window, which is why the corpus is "
      ++ "quadratic." },
  { key := "vattay-2026-comment"
    who := "Gabor Vattay 2026"
    work := "Comment on On computing quantum waves exactly from classical action"
    where? := "arXiv:2605.02621"
    contribution :=
      "The refutation. Neglecting the spatial derivatives of the density "
      ++ "amplitude drops Madelung's quantum potential, so the claimed "
      ++ "identity is the standard semiclassical approximation." },
  { key := "vattay-2026-reply"
    who := "Gabor Vattay 2026"
    work := "Reply to Interpreting Bohm quantum potentials"
    where? := "arXiv:2606.05197, 2026-05-22"
    contribution :=
      "The chain rule written out against the position-dependent time "
      ++ "transformation, and the theorem the corpus rests on: the "
      ++ "semiclassical trace formula is exact if and only if the "
      ++ "Hamiltonian is at most quadratic in position and momentum." },
  { key := "schrodinger-1926"
    who := "Erwin Schrodinger 1926"
    work := "Quantisierung als Eigenwertproblem"
    where? := "Ann. Phys. 79-81, 1926; An Undulatory Theory of the Mechanics "
      ++ "of Atoms and Molecules, Phys. Rev. 28, 1049-1070"
    contribution :=
      "The equation whose polar-form split is the Madelung pair "
      ++ "(continuity and the quantum Hamilton-Jacobi equation). The "
      ++ "corpus states its one-dimensional pointwise residual in "
      ++ "`Physics.Models.Schrodinger` and proves its polar substitution "
      ++ "equivalent to the two Madelung residuals under explicit positive "
      ++ "scale, density, and derivative hypotheses." },
  { key := "madelung-1927"
    who := "Erwin Madelung 1927"
    work := "Quantentheorie in hydrodynamischer Form"
    where? := "Z. Phys. 40, 322-326"
    contribution :=
      "The hydrodynamic form of the wave: ψ = √ρ e^{iS/ℏ} split into a "
      ++ "continuity equation and a Hamilton-Jacobi equation whose "
      ++ "remainder is the amplitude-curvature term Q = −(ℏ²/2m)∇²√ρ/√ρ. "
      ++ "The polar decomposition `Physics.Madelung.lean` derives over "
      ++ "`Cell σ` as `R · flow σ(φ)`." },
  { key := "bohm-1952"
    who := "David Bohm 1952"
    work := "A suggested interpretation of the quantum theory in terms of "
      ++ "hidden variables, I"
    where? := "Phys. Rev. 85, 166-179"
    contribution :=
      "The quantum potential Q, named and made the carrier of the "
      ++ "nonlocal correction. In this corpus it is the residue "
      ++ "`residue_is_curvature` leaves when the kinetic term is removed "
      ++ "from the wave's first slot." },
  { key := "hamilton-jacobi-1834"
    who := "William Rowan Hamilton 1834, completed by Carl Gustav Jacob "
      ++ "Jacobi 1842"
    work := "On a General Method in Dynamics; Vorlesungen uber Dynamik"
    where? := "Phil. Trans. R. Soc. 124, 1834"
    contribution :=
      "The classical equation of the phase: −∂φ/∂t = (∇φ)²/2m + V. Its "
      ++ "quantum correction is the amplitude-curvature term, and the "
      ++ "sign of σ in `hamilton_jacobi` is which reading of the kinetic "
      ++ "term is meant." },
  { key := "van-vleck-1928"
    who := "John H. Van Vleck 1928"
    work := "The correspondence principle in the statistical interpretation "
      ++ "of quantum mechanics"
    where? := "Proc. Natl. Acad. Sci. USA 14, 178-188"
    contribution :=
      "The determinant carrying the classical density along an extremal "
      ++ "path, the propagator's prefactor. The construction Lohmiller and "
      ++ "Slotine re-derive, and the object Vattay shows is exact only "
      ++ "inside the quadratic window." },
  { key := "morette-1951"
    who := "Cecile Morette 1951"
    work := "On the definition and approximation of Feynman's path integrals"
    where? := "Phys. Rev. 81, 848-852"
    contribution :=
      "The semiclassical kernel in closed form, the Van Vleck-Pauli-"
      ++ "Morette formula: the sum over classical paths of the density "
      ++ "weight times the phase. The exactness window (at-most-quadratic "
      ++ "Hamiltonians) is the statement about this formula the corpus "
      ++ "quadratic window rests on." },
  { key := "feynman-hibbs-1965"
    who := "Richard P. Feynman and Albert R. Hibbs 1965"
    work := "Quantum Mechanics and Path Integrals"
    where? := "McGraw-Hill"
    contribution :=
      "The two-step construction: compute the propagator from a point "
      ++ "source, then superpose initial conditions. Lohmiller and "
      ++ "Slotine's per-branch waves are this kernel, and the Gaussian "
      ++ "exactness of the quadratic window is the case where the "
      ++ "superposition needs no correction." },
  { key := "cubitt-2015"
    who := "Toby Cubitt, David Perez-Garcia and Michael Wolf 2015"
    work := "Undecidability of the spectral gap"
    where? := "Nature 528, 207-211, arXiv:1502.04573"
    contribution :=
      "Undecidability inside physics: nearest-neighbour spin lattices whose "
      ++ "gapped status is undecidable, and for some of which it is "
      ++ "independent of the axioms of mathematics. Relies on the "
      ++ "thermodynamic limit, so no finite system carries it." },
  { key := "mir-2026"
    who := "Mir, Marino, Shabir, Krauss and Faizal 2026"
    work := "Undecidability in spacetime geometry via the AdS/CFT correspondence"
    where? := "Phys. Rev. D 113, L021904, arXiv:2601.22761"
    contribution :=
      "The holographic dictionary transmits that undecidability into the "
      ++ "bulk: which smooth spacetime emerges can be beyond computability. "
      ++ "The bulk is a conduit for undecidability, not a shelter." },
  { key := "cardona-2021"
    who := "Cardona, Miranda, Peralta-Salas and Presas 2021"
    work := "Constructing Turing complete Euler flows in dimension 3"
    where? := "PNAS 118, arXiv:2012.12828"
    contribution :=
      "Stationary ideal fluid flows that are Turing complete, with "
      ++ "undecidable particle trajectories. Fluid dynamics is "
      ++ "computationally universal, which bears on ../whispering-tides." },
  { key := "ramallo-2013"
    who := "Alfonso V. Ramallo 2013"
    work := "Introduction to the AdS/CFT correspondence"
    where? := "Springer Proc. Phys. 161, 411-474, arXiv:1310.4319"
    contribution :=
      "The reference specification of anti-de Sitter space used here. Its "
      ++ "equation (8) writes the most general metric with boundary Poincaré "
      ++ "invariance as a conformal factor times a flat bracket, and "
      ++ "equations (9) to (11) derive the factor L/z from requiring the "
      ++ "dual theory be scale invariant. That split, factor times bracket, "
      ++ "is what lineElement follows: the factor does not depend on σ and "
      ++ "the bracket is the norm form, so σ's only contribution is the "
      ++ "signature." },
  { key := "kisil-2010"
    who := "Vladimir V. Kisil 2010"
    work := "Erlangen Program at Large-1: Geometry of Invariants"
    where? := "SIGMA 6 (2010) 076, arXiv:math/0512416"
    contribution :=
      "This corpus's algebra, worked as geometry. The three cases of "
      ++ "ℝ[x]/(x² − σ) give the elliptic, parabolic and hyperbolic "
      ++ "geometries of the SL(2,ℝ) action by Möbius transformations, with "
      ++ "the invariant curves he calls cycles. The projective reading of a "
      ++ "point is his, not this corpus's." },
  { key := "kisil-2007"
    who := "Vladimir V. Kisil 2007"
    work := "Two-Dimensional Conformal Models of Space-Time and Their Compactification"
    where? := "J. Math. Phys. 48, 073506, arXiv:math-ph/0611053"
    contribution :=
      "The conformal completion of the two-dimensional model over each of "
      ++ "the three algebras, and the time-arrow condition the hyperbolic "
      ++ "case needs to compactify correctly. The cone this corpus proves "
      ++ "is two rays is that case's light cone." },
  { key := "kisil-2012"
    who := "Vladimir V. Kisil 2012"
    work := "Hypercomplex Representations of the Heisenberg Group and Mechanics"
    where? := "Internat. J. Theoret. Phys. 51, 964-984, arXiv:1005.5057"
    contribution :=
      "The physics reading of the same three algebras: the elliptic "
      ++ "character gives quantum mechanics, the hyperbolic gives hyperbolic "
      ++ "mechanics, and the parabolic gives classical mechanics with no "
      ++ "limit taken in Planck's constant." },
  { key := "yaglom-1979"
    who := "Isaak M. Yaglom 1979"
    work := "A Simple Non-Euclidean Geometry and its Physical Basis"
    where? := "Springer-Verlag, New York"
    contribution :=
      "The classical reference for the three geometries built from the complex, dual and "
      ++ "double numbers: elliptic, parabolic and hyperbolic, one per sign of the same "
      ++ "generator. Kisil's programme cites Yaglom for this; the trichotomy's three faces "
      ++ "are Yaglom's three geometries at σ = −1, 0, +1." },
  { key := "ryu-takayanagi-2006"
    who := "Shinsei Ryu and Tadashi Takayanagi 2006"
    work := "Holographic Derivation of Entanglement Entropy from AdS/CFT"
    where? := "Phys. Rev. Lett. 96, 181602, arXiv:hep-th/0603001"
    contribution :=
      "Entanglement entropy of a boundary region is the area of a minimal "
      ++ "surface in the bulk. In AdS₃ that surface is a geodesic, and "
      ++ "geodesics of the hyperbolic plane are cycles in Kisil's sense." },
  { key := "heydeman-2016"
    who := "Heydeman, Marcolli, Saberi and Stoica 2016"
    work := "Tensor networks, p-adic fields, and algebraic curves"
    where? := "arXiv:1605.07639"
    contribution :=
      "Holography with a discrete bulk: the Bruhat-Tits tree for "
      ++ "PGL(2,ℚ_p), whose boundary is a projective line, with the bulk "
      ++ "isometries unbroken by the discretisation and Ryu-Takayanagi "
      ++ "appearing naturally. A tree over a projective line, which is the "
      ++ "shape Tower and ratio have." },
  { key := "gubser-2016"
    who := "Gubser, Knaute, Parikh, Samberg and Witaszczyk 2016"
    work := "p-adic AdS/CFT"
    where? := "Comm. Math. Phys. 352, arXiv:1605.01061"
    contribution :=
      "The same construction reached independently and in the same month. "
      ++ "Two groups arriving at a discrete bulk at once is the measure of "
      ++ "how available the idea was." },
  { key := "freedman-headrick-2016"
    who := "Michael Freedman and Matthew Headrick 2016"
    work := "Bit threads and holographic entanglement"
    where? := "Comm. Math. Phys. 352, 407, arXiv:1604.00354"
    contribution :=
      "Ryu-Takayanagi restated by max-flow min-cut, so the entropy is a "
      ++ "flow through a network rather than the area of a surface. The "
      ++ "graph-theoretic face of holography." },
  { key := "chen-czech-wang-2022"
    who := "Bowen Chen, Bartlomiej Czech and Zi-zhi Wang 2022"
    work := "Quantum Information in Holographic Duality"
    where? := "Rept. Prog. Phys. 85, 046001, arXiv:2108.09188"
    contribution :=
      "The survey to read first for the holographic side: Ryu-Takayanagi, "
      ++ "subregion duality, tensor network architectures, bit threads, and "
      ++ "what boundary entanglement fixes about a bulk." },
  { key := "artin-schreier-1927"
    who := "Emil Artin and Otto Schreier 1927"
    work := "Algebraische Konstruktion reeller Körper"
    where? := "Abh. Math. Sem. Univ. Hamburg 5, 85-99"
    contribution :=
      "Real closed fields: the order is determined by the squares, so "
      ++ "x ≤ y exactly when y − x is a square. The reason an order on a "
      ++ "null channel is the algebra's own and not structure added to it." },
  { key := "curry-feys-1958"
    who := "Haskell Curry and Robert Feys 1958"
    work := "Combinatory Logic, Volume I"
    where? := "North-Holland"
    contribution :=
      "The systematic treatment of the calculus, including the family of "
      ++ "bracket abstraction algorithms and what each clause costs." },
  { key := "tomita-2023"
    who := "Haruka Tomita 2023"
    work := "Categorical Realizability for Non-symmetric Closed Structures"
    where? := "arXiv:2307.04119"
    contribution :=
      "Proposition 1 and Proof 2.1: combinatory completeness for partial "
      ++ "combinatory algebras, with the three-clause bracket algorithm the "
      ++ "corpus uses, and the statement that the result keeps every free "
      ++ "variable of the body but the abstracted one." },
  { key := "selinger-2002"
    who := "Peter Selinger 2002"
    work := "The lambda calculus is algebraic"
    where? := "Journal of Functional Programming 12(6), 549-566"
    contribution :=
      "The axiomatisation of lambda algebras, and why an algebraic "
      ++ "treatment of the calculus is delicate rather than routine." },
  { key := "dallago-2022"
    who := "Dal Lago, Honsell, Lenisa and Pistone 2022"
    work := "On Quantitative Algebraic Higher-Order Theories"
    where? := "arXiv:2204.13654"
    contribution :=
      "Propositions 48 and 49: a metric space without enough points is no "
      ++ "quantitative lambda-algebra, and no non-trivial one-sorted weak "
      ++ "quantitative lambda-algebra in Met is compact. The obstruction to "
      ++ "carrying application by a continuous operation on a real space." },
  { key := "hofstra-vanoosten-2003"
    who := "Pieter Hofstra and Jaap van Oosten 2003"
    work := "Ordered partial combinatory algebras"
    where? := "Mathematical Proceedings of the Cambridge Philosophical Society 134, 445-463"
    contribution :=
      "Application carried by a preorder rather than by an equality, the "
      ++ "first of the three shapes left open in theory/DEAD_ENDS.md." },
  { key := "vanoosten-2008"
    who := "Jaap van Oosten 2008"
    work := "Realizability: An Introduction to its Categorical Side"
    where? := "Studies in Logic 152, Elsevier"
    contribution :=
      "The survey for partial combinatory algebras, applicative morphisms "
      ++ "and realizability toposes." },
  { key := "ferrer-santos-2014"
    who := "Ferrer Santos, Frey, Guillermo, Malherbe and Miquel 2014"
    work := "Ordered combinatory algebras and realizability"
    where? := "arXiv:1410.5034"
    contribution :=
      "Krivine ordered combinatory algebras, and the proof that the "
      ++ "competing ordered structures give the same triposes." },
  { key := "miquel-2018"
    who := "Alexandre Miquel 2018"
    work := "Implicative algebras: a new foundation for realizability and forcing"
    where? := "Mathematical Structures in Computer Science 30, 458-510, arXiv:1802.00528"
    contribution :=
      "A complete meet-semilattice with an implication, in which "
      ++ "application is derived as a meet rather than given as an "
      ++ "operation. Definition 2.1. The published form of the order-based "
      ++ "route, and the reason the associativity obstruction does not "
      ++ "close the question." },
  { key := "palmieri-2026"
    who := "Stefano Palmieri 2026"
    work :=
      "Pairwise Independence of Representation, Classification, and "
      ++ "Composition in Finite Extensional Magmas"
    where? := "arXiv:2603.27007"
    contribution :=
      "Theorem 3.7, a nontrivial S+K algebra is infinite. Theorems 3.5 and "
      ++ "3.6, no semigroup, monoid, group or ring carries a classifier "
      ++ "with a retraction pair. The published form of the obstruction "
      ++ "recorded in theory/DEAD_ENDS.md." },
  { key := "mathlib-findim"
    who := "Mathlib"
    work := "Mathlib.LinearAlgebra.FiniteDimensional.Lemmas"
    where? := ""
    contribution :=
      "Module.finrank, Basis, basisOfLinearIndependentOfCardEqFinrank." },
  { key := "mathlib-sqrt"
    who := "Mathlib"
    work := "Mathlib.Analysis.SpecialFunctions.Sqrt"
    where? := ""
    contribution :=
      "Real.sqrt and Real.mul_self_sqrt, used to normalise the generator." },
  { key := "mathlib-complex"
    who := "Mathlib"
    work := "Mathlib.Data.Complex.Basic"
    where? := ""
    contribution :=
      "ℂ and Complex.normSq, for the elliptic-face contrast theorems." },
  { key := "farkas-1902"
    who := "Gyula Farkas 1902"
    work := "Theorie der einfachen Ungleichungen"
    where? := "Journal fur die reine und angewandte Mathematik 124, 1-27"
    contribution :=
      "A system of linear inequalities is unsatisfiable exactly when a "
      ++ "nonnegative combination of them is a negative constant. The "
      ++ "soundness half is what certcheck replays and what "
      ++ "unsat_of_negative_combo states over the row model." },
  { key := "presburger-1929"
    who := "Mojzesz Presburger 1929"
    work :=
      "Uber die Vollstandigkeit eines gewissen Systems der Arithmetik "
      ++ "ganzer Zahlen, in welchem die Addition als einzige Operation "
      ++ "hervortritt"
    where? := "Comptes Rendus du I Congres de Mathematiciens des Pays Slaves, 92-101"
    contribution :=
      "Addition and order over the whole numbers is decidable and "
      ++ "complete, by quantifier elimination. The unbounded fragment "
      ++ "Fragment/Basic.lean leaves open." },
  { key := "matiyasevich-1970"
    who := "Yuri Matiyasevich 1970"
    work := "Enumerable sets are Diophantine"
    where? := "Doklady Akademii Nauk SSSR 191, 279-282"
    contribution :=
      "Integer polynomial solvability is undecidable, completing "
      ++ "Davis, Putnam and Robinson. The line the monomial abstraction "
      ++ "keeps facec's rows on the decidable side of." },
  { key := "cooper-1972"
    who := "David C. Cooper 1972"
    work := "Theorem proving in arithmetic without multiplication"
    where? := "Machine Intelligence 7, 91-99"
    contribution :=
      "The quantifier elimination procedure for Presburger arithmetic "
      ++ "that carries divisibility atoms through the elimination step. "
      ++ "Route 2 of theory/DEAD_ENDS.md's unbounded entry." },
  { key := "chaieb-nipkow-2005"
    who := "Amine Chaieb and Tobias Nipkow 2005"
    work := "Verifying and reflecting quantifier elimination for Presburger arithmetic"
    where? := "LPAR 2005, LNCS 3835, 367-380; AFP entry LinearQuantifierElim"
    contribution :=
      "Cooper's algorithm formalised and executable in Isabelle/HOL. "
      ++ "The published closure of the unbounded fragment, in a system "
      ++ "other than Lean." },
  { key := "mathlib-presburger"
    who := "Mathlib"
    work := "Mathlib.ModelTheory.Arithmetic.Presburger.Definability"
    where? := ""
    contribution :=
      "presburger.definable_iff_isSemilinearSet and "
      ++ "presburger.mul_not_definable. Which sets are definable, with "
      ++ "no decision procedure and no Decidable instance." },
  { key := "herranz-ortega-santander-2000"
    who := "Francisco Herranz, Ramon Ortega and Mariano Santander 2000"
    work := "Trigonometry of spacetimes: a new self-dual approach to a "
      ++ "curvature/signature (in)dependent trigonometry"
    where? := "J. Phys. A 33, 4525, arXiv:math-ph/9910041"
    contribution :=
      "Equations (2.14) and (2.15) define the labelled cosine and sine as "
      ++ "power series in one curvature parameter, and (A.1) is the identity "
      ++ "C² + κS² = 1. That is this corpus's flow, written before it, from "
      ++ "graded contractions of the rotation group rather than from a "
      ++ "two-dimensional algebra. Their κ is this corpus's σ negated." },
  { key := "kocik-1999"
    who := "Jerzy Kocik 1999"
    work := "Duplex numbers, diffusion systems, and generalized quantum mechanics"
    where? := "Int. J. Theor. Phys. 38, 2221, arXiv:1201.3069"
    contribution :=
      "Theorem 1 and equations (2.2), (2.4) and (2.10) split a wave written "
      ++ "in polar form over a two-dimensional real algebra into a "
      ++ "Hamilton-Jacobi equation whose kinetic sign is the algebra's, and a "
      ++ "continuity equation that is the same on both faces. Section 3, "
      ++ "equation (3.2), states that sameness. The operator carries the "
      ++ "generator's inverse, so the face where the generator squares to "
      ++ "zero is outside his statement." },
  { key := "gibbons-hawking-1977"
    who := "Gary Gibbons and Stephen Hawking 1977"
    work := "Action integrals and partition functions in quantum gravity"
    where? := "Phys. Rev. D 15, 2752"
    contribution :=
      "The rotated horizon is a plane in polar coordinates whose angle runs "
      ++ "at the surface gravity, and the centre is a regular point only if "
      ++ "the angle closes. That fixes the period and hence the temperature. "
      ++ "The argument assumes the closing, which is what this corpus proves "
      ++ "is available on one face only." },
  { key := "baldazzi-percacci-skrinjar-2018"
    who := "Alessio Baldazzi, Roberto Percacci and Vedran Skrinjar 2018"
    work := "Wicked metrics"
    where? := "arXiv:1811.03369"
    contribution :=
      "Equation (1.15) interpolates between a Lorentzian and a Euclidean "
      ++ "metric along one real parameter and records that the midpoint is "
      ++ "degenerate. The nearest published statement of Wick rotation as a "
      ++ "continuous parameter, on metrics rather than on algebras." },
  { key := "foot-1994"
    who := "Robert Foot 1994"
    work := "A note on Koide's lepton mass relation"
    where? := "arXiv:hep-ph/9402242"
    contribution :=
      "Equations (2) and (3): Koide's relation is the statement that the "
      ++ "vector of square-root masses stands at 45 degrees to the "
      ++ "direction with all three coordinates equal. The cone reading of "
      ++ "the quotient, published." },
  { key := "koide-1982"
    who := "Yoshio Koide 1982"
    work := "Fermion-boson two-body model of quarks and leptons and Cabibbo mixing"
    where? := "Lett. Nuovo Cimento 34, 201"
    contribution :=
      "The quotient of the sum of three masses by the square of the sum of "
      ++ "their square roots, and the observation that the charged leptons "
      ++ "put it at two thirds." },
  { key := "koide-2000"
    who := "Yoshio Koide 2000"
    work := "Universal seesaw mass matrix model with an S3 symmetry"
    where? := "arXiv:hep-ph/0005137"
    contribution :=
      "Equation (1.5) writes the three square-root masses as one amplitude "
      ++ "sampled at three phases a third of a turn apart, and equation "
      ++ "(2.5) computes the quotient as one plus the squared amplitude "
      ++ "ratio, over three. Table 1 reads that ratio per sector." },
  { key := "brannen-2006"
    who := "Carl Brannen 2006"
    work := "The lepton masses"
    where? := "brannenworks.com/MASSES2.pdf"
    contribution :=
      "Equations (3), (4) and (7): the same three-phase form derived as the "
      ++ "eigenvalues of a circulant matrix, with the quotient as one plus "
      ++ "twice the squared amplitude, over three. Equation (8) is the "
      ++ "amplitude the measured quotient forces." },
  { key := "sumino-2009"
    who := "Yukinari Sumino 2009"
    work := "Family gauge symmetry as an origin of Koide's mass formula and charged lepton spectrum"
    where? := "JHEP 0905, 075, arXiv:0812.2103"
    contribution :=
      "Section 4 assumes the Koide condition on the vacuum and proves that a "
      ++ "family gauge correction cancels the electromagnetic one that would "
      ++ "spoil it. A stability result for the relation, not a derivation "
      ++ "of it; the paper says so." },
  { key := "rodejohann-zhang-2011"
    who := "Werner Rodejohann and He Zhang 2011"
    work := "Extension of an empirical charged lepton mass relation to the neutrino sector"
    where? := "Phys. Lett. B 698, 152, arXiv:1101.5525"
    contribution :=
      "Equation (3): the light neutrino masses cannot carry the charged "
      ++ "lepton quotient, which stays below about 0.55 in normal ordering. "
      ++ "Equations (8) to (11) move the relation onto the seesaw matrices "
      ++ "or add Majorana phases instead." },
  { key := "nufit-6-2024"
    who := "Ivan Esteban, Concha Gonzalez-Garcia, Michele Maltoni, Thomas Schwetz,\n"
      ++ "Albert Zhou 2024"
    work := "NuFit-6.0: updated global analysis of three-flavor neutrino oscillations"
    where? := "JHEP 12, 216, arXiv:2410.05380"
    contribution :=
      "Table 1: the two measured mass-squared splittings and the mixing "
      ++ "angles, with their three-sigma ranges. The empirical input the "
      ++ "conditional mass theorems carry as hypotheses." },
  { key := "desi-dr2-2025"
    who := "DESI Collaboration 2025"
    work := "Constraints on neutrino physics from DESI DR2 BAO and DR1 full shape"
    where? := "Phys. Rev. D 112, 083513, arXiv:2503.14744"
    contribution :=
      "The sum of the neutrino masses is below 64.2 meV at 95 per cent in "
      ++ "the standard cosmology and below 163 meV once the dark energy "
      ++ "equation of state is allowed to run. The number the sum-rule "
      ++ "prediction is measured against." },
  { key := "gatto-sartori-tonin-1968"
    who := "Raoul Gatto, Guido Sartori and Mario Tonin 1968"
    work := "Weak self-masses, Cabibbo angle, and broken SU(2) x SU(2)"
    where? := "Phys. Lett. B 28, 128"
    contribution :=
      "The Cabibbo angle's sine is the square root of the down to strange "
      ++ "mass ratio. Stated here as a named hypothesis, not derived." },
  { key := "pdg-2024"
    who := "Particle Data Group 2024"
    work := "Review of Particle Physics"
    where? := "Phys. Rev. D 110, 030001"
    contribution :=
      "The charged lepton masses, the light quark mass ratio, and the "
      ++ "quark mixing grid. The measured numbers the conditional theorems "
      ++ "carry as interval hypotheses." },
  { key := "watson-1918"
    who := "George Neville Watson 1918"
    work := "The problem of the square pyramid"
    where? := "Messenger of Mathematics 48, 1"
    contribution :=
      "The sum of the first n squares is itself a square only at n = 1 and "
      ++ "n = 24, where it is 70 squared. Cited for the arithmetic fact "
      ++ "alone; theory/DEAD_ENDS.md records why no theorem here rests "
      ++ "on it." },
  { key := "conway-sloane-1999"
    who := "John Conway and Neil Sloane 1999"
    work := "Sphere Packings, Lattices and Groups, third edition, chapter 26"
    where? := "Springer, Grundlehren 290"
    contribution :=
      "The even unimodular Lorentzian lattices exist in ranks two more than "
      ++ "a multiple of eight, and in rank 26 the vector whose coordinates "
      ++ "are 0 through 24 with 70 in the time slot is null exactly by "
      ++ "Watson's identity. The rank condition is assumed, not derived, "
      ++ "which is the step theory/DEAD_ENDS.md records as not closing." },
  { key := "graves-cayley-1845"
    who := "John Graves 1843 and Arthur Cayley 1845"
    work := "On Jacobi elliptic functions, in reply to the Rev. B. Bronwin, and on quadratic forms"
    where? := "Philosophical Magazine 26, 208"
    contribution :=
      "The eight-dimensional algebra reached by doubling the quaternions, "
      ++ "with its eight-square norm identity. Graves found it in 1843 and "
      ++ "did not publish; Cayley published in 1845, which is why the "
      ++ "construction is read here as the doubling and not as a table." },
  { key := "dickson-1919"
    who := "Leonard Eugene Dickson 1919"
    work := "On quaternions and their generalization and the history of the eight square theorem"
    where? := "Annals of Mathematics 20, 155"
    contribution :=
      "The doubling written once as a rule on pairs, with the sign carried "
      ++ "as a parameter, so that the quaternions, the octonions and their "
      ++ "split forms are one construction run again. Apeiron.Doubling is "
      ++ "that rule with the sign named σ." },
  { key := "hurwitz-1898"
    who := "Adolf Hurwitz 1898"
    work := "Ueber die Composition der quadratischen Formen von beliebig vielen Variabeln"
    where? := "Nachrichten der Gesellschaft der Wissenschaften Goettingen 1898, 309"
    contribution :=
      "A sum of n squares composes multiplicatively only for n in 1, 2, 4 "
      ++ "and 8. Cited for the cap. The corpus proves the two rungs it "
      ++ "needs, the octonion composing and the sedenion admitting no "
      ++ "anisotropic composition norm, and does not reprove the cap." },
  { key := "zorn-1933"
    who := "Max Zorn 1933"
    work := "Alternativkoerper und quadratische Systeme"
    where? := "Abhandlungen aus dem Mathematischen Seminar der Universitaet Hamburg 9, 395"
    contribution :=
      "The split octonion as the second real form of the eight-dimensional "
      ++ "composition algebra, with an indefinite norm and zero divisors. "
      ++ "The name for what Apeiron.Doubling.SplitOctonion exhibits at "
      ++ "σ = 1." },
  { key := "bott-1959"
    who := "Raoul Bott 1959"
    work := "The stable homotopy of the classical groups"
    where? := "Annals of Mathematics 70, 313"
    contribution :=
      "The stable homotopy of the real classical groups repeats with "
      ++ "period eight. Cited as the name for the period. Nothing here "
      ++ "computes a homotopy group, and Apeiron.Doubling.Bott says so." },
  { key := "atiyah-bott-shapiro-1964"
    who := "Michael Atiyah, Raoul Bott and Arnold Shapiro 1964"
    work := "Clifford modules"
    where? := "Topology 3, Supplement 1, 3"
    contribution :=
      "The real Clifford algebras repeat with period eight, the rung "
      ++ "Cl(0,8) being the sixteen by sixteen real matrices. This is the "
      ++ "identification the corpus does not make: the eight it derives is "
      ++ "the octonion's dimension, and that this is that eight is cited." },
  { key := "vanderblij-1959"
    who := "Frederik van der Blij 1959"
    work := "An invariant of quadratic forms mod 8"
    where? := "Indagationes Mathematicae 21, 291"
    contribution :=
      "For a unimodular integral symmetric form of signature (p, q) with "
      ++ "characteristic vector w, the value of w paired with itself is "
      ++ "congruent to p minus q modulo eight. The corpus proves the "
      ++ "invariant is well defined and vanishes on even forms, and cites "
      ++ "this congruence for the bridge to the signature." },
  { key := "milnor-husemoller-1973"
    who := "John Milnor and Dale Husemoller 1973"
    work := "Symmetric Bilinear Forms, chapter II"
    where? := "Springer, Ergebnisse 73"
    contribution :=
      "The even unimodular lattices, the hyperbolic plane, and the "
      ++ "signature congruence in textbook form. Read for the definitions "
      ++ "the corpus builds natively, since Mathlib carries none of them." },
  { key := "bourbaki-1968"
    who := "Nicolas Bourbaki 1968"
    work := "Groupes et algebres de Lie, chapitres IV, V, VI, plate VII"
    where? := "Hermann, Paris"
    contribution :=
      "The E8 root system and its Cartan matrix, which is the Gram matrix "
      ++ "Mathlib carries as CartanMatrix.E8 and the corpus uses as the "
      ++ "definite block." },
  { key := "mathlib-cartan"
    who := "Mathlib contributors"
    work := "Mathlib/Data/Matrix/Cartan.lean"
    where? := "CartanMatrix.E8, E8_isSymm, E8_diag, proof_wanted E8_det"
    contribution :=
      "The E8 Cartan matrix and its symmetry and diagonal. Its determinant "
      ++ "is a proof_wanted there and is proved here from an explicit "
      ++ "integer inverse." },
  { key := "feigenbaum-1978"
    who := "Mitchell Feigenbaum 1978"
    work := "Quantitative universality for a class of nonlinear transformations"
    where? := "Journal of Statistical Physics 19, 25"
    contribution :=
      "The period doubling cascade of a unimodal map has a universal ratio, "
      ++ "reported numerically as 4.6692. Cited for the definition of the "
      ++ "renormalisation operator. The number is not claimed here." },
  { key := "lanford-1982"
    who := "Oscar Lanford 1982"
    work := "A computer-assisted proof of the Feigenbaum conjectures"
    where? := "Bulletin of the American Mathematical Society 6, 427"
    contribution :=
      "The doubling operator has a fixed point, and its derivative there "
      ++ "has one eigenvalue outside the unit disc. That eigenvalue is "
      ++ "what the corpus defines the ratio to be. The proof is machine "
      ++ "assisted with interval arithmetic, which Mathlib does not carry "
      ++ "at that scale." },
  { key := "pomeau-manneville-1980"
    who := "Yves Pomeau and Paul Manneville 1980"
    work := "Intermittent transition to turbulence in dissipative dynamical systems"
    where? := "Communications in Mathematical Physics 74, 189"
    contribution :=
      "The route to chaos through a saddle-node tangency, where a real "
      ++ "multiplier passes plus one and the laminar length grows as the "
      ++ "inverse square root of the distance past it. The name for the "
      ++ "parabolic face's route." },
  { key := "shenker-1982"
    who := "Scott Shenker 1982"
    work := "Scaling behavior in a map of a circle onto itself: empirical results"
    where? := "Physica D 5, 405"
    contribution :=
      "The critical circle map at golden rotation number scales with a "
      ++ "universal constant. Cited as the name of the route the golden "
      ++ "fixed point of the continued fraction step belongs to." },
  { key := "bhatt-kaushik-raj-singh-2022"
    who := "Sukrat Bhatt, Suvarna Kaushik, Ansh Raj and Tejinder Singh 2022"
    work := "Majorana neutrinos, exceptional Jordan algebra, and mass ratios for charged fermions"
    where? := "J. Phys. G 49, 045007, arXiv:2108.05787"
    contribution :=
      "Equation (63): the roots of one characteristic cubic of the "
      ++ "exceptional Jordan algebra put the quotient at exactly two "
      ++ "thirds. The only algebraic occurrence of that value found in the "
      ++ "sweep, stated as an observation rather than a theorem, and their "
      ++ "own mass ratios miss the measured value by four parts in a "
      ++ "thousand." },
  { key := "engeler-1981"
    who := "Erwin Engeler 1981"
    work := "Algebras and combinators"
    where? := "Algebra Universalis 13, 389-392"
    contribution :=
      "The graph model: from any set of atoms, a combinatory algebra "
      ++ "whose carrier is a powerset and whose application collects the "
      ++ "arrows of the left set whose sources lie in the right one. The "
      ++ "route that builds an applicative structure over a set without "
      ++ "asking that set to carry an operation." },
  { key := "scott-1976"
    who := "Dana Scott 1976"
    work := "Data types as lattices"
    where? := "SIAM Journal on Computing 5, 522-587"
    contribution :=
      "The universal domain of all subsets of the integers, and the graph "
      ++ "model on it. The predecessor of Engeler's construction and the "
      ++ "reason the carrier of such a model is a powerset." },
  { key := "rodenburg-2023"
    who := "Piet Rodenburg 2023"
    work := "Combinatory completeness in partial groupoids"
    where? := "arXiv:2302.09982"
    contribution :=
      "The combinatorially complete partial applicative systems are the "
      ++ "ones expandable by two constants satisfying the S and K "
      ++ "identities, and that class is larger than the reducts of partial "
      ++ "combinatory algebras. The reference for asking whether a partial "
      ++ "operation on the reals is combinatorially complete." },
  { key := "hyland-johnstone-pitts-1980"
    who := "Hyland, Johnstone and Pitts 1980"
    work := "Tripos theory"
    where? := "Math. Proc. Cambridge Philosophical Society 88, 205-232"
    contribution :=
      "The tripos-to-topos construction. A complete Heyting algebra gives "
      ++ "a forcing tripos by the same machinery a partial combinatory "
      ++ "algebra gives a realizability one, which is where the four "
      ++ "idempotents of the split face land." },
  { key := "aitken-1926"
    who := "A. C. Aitken 1926"
    work := "On Bernoulli's numerical solution of algebraic equations"
    where? := "Proceedings of the Royal Society of Edinburgh 46, 289-305"
    contribution :=
      "The delta-squared extrapolation. Three consecutive terms of a "
      ++ "sequence whose error is a single geometric mode determine that "
      ++ "mode's limit exactly, by a formula dividing the squared first "
      ++ "difference by the second difference. The depth-two collapse "
      ++ "this corpus applies to the partial sums of Girard's execution "
      ++ "formula, where the division is the loop's own resolvent." },
  { key := "girard-1989"
    who := "Jean-Yves Girard 1989"
    work := "Geometry of interaction I: interpretation of System F"
    where? := "Logic Colloquium '88, North-Holland, 221-260"
    contribution :=
      "Application as composition with feedback, and the execution "
      ++ "formula U(1 - sigma U)inv = U + U sigma U + ..., whose length is "
      ++ "the order of nilpotency of sigma U. The route in which "
      ++ "application is a sum over paths rather than a product." },
  { key := "abramsky-haghverdi-scott-2002"
    who := "Abramsky, Haghverdi and Scott 2002"
    work := "Geometry of Interaction and linear combinatory algebras"
    where? := "Mathematical Structures in Computer Science 12, 625-665"
    contribution :=
      "Every GoI situation yields a linear combinatory algebra, with "
      ++ "application built from the trace. The construction needs a "
      ++ "traced functor standing for the exponential of linear logic, "
      ++ "which is what supplies duplication and is what this corpus "
      ++ "does not have." },
  { key := "joyal-street-verity-1996"
    who := "Joyal, Street and Verity 1996"
    work := "Traced monoidal categories"
    where? := "Math. Proc. Cambridge Philosophical Society 119, 447-468"
    contribution :=
      "The trace axioms: tightening, sliding, vanishing over the unit and "
      ++ "over a pair, strength, and yanking. What a feedback operator "
      ++ "must satisfy before it may be called a trace." },
  { key := "haghverdi-scott-2010"
    who := "Haghverdi and Scott 2010"
    work := "Towards a typed geometry of interaction"
    where? := "Mathematical Structures in Computer Science 20, 1-49"
    contribution :=
      "Partial traces and the structured approach to solving feedback "
      ++ "equations, which is why a trace defined only where an inverse "
      ++ "exists is still a trace." },
  { key := "ehrhard-regnier-2003"
    who := "Ehrhard and Regnier 2003"
    work := "The differential lambda-calculus"
    where? := "Theoretical Computer Science 309, 1-41"
    contribution :=
      "A calculus with a term applying one argument to another linearly, "
      ++ "once, with confluence, strong normalisation in the typed case, "
      ++ "and Taylor expansion against linear head reduction. The syntax "
      ++ "the dual numbers supply arithmetic for." },
  { key := "shapiro-preguica-baquero-zawirski-2011"
    who := "Marc Shapiro, Nuno Preguica, Carlos Baquero and Marek Zawirski 2011"
    work := "Conflict-free Replicated Data Types"
    where? := "SSS 2011, LNCS 6976, 386-400, DOI 10.1007/978-3-642-24550-3_29"
    contribution :=
      "State-based CRDT convergence uses states in a join-semilattice, "
      ++ "monotone local updates, and merge as least upper bound. The join's "
      ++ "associativity, commutativity and idempotence remove delivery order "
      ++ "and duplication from the converged state. These are operation-level "
      ++ "conditions and do not classify a trichotomy face." },
  -- ── The comparators ─────────────────────────────────────────────────────────────────────────
  -- These are NOT sources the corpus draws on. They are the nearest prior art for the
  -- mechanisms the corpus realises, recorded here so that a `@cite` in a docstring can name
  -- the comparator and the comparison can be stated rather than implied. Each was resolved
  -- against its primary registry on 2026-11-23; none has been read in full, so each is a
  -- pointer and carries no evidential weight (see book/src/reference/bibliography.md).
  { key := "schlag-fast-weights-2021"
    who := "Imanol Schlag, Kazuki Irie and Jürgen Schmidhuber 2021"
    work := "Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021"
    where? := "arXiv:2102.11174"
    contribution :=
      "Comparator, not source. The nearest prior art for a forward computation that "
      ++ "writes the weights it later reads, and for the delta rule that corrects an "
      ++ "associative memory. The corpus's mechanism is NOT this: it changes the "
      ++ "REPRESENTATION, with an explicit inverse, and can fold the change into the "
      ++ "weights (rotor_folds_into_weight). A memory update is not a representation "
      ++ "change, and the memory's capacity and interference arguments do not transfer "
      ++ "to a prescribed basis." },
  { key := "katharopoulos-linear-attention-2020"
    who := "Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas and François Fleuret 2020"
    work := "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, NeurIPS 2020"
    where? := "arXiv:2006.16236"
    contribution :=
      "Comparator, not source. The nearest prior art for the predictive READ: a ratio "
      ++ "of a numerator to a normaliser over accumulated state. The corpus's read has "
      ++ "the same shape (attnOut_is_bulk_ratio) and differs in what the numerator and "
      ++ "the drag ARE: channel coordinates of a cell, not inner products with a feature "
      ++ "map; the state is O(1) and exact, not linear in the sequence length; and the "
      ++ "read has a refusal where a normaliser can vanish." },
  { key := "zhuang-acil-2022"
    who := "Huiping Zhuang, Zhenyu Weng, Hongxin Wei, Renchunzi Xie, Kar-Ann Toh and Zhiping Lin 2022"
    work := "ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection, NeurIPS 2022"
    where? := "arXiv:2205.14922"
    contribution :=
      "Comparator, not source. The nearest prior art for the claim that retaining an "
      ++ "analytic state reproduces the joint solution without retaining the samples. "
      ++ "The corpus's additive face makes that statement for the Gram ledger "
      ++ "(stage_removal_exact, normal_equations_order_free); the overlap is on the "
      ++ "CLAIM, and the corpus's addition is that the state is the cell's product as "
      ++ "well as its sum, so that the same inverse serves the execution face." },
  { key := "chen-liu-bls-2018"
    who := "C. L. Philip Chen and Zhulin Liu 2018"
    work := "Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture, IEEE TNNLS 29"
    where? := "DOI 10.1109/TNNLS.2017.2716952"
    contribution :=
      "Comparator, not source. The older incremental-analytic line: mapped features, "
      ++ "enhancement nodes, and incremental update without complete retraining. It is "
      ++ "the ancestor of the 'keep the state, not the samples' claim and is named here "
      ++ "so that the corpus's version of that claim is not read as the origin of it." },
  { key := "quan-federated-unlearning-2026"
    who := "Yijun Quan, Wentai Wu and Giovanni Montana 2026"
    work := "Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models"
    where? := "arXiv:2603.12977"
    contribution :=
      "Comparator, not source. The closest prior art for the corpus's Gram ledger: "
      ++ "fixed-size additive sufficient statistics from clients, a server supporting "
      ++ "interleaved additions and deletions, and a head recovered pointwise identical "
      ++ "to centralised retraining IN EXACT ARITHMETIC. The corpus's mechanism is NOT "
      ++ "novel here; what the corpus adds is the arithmetic (exact in a fixed i64 ring, "
      ++ "not exact arithmetic over the reals), the cost (O(d^2), no re-scan of the "
      ++ "remaining peers), and the machine-checked link from the Rust to the canonical "
      ++ "definition." },
  { key := "alsulaimawi-federated-ridge-2026"
    who := "Zahir Alsulaimawi 2026"
    work := "One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation"
    where? := "arXiv:2601.08216"
    contribution :=
      "Comparator, not source. Aggregating Gram and feature-response statistics and "
      ++ "solving the global ridge problem once is the corpus's additive face, and this "
      ++ "is the prior art for it. Named so that the accumulation law is not presented as "
      ++ "the corpus's own." },
  { key := "jayraj-unlearning-curation-2026"
    who := "Priyank Jayraj, Devanshu Sahoo and Uday Mittal 2026"
    work := "Exact Unlearning After Data Curation"
    where? := "arXiv:2610.33426"
    contribution :=
      "Comparator, not source. Prior art for exact removal of a contributed block. The "
      ++ "corpus's subtraction (stage_removal_exact) is an instance of this established "
      ++ "shape; the contribution is the cost and the verification, not the shape." },
  { key := "tavory-unlearning-quantized-2026"
    who := "Ami Tavory, Shripad Gade, Tal Sarig, Noam Touitou and Ido Guy 2026"
    work := "Exact Unlearning via Quantized Sufficient Statistics"
    where? := "arXiv:2610.07197"
    contribution :=
      "Comparator, not source. Prior art for exact removal from sufficient statistics, "
      ++ "in the quantized case. It is the nearest neighbour of the corpus's own "
      ++ "quantized carriers and of statfold's exact-merge and exact-unmerge, and it is "
      ++ "named here rather than left to be rediscovered." },
  { key := "gonzalez-linear-dynamical-2025"
    who := "Xavier Gonzalez, E. Kelly Buchanan, Hyun Dong Lee, Jerry Weihong Liu, Ke Alexander Wang, David M. Zoltowski and Leo Kozachkov 2025"
    work := "A Unifying Framework for Parallelizing Sequential Models with Linear Dynamical Systems"
    where? := "arXiv:2509.21716"
    contribution :=
      "Comparator, not source. The survey-level account of associative scan and of the "
      ++ "composition law for affine recurrences, which the corpus's affine laws "
      ++ "(AnyRing.lean) formalise. The corpus's contribution is the verified "
      ++ "integration, not the invention of the composition law." },
  { key := "sun-test-time-learning-2024"
    who := "Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois and others 2024"
    work := "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
    where? := "arXiv:2407.04620"
    contribution :=
      "Comparator, not source. Prior art for blurring the training and inference "
      ++ "boundary by updating a state during the forward computation. The corpus's "
      ++ "update is a composable accumulation with a supported inverse, which is a "
      ++ "different update rule from a learning step on a hidden model." },
  { key := "guan-merging-vs-ensembling-2019"
    who := "Zoe Guan, Giovanni Parmigiani and Prasad Patil 2019"
    work := "Merging versus Ensembling in Multi-Study Prediction: Theoretical Insight from Random Effects"
    where? := "arXiv:1905.07382"
    contribution :=
      "Comparator, not source. The statistical account of when pooling per-unit "
      ++ "estimates differs from averaging their predictions. It is the prior art for "
      ++ "the distinction the corpus's shard law rests on: a shard is a contribution to a "
      ++ "shared state, not a separately fitted model whose predictions are averaged." }
]

/-- Is `k` a registered citation key? -/
def knownSource (k : String) : Bool := sources.any (fun s => s.key == k)

end Apeiron.Linter
