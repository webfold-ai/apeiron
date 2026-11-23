# Bibliography

A citation is a pointer until the source is retrieved and read. The status column says what was
done for this book.

| source | used for | status |
|---|---|---|
| Ho, S. and Protzenko, J. *Aeneas: Rust Verification by Functional Translation.* Proc. ACM Program. Lang. 6 (ICFP), 2022. doi:10.1145/3547647, arXiv:2206.07185 | Rust to Lean extraction | abstract page retrieved; paper not read in full |
| Ho, S., Boisseau, G., Franceschino, L., Prak, Y., Fromherz, A., Protzenko, J. *Charon: An Analysis Framework for Rust.* CAV 2025 (LNCS), doi:10.1007/978-3-031-98685-7_18; arXiv:2410.18042, 2024 | front end of the extraction | abstract page retrieved; paper not read in full |
| Gauss, C. F. *Disquisitiones Arithmeticae.* Leipzig, 1801 (book, no DOI) | norm form | pointer |
| Cockle, J. "On certain functions resembling quaternions, and on a new imaginary in algebra." *Philosophical Magazine* 33, 1848, 435–439. doi:10.1080/14786444808646139 | the hyperbolic face | pointer |
| Clifford, W. K. "Preliminary Sketch of Biquaternions." *Proc. London Math. Soc.* s1-4, 1873, 381–395. doi:10.1112/plms/s1-4.1.381 | the parabolic face | pointer |
| Knuth, D. E. *The Art of Computer Programming*, vol. 3, section 6.4 (book, no DOI; ISBN 978-0-201-89685-5) | multiplicative hashing | pointer |
| de Moura, L. and Ullrich, S. *The Lean 4 Theorem Prover and Programming Language.* CADE-28 (LNCS), 2021. doi:10.1007/978-3-030-79876-5_37. The Mathlib Community. *The Lean Mathematical Library.* CPP 2020. doi:10.1145/3372885.3373824. Pinned at `v4.31.0` | the proof corpus | the pinned toolchain is built and run by `proof/regen.sh` and `lake build` |
| ggml and llama.cpp (github.com/ggml-org/llama.cpp; no DOI) | GGUF and the quantisation block layouts; the throughput comparator | the repository pins the comparator revision in each receipt |
| Fisher, R. A. "On the mathematical foundations of theoretical statistics." *Phil. Trans. R. Soc. A* 222, 1922, 309–368. doi:10.1098/rsta.1922.0009 | Fisher information ("intrinsic accuracy"), `Numeric/Quantum/Ledger/ExactUncertainty.lean` | pointer |
| Fisher, R. A. "Theory of statistical estimation." *Proc. Camb. Phil. Soc.* 22, 1925, 700–725. doi:10.1017/S0305004100009580 | Fisher information `I_F = ∫ ρ'²/ρ`, `Numeric/Quantum/Ledger/ExactUncertainty.lean` | pointer |
| Cauchy, A.-L. *Cours d'analyse de l'École royale polytechnique.* Paris, 1821 (book, no DOI) | Cauchy–Schwarz for finite sums, `cauchy_schwarz_fisher` | pointer |
| Bunyakovsky, V. Mémoire in *Mém. Acad. Imp. Sci. St.-Pétersbourg* (7) 1, no. 9, 1859 (no DOI) | Cauchy–Schwarz for integrals, `cauchy_schwarz_fisher` | pointer |
| Schwarz, H. A. "Über ein die Flächen kleinsten Flächeninhalts betreffendes Problem der Variationsrechnung." *Acta Soc. Sci. Fennicae* 15, 1888, 315–362 (Weierstrass Festschrift, dated 1885; no DOI) | Cauchy–Schwarz in inner-product form, `cauchy_schwarz_fisher` | pointer |
| Aitken, A. C. and Silverstone, H. "On the estimation of statistical parameters." *Proc. R. Soc. Edinburgh A* 61, 1942, 186–194. doi:10.1017/S008045410000618X | the Cramér–Rao bound, `cramer_rao` | pointer |
| Fréchet, M. "Sur l'extension de certaines évaluations statistiques au cas de petits échantillons." *Rev. Inst. Int. Stat.* 11, 1943, 182. doi:10.2307/1401114 | the Cramér–Rao bound, one parameter, `cramer_rao` | pointer |
| Rao, C. R. "Information and the accuracy attainable in the estimation of statistical parameters." *Bull. Calcutta Math. Soc.* 37, 1945, 81–91 (no DOI); reprinted in *Breakthroughs in Statistics* I, Springer, 1992, 235–247, doi:10.1007/978-1-4612-0919-5_16 | the Cramér–Rao bound, `cramer_rao` | pointer |
| Darmois, G. "Sur les limites de la dispersion de certaines estimations." *Rev. Inst. Int. Stat.* 13, 1945, 9. doi:10.2307/1400974 | the Cramér–Rao bound, several parameters | pointer |
| Cramér, H. *Mathematical Methods of Statistics.* Princeton University Press, 1946 (book, no DOI) | the Cramér–Rao bound, `cramer_rao` | pointer |
| Heisenberg, W. "Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik." *Z. Phys.* 43, 1927, 172–198. doi:10.1007/BF01397280 | the uncertainty relation, heuristic form | pointer |
| Kennard, E. H. "Zur Quantenmechanik einfacher Bewegungstypen." *Z. Phys.* 44, 1927, 326–352. doi:10.1007/BF01391200 | first proof of `σ_x σ_p ≥ ℏ/2` | pointer |
| Madelung, E. "Quantentheorie in hydrodynamischer Form." *Z. Phys.* 40, 1927, 322–326. doi:10.1007/BF01400372 | the Madelung equations, `ψ = √ρ e^{iS/ℏ}` | pointer |
| Weyl, H. *Gruppentheorie und Quantenmechanik.* Hirzel, Leipzig, 1928 (book, no DOI) | independent proof of the position–momentum relation | pointer |
| Robertson, H. P. "The uncertainty principle." *Phys. Rev.* 34, 1929, 163–164. doi:10.1103/PhysRev.34.163 | the uncertainty relation for any two observables, by Cauchy–Schwarz | pointer |
| Bohm, D. "A suggested interpretation of the quantum theory in terms of 'hidden' variables. I, II." *Phys. Rev.* 85, 1952, 166–179 and 180–193. doi:10.1103/PhysRev.85.166, doi:10.1103/PhysRev.85.180 | the quantum potential `Q`, `Numeric/Quantum/Ledger/ExactUncertainty.lean` | pointer |
| Helstrom, C. W. *Quantum Detection and Estimation Theory.* Academic Press, 1976 (book, no DOI) | quantum estimation theory | pointer |
| Braunstein, S. L. and Caves, C. M. "Statistical distance and the geometry of quantum states." *Phys. Rev. Lett.* 72, 1994, 3439–3443. doi:10.1103/PhysRevLett.72.3439 | quantum Fisher information | pointer |
| Reginatto, M. "Derivation of the equations of nonrelativistic quantum mechanics using the principle of minimum Fisher information." *Phys. Rev. A* 58, 1998, 1775–1778. doi:10.1103/PhysRevA.58.1775 | `∫ρQ = (ℏ²/8m) I_F`, `integral_rho_Q` | pointer |
| Frieden, B. R. *Physics from Fisher Information.* Cambridge University Press, 1998. doi:10.1017/CBO9780511622670 | Fisher information in physics | pointer |
| Hall, M. J. W. and Reginatto, M. "Schrödinger equation from an exact uncertainty principle." *J. Phys. A* 35, 2002, 3289–3303. doi:10.1088/0305-4470/35/14/310; arXiv:quant-ph/0102069 | `exact_uncertainty` (their eq. 19, `δx ΔN = ℏ/2`), the Cramér–Rao route to Heisenberg (`cramer_rao`, `heisenberg_from_cr`) | arXiv v2 retrieved; §3.4 (eq. 19 and the Cramér–Rao step) and the references read |
| Werner, R. F. and Farrelly, T. "Uncertainty from Heisenberg to today." *Found. Phys.* 49, 2019, 460–491. doi:10.1007/s10701-019-00265-z; arXiv:1904.06139 | the Heisenberg, Kennard, Weyl and Robertson history | arXiv v1 retrieved; the history of the preparation relation read |
| Bloch, Y. and Cohen, E. "The identification of mean quantum potential with Fisher information leads to a strong uncertainty relation." *Found. Phys.* 52, 2022, 117. doi:10.1007/s10701-022-00638-x; arXiv:2210.07732 | `integral_rho_Q` (their eq. 13) and its translation-invariance premise | arXiv v1 retrieved; §§I–II read |

## The comparators

These are **not** sources the corpus draws on. They are the nearest prior art for the mechanisms
the corpus realises, recorded here so that the comparison can be stated rather than implied.
Each identifier was resolved against its primary registry on **2026-11-23** — the arXiv abstract
page for the arXiv entries, Crossref for the DOI — and the title, authors and year below are
copied from that record. **None has been read in full**, so every one of them is a **pointer**
under the rule at the top of this file, and none carries evidential weight in the
[grades](../verification/evidence.md). The full six-axis comparison is at
`Ptyktos/research` `drafts/apeiron-vs-priors/`.

| source | used for | status |
|---|---|---|
| Schlag, I., Irie, K. and Schmidhuber, J. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. arXiv:2102.11174 | the comparator for a forward computation that writes the weights it later reads, and for the delta rule (`SpinorFold.lean`, `Nous/Combinator/CellTransform.lean`) | arXiv abstract page retrieved 2026-11-23; not read in full |
| Katharopoulos, A., Vyas, A., Pappas, N. and Fleuret, F. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* NeurIPS 2020. arXiv:2006.16236 | the comparator for the predictive read (`NullTransformer`, `Nous/Model/Weighting.lean`) | arXiv abstract page retrieved 2026-11-23; not read in full |
| Zhuang, H., Weng, Z., Wei, H., Xie, R., Toh, K.-A. and Lin, Z. *ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection.* NeurIPS 2022. arXiv:2205.14922 | the comparator for retaining analytic state in place of samples (`TrainingCollapse.lean`) | arXiv abstract page retrieved 2026-11-23; not read in full |
| Chen, C. L. P. and Liu, Z. *Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture.* IEEE TNNLS 29 (2018). doi:10.1109/TNNLS.2017.2716952 | the ancestor of the incremental-analytic line, named so the corpus's version of that claim is not read as the origin of it | Crossref record resolved 2026-11-23; not read in full |
| Quan, Y., Wu, W. and Montana, G. *Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models.* arXiv:2603.12977 | the closest prior art for the Gram ledger and its removal (`TrainingCollapse.lean`, `webtrainer-core`) | arXiv abstract page retrieved 2026-11-23; not read in full |
| Alsulaimawi, Z. *One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation.* arXiv:2601.08216 | the prior art for aggregating Gram and cross-moment statistics and solving once | arXiv abstract page retrieved 2026-11-23; not read in full |
| Jayraj, P., Sahoo, D. and Mittal, U. *Exact Unlearning After Data Curation.* arXiv:2610.33426 | the prior art for exact removal of a contributed block | arXiv abstract page retrieved 2026-11-23; not read in full |
| Tavory, A., Gade, S., Sarig, T., Touitou, N. and Guy, I. *Exact Unlearning via Quantized Sufficient Statistics.* arXiv:2610.07197 | the prior art for exact removal from quantized sufficient statistics — the nearest neighbour of the corpus's own quantized carriers | arXiv abstract page retrieved 2026-11-23; not read in full |
| Gonzalez, X., Buchanan, E. K., Lee, H. D., Liu, J. W., Wang, K. A., Zoltowski, D. M. and Kozachkov, L. *A Unifying Framework for Parallelizing Sequential Models with Linear Dynamical Systems.* arXiv:2509.21716 | the account of associative scan and the affine composition law the corpus's `AnyRing.lean` laws formalise | arXiv abstract page retrieved 2026-11-23; not read in full |
| Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y. and others. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv:2407.04620 | the prior art for updating a state during the forward computation | arXiv abstract page retrieved 2026-11-23; not read in full |
| Guan, Z., Parmigiani, G. and Patil, P. *Merging versus Ensembling in Multi-Study Prediction: Theoretical Insight from Random Effects.* arXiv:1905.07382 | the statistical account of when pooling estimates differs from averaging predictions — the distinction the shard law rests on | arXiv abstract page retrieved 2026-11-23; not read in full |

Each DOI above was resolved against Crossref's record for the cited title, authors and year; no DOI means the source has none. Retrieving and reading the pointer entries is open work. Until then each is a pointer and carries
no evidential weight in the [grades](../verification/evidence.md).
