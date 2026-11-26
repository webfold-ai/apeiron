# marketing

Positioning, not API. Nothing in this directory is a contract: the laws are in
`caretaker/docs/{ml-architecture-standard,rust-implementation-spec}.md`, the primitives are in
`src/`, and the evidence is in `bench/` and `bench-results/`.

| file | what it is | where it also belongs |
|---|---|---|
| `ATLAS-PATTERN.md` | the four moves every workload is solved with, and the three mistakes they prevent - the guide that gets extracted from the code and cited by it | the docs repository (`webfold-documentation-saas`), which is where positioning should live; it was in `docs/`, between the GGUF format and the licensing notes, which is the wrong shelf |

It was moved out of `docs/` because a core crate's `docs/` is for the things a caller has to know to
use the crate. This document is addressed to someone deciding whether the approach is worth adopting.
