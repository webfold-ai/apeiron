# Security policy

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting:
<https://github.com/webfold-ai/apeiron/security/advisories/new>, or by email to
security@twn.systems. Do not open a public issue for a suspected vulnerability.

Include the affected revision, the input or model file that triggers it (or how to build one), and
what you observed. A report is acknowledged within 3 days. A confirmed issue gets a fix or a
published advisory within 90 days of the report, sooner when it is being exploited. Reporters are
credited in the advisory unless they ask otherwise.

The project signs no non-disclosure agreement. A vulnerability found with AI tools is treated as
already public: anyone running the same tools can find it, so its fix is not held back for a
coordinated date.

## Supported versions

| version | supported |
|---|---|
| `main` | yes |
| `0.1.x` | yes, latest patch only |
| older | no |

Before 1.0, fixes land on `main` and in the next `0.1.x` patch release; there are no backports.

## Scope

In scope:

- **GGUF input.** The GGUF reader (`src/format/`) and `Engine::open` (`src/engine/`) take untrusted
  model files. A crash, hang, unbounded allocation or out-of-bounds access from a crafted file is a
  vulnerability. Both are fuzzed (`fuzz/`).
- **`unsafe` code** in `src/engine/`, `src/gpu/`, `src/map/hashtrinity/` and `src/map/table/`. Undefined
  behaviour reachable from safe public API is a vulnerability.
- **The hash tables** (`src/map/table.rs`, `src/map/hashtrinity/`): a key set that drives a lookup or insert
  into unbounded probing or allocation is a vulnerability.

Stated limits, reported as known rather than as vulnerabilities:

- **Cryptography is unaudited.** `aes`, `chacha`, `sha256` and `crypto` are tested against known-answer
  vectors only, are not side-channel resistant (`aes` indexes its S-box by secret bytes), and have not
  been reviewed. Do not use them to protect real data. Their rooting onto `R[x]/(x² − σ)` is in
  progress.
- **The GPU path is single-tenant.** Vulkan buffers run without `robustBufferAccess` and workgroup
  memory is not cleared between dispatches. Do not share one device between mutually untrusted users
  through this crate.

## No network listener

The crate opens no socket: there is no network listener, server or client in `src/` or `examples/`.
Any network exposure is the embedding application's. A change that adds one is a security-relevant
change and is reviewed as such.
