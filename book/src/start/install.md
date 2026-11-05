# Install

The crate needs Rust 1.89 or later. The default build has no dependencies beyond `libc` on Linux.

```sh
git clone https://github.com/webfold-ai/apeiron
cd apeiron
cargo build --lib
```

The default build has no dependencies on any platform; on Linux it adds `libc` for thread affinity.
Two features are off by default:

| feature | adds |
|---|---|
| `gpu` | the Vulkan runtime, loaded with `dlopen` at run time; no crates |
| `derive` | the `apeiron-derive` proc-macro crate |

Run the tests:

```sh
cargo test --workspace --all-features --locked
```

The Lean development needs the pinned toolchain; see [Extraction](../verification/extraction.md)
and `proof/README.md`.

Next: [A Cell](cell.md).
