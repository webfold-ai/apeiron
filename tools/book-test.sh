#!/usr/bin/env bash
# Build the crate, then build the book and compile every Rust block in it against the crate.
#
#   tools/book-test.sh
#
# Needs `mdbook` on PATH (cargo install mdbook). Exits non-zero if the crate, the book, or any
# example fails. Also runs tools/book-check.py (names and numbers the book states).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

command -v mdbook >/dev/null || { echo "book-test: mdbook not found (cargo install mdbook)" >&2; exit 127; }

cargo build --lib --locked
python3 tools/book-check.py
mdbook build book
# The doctest crate path must be the one cargo actually wrote to, AND the crate
# set must be unambiguous.
#
# A relative `target/debug/deps` is wrong whenever CARGO_TARGET_DIR points
# elsewhere (this box shares /home/ctown/.cache/cargo-target across many
# checkouts). Worse, that shared dir holds one libapeiron-*.rlib per checkout,
# so rustdoc reports "multiple candidates for `rlib` dependency `apeiron`" and
# every doctest in the book fails for a reason that has nothing to do with the
# book's content.
#
# mdbook's `test` subcommand takes no extra rustdoc arguments (no --extern),
# so the ambiguity cannot be resolved from the command line. The gate therefore
# owns its target directory: deterministic, per checkout, and reproducible.
export CARGO_TARGET_DIR="${BOOK_TEST_TARGET_DIR:-$(pwd)/target}"
cargo build --lib --locked
mdbook test book -L "$CARGO_TARGET_DIR/debug/deps"
echo "book-test: PASS"
