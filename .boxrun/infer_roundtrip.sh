#!/bin/sh
# Box-side: the INFERENCE roundtrip, like-for-like.
#
# Ours: apeiron's engine (examples/engine_profile) on a real checkpoint.
# Incumbent: llama.cpp's llama-bench on the SAME file, SAME host, interleaved.
# The harness (examples/engine_profile.rs) already does the interleaving, the
# warm-ups and the quiet-host wait; this script only builds and runs it.
#
# One job. The build is cached in /home/ctown-admin/factory/.target/<mirror>/.
set -u
export PATH=$HOME/.cargo/bin:$PATH

say() { echo "=== $* ==="; }

MODEL=${MODEL:-/home/ctown-admin/models/lfm2.5-8b-a1b/LFM2.5-8B-A1B-Q4_K_M.gguf}
LLAMA=${LLAMA:-/home/ctown-admin/llama.cpp}
REPS=${REPS:-3}
PP=${PP:-200}
TG=${TG:-64}

mkdir -p .box/out

say "host"
echo "model: $MODEL"
ls -l "$MODEL" 2>&1 | head -2
nproc
df -h / | tail -1

say "build llama-bench (the incumbent)"
cmake --build "$LLAMA/build" --target llama-bench -j 4 > .box/llama_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "LLAMA-BENCH BUILD FAILED (exit $rc)"
  tail -n 40 .box/llama_build.log
  exit $rc
fi
ls -l "$LLAMA/build/bin/llama-bench"

say "build our engine_profile + generate"
cargo build --release --example engine_profile --example generate > .box/build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAILED (exit $rc) - not running a stale binary"
  grep -n "error" -A 12 .box/build.log | head -n 80
  exit $rc
fi
echo "build ok"
ls -l "${CARGO_TARGET_DIR:-$PWD/target}/release/examples/" | grep -E "engine_profile|generate" | head -4

say "run: engine_profile, ours vs llama-bench, interleaved, $REPS reps"
BIN="${CARGO_TARGET_DIR:-$PWD/target}/release/examples/engine_profile"
"$BIN" \
  --model "$MODEL" \
  --ids tests/fixtures/qwen35-prompt2.ids \
  --pp "$PP" --tg "$TG" --reps "$REPS" \
  --llama "$LLAMA/build/bin/llama-bench" \
  --label "LFM2.5-8B-A1B Q4_K_M / RTX3060 host" \
  --out .box/out > .box/engine_profile.log 2>&1
rc=$?
echo "engine_profile exit: $rc"
tail -n 40 .box/engine_profile.log
ls -l .box/out 2>&1 | head -10

say "done"
