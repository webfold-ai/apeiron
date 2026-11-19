#!/bin/sh
# Box-side: THE WHOLE MODEL.
#
# Every two-dimensional weight of a real PyTorch checkpoint goes through the monoid
# round-trip: read -> lift -> fold a domain shard -> fold a revoked shard -> subtract the
# revoked shard -> solve -> export. The export is F32, so the round-trip's own error is
# preserved (an f16 or q8_0 container would quantise 3.8e-6 away).
#
# The rows are streamed to rows.jsonl as they happen, so a disk failure cannot lose the
# measurements. The behavioural check runs afterwards, in PyTorch, in memory.
set -u
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ctown-admin/factory/.target/apeiron-rt

BOXDIR=/home/ctown-admin/factory/apeiron-rt
SRC=/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct
# The box root is full (other jobs), so the export goes to RAM-backed tmpfs:
# 36 GB of /dev/shm, and the export is ~2 GB. The measurements are copied back afterwards.
OUT=/dev/shm/rt_full
say() { echo "=== $* ==="; }

mkdir -p "$OUT"

say "build"
cd "$BOXDIR" || exit 1
cargo build --release --example checkpoint_roundtrip > "$BOXDIR/.box/rt2_build.log" 2>&1 || {
  echo "BUILD FAILED"; grep -n "error" -A 10 "$BOXDIR/.box/rt2_build.log" | head -50; exit 1; }
echo "build ok"

say "whole-model round-trip (background: this takes a while)"
nohup "$CARGO_TARGET_DIR/release/examples/checkpoint_roundtrip" \
  --input "safetensors:$SRC" --out "$OUT" --all --calib-rows 512 --seed 7 --threads 8 \
  > /dev/shm/rt_full.log 2>&1 &
echo "pid $!"
sleep 45
tail -n 5 /dev/shm/rt_full.log
wc -l "$OUT/rows.jsonl" 2>/dev/null || true
df -h / | tail -1
