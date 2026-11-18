#!/bin/sh
# The box root is FULL (other jobs), so this script is piped over stdin (no file write)
# and the export goes to RAM-backed tmpfs (/dev/shm, 36 GB). The example binary is
# already built in /home/ctown-admin/factory/.target/apeiron-rt.
set -u
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ctown-admin/factory/.target/apeiron-rt
BOXDIR=/home/ctown-admin/factory/apeiron-rt
SRC=/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct
OUT=/dev/shm/rt_full
mkdir -p "$OUT"
echo "=== whole-model round-trip (background) ==="
nohup "$CARGO_TARGET_DIR/release/examples/checkpoint_roundtrip" \
  --input "safetensors:$SRC" --out "$OUT" --all --calib-rows 512 --seed 7 --threads 8 \
  > /dev/shm/rt_full.log 2>&1 &
echo "pid $!"
sleep 60
tail -n 4 /dev/shm/rt_full.log
wc -l "$OUT/rows.jsonl" 2>/dev/null || true
df -h /dev/shm | tail -1
