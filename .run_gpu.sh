#!/usr/bin/env bash
set -uo pipefail
echo "== the GPU on the box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -2 || echo "no nvidia-smi"
echo "== vulkan ICDs"
ls /usr/share/vulkan/icd.d/ 2>/dev/null | head -4
echo "== build with the gpu feature"
cargo build --release --features gpu --example engine_profile 2>&1 | tail -3
BIN=$(find /home/ctown-admin/factory/.target/apeiron -name engine_profile -type f 2>/dev/null | head -1)
echo "binary: $BIN"
"$BIN" --model /home/ctown-admin/models/q05b-q8_0.gguf --ids /home/ctown-admin/models/q05b.ids \
  --pp 512 --tg 64 --reps 3 --label new-engine-gpu --out /home/ctown-admin/results/paired/new-engine-gpu 2>&1 | grep -iE 'vulkan|device|gpu|error|wrote' | head -8
echo "===== GPU RUN ====="; sed -n '/^| metric/,/^$/p' /home/ctown-admin/results/paired/new-engine-gpu/*.md 2>/dev/null | head -8
