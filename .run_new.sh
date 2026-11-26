#!/usr/bin/env bash
set -uo pipefail
cargo build --release --example engine_profile 2>&1 | tail -2
BIN=$(find /home/ctown-admin/factory/.target /home/ctown-admin/factory/apeiron/target -name engine_profile -type f 2>/dev/null | head -1)
echo "binary: $BIN"
"$BIN" --model /home/ctown-admin/models/q05b-q8_0.gguf --ids /home/ctown-admin/models/q05b.ids \
  --pp 512 --tg 64 --reps 3 --label new-engine --out /home/ctown-admin/results/paired/new-engine 2>&1 | tail -4
echo "===== OLD ====="; sed -n '/^| metric/,/^$/p' /home/ctown-admin/results/paired/old-engine/*.md | head -9
echo "===== NEW ====="; sed -n '/^| metric/,/^$/p' /home/ctown-admin/results/paired/new-engine/*.md | head -9
