#!/usr/bin/env bash
set -uo pipefail
LABEL="${1:-old-engine}"
OUT="/home/ctown-admin/results/paired/${LABEL}"
mkdir -p "$OUT"
BIN=$(find /home/ctown-admin/factory/.target /home/ctown-admin/factory/apeiron/target -name engine_profile -type f 2>/dev/null | head -1)
echo "binary: $BIN"
[ -n "$BIN" ] || { echo "no binary"; exit 2; }
"$BIN" --model /home/ctown-admin/models/q05b-q8_0.gguf --ids /home/ctown-admin/models/q05b.ids \
  --pp 512 --tg 64 --reps 3 --label "$LABEL" --out "$OUT" 2>&1 | tail -10
echo "== receipts"; ls "$OUT" | head -4
