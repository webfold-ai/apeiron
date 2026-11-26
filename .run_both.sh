#!/usr/bin/env bash
set -uo pipefail
cat > /tmp/tbl.sh <<'INNER'
sed -n '/^| metric/,/^$/p' "$1" | head -12
INNER
bash /tmp/tbl.sh /home/ctown-admin/results/paired/old-engine/*.md
echo "== building the new engine"
cargo build --release --example engine_profile 2>&1 | tail -3
BIN=$(find /home/ctown-admin/factory/.target /home/ctown-admin/factory/apeiron/target -name engine_profile -type f 2>/dev/null | head -1)
"$BIN" --model /home/ctown-admin/models/q05b-q8_0.gguf --ids /home/ctown-admin/models/q05b.ids \
  --pp 512 --tg 64 --reps 3 --label new-engine --out /home/ctown-admin/results/paired/new-engine 2>&1 | tail -6
bash /tmp/tbl.sh /home/ctown-admin/results/paired/new-engine/*.md
