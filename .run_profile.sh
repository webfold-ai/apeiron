#!/usr/bin/env bash
set -uo pipefail
GGUF=/home/ctown-admin/models/q05b-q8_0.gguf
IDS=/home/ctown-admin/models/q05b.ids
mkdir -p /home/ctown-admin/models /home/ctown-admin/results/paired
if [ ! -f "$GGUF" ]; then
  echo "== downloading the weights"
  curl -fL --retry 2 -o "$GGUF" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf" || { echo "download failed"; exit 2; }
fi
ls -la "$GGUF"
python3 -c "open('${IDS}','w').write(' '.join(str(100 + (i % 500)) for i in range(512)))"
echo "== tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
cargo build --release --example engine_profile 2>&1 | tail -3
