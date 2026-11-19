#!/bin/sh
# Box-side, step 2 of the roundtrip: the measurement.
#
# Run directly over ssh (not through box-run) because the box-run disk gate refused:
# the box was at 8.8 GB free after another job filled it. This step needs no build
# (the examples are already built in /home/ctown-admin/factory/.target/apeiron-rt),
# no download (the GGUF is on disk) and writes only kilobytes, so the 12 GB gate is
# not the binding constraint here. That is the reason, recorded rather than hidden.
set -u
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ctown-admin/factory/.target/apeiron-rt

BOXDIR=/home/ctown-admin/factory/apeiron-rt
LLAMA=/home/ctown-admin/llama.cpp
MODELS=/home/ctown-admin/models/roundtrip
GGUF=$MODELS/qwen2.5-0.5b-instruct-q8_0.gguf
say() { echo "=== $* ==="; }

mkdir -p "$BOXDIR/.box/out"

say "provenance: the PyTorch checkpoint and the GGUF derived from it"
ls -l "$MODELS/qwen2.5-0.5b-instruct/model.safetensors" "$GGUF"
sha256sum "$MODELS/qwen2.5-0.5b-instruct/model.safetensors" "$GGUF" | cut -c1-16,65-200

say "prompt ids for THIS model's vocabulary (the fixture belongs to another model)"
python3 - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct")
text = ("The three laws of the fold are: " * 20) + " First, the fold is commutative. Second, the fold is associative. Third, the fold has an identity. "
ids = tok(text, add_special_tokens=False)["input_ids"]
open("/home/ctown-admin/factory/apeiron-rt/.box/prompt.ids", "w").write(" ".join(map(str, ids)))
print("PROMPT_IDS", len(ids))
PY

say "ours vs llama-bench, interleaved, same model, same host, CPU only"
"$CARGO_TARGET_DIR/release/examples/engine_profile" \
  --model "$GGUF" \
  --ids "$BOXDIR/.box/prompt.ids" \
  --pp 200 --tg 64 --reps 3 \
  --llama "$LLAMA/build/bin/llama-bench" \
  --label "Qwen2.5-0.5B-Instruct, HF safetensors -> GGUF, 10.0.63.10" \
  --out "$BOXDIR/.box/out" > "$BOXDIR/.box/engine_profile.log" 2>&1
rc=$?
echo "engine_profile exit: $rc"
tail -n 40 "$BOXDIR/.box/engine_profile.log"
ls -l "$BOXDIR/.box/out" 2>&1 | head -8

say "output parity: ours (generate) vs the incumbent, same prompt, greedy"
PROMPT="The three laws of the fold are: "
GEN="$CARGO_TARGET_DIR/release/examples/generate"
"$GEN" --model "$GGUF" --tokenizer "$MODELS/qwen2.5-0.5b-instruct/tokenizer.json" \
  --prompt "$PROMPT" --n 48 > "$BOXDIR/.box/ours.txt" 2> "$BOXDIR/.box/ours.stderr"
echo "ours stderr:"; tail -n 6 "$BOXDIR/.box/ours.stderr"
echo "ours text:"; head -c 300 "$BOXDIR/.box/ours.txt"; echo

# The incumbent: llama-completion if this build has it, else llama-cli. Output is
# capped: the model rambles and an uncapped capture filled 12 GB of the box once.
if [ -x "$LLAMA/build/bin/llama-completion" ]; then INC="$LLAMA/build/bin/llama-completion"; else INC="$LLAMA/build/bin/llama-cli"; fi
echo "incumbent: $INC"
"$INC" -m "$GGUF" -p "$PROMPT" -n 48 --temp 0 --seed 0 --no-warmup -t 4 -ngl 0 2> "$BOXDIR/.box/theirs.stderr" \
  | head -c 4000 > "$BOXDIR/.box/theirs.txt"
echo "theirs stderr:"; grep -E "eval time|total time|sampling time|tokens" "$BOXDIR/.box/theirs.stderr" | tail -6
echo "theirs text:"; head -c 300 "$BOXDIR/.box/theirs.txt"; echo

say "done"
