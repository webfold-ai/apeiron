#!/bin/sh
# Box-side: the ROUNDTRIP, measured, on a real PyTorch checkpoint.
#
#   HF safetensors (PyTorch)  --llama.cpp's converter-->  GGUF  -->  OUR engine (apeiron)
#                                                          \-->  llama.cpp (llama-bench / llama-cli)
#
# The same file, the same host, the same prompt ids, interleaved: ours vs the incumbent.
# The PyTorch origin is the point: the GGUF is DERIVED from the HF safetensors, and the
# weights our engine serves are the weights PyTorch would load.
set -u
export PATH=$HOME/.cargo/bin:$PATH

say() { echo "=== $* ==="; }
BOXDIR=$PWD
LLAMA=/home/ctown-admin/llama.cpp
MODELS=/home/ctown-admin/models/roundtrip
HF=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p "$MODELS" .box/out

say "1. fetch the PyTorch checkpoint (HF safetensors)"
python3 - <<'PY' 2>&1 | tail -5
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"],
                      local_dir="/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct")
print("SNAPSHOT", p)
PY
ls -l "$MODELS/qwen2.5-0.5b-instruct" | head -8

say "2. convert to GGUF (torch CPU, llama.cpp's converter)"
python3 -c "import sentencepiece" 2>/dev/null || python3 -m pip install --quiet --disable-pip-version-check sentencepiece 2>&1 | tail -2
if [ ! -f "$MODELS/qwen2.5-0.5b-instruct-q8_0.gguf" ] && [ ! -f "$MODELS/qwen2.5-0.5b-instruct-f16.gguf" ]; then
  PYTHONPATH="$LLAMA/gguf-py:$LLAMA" python3 "$LLAMA/convert_hf_to_gguf.py" \
    "$MODELS/qwen2.5-0.5b-instruct" \
    --outfile "$MODELS/qwen2.5-0.5b-instruct-q8_0.gguf" \
    --outtype q8_0 > "$BOXDIR/.box/convert.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "q8_0 conversion failed (exit $rc); trying f16"
    tail -n 12 "$BOXDIR/.box/convert.log"
    PYTHONPATH="$LLAMA/gguf-py:$LLAMA" python3 "$LLAMA/convert_hf_to_gguf.py" \
      "$MODELS/qwen2.5-0.5b-instruct" \
      --outfile "$MODELS/qwen2.5-0.5b-instruct-f16.gguf" \
      --outtype f16 > "$BOXDIR/.box/convert.log" 2>&1 || { tail -n 25 "$BOXDIR/.box/convert.log"; exit 1; }
  fi
fi
GGUF=$(ls "$MODELS"/*.gguf 2>/dev/null | head -1)
echo "GGUF: $GGUF"
ls -l "$GGUF"

say "3. build our engine_profile + generate"
cargo build --release --example engine_profile --example generate > "$BOXDIR/.box/build.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAILED (exit $rc)"
  grep -n "error" -A 12 "$BOXDIR/.box/build.log" | head -n 60
  exit $rc
fi
echo "build ok"

say "4. ours vs llama-bench, interleaved, same model, same host"
python3 - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct")
text = ("The three laws of the fold are: " * 20) + " First, the fold is commutative. "
        "Second, the fold is associative. Third, the fold has an identity. ")
ids = tok(text, add_special_tokens=False)["input_ids"]
with open("/home/ctown-admin/factory/apeiron-rt/.box/prompt.ids", "w") as f:
    f.write(" ".join(str(i) for i in ids))
print("PROMPT_IDS", len(ids))
PY
BIN="${CARGO_TARGET_DIR:-$BOXDIR/target}/release/examples/engine_profile"
"$BIN" \
  --model "$GGUF" \
  --ids "$BOXDIR/.box/prompt.ids" \
  --pp 200 --tg 64 --reps 3 \
  --llama "$LLAMA/build/bin/llama-bench" \
  --label "Qwen2.5-0.5B-Instruct (HF safetensors -> GGUF), 10.0.63.10" \
  --out "$BOXDIR/.box/out" > "$BOXDIR/.box/engine_profile.log" 2>&1
rc=$?
echo "engine_profile exit: $rc"
tail -n 30 "$BOXDIR/.box/engine_profile.log"
ls -l "$BOXDIR/.box/out" 2>&1 | head -8

say "5. output parity: ours (generate) vs llama-cli, same prompt, greedy"
PROMPT="The three laws of the fold are: "
printf '%s' "$PROMPT" > .box/prompt.txt
"$BIN" >/dev/null 2>&1 || true
GEN="${CARGO_TARGET_DIR:-$BOXDIR/target}/release/examples/generate"
"$GEN" --model "$GGUF" --tokenizer "$MODELS/qwen2.5-0.5b-instruct/tokenizer.json" \
  --prompt "$PROMPT" --n 48 > "$BOXDIR/.box/ours.txt" 2> "$BOXDIR/.box/ours.stderr"
echo "ours stderr:"; tail -n 6 "$BOXDIR/.box/ours.stderr"
if [ -x "$LLAMA/build/bin/llama-completion" ]; then
  INC="$LLAMA/build/bin/llama-completion"
else
  INC="$LLAMA/build/bin/llama-cli"
fi
echo "incumbent binary: $INC"
"$INC" -m "$GGUF" -p "$PROMPT" -n 48 --temp 0 --seed 0 \
  --no-warmup -t 8 > "$BOXDIR/.box/theirs.txt" 2> "$BOXDIR/.box/theirs.stderr"
echo "theirs timings:"; grep -E "eval time|total time|sampling" "$BOXDIR/.box/theirs.stderr" | tail -4
echo "--- ours:"; head -c 400 "$BOXDIR/.box/ours.txt"; echo
echo "--- theirs:"; grep -v "^llama_\|^load\|^print_info\|^init:\|^main:" "$BOXDIR/.box/theirs.txt" | head -c 400; echo

say "done"
