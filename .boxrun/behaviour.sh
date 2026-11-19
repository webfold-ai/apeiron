#!/bin/sh
# Box-side: DOES THE MODEL STILL RUN WELL AFTER THE ROUND-TRIP?
#
# The weight-level round-trip is measured (max abs diff 3.8e-6, below the q8_0 grid).
# This asks the behavioural question: does the model still produce the same output?
#
# Two arms, one GGUF each, F32 output so the round-trip's own error is preserved
# (an F16 or q8_0 container would quantise 3.8e-6 away and the test would prove nothing):
#   arm A: the ORIGINAL HF safetensors -> GGUF f32
#   arm B: the same checkpoint with FOUR of its matrices replaced by our round-trip
#          export -> GGUF f32
# Then: the same three prompts through our engine, greedy, 64 tokens, and the generated
# token ids are compared. "Identical for all 64 tokens" is the answer we want.
set -u
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ctown-admin/factory/.target/apeiron-rt

BOXDIR=/home/ctown-admin/factory/apeiron-rt
LLAMA=/home/ctown-admin/llama.cpp
MODELS=/home/ctown-admin/models/roundtrip
SRC=$MODELS/qwen2.5-0.5b-instruct
RT=$BOXDIR/.box/rt
WORK=$MODELS/rt
say() { echo "=== $* ==="; }

mkdir -p "$WORK"

say "build generate with --dump-ids"
cd "$BOXDIR" || exit 1
cargo build --release --example generate > "$BOXDIR/.box/gen_build.log" 2>&1 || {
  echo "BUILD FAILED"; grep -n "error" -A 8 "$BOXDIR/.box/gen_build.log" | head -40; exit 1; }
echo "build ok"

say "arm B: the original checkpoint with our round-tripped matrices substituted"
python3 - <<'PY'
import json, shutil, os
from safetensors.torch import load_file, save_file
src = "/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct"
dst = "/home/ctown-admin/models/roundtrip/rt"
rt = load_file("/home/ctown-admin/factory/apeiron-rt/.box/rt/roundtrip.safetensors")
orig = load_file(src + "/model.safetensors")
MAP = {
  "token_embd.weight": "model.embed_tokens.weight",
  "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
  "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
  "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
}
replaced = []
for gguf_name, hf_name in MAP.items():
    if gguf_name not in rt:
        continue
    t = rt[gguf_name].float()
    o = orig[hf_name]
    assert tuple(t.shape) == tuple(o.shape), (gguf_name, t.shape, o.shape)
    orig[hf_name] = t
    replaced.append(hf_name)
os.makedirs(dst, exist_ok=True)
save_file(orig, dst + "/model.safetensors")
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    if os.path.exists(src + "/" + f):
        shutil.copy(src + "/" + f, dst + "/" + f)
print("replaced:", replaced)
print("dtypes:", {k: str(v.dtype) for k, v in list(orig.items())[:2]})
PY

say "convert both arms to GGUF f32"
for arm in orig rt; do
  if [ "$arm" = orig ]; then IN=$SRC; else IN=$WORK; fi
  OUT="$MODELS/$arm-f32.gguf"
  if [ ! -f "$OUT" ]; then
    PYTHONPATH="$LLAMA/gguf-py:$LLAMA" python3 "$LLAMA/convert_hf_to_gguf.py" "$IN" \
      --outfile "$OUT" --outtype f32 > "$BOXDIR/.box/convert_$arm.log" 2>&1 || {
        echo "convert $arm FAILED"; tail -n 12 "$BOXDIR/.box/convert_$arm.log"; exit 1; }
  fi
  ls -l "$OUT"
done

say "run both arms through our engine, greedy, same three prompts"
GEN="$CARGO_TARGET_DIR/release/examples/generate"
PROMPTS="The three laws of the fold are: |A triangle has three sides|What is the capital of Tasmania and why is it famous?|Write one sentence explaining a closed-form ridge solve to a non-mathematician."
i=0
OLDIFS=$IFS; IFS='|'
for p in $PROMPTS; do
  i=$((i+1))
  for arm in orig rt; do
    "$GEN" --model "$MODELS/$arm-f32.gguf" --tokenizer "$SRC/tokenizer.json" \
      --prompt "$p" --n 64 --dump-ids "$BOXDIR/.box/ids.$arm.$i" \
      > "$BOXDIR/.box/text.$arm.$i" 2> "$BOXDIR/.box/gen.$arm.$i.stderr"
    echo "arm=$arm prompt=$i: $(tail -n 2 "$BOXDIR/.box/gen.$arm.$i.stderr" | tr '\n' ' ')"
  done
done
IFS=$OLDIFS

say "compare: the generated token ids, arm A (original) vs arm B (round-tripped)"
python3 - <<'PY'
import os
box = "/home/ctown-admin/factory/apeiron-rt/.box"
total = same = 0
for i in (1, 2, 3):
    a = [l.strip() for l in open(f"{box}/ids.orig.{i}") if l.strip()]
    b = [l.strip() for l in open(f"{box}/ids.rt.{i}") if l.strip()]
    n = min(len(a), len(b))
    first = next((k for k in range(n) if a[k] != b[k]), n)
    same += first
    total += n
    print(f"prompt {i}: {len(a)} vs {len(b)} tokens; first divergence at token {first}"
          + (f" (a={a[first]}, b={b[first]})" if first < n else " (identical throughout)"))
    ta = open(f"{box}/text.orig.{i}").read().strip()
    tb = open(f"{box}/text.rt.{i}").read().strip()
    print("  original  :", ta[:160].replace("\n", " "))
    print("  roundtrip :", tb[:160].replace("\n", " "))
print(f"TOKENS IDENTICAL: {same}/{total}")
PY

say "done"
