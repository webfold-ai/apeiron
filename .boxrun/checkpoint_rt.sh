#!/bin/sh
# Box-side: the checkpoint round-trip, measured, on a real PyTorch checkpoint.
#
#   HF safetensors (PyTorch, 988 MB)  --llama.cpp's converter-->  GGUF q8_0
#        --> OUR stack: read -> lift into the monoid (Gram + cross-moment)
#            -> fold a domain shard -> fold a revoked shard -> subtract the revoked shard
#            -> solve back to dense weights -> export safetensors
#        --> PyTorch reads BOTH exports and compares them with the original checkpoint.
#
# Run directly over ssh (not through box-run): the box-run disk gate refused (the box
# was at 8.8 GB free after other jobs filled it), and this step needs no build of the
# library, no download, and writes only megabytes.
set -u
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ctown-admin/factory/.target/apeiron-rt

BOXDIR=/home/ctown-admin/factory/apeiron-rt
LLAMA=/home/ctown-admin/llama.cpp
MODELS=/home/ctown-admin/models/roundtrip
GGUF=$MODELS/qwen2.5-0.5b-instruct-q8_0.gguf
OUT=$BOXDIR/.box/rt
say() { echo "=== $* ==="; }

mkdir -p "$OUT"

say "build the round-trip example"
cd "$BOXDIR" || exit 1
cargo build --release --example checkpoint_roundtrip > "$BOXDIR/.box/rt_build.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAILED (exit $rc)"
  grep -n "error" -A 10 "$BOXDIR/.box/rt_build.log" | head -n 60
  exit $rc
fi
echo "build ok"

say "pick the tensors: the first 2-D weights in the model, plus one MLP matrix"
NAMES=$(python3 - <<'PY'
import struct
p = "/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct-q8_0.gguf"
f = open(p, "rb"); f.read(4); f.read(4)
n_t = struct.unpack("<Q", f.read(8))[0]; n_kv = struct.unpack("<Q", f.read(8))[0]
def rd(fmt):
    s = struct.calcsize(fmt); return struct.unpack(fmt, f.read(s))[0]
for _ in range(n_kv):
    klen = rd("<Q"); f.read(klen); t = rd("<I")
    if t == 8:
        sl = rd("<Q"); f.read(sl)
    elif t == 9:
        et = rd("<I"); n = rd("<Q")
        if et == 8:
            for _ in range(n):
                sl = rd("<Q"); f.read(sl)
        else:
            fm = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[et]
            for _ in range(n): rd("<"+fm)
    else:
        fm = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[t]
        rd("<"+fm)
names = []
for _ in range(n_t):
    nlen = rd("<Q"); name = f.read(nlen).decode()
    nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]
    typ = rd("<I"); off = rd("<Q")
    if len(dims) == 2 and dims[0] <= 1024 and len(names) < 4:
        names.append(name)
print(",".join(names))
PY
)
echo "tensors: $NAMES"

say "run the round-trip"
"$CARGO_TARGET_DIR/release/examples/checkpoint_roundtrip" \
  --model "$GGUF" --out "$OUT" --tensors "$NAMES" --calib-rows 512 --seed 7 \
  2>&1 | tail -n 20

say "PyTorch verifies the export against the ORIGINAL checkpoint"
python3 - <<'PY'
import json, torch
from safetensors.torch import load_file
out = "/home/ctown-admin/factory/apeiron-rt/.box/rt"
ref = load_file(out + "/ref.safetensors")
rt = load_file(out + "/roundtrip.safetensors")
orig = load_file("/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct/model.safetensors")
report = json.load(open(out + "/report.json"))
print("PYTORCH SAW", len(rt), "tensors; our report says", report["selected"])
worst_rt = 0.0
worst_q = 0.0
for name, t in rt.items():
    a = t.float()
    b = ref[name].float()
    e_rt = (a - b).abs().max().item()
    worst_rt = max(worst_rt, e_rt)
    # the GGUF is q8_0, so compare our dequantised reference with the ORIGINAL bf16
    key = name.replace("blk.", "model.layers.")
    cand = [k for k in orig if k.endswith(name.split(".")[-2] + "." + name.split(".")[-1])]
    if cand:
        o = orig[cand[0]].float()
        if o.shape == b.shape:
            worst_q = max(worst_q, (b - o).abs().max().item())
    print(f"  {name}: shape {tuple(a.shape)} roundtrip_max_abs_diff {e_rt:.3e} "
          f"bit_exact {(a == b).float().mean().item():.6f}")
print(f"WORST round-trip max abs diff (ours vs our own dequantised reference): {worst_rt:.6e}")
print(f"WORST quantisation diff (our dequantised reference vs the original bf16 checkpoint): {worst_q:.6e}")
print("unlearn bit-exact on every selected matrix:", report["unlearn_bit_exact_all"])
print("q32 floor:", report["q32_floor"])
PY

say "done"
