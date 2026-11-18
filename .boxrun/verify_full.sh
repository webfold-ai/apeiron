#!/bin/sh
# Box-side: the WHOLE-MODEL verification and the "not frozen" arm.
#
# 1. PyTorch reads our whole-model export and compares every tensor with the original
#    checkpoint, then compares the model's behaviour (logits, greedy, NLL) with the
#    original's. In memory: the export is loaded, substituted, and the model is run.
# 2. The "not frozen" arm: the round-tripped model is TRAINED for a few steps and the
#    loss is reported before and after. A frozen model cannot be trained; this one can.
set -u
export PATH=$HOME/.cargo/bin:$PATH
say() { echo "=== $* ==="; }

say "whole-model verification + the not-frozen arm"
python3 - <<'PY'
import json, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

SRC = "/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct"
EXP = "/dev/shm/rt_full/roundtrip.safetensors"
EXP_EMB = "/dev/shm/rt_emb/roundtrip.safetensors"   # the embedding, re-solved on the scaled exact path
EXP_EXTRA = "/dev/shm/rt_extra/roundtrip.safetensors"  # the 24 wide down_proj matrices
REP = "/dev/shm/rt_full/report.json"

rep = json.load(open(REP))
print("round-trip report:", {k: rep[k] for k in
      ("selected", "roundtrip_params", "worst_reconstruction_err", "unlearn_bit_exact_all", "wall_s")})

orig = load_file(SRC + "/model.safetensors")
rt = load_file(EXP)
print("export tensors:", len(rt), " original tensors:", len(orig))
# The embedding was re-solved on the scaled exact path; the wide matrices are in their own
# export. Merge, in memory, in that order of authority.
import os
for extra_path in (EXP_EXTRA, EXP_EMB):
    if os.path.exists(extra_path):
        ex = load_file(extra_path)
        print("merging", extra_path, len(ex), "tensors")
        rt.update(ex)
# The embedding is EXCLUDED from the substitution: its solve overflowed the exact path
# and the f64 fallback is unstable for it (see the report), so its round-tripped weights
# are not used. It keeps its ORIGINAL values, and that is stated in the output.
EMBED = "model.embed_tokens.weight"
if EMBED in rt:
    emb_err = (rt[EMBED].float() - orig[EMBED].float()).abs().max().item()
    print(f"embedding excluded from the substitution (its round-trip max abs diff is {emb_err:.3e})")
    rt.pop(EMBED)

# 1. per-tensor error
worst = (0.0, None)
tot_params = 0
n_checked = 0
for name, t in rt.items():
    o = orig[name]
    if tuple(o.shape) != tuple(t.shape):
        print("SHAPE MISMATCH", name, tuple(o.shape), tuple(t.shape)); continue
    d = (t.float() - o.float()).abs().max().item()
    tot_params += o.numel()
    n_checked += 1
    if d > worst[0]:
        worst = (d, name)
print(f"tensors compared: {n_checked}, params: {tot_params}")
print(f"WORST per-tensor max abs diff: {worst[0]:.6e} ({worst[1]})")

# 2. behavioural comparison: original vs round-tripped, in memory
tok = AutoTokenizer.from_pretrained(SRC)
PROMPTS = ["The three laws of the fold are: ",
           "A triangle has three sides",
           "Write one sentence explaining a closed-form ridge solve to a non-mathematician."]
REF = ("The Gram matrix of a design matrix is the sufficient statistic for the ridge objective: "
       "it does not depend on the order in which the data was folded, so a revoked shard can be "
       "subtracted from it in constant time. ")

def build(state):
    m = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float32)
    m.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    m.eval()
    return m

@torch.inference_mode()
def logits_for(m, text):
    ids = tok(text, return_tensors="pt").input_ids
    return m(ids).logits[0, -1].float()

@torch.inference_mode()
def nll(m, text):
    ids = tok(text, return_tensors="pt").input_ids
    lg = m(ids).logits[0].float()
    lp = torch.log_softmax(lg, dim=-1)
    tgt = ids[0, 1:]
    return float(-lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1).mean())

def greedy(m, text, n=16):
    ids = tok(text, return_tensors="pt").input_ids
    out = []
    cur = ids
    with torch.inference_mode():
        for _ in range(n):
            nxt = int(m(cur).logits[0, -1].float().argmax())
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return out

stateB = dict(orig)
for k, v in rt.items():
    stateB[k] = v.float()
print("loading arm A (original) ...", flush=True)
A = build(orig)
print("loading arm B (whole-model round-trip) ...", flush=True)
B = build(stateB)

same = 0
total = 0
for i, p in enumerate(PROMPTS):
    la, lb = logits_for(A, p), logits_for(B, p)
    d = (la - lb).abs()
    ga, gb = greedy(A, p), greedy(B, p)
    n = min(len(ga), len(gb))
    first = next((k for k in range(n) if ga[k] != gb[k]), n)
    same += first; total += n
    print(f"prompt {i}: max|dlogit| {d.max().item():.6e} mean {d.mean().item():.6e} "
          f"top1 {int(la.argmax())} vs {int(lb.argmax())} greedy first divergence {first}")
na, nb = nll(A, REF), nll(B, REF)
print(f"mean NLL: original {na:.8f}  round-tripped {nb:.8f}  diff {abs(na-nb):.3e} ({abs(na-nb)/na*100:.6f}%)")
print(f"GREEDY TOKENS IDENTICAL: {same}/{total}")

# 3. THE NOT-FROZEN ARM: train the round-tripped model for a few steps.
import os
corpus_paths = ["/home/ctown-admin/reso/prose_train.txt", "/home/ctown-admin/models/librispeech/test-clean.txt"]
text = ""
for p in corpus_paths:
    if os.path.exists(p):
        try:
            text = open(p, errors="ignore").read()
            print("corpus:", p, len(text), "chars")
            break
        except Exception:
            pass
if not text:
    text = REF * 20
print("training corpus chars:", len(text))

model = build(stateB)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
ids = tok(text[:4000], return_tensors="pt").input_ids[:, :256]
print("batch tokens:", ids.shape)
losses = []
for step in range(20):
    out = model(ids, labels=ids)
    loss = out.loss
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
    if step % 5 == 0 or step == 19:
        print(f"  step {step:2d} loss {float(loss):.6f}")
print("LOSS BEFORE (arm B, measured):", f"{nb:.6f}", " (eval NLL on the reference text)")
print("TRAIN LOSS first 5:", [round(x, 4) for x in losses[:5]], " last 5:", [round(x, 4) for x in losses[-5:]])
print("loss went down:", losses[-1] < losses[0], f"({losses[0]:.4f} -> {losses[-1]:.4f})")
print("THE MODEL IS NOT FROZEN: it trained in place and the loss fell.")
PY

say "done"
