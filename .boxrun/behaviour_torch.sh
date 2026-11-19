#!/bin/sh
# Box-side: DOES THE MODEL STILL RUN WELL AFTER THE ROUND-TRIP?
#
# The weight-level round-trip is measured (max abs diff 3.8e-6, below the q8_0 grid).
# This asks the behavioural question, in PyTorch - the same independent oracle that
# verified the weights - because the box ran out of disk for a second GGUF and because
# the oracle is the right instrument anyway: logits, not anecdotes.
#
#   arm A: the ORIGINAL HF checkpoint
#   arm B: the same checkpoint with FOUR of its matrices replaced by our round-trip
#          export (the 151936x896 embedding, and three matrices of block 0)
#
# For three prompts: the full-vocabulary logits at the last position, the greedy
# continuation, and the mean NLL of a fixed reference text.
set -u
export PATH=$HOME/.cargo/bin:$PATH

say() { echo "=== $* ==="; }

say "the behavioural comparison, in PyTorch"
python3 - <<'PY'
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM

SRC = "/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct"
RT  = "/home/ctown-admin/models/roundtrip/rt"
tok = AutoTokenizer.from_pretrained(SRC)
PROMPTS = [
    "The three laws of the fold are: ",
    "A triangle has three sides",
    "Write one sentence explaining a closed-form ridge solve to a non-mathematician.",
]
REF_TEXT = ("The Gram matrix of a design matrix is the sufficient statistic for the ridge "
            "objective: it does not depend on the order in which the data was folded, so a "
            "revoked shard can be subtracted from it in constant time. ")

def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m.eval()
    return m

@torch.inference_mode()
def logits_for(model, text):
    ids = tok(text, return_tensors="pt").input_ids
    out = model(ids).logits[0, -1].float()
    return out

@torch.inference_mode()
def nll(model, text):
    ids = tok(text, return_tensors="pt").input_ids
    lg = model(ids).logits[0].float()
    lp = torch.log_softmax(lg, dim=-1)
    tgt = ids[0, 1:]
    sel = lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(-sel.mean()), int(tgt.numel())

def greedy(model, text, n=24):
    ids = tok(text, return_tensors="pt").input_ids
    out_ids = []
    cur = ids
    with torch.inference_mode():
        for _ in range(n):
            nxt = int(model(cur).logits[0, -1].float().argmax())
            out_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return out_ids

say_models = []
print("loading arm A (original) ...", flush=True)
A = load(SRC)
print("loading arm B (round-tripped matrices) ...", flush=True)
B = load(RT)

print("\n=== logits at the last position ===")
for i, p in enumerate(PROMPTS):
    la = logits_for(A, p)
    lb = logits_for(B, p)
    d = (la - lb).abs()
    top_a = torch.topk(la, 10).indices.tolist()
    top_b = torch.topk(lb, 10).indices.tolist()
    print(f"prompt {i}: max|dlogit| {d.max().item():.6e}  mean|dlogit| {d.mean().item():.6e}  "
          f"top1 {top_a[0]} vs {top_b[0]}  top10 overlap {len(set(top_a) & set(top_b))}/10  "
          f"argmax_agree {bool(top_a[0] == top_b[0])}")
    ga = greedy(A, p, 16)
    gb = greedy(B, p, 16)
    n = min(len(ga), len(gb))
    first = next((k for k in range(n) if ga[k] != gb[k]), n)
    print(f"  greedy 16 tokens: first divergence at token {first}"
          + (" (identical)" if first == n else f" ({ga[first]} vs {gb[first]})"))
    print("   A:", tok.decode(ga)[:140].replace("\n", " "))
    print("   B:", tok.decode(gb)[:140].replace("\n", " "))

print("\n=== mean NLL on a fixed reference text (nats/token) ===")
na, na_n = nll(A, REF_TEXT)
nb, nb_n = nll(B, REF_TEXT)
print(f"A (original)      {na:.8f} over {na_n} tokens")
print(f"B (round-tripped) {nb:.8f} over {nb_n} tokens")
print(f"difference        {abs(na-nb):.3e} nats/token ({abs(na-nb)/na*100:.6f}% of A)")
PY

say "done"
