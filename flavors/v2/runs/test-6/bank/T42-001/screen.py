# Certified (exact) argmax screening of the target lm_head, CPU only.
# For hidden h and vocab row w: h.w = (V_r^T h).(V_r^T w) + sum_b h'_b . w'_b  (V = eigenbasis of H's covariance)
# |sum_b h'_b.w'_b| <= sum_b ||h'_b|| ||w'_b||  (Cauchy-Schwarz per block of the complement): rigorous bound.
import torch, json, glob, os, sys, time
torch.set_num_threads(16)
from huggingface_hub import hf_hub_download
from safetensors import safe_open
R = '/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1/data'
OUT = os.path.dirname(os.path.abspath(__file__))
split = json.load(open(f'{R}/split.json'))
print({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})
tr = split.get('train') or split.get('train_ids'); he = split.get('heldout') or split.get('heldout_ids')

def load(ids, per, gen):
    hs = []
    for i in ids:
        d = torch.load(f'{R}/seq/{i}.pt', map_location='cpu')
        h = d['hidden'][d['P']:]  # response positions (decode-like)
        if len(h) > per:
            h = h[torch.randperm(len(h), generator=gen)[:per]]
        hs.append(h)
    return torch.cat(hs)
g = torch.Generator().manual_seed(0)
Htr = load(tr, 40, g).float()
Hte = load(he, 50, g).float()
print('train', Htr.shape, 'test', Hte.shape)

f = hf_hub_download('Qwen/Qwen3.8-27B-FP8', 'outside.safetensors')
with safe_open(f, 'pt') as s:
    W = s.get_tensor('lm_head.weight').float()  # [V, 5120]
V_, D = W.shape
print('W', W.shape)

# exact reference logits (fp32 accumulate of bf16 inputs) and bf16-rounded argmax (first index on ties)
t = time.time()
Lex = Hte @ W.T
Lbf = Lex.bfloat16().float()
ref = Lbf.argmax(1)
top = Lex.max(1).values
print('ref logits', time.time() - t, 's; top logit mean', top.mean().item())
srt = Lex.topk(2, 1).values
print('top1-top2 gap: median %.3f p10 %.3f' % ((srt[:, 0] - srt[:, 1]).median(), (srt[:, 0] - srt[:, 1]).quantile(0.1)))

# eigenbasis of hidden covariance (uncentered second moment: we bound dot products, not deviations)
C = Htr.T @ Htr / len(Htr)
ev, Vb = torch.linalg.eigh(C)
Vb = Vb.flip(1); ev = ev.flip(0)
frac = ev.cumsum(0) / ev.sum()
for r in [32, 64, 128, 256, 512, 1024]:
    print(f'energy captured r={r}: {frac[r-1]:.4f}')
Wp = W @ Vb          # [V, D] rotated rows
Hp = Hte @ Vb
res = {}
for r in [64, 128, 256, 512]:
    for B in [8, 32, 128]:
        rest = D - r
        bs = (rest + B - 1) // B
        wn = torch.stack([Wp[:, r + b*bs: r + (b+1)*bs].norm(dim=1) for b in range(B)], 1)  # [V,B]
        hn = torch.stack([Hp[:, r + b*bs: r + (b+1)*bs].norm(dim=1) for b in range(B)], 1)  # [N,B]
        approx = Hp[:, :r] @ Wp[:, :r].T          # [N,V]
        slack = hn @ wn.T                          # [N,V]
        lb = (approx - slack).max(1, keepdim=True).values
        tol = lb.abs() * 2**-7 + 1e-3              # bf16 rounding of logits + fp error margin
        cand = (approx + slack) >= (lb - tol)
        n = cand.sum(1).float()
        ok = cand.gather(1, ref[:, None]).all().item()
        q = n.quantile(torch.tensor([0.5, 0.9, 0.99]))
        key = f'r{r}_B{B}'
        res[key] = dict(mean=n.mean().item(), p50=q[0].item(), p90=q[1].item(), p99=q[2].item(), max=n.max().item(), ref_in_cand=ok)
        print(key, res[key], flush=True)
        del approx, slack, cand
json.dump(res, open(f'{OUT}/screen_res.json', 'w'), indent=1)
