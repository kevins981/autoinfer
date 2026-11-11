import torch
torch.set_num_threads(32)
from huggingface_hub import hf_hub_download
from safetensors import safe_open
R = '/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1/data'
f = hf_hub_download('Qwen/Qwen3.8-27B-FP8', 'outside.safetensors')
with safe_open(f, 'pt') as s: Wb = s.get_tensor('lm_head.weight')
g = torch.Generator().manual_seed(0); hs = []; resp = []
for i in range(0, 400, 20):
    d = torch.load(f'{R}/seq/{i}.pt', map_location='cpu'); h = d['hidden']; p = torch.randperm(len(h), generator=g)[:100]
    hs.append(h[p]); resp.append(p >= d['P'])
H = torch.cat(hs).float(); resp = torch.cat(resp); N = len(H)
h2 = H.norm(dim=1, keepdim=True)
def run(b, G, pow2=False):
    NG = 5120 // G; qmax = 2**(b-1) - 1
    hg1 = H.abs().view(N, NG, G).sum(2)
    U = []; L = []; Uc = []; Lc = []
    for half in range(2):
        W = Wb[half*124160:(half+1)*124160].float(); Wg = W.view(-1, NG, G); mx = Wg.abs().amax(2).clamp(min=1e-30)
        s = torch.exp2(torch.ceil(torch.log2(mx / qmax))) if pow2 else mx / qmax
        q = torch.round(Wg / s[..., None]).clamp(-qmax-1, qmax)
        What = (q * s[..., None]).view_as(W)
        r2 = (W - What).norm(dim=1)
        approx = H @ What.T
        hold = hg1 @ (s.T / 2) * 1.06
        cs = h2 * r2[None] * 1.06
        bnd = torch.minimum(hold, cs)
        U += [approx + bnd, approx + cs]; L += [approx - bnd, approx - cs]
        del What, approx
    out = []
    for k in range(2):  # 0: min(hoelder, cs), 1: cs only
        Ua, Ub, La, Lb = U[k], U[k+2], L[k], L[k+2]
        lb = torch.maximum(La.max(1, keepdim=True).values, Lb.max(1, keepdim=True).values)
        lb = lb - lb.abs() * 2**-8
        n = ((Ua + Ua.abs()*2**-8) >= lb).sum(1).float() + ((Ub + Ub.abs()*2**-8) >= lb).sum(1).float()
        x = n[resp]
        out.append(f"{['min','cs'][k]}: mean {x.mean():.1f} p50 {x.median():.0f} p90 {x.quantile(.9):.0f} p99 {x.quantile(.99):.0f} max {x.max():.0f} >64:{(x>64).float().mean()*100:.1f}% >256:{(x>256).float().mean()*100:.2f}%")
    bpw = b/8 + 2/G
    print(f'b={b} G={G} pow2={pow2} B/w={bpw:.3f} GB/rank={bpw*124160*5120/1e9:.3f} | ' + ' | '.join(out), flush=True)
for b, G in [(7,128),(7,5120),(8,5120),(8,128),(6,64),(7,64),(6,32)]:
    run(b, G)
