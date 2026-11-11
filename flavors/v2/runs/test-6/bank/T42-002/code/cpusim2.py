import torch, json
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
H = torch.cat(hs).float(); resp = torch.cat(resp); N = len(H); print('rows', N, 'resp', resp.sum().item())
G = 128; NG = 40; DTC = 2.0**-11
hg1 = H.abs().view(N, NG, G).sum(2); hmax = H.abs().view(N, NG, G).amax(2)
sh = torch.exp2(torch.ceil(torch.log2(hmax.clamp(min=1e-30) / 16256)))
Hp = hg1 * (0.5 + 63 * DTC) + 4032 * sh
for pow2 in [True, False]:
    U = []; L = []; LEX = []
    for half in range(2):
        W = Wb[half*124160:(half+1)*124160].float(); Wg = W.view(-1, NG, G); mx = Wg.abs().amax(2)
        s = torch.exp2(torch.ceil(torch.log2(mx / 63))) if pow2 else mx / 63
        q = torch.round(Wg / s[..., None]).clamp(-64, 63)
        approx = H @ (q * s[..., None]).view_as(W).T; R_ = Hp @ s.T
        U.append(approx + R_ * 1.001); L.append(approx - R_ * 1.001); LEX.append((H @ W.T).bfloat16().float())
    for name, lbs in [('local', [L[0].max(1, keepdim=True).values, L[1].max(1, keepdim=True).values]),
                      ('global', [torch.maximum(L[0].max(1, keepdim=True).values, L[1].max(1, keepdim=True).values)] * 2)]:
        n = sum(((U[k] + U[k].abs() * 2**-8) >= (lbs[k] - lbs[k].abs() * 2**-8)).sum(1).float() for k in range(2))
        for sel, nm in [(slice(None), 'all'), (resp, 'resp'), (~resp, 'prompt')]:
            x = n[sel]; print(f'pow2={pow2} {name:6s} {nm:6s}: mean {x.mean():.1f} p50 {x.median():.0f} p90 {x.quantile(.9):.0f} p99 {x.quantile(.99):.0f} max {x.max():.0f}', flush=True)
