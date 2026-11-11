# Lossless bit-plane split of the bf16 lm_head: high plane q (b-bit ints, scale per group of G along K),
# rigorous Hoelder bound |h.(w - s q)| <= sum_g ||h_g||_1 * s_g/2. Candidates = {v: approx_v + slack_v >= max(approx - slack) - eps}.
import torch, json
torch.set_num_threads(16)
exec(open('screen.py').read().split('# exact reference logits')[0].replace("Hte = load(he, 50, g)", "Hte = load(he, 20, g)"))
N = len(Hte)
Lex = Hte @ W.T
ref = Lex.bfloat16().float().argmax(1)
out = {}
Ha = Hte.abs()
for b in [4, 5, 6, 7, 8]:
    for G in [5120, 512, 128, 32]:
        qmax = 2**(b-1) - 1
        Wg = W.view(W.shape[0], 5120 // G, G)
        s = Wg.abs().amax(2, keepdim=True) / qmax
        s = torch.where(s == 0, torch.ones_like(s), s)
        What = (torch.round(Wg / s).clamp(-qmax, qmax) * s).view_as(W)
        approx = Hte @ What.T
        hg = Ha.view(N, 5120 // G, G).sum(2)                 # [N, ngroups]
        slack = hg @ (s.squeeze(2) / 2).T                     # [N, V]
        lb = (approx - slack).max(1, keepdim=True).values
        eps = 1e-3 + lb.abs() * 2**-8                         # fp32 screen error + bf16 rounding of logits/ties
        cand = (approx + slack) >= (lb - eps)
        n = cand.sum(1).float()
        ok = bool(cand.gather(1, ref[:, None]).all())
        bytes_per_w = b / 8 + (16 / G)                        # plane + bf16 scale per group
        key = f'b{b}_G{G}'
        out[key] = dict(bytes_per_w=round(bytes_per_w, 4), mean=n.mean().item(), p50=n.median().item(),
                        p99=n.quantile(0.99).item(), max=n.max().item(), ref_in=ok)
        print(key, out[key], flush=True)
        del What, approx, slack, cand
json.dump(out, open('sweep_res.json', 'w'), indent=1)
