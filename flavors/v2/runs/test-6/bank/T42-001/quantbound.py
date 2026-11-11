# Rigorous slack of quantized lm_head approximations vs the top1-top2 logit gap (CPU).
import torch, json
torch.set_num_threads(16)
exec(open('screen.py').read().split('# exact reference logits')[0].replace("Hte = load(he, 50, g)", "Hte = load(he, 10, g)"))
L = Hte @ W.T
top2 = L.topk(2, 1)
gap = top2.values[:, 0] - top2.values[:, 1]
out = {}
Ha = Hte.abs()
for name, bits in [('int8_row', 8), ('int12_row', 12), ('int16_row', 16)]:
    s = W.abs().max(1).values / (2**(bits-1) - 1)        # per-row scale, |e_i| <= s/2
    slack = Ha.sum(1, keepdim=True) * (s / 2)[None, :]    # Hoelder: |h.e| <= ||h||_1 * max|e|
    approx = L  # exact approx stands in; slack decides the candidate count
    lb = (approx - slack).max(1, keepdim=True).values
    n = ((approx + slack) >= lb).sum(1).float()
    out[name] = dict(cand_p50=n.median().item(), cand_mean=n.mean().item(), slack_top_med=slack.gather(1, top2.indices[:, :1]).median().item())
    print(name, out[name], flush=True)
# e4m3 per-element relative error 2^-4: |h.e| <= 2^-4 sum|h_i w_i|
for name, rel in [('e4m3', 2**-4), ('bf16->10bit', 2**-10)]:
    S = Ha @ W.abs().T
    slack = rel * S
    lb = (L - slack).max(1, keepdim=True).values
    n = ((L + slack) >= lb).sum(1).float()
    out[name] = dict(cand_p50=n.median().item(), cand_mean=n.mean().item(), slack_top_med=slack.gather(1, top2.indices[:, :1]).median().item())
    print(name, out[name], flush=True)
out['gap_median'] = gap.median().item(); out['gap_p10'] = gap.quantile(0.1).item()
print(out); json.dump(out, open('quantbound_res.json', 'w'), indent=1)
