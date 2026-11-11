# Offline bitwise compare of the T29 fused producers against real activations captured from a served
# run (T29_CAPTURE dumps: unfused path inputs + outputs). For each dump:
#   reference = the in-server unfused outputs (flashinfer norm y / residual; vLLM silu_and_mul), then
#               quant_perm (the T29_W8A8=2 unfused quant) on them -> q_ref, s_ref
#   fused     = t29_fused.norm_quant / silu_mul_quant on the dumped inputs
#   also: offline unfused recompute == in-server output (the reference is reproducible)
# usage: python cmp.py <capture dir>   (env T29_FUSED=1 T29_NORM_VARIANT=<v> T29_FUSED_ALL_M=1)
import sys, glob, collections, torch
import flashinfer.norm as fn
from vllm.model_executor.layers.quantization.utils import t29_fused as F, t29_marlin_w8a8 as W
torch.cuda.set_device(0)
def nb(a, b):
    if a is None and b is None: return 0
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item())
tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # n, y, res, q, s, offline-ref
Ms = collections.defaultdict(set)
bad = 0
for f in sorted(glob.glob(sys.argv[1] + '/cap_*.pt')):
    d = torch.load(f)
    if d['kind'] == 'norm':
        x, r, w = d['x'].cuda(), d['residual'], d['weight'].cuda()
        r = r.cuda() if r is not None else None
        key = (f"norm {'res' if r is not None else 'nores'} write_y={int(d['write_y'])} {d['prefix'].split('.')[-1] or d['prefix']}"
               f" reduced={int(d.get('reduced', 0))} ar={int(d.get('allreduce', 0))}")
        y_ref, r_ref = d['y_out'].cuda(), (d['residual_out'].cuda() if d['residual_out'] is not None else None)
        # offline unfused recompute
        if r is not None:
            x1, r1 = x.clone(), r.clone(); fn.gemma_fused_add_rmsnorm(x1, r1, w, d['eps']); off = nb(x1, y_ref) + nb(r1, r_ref)
        else:
            off = nb(fn.gemma_rmsnorm(x, w, d['eps']), y_ref)
        q_ref, s_ref = W.quant_perm(y_ref)
        r2 = r.clone() if r is not None else None
        qa, y = F.norm_quant(x, r2, w, d['eps'], True)
        c = [1, nb(y, y_ref), nb(r2, r_ref) if r is not None else 0, nb(qa.data, q_ref), int((qa.scale != s_ref).sum()), off]
    else:
        gu, out = d['gate_up'].cuda(), d['out'].cuda()
        key = f"silu {d['prefix'].split('.')[-2] if '.' in d['prefix'] else d['prefix']}"
        v = torch.empty_like(out); torch.ops._C.silu_and_mul(v, gu)
        q_ref, s_ref = W.quant_perm(out)
        q, s = F.silu_mul_quant(gu)
        c = [1, 0, 0, nb(q, q_ref), int((s != s_ref).sum()), nb(v, out)]
        x = gu
    Ms[key].add(x.shape[0])
    t = tot[key]
    for i in range(6): t[i] += c[i]
    bad += sum(c[1:])
print('site | dumps | y diff | residual diff | q diff | s diff | offline-unfused vs server | M values')
for k, t in sorted(tot.items()):
    print(f'{k} | {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} | {t[5]} | {sorted(Ms[k])}')
print('CAPTURE COMPARE', 'BITWISE PASS' if bad == 0 else f'FAIL ({bad} differing elements)')
