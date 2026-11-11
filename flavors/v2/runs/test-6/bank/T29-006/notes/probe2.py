# new PI store: bitwise re-check vs flashinfer + quant_perm, and timings incl. M <= 96 (T28)
import math, torch, statistics as st
import flashinfer.norm as fn
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as T, t29_fused as F
torch.cuda.set_device(0); dev = 'cuda'; g = torch.Generator(device=dev).manual_seed(1); H = 5120; EPS = 1e-6
def rows(M):
    sc = torch.exp(torch.empty(M, 1, device=dev).uniform_(math.log(1e-2), math.log(1e2), generator=g))
    x = torch.randn(M, H, device=dev, generator=g) * sc; x[:, torch.randperm(H, device=dev, generator=g)[:80]] *= 30
    return x.bfloat16()
def nb(a, b): return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item())
w = (torch.randn(H, device=dev, generator=g) * 0.3).bfloat16(); bad = 0
for M in (1, 3, 16, 17, 63, 96, 97, 113, 128, 255, 777, 1536, 2049):
    x, r = rows(M), rows(M)
    x1, r1 = x.clone(), r.clone(); fn.gemma_fused_add_rmsnorm(x1, r1, w, EPS); qr, sr = T.quant_perm(x1)
    r2 = r.clone(); qa, y = F.norm_quant(x, r2, w, EPS, True)
    y0 = fn.gemma_rmsnorm(x, w, EPS); q0, s0 = T.quant_perm(y0); qb, yb = F.norm_quant(x, None, w, EPS, True)
    c = nb(y, x1) + nb(r2, r1) + nb(qa.data, qr) + int((qa.scale != sr).sum()) + nb(yb, y0) + nb(qb.data, q0) + int((qb.scale != s0).sum())
    bad += c; print(f'M={M}: differing {c}', flush=True)
print('NORM PI-STORE', 'BITWISE PASS' if bad == 0 else f'FAIL {bad}')
R = 20
def tm(fn_):
    s_ = torch.cuda.Stream()
    with torch.cuda.stream(s_):
        fn_(); s_.synchronize(); gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s_):
            for _ in range(R): fn_()
    s_.synchronize()
    for _ in range(3): gr.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(15):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); gr.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000 / R)
    return st.median(ts)
I = 8704
for M in (16, 48, 96, 112, 128, 192, 256, 512, 1024):
    x, r = rows(M), rows(M); gu = (torch.randn(M, 2 * I, device=dev, generator=g)).bfloat16(); v_ = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    t_un = tm(lambda: (fn.gemma_fused_add_rmsnorm(x, r, w, EPS), T.quant_perm(x)))
    t_f = tm(lambda: F.norm_quant(x, r, w, EPS, False)); t_fy = tm(lambda: F.norm_quant(x, r, w, EPS, True))
    t_su = tm(lambda: (torch.ops._C.silu_and_mul(v_, gu), T.quant_perm(v_))); t_sf = tm(lambda: F.silu_mul_quant(gu))
    print(f'M={M}: norm+quant_perm {t_un:.1f} fused {t_f:.1f} fused+y {t_fy:.1f} | silu+quant_perm {t_su:.1f} fused {t_sf:.1f}', flush=True)
