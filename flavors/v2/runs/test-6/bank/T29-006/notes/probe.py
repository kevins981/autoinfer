# T29 fused-producer probe (GPU, ~1 min): which norm_quant variant is bitwise vs flashinfer's gemma norms;
# q/s vs quant_perm(flashinfer y) and vLLM quant; silu_mul+quant(+PI) vs silu_and_mul + quant_perm; timings.
import sys, os, math, torch, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashinfer.norm as fn
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as T
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
import t29_act_quant as A
torch.cuda.set_device(0); dev = 'cuda'; g = torch.Generator(device=dev).manual_seed(0)
E = T.nq_ext(); H = 5120; EPS = 1e-6
print('flashinfer cuda-norm path:', fn._use_cuda_norm(torch.device(dev)))
def rows(M, outl=True):
    sc = torch.exp(torch.empty(M, 1, device=dev).uniform_(math.log(1e-2), math.log(1e2), generator=g))
    x = torch.randn(M, H, device=dev, generator=g) * sc
    if outl: x[:, torch.randperm(H, device=dev, generator=g)[:H // 64]] *= 30
    return x.bfloat16()
def nb(a, b): return int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item()) if a.element_size() == 1 else int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
w = (torch.randn(H, device=dev, generator=g) * 0.3).bfloat16()
res = {}
for M in (97, 130, 257, 1536):
    x, r = rows(M), rows(M)
    x1, r1 = x.clone(), r.clone(); fn.gemma_fused_add_rmsnorm(x1, r1, w, EPS)   # x1 = y, r1 = residual
    y0 = fn.gemma_rmsnorm(x, w, EPS)
    qp_ref, sp_ref = T.quant_perm(x1); qn_ref, sn_ref = per_token_group_quant_fp8(x1, 128, use_ue8m0=False)
    qp0, sp0 = T.quant_perm(y0)
    for v in range(8):
        r2 = r.clone(); y2 = torch.empty_like(x); q = torch.empty(M, H, device=dev, dtype=torch.float8_e4m3fn); s = torch.empty(M, H // 128, device=dev)
        E.t29_norm_quant(x, r2, w, y2, q, s, EPS, True, v)
        qn = torch.empty_like(q); sn = torch.empty_like(s); r3 = r.clone()
        E.t29_norm_quant(x, r3, w, torch.empty(0, device=dev), qn, sn, EPS, False, v)
        ya = torch.empty_like(x); qa = torch.empty_like(q); sa = torch.empty_like(s)
        E.t29_norm_quant(x, torch.empty(0, device=dev), w, ya, qa, sa, EPS, True, v)
        d = res.setdefault(v, [0] * 8)
        for i, c in enumerate((nb(y2, x1), nb(r2, r1), nb(q, qp_ref), int((s != sp_ref).sum()), nb(qn, qn_ref) + int((sn != sn_ref).sum()) + nb(r3, r1),
                               nb(ya, y0), nb(qa, qp0) + int((sa != sp0).sum()), 0)):
            d[i] += c
print('variant (RED + 2*FMA + 4*DIVA): mismatches summed over M in 97,130,257,1536 x 5120')
print('  v | add:y  residual  q(PI)  s  | q,s(plain)+res2 | noadd:y  q,s(PI)')
for v, d in res.items():
    print(f'  {v} | {d[0]:6d} {d[1]:6d} {d[2]:6d} {d[3]:4d} | {d[4]:6d} | {d[5]:6d} {d[6]:6d}', '<-- BITWISE' if sum(d) == 0 else '')
# silu
I = 8704
for M in (97, 513):
    gu = (torch.randn(M, 2 * I, device=dev, generator=g) * 2).bfloat16()
    v_ = torch.empty(M, I, device=dev, dtype=torch.bfloat16); torch.ops._C.silu_and_mul(v_, gu)
    qr, sr = T.quant_perm(v_); q, s = A.silu_mul_quant_fp8(gu, perm=True)
    print(f'silu_mul+quant PI M={M}: q diff {nb(q, qr)} s diff {int((s != sr).sum())}')
# timing
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
best = min(res, key=lambda v: sum(res[v]))
print('timing (us, CUDA graph x20) with variant', best)
for M in (128, 256, 512, 1024, 1536):
    x, r = rows(M), rows(M); y = torch.empty_like(x); q = torch.empty(M, H, device=dev, dtype=torch.float8_e4m3fn); s = torch.empty(M, H // 128, device=dev)
    gu = (torch.randn(M, 2 * I, device=dev, generator=g)).bfloat16(); v_ = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    e0 = torch.empty(0, device=dev)
    t_un = tm(lambda: (fn.gemma_fused_add_rmsnorm(x, r, w, EPS), T.quant_perm(x)))
    t_n = tm(lambda: fn.gemma_fused_add_rmsnorm(x, r, w, EPS))
    t_f = tm(lambda: E.t29_norm_quant(x, r, w, e0, q, s, EPS, True, best))
    t_fy = tm(lambda: E.t29_norm_quant(x, r, w, y, q, s, EPS, True, best))
    t_su = tm(lambda: (torch.ops._C.silu_and_mul(v_, gu), T.quant_perm(v_)))
    t_sf = tm(lambda: A.silu_mul_quant_fp8(gu, perm=True))
    print(f'M={M}: norm+quant_perm {t_un:.1f} (norm {t_n:.1f}) fused {t_f:.1f} fused+y {t_fy:.1f} | silu+quant_perm {t_su:.1f} fused {t_sf:.1f}', flush=True)
