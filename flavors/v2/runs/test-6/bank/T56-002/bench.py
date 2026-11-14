import os, sys, json, statistics, ctypes
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import t31_cfg
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as T
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
MB = 1 << 20
SH = {"qkvz": (8192, 5120), "out": (5120, 3072), "aqkv": (7168, 5120), "o": (5120, 3072), "down": (5120, 8704), "gate_up": (17408, 5120)}
W, WS = {}, {}
for k, (N, K) in SH.items():
    if (N, K) in [(SH[j]) for j in W]:
        j = [j for j in W if SH[j] == (N, K)][0]; W[k], WS[k] = W[j], WS[j]; continue
    W[k] = torch.randn(N * K, device=dev).to(torch.float8_e4m3fn).view(torch.int32)
    WS[k] = torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32)
flush = torch.empty(512 * MB // 4, dtype=torch.int32, device=dev)
try:
    _rt = ctypes.CDLL("libcudart.so.12"); _rt.cudaCtxResetPersistingL2Cache()
    def reset(): _rt.cudaCtxResetPersistingL2Cache()
except Exception as e:
    print("no cudaCtxResetPersistingL2Cache", e); reset = lambda: None
_ext = T.ext()
def sync_flush():
    flush.sum(); reset()
cache_ws = {}
def t31_ws(N, K):
    key = (N, K)
    if key not in cache_ws:
        need = max(t31_cfg.ws_numel(N, K, m, c) for m in range(1, 97) for c in {t31_cfg.pick(N, K, m, r) for r in ("cold", "mid", "r48", "warm")})
        cache_ws[key] = (torch.zeros(max(need, 1), device=dev, dtype=torch.float32), torch.zeros(4096, device=dev, dtype=torch.int32))
    return cache_ws[key]
bsw = (torch.empty(1, device=dev), torch.zeros(16384, device=dev, dtype=torch.int32))
def slices(M, scheme):
    if scheme == "s64": s = [64] * (M // 64) + ([M % 64] if M % 64 else [])
    elif scheme == "s96": s = [96] * (M // 96) + ([M % 96] if M % 96 else [])
    elif scheme == "even":
        n = -(-M // 96); b = M // n; r = M - b * n; s = [b + 1] * r + [b] * (n - r)
    return s
def quant(x): return T.quant_perm(x)
def run_A(k, x, outs):
    N, K = SH[k]; xq, xs = quant(x)
    out = torch.empty(x.shape[0], N, device=dev, dtype=torch.bfloat16)
    _ext.bs_gemm_marlin(xq, W[k], xs, WS[k], out, T.pick(N, K, x.shape[0]), 1, *bsw); outs[k] = out
def run_B(k, x, outs, scheme, regs):
    N, K = SH[k]; xq, xs = quant(x); ws, cnt = t31_ws(N, K)
    i = 0; res = []
    for si, m in enumerate(slices(x.shape[0], scheme)):
        reg = regs[0] if si == 0 else regs[1]
        res.append(ops.t31_gemm_e4m3(xq[i:i+m], xs[i:i+m], W[k], WS[k], ws, cnt, m, N, K, t31_cfg.pick(N, K, m, reg))); i += m
    outs[k] = torch.cat(res) if len(res) > 1 else res[0]
def graph_time(fn, reps=15):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps + 2):
        sync_flush(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts[2:])
VARS = {"A": None}
for sch in ("s64", "s96", "even"):
    for rg in (("cold", "cold"), ("cold", "warm")):
        VARS[f"B_{sch}_{rg[0][0]}{rg[1][0]}"] = (sch, rg)
GDN = ["qkvz", "out", "gate_up", "down"]; ATT = ["aqkv", "o", "gate_up", "down"]
res = {}
for M in [int(v) for v in os.environ.get("MS", "112,128,160,192").split(",")]:
    torch.manual_seed(M)
    xs_in = {k: torch.randn(M, SH[k][1], device=dev, dtype=torch.bfloat16) for k in SH}
    for vn, v in VARS.items():
        outs = {}
        def f(k):
            return (lambda: run_A(k, xs_in[k], outs)) if v is None else (lambda: run_B(k, xs_in[k], outs, v[0], v[1]))
        for k in SH:
            res[f"{M}/{vn}/{k}"] = graph_time(f(k))
        for nm, L in (("GDNchain", GDN), ("ATTchain", ATT)):
            def chain(L=L): [f(k)() for k in L]
            res[f"{M}/{vn}/{nm}"] = graph_time(chain)
        res[f"{M}/{vn}/GDNsum"] = sum(res[f"{M}/{vn}/{k}"] for k in GDN)
        res[f"{M}/{vn}/ATTsum"] = sum(res[f"{M}/{vn}/{k}"] for k in ATT)
        print(M, vn, " ".join(f"{k}={res[f'{M}/{vn}/{k}']:.1f}" for k in list(SH) + ["GDNsum", "GDNchain", "ATTsum", "ATTchain"]), flush=True)
    # correctness
    for k in SH:
        N, K = SH[k]
        o = {}
        run_A(k, xs_in[k], o); a = o[k].float()
        for vn in ("B_s64_cc", "B_s96_cc", "even"):
            if vn == "even": continue
            run_B(k, xs_in[k], o, *VARS[vn]); b = o[k].float()
            # t31 at full-M-ish reference: M=96 vs 48-row halves
            res[f"{M}/diff/{k}/{vn}_vs_bs"] = [(a - b).abs().max().item(), a.abs().max().item(), ((a - b).norm() / a.norm()).item()]
    # slice dependence of t31b on identical rows
    for k in ("qkvz", "down"):
        N, K = SH[k]; x = xs_in[k][:96]; xq, xs = quant(x); ws, cnt = t31_ws(N, K)
        full = ops.t31_gemm_e4m3(xq, xs, W[k], WS[k], ws, cnt, 96, N, K, t31_cfg.pick(N, K, 96, "cold")).float()
        for m in (64, 48, 32):
            part = torch.cat([ops.t31_gemm_e4m3(xq[i:i+m], xs[i:i+m], W[k], WS[k], ws, cnt, xq[i:i+m].shape[0], N, K, t31_cfg.pick(N, K, xq[i:i+m].shape[0], "cold")) for i in range(0, 96, m)]).float()
            res[f"slicedep/{k}/96_vs_{m}"] = (full - part).abs().max().item()
        full2 = ops.t31_gemm_e4m3(xq, xs, W[k], WS[k], ws, cnt, 96, N, K, t31_cfg.pick(N, K, 96, "warm")).float()
        res[f"slicedep/{k}/cold_vs_warm_cfg"] = (full - full2).abs().max().item()
json.dump(res, open(os.environ.get("OUT", "res.json"), "w"), indent=1)
