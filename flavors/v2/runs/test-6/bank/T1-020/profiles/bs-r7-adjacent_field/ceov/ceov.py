# bs-r7-adjacent_field: can copy-engine (DMA) PCIe transfers of prefill-AR size run under a compute-bound Marlin GEMM for free?
import os, sys, math, torch
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
import vllm._custom_ops  # noqa
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
log = open(f"{OUT}/log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
wsp = marlin_make_workspace_new(dev); FP8 = scalar_types.float8_e4m3fn.id
N, K = 17408, 5120
_, q, s = marlin_quant_fp8_torch(torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02, 128)
M = 320
X = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
def gemm(): return torch.ops._C.marlin_gemm(X, None, q, None, s, None, None, wsp, FP8, M, N, K, False, True, False)
NB = M * 5120 * 2  # one prefill AR payload at M=320 (3.3 MB)
dsrc = torch.randn(NB // 2, dtype=torch.bfloat16, device=dev); ddst = torch.empty_like(dsrc)
h1 = torch.empty(NB // 2, dtype=torch.bfloat16, pin_memory=True); h2 = torch.empty(NB // 2, dtype=torch.bfloat16, pin_memory=True)
sA, sB, sC = torch.cuda.Stream(dev), torch.cuda.Stream(dev), torch.cuda.Stream(dev)
def ev(): return torch.cuda.Event(enable_timing=True)
def run(gemm_n=0, d2h_n=0, h2d_n=0):
    torch.cuda.synchronize()
    ea0, ea1, eb0, eb1, ec0, ec1 = ev(), ev(), ev(), ev(), ev(), ev()
    with torch.cuda.stream(sA):
        ea0.record()
        for _ in range(gemm_n): gemm()
        ea1.record()
    with torch.cuda.stream(sB):
        eb0.record()
        for _ in range(d2h_n): h1.copy_(dsrc, non_blocking=True)
        eb1.record()
    with torch.cuda.stream(sC):
        ec0.record()
        for _ in range(h2d_n): ddst.copy_(h2, non_blocking=True)
        ec1.record()
    torch.cuda.synchronize()
    return (ea0.elapsed_time(ea1) if gemm_n else 0), (eb0.elapsed_time(eb1) if d2h_n else 0), (ec0.elapsed_time(ec1) if h2d_n else 0)
for _ in range(3): run(5, 5, 5)
G = 40
g, _, _ = run(G); P(f"gemm alone: {g*1000/G:.1f} us per gate_up M={M}")
_, d, _ = run(0, 20); P(f"D2H alone: {d*1000/20:.1f} us per {NB/1e6:.2f} MB = {NB/(d/20)/1e6:.1f} GB/s")
_, _, h = run(0, 0, 20); P(f"H2D alone: {h*1000/20:.1f} us = {NB/(h/20)/1e6:.1f} GB/s")
_, d, h = run(0, 20, 20); P(f"D2H||H2D: D2H {d*1000/20:.1f} us, H2D {h*1000/20:.1f} us per copy (duplex)")
for nc in (10, 20):
    g, d, h = run(G, nc, nc)
    P(f"gemm x{G} || {nc} D2H + {nc} H2D: gemm {g*1000/G:.1f} us/launch, D2H {d*1000/nc:.1f} us, H2D {h*1000/nc:.1f} us per copy")
