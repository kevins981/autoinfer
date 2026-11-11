# bs-r7-adjacent_field: Marlin FP8 (W8A16) at mixed-step M: stock (main 64*k launch + remainder launch) vs one launch padded to 64*k.
import os, sys, math, json, torch
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
SH = {"gate_up": (17408, 5120, 64), "down": (5120, 8704, 64), "in_proj": (8192, 5120, 48), "out_proj": (5120, 3072, 48),
      "qkv": (7168, 5120, 16), "o_proj": (5120, 3072, 16)}   # (N, K, calls per forward) per rank
torch.manual_seed(0)
W = {}
for k, (N, K, n) in SH.items():
    _, q, s = marlin_quant_fp8_torch(torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02, 128)
    byts = q.numel() * 4 + s.numel() * 2
    C = max(2, math.ceil(200e6 / byts))
    W[k] = dict(N=N, K=K, n=n, C=C, q=[q] + [q.clone() for _ in range(C - 1)], s=[s] + [s.clone() for _ in range(C - 1)])
torch.cuda.empty_cache()
MS = [int(x) for x in os.environ.get("MS", "48,64,80,96,105,120,128,150,177,192,220,256,262,300,306,320,350,384,403,448,474,527").split(",")]
MAXM = max(math.ceil(m / 64) * 64 for m in MS)
X = {K: torch.randn(MAXM, K, dtype=torch.bfloat16, device=dev) for K in (3072, 5120, 8704)}
s1 = torch.cuda.Stream(dev)
def gemm(k, c, M, Xk):
    w = W[k]; return torch.ops._C.marlin_gemm(Xk[:M], None, w["q"][c], None, w["s"][c], None, None, wsp, FP8, M, w["N"], w["K"], False, True, False)
def timeit(k, M, reps=2, R=6):
    C = W[k]["C"]; Xk = X[W[k]["K"]]
    def fn():
        for _ in range(reps):
            for c in range(C): gemm(k, c, M, Xk)
    with torch.cuda.stream(s1):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s1): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(R): g.replay()
    e1.record(); torch.cuda.synchronize()
    del g
    return e0.elapsed_time(e1) * 1000 / (R * reps * C)
res = {}
for M in MS:
    Mp = math.ceil(M / 64) * 64
    tot_a = tot_b = 0.0; row = {}
    for k in SH:
        a = timeit(k, M); b = timeit(k, Mp) if Mp != M else a
        # exactness of real rows: stock vs padded
        Xk = X[W[k]["K"]]
        ya = gemm(k, 0, M, Xk); yb = gemm(k, 0, Mp, Xk)[:M]
        d = (ya.float() - yb.float()).abs().max().item(); neq = (ya != yb).sum().item()
        row[k] = (round(a, 1), round(b, 1), d, neq)
        tot_a += a * W[k]["n"]; tot_b += b * W[k]["n"]
    res[M] = dict(stock_ms=tot_a / 1e3, pad_ms=tot_b / 1e3, per=row)
    P(f"M={M:4d} Mp={Mp:4d} per-forward Marlin stock {tot_a/1e3:6.2f} ms  padded {tot_b/1e3:6.2f} ms  "
      + " ".join(f"{k}:{v[0]}/{v[1]}(d{v[2]:.3g},n{v[3]})" for k, v in row.items()))
json.dump(res, open(f"{OUT}/res.json", "w"), indent=1)
