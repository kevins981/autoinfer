# T45 stage 1b: e4m3-MMA gate_up (17408x5120 per rank) at M=48..128, cold (read flush), vs Marlin bf16-MMA.
import os, json, statistics, torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import t31_cfg
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as t29
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.scalar_type import scalar_types
dev = torch.device("cuda:0"); N, K = 17408, 5120; NW = 8
W = [torch.empty(K // 16, N * 4, dtype=torch.int32, device=dev).random_() for _ in range(NW)]
WSb = [(torch.rand(K // 128, N, device=dev) * 0.02 + 0.001).to(torch.bfloat16) for _ in range(NW)]
WSf = [torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32) for _ in range(NW)]
flush = torch.empty(512 << 18, dtype=torch.int32, device=dev); sink = torch.zeros(1, device=dev, dtype=torch.int64)
mws = marlin_make_workspace_new(dev, 4); ext = t29.ext()
dz = (torch.empty(1, device=dev, dtype=torch.float32), torch.zeros(64, device=dev, dtype=torch.int32))
def fl(): sink.add_(flush.sum())
def tm(fn, reps=12):
    ts = []
    for i in range(reps + 2):
        fl(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(i % NW); b.record(); torch.cuda.synchronize()
        if i >= 2: ts.append(a.elapsed_time(b) * 1000)
    return round(statistics.median(ts), 1)
res = {}
T31C = [322, 4388, 578, 290, 292, 385, 548, 804, 4386]
BSC = list(range(32)) + [103, 107, 109, 112, 115, 121, 124, 125, 126, 127, 128]
for M in (48, 64, 80, 96, 112, 128):
    r = {}
    xg = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    r["marlin_bf16mma"] = tm(lambda l: torch.ops._C.marlin_gemm(xg, None, W[l], None, WSb[l], None, None, mws,
                              scalar_types.float8_e4m3fn.id, M, N, K, False, True, False))
    xq = torch.randn(M, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
    xs = torch.full((M, K // 128), 0.05, device=dev, dtype=torch.float32)
    if M <= 96:
        best = None
        for c in T31C:
            try:
                ws = torch.zeros(max(t31_cfg.ws_numel(N, K, M, c), 1), device=dev); cnt = torch.zeros(4096, device=dev, dtype=torch.int32)
                t = tm(lambda l: ops.t31_gemm_e4m3(xq, xs, W[l], WSf[l], ws, cnt, M, N, K, c))
                r[f"t31_{c}"] = t; best = t if best is None or t < best else best
            except Exception as e:
                r[f"t31_{c}"] = "err"
        r["t31_best"] = best
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16); best = None
    for c in BSC:
        try:
            t = tm(lambda l: ext.bs_gemm_marlin(xq, W[l], xs, WSf[l], out, c, 1, *dz), reps=6)
            r[f"bs_{c}"] = t; best = (t, c) if best is None or t < best[0] else best
        except Exception as e:
            pass
    r["bs_best"] = best
    r["floor_us_0.98TBps"] = round(N * K / 0.98e6, 1)
    r["bf16mma_roof_us_146TF"] = round(2 * M * N * K / 146e12 * 1e6, 1)
    r["e4m3mma_roof_us_291TF"] = round(2 * M * N * K / 291e12 * 1e6, 1)
    res[M] = r
    print(M, {k: v for k, v in r.items() if not k.startswith(("bs_", "t31_")) or k.endswith("best")}, flush=True)
    json.dump(res, open("gu.json", "w"), indent=1)
