# best bs_gemm_marlin cfg per TP decode shape and M (cold, read flush); writes bs_best.json
import os, json, statistics, torch
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as t29
dev = torch.device("cuda:0"); ext = t29.ext(); NW = 6
SH = {"qkvz": (8192, 5120), "gdn_out": (5120, 3072), "gate_up": (17408, 5120), "down": (5120, 8704)}
flush = torch.empty(512 << 18, dtype=torch.int32, device=dev); sink = torch.zeros(1, device=dev, dtype=torch.int64)
dz = (torch.empty(1, device=dev, dtype=torch.float32), torch.zeros(64, device=dev, dtype=torch.int32))
BSC = list(range(32)) + [103, 107, 109, 112, 115, 121, 124, 125, 126, 127, 128]
res = {}
for k, (N, K) in SH.items():
    W = [torch.empty(K // 16, N * 4, dtype=torch.int32, device=dev).random_() for _ in range(NW)]
    S = [torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32) for _ in range(NW)]
    for M in (64, 96, 128):
        xq = torch.randn(M, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
        xs = torch.full((M, K // 128), 0.05, device=dev, dtype=torch.float32)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16); best = None
        for c in BSC:
            try:
                ts = []
                for i in range(7):
                    sink.add_(flush.sum()); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); ext.bs_gemm_marlin(xq, W[i % NW], xs, S[i % NW], out, c, 1, *dz); b.record(); torch.cuda.synchronize()
                    if i >= 2: ts.append(a.elapsed_time(b) * 1000)
                t = statistics.median(ts)
                if best is None or t < best[0]: best = (round(t, 1), c)
            except Exception:
                pass
        res.setdefault(k, {})[M] = best
        print(k, M, best, "floor", round(N * K / 0.98e6, 1), flush=True)
    del W
json.dump(res, open("bs_best.json", "w"), indent=1)
