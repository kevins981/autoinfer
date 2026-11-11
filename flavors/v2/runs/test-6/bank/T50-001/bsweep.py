# T50: bs_gemm_marlin (W8A8, champion mixed-step GEMM) efficiency at prefill-chunk M, all tile configs,
# vs per-tensor FP8 cuBLASLt (torch._scaled_mm) and a raw e4m3 mma roof (329 TF, KB69). Cold (L2 flush).
import json, statistics, sys, torch
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as t29
dev = torch.device("cuda:0")
ext = t29.ext()
dz = (torch.empty(1, device=dev, dtype=torch.float32), torch.zeros(64, device=dev, dtype=torch.int32))
flush = torch.empty(512 << 18, dtype=torch.int32, device=dev); sink = torch.zeros(1, device=dev, dtype=torch.int64)
def fl(): sink.add_(flush.sum())
def tm(fn, reps):
    ts = []
    for i in range(reps + 2):
        fl(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        if i >= 2: ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
SH = [(17408, 5120), (5120, 8704), (8192, 5120), (7168, 5120), (5120, 3072)]
MS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048]
CF = list(range(32)) + [103, 107, 109, 112, 115, 121, 124, 125, 126, 127, 128]
res = {}
for N, K in SH:
    W = torch.empty(K // 16, N * 4, dtype=torch.int32, device=dev).random_()
    WSf = torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32)
    Wt = torch.randn(N, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
    one = torch.ones(1, device=dev)
    for M in MS:
        xq = torch.randn(M, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
        xs = torch.full((M, K // 128), 0.05, device=dev, dtype=torch.float32)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        r = {}
        for c in CF:
            try:
                r[c] = round(tm(lambda: ext.bs_gemm_marlin(xq, W, xs, WSf, out, c, 1, *dz), 5), 1)
            except Exception:
                pass
        tab = t29.pick(N, K, M)
        best = min(r, key=r.get)
        try:
            cub = round(tm(lambda: torch._scaled_mm(xq, Wt.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 5), 1)
        except Exception as e:
            cub = str(e)[:60]
        fl_ = 2 * M * N * K
        e = dict(table_cfg=tab, table_us=r.get(tab), best_cfg=best, best_us=r[best], cublas_pt_us=cub,
                 roof329_us=round(fl_ / 329e12 * 1e6, 1), bytes_floor_us=round(N * K / 0.95e6, 1),
                 table_TF=round(fl_ / r[tab] / 1e6, 0) if tab in r else None, best_TF=round(fl_ / r[best] / 1e6, 0),
                 all=r)
        res[f"{N}x{K}@{M}"] = e
        print(f"{N}x{K} M={M}: table cfg {tab} {e['table_us']} us ({e['table_TF']} TF) | best cfg {best} {r[best]} us ({e['best_TF']} TF) | cublas per-tensor {cub} | roof {e['roof329_us']} | bytes {e['bytes_floor_us']}", flush=True)
        json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "bsweep.json", "w"), indent=1)
