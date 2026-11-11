"""T47-002 microbench (GPU0 only): (1) DRAM read BW vs active SMs, (2) champion M=64 decode GEMMs cold vs SMs (green ctx)."""
import os, sys, json, statistics
import torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
X = load("t47bw", [f"{HERE}/bw.cu"], build_directory=f"{HERE}/build", extra_cuda_cflags=["-O3"], verbose=False)
from torch.cuda.green_contexts import GreenContext
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
GB = 1 << 30
PART = os.environ.get("PART", "12")
res = {"bw": {}, "bw_green": {}, "gemm": {}}
out = torch.zeros(4, dtype=torch.int64, device=dev)
TILE16 = {0: 1024 * 8, 1: 512 * 4, 2: 1024 * 16, 3: 512 * 2, 4: 256 * 4, 5: 512 * 2}
NAMES = {0: "ldg 1024thr U8 (128KB/SM)", 1: "ldg 512thr U4 (32KB/SM)", 2: "ldg 1024thr U16 (256KB/SM)",
         3: "cp.async 512thr S6 (96KB ring)", 4: "cp.async 256thr S4 (64KB ring)", 5: "cp.async 512thr S3 (48KB ring)"}

def ev_time(fn, stream, reps, pre=None):
    ts = []
    for _ in range(reps):
        if pre: pre()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)  # us
    return statistics.median(ts), min(ts)

default = torch.cuda.current_stream()
print("blocks/SM:", {v: X.blocks_per_sm(v) for v in (0, 3, 4)}, flush=True)
if "1" in PART:
    buf = torch.empty(GB // 4, dtype=torch.int32, device=dev).random_()
    SMS = [16, 32, 48, 64, 80, 96, 112, 128]
    for v in range(6):
        nbytes = (buf.numel() * 4 // 16 // TILE16[v]) * TILE16[v] * 16
        row = {}
        for n in SMS:
            fn = lambda: X.run(buf, v, n, out)
            fn(); torch.cuda.synchronize()
            med, best = ev_time(fn, default, 5)
            row[n] = nbytes / (best * 1e-6) / 1e9
        res["bw"][NAMES[v]] = row
        top = max(row.values())
        n95 = min(n for n in SMS if row[n] >= 0.95 * top)
        print(f"BW {NAMES[v]:32s} " + " ".join(f"{n}:{row[n]:.0f}" for n in SMS) + f"  GB/s | best {top:.0f} | >=95% at {n95} SMs", flush=True)
    # cross-check: grid = n blocks inside a green context of n SMs
    for v in (0, 3):
        nbytes = (buf.numel() * 4 // 16 // TILE16[v]) * TILE16[v] * 16
        for n in (32, 64, 96):
            gc = GreenContext(num_sms=n); s = gc.Stream()
            fn = lambda: X.run(buf, v, 128, out)  # full 128-block grid, confined to n SMs
            with torch.cuda.stream(s): fn()
            torch.cuda.synchronize()
            med, best = ev_time(fn, s, 5)
            res["bw_green"][f"{NAMES[v]} @green{n}"] = nbytes / (best * 1e-6) / 1e9
            print(f"BW-green {NAMES[v]:32s} grid128 in {n}-SM green ctx: {nbytes / (best * 1e-6) / 1e9:.0f} GB/s", flush=True)
    del buf
    torch.cuda.empty_cache()

if "2" in PART:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import t31_cfg, t29_marlin_w8a8 as T29
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    M = 64
    NL = 3
    SH = {"qkvz": (8192, 5120), "down": (5120, 8704), "gate_up": (17408, 5120)}
    W = {k: [torch.empty(N * K // 4, dtype=torch.int32, device=dev).random_() for _ in range(NL)] for k, (N, K) in SH.items()}
    WS = {k: [torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32) for _ in range(NL)] for k, (N, K) in SH.items()}
    act = {}
    for k in ("qkvz", "down"):
        N, K = SH[k]
        xq = torch.randn(M, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
        xs = torch.full((M, K // 128), 0.05, device=dev, dtype=torch.float32)
        cfg = t31_cfg.pick(N, K, M, "cold")
        ws = torch.zeros(max(t31_cfg.ws_numel(N, K, M, cfg), 1), device=dev, dtype=torch.float32)
        cnt = torch.zeros(4096, device=dev, dtype=torch.int32)
        act[k] = (xq, xs, ws, cnt, cfg)
    xg = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
    mws = marlin_make_workspace_new(dev, 4)
    MS = [(torch.rand(5120 // 128, 17408, device=dev) * 0.02 + 0.001).to(torch.bfloat16) for _ in range(NL)]
    WG = [w.view(5120 // 16, 17408 * 4) for w in W["gate_up"]]
    li = [0]
    def t31(k):
        N, K = SH[k]; xq, xs, ws, cnt, cfg = act[k]; l = li[0] % NL
        return ops.t31_gemm_e4m3(xq, xs, W[k][l], WS[k][l], ws, cnt, M, N, K, cfg)
    def gu_champ():  # T45-005 path at 48 < M <= 96: quant_perm + bs_gemm_marlin
        l = li[0] % NL
        return T29.w8a8_marlin(xg, W["gate_up"][l], WS["gate_up"][l], 17408, 5120)
    def gu_marlin():  # plain bf16-activation Marlin (pre-T45 path), reference
        l = li[0] % NL
        return torch.ops._C.marlin_gemm(xg, None, WG[l], None, MS[l], None, None, mws,
                                        scalar_types.float8_e4m3fn.id, M, 17408, 5120, False, True, False)
    KERN = {"qkvz t31b": (lambda: t31("qkvz"), 8192 * 5120), "down t31b": (lambda: t31("down"), 5120 * 8704),
            "gate_up champion (quant_perm + bs_gemm_marlin)": (gu_champ, 17408 * 5120),
            "gate_up plain Marlin (ref)": (gu_marlin, 17408 * 5120)}
    flush = torch.empty(256 << 20 >> 2, dtype=torch.int32, device=dev).random_()
    sink = torch.zeros(1, dtype=torch.int64, device=dev)
    def pre():  # READ flush (KB76) + reset persisting L2 lines (KB103); next weight copy
        li[0] += 1
        X.run(flush, 0, 128, sink)
        X.reset_persisting()
    for name, (fn, wb) in KERN.items():
        for _ in range(3): pre(); fn()
        torch.cuda.synchronize()
    for n in (128, 112, 96, 80, 64, 48, 32):
        if n == 128:
            s = default
        else:
            gc = GreenContext(num_sms=n); s = gc.Stream()
        for name, (fn, wb) in KERN.items():
            with torch.cuda.stream(s):
                pre(); fn()
            torch.cuda.synchronize()
            med, best = ev_time(fn, s, 15, pre)
            res["gemm"].setdefault(name, {})[n] = {"med_us": med, "min_us": best, "bytes": wb}
            print(f"GEMM {name:46s} SMs {n:3d}{' (default stream)' if n == 128 else ' (green ctx)':17s} cold med {med:7.1f} us (min {best:7.1f}) "
                  f"weights {wb / 1e6:.1f} MB -> {wb / (med * 1e-6) / 1e9:.0f} GB/s", flush=True)
json.dump(res, open(os.environ.get("OUT", f"{HERE}/res.json"), "w"), indent=1)
print("done", flush=True)
