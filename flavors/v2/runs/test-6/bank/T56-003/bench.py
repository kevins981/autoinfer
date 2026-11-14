import os, json, statistics, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", device_id=dev)
from vllm.utils import numa_utils
try: numa_utils.self_bind_to_device_node(rank, f"rank{rank}")
except Exception as e: print("numa bind failed", e)
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
from vllm.distributed.device_communicators.dma_staged_ar import DmaStagedAllReduce
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as T, t31_cfg
from vllm import _custom_ops as ops
g = dist.group.WORLD
host = HostStagedAllReduce(g, dev); dma = DmaStagedAllReduce(g, dev)
MB = 1 << 20; N = 5120; KH = 3072; KF = 6144
_ext = T.ext()
flush = torch.empty(512 * MB // 4, dtype=torch.int32, device=dev)
def timed(fn, reps=60, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / reps
def timed_cold(fn, reps=15):
    ts = []
    for _ in range(3): flush.sum(); fn()
    for _ in range(reps):
        flush.sum(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
W = torch.randn(N * KF, device=dev).to(torch.float8_e4m3fn).view(torch.int32)   # full K, synthetic
WH = W[: N * KH // 4].contiguous()
WS_F = torch.full((N // 128, KF // 128), 0.01, device=dev, dtype=torch.float32)
WS_H = WS_F[:, : KH // 128].contiguous()
bsw = (torch.empty(1, device=dev), torch.zeros(16384, device=dev, dtype=torch.int32))
res = {}
for M in (64, 128, 256, 512):
    torch.manual_seed(rank * 100 + M)
    xh = (torch.randn(M, KH, device=dev) * 2).to(torch.bfloat16)
    xq, xs = T.quant_perm(xh)                                      # this rank's K-half, as today
    # real-ish bf16 partial output for AR (GEMM of the half-K), compression is data dependent
    outh = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    _ext.bs_gemm_marlin(xq, WH, xs, WS_H, outh, T.pick(N, KH, M), 1, *bsw)
    ar_in = outh.contiguous()
    r = {}
    r["ar_host"] = timed(lambda: host.all_reduce(ar_in))
    r["ar_dma"] = timed(lambda: dma.all_reduce(ar_in)) if dma.should_use(ar_in) else None
    # all-gather: packed [M, 3168] bytes (xq row 3072 B | xs row 24 fp32 = 96 B)
    packed = torch.empty(M, 3168, dtype=torch.uint8, device=dev)
    packed[:, :3072] = xq.view(torch.uint8); packed[:, 3072:] = xs.view(torch.uint8)
    r["ag_packed"] = timed(lambda: host.all_gather(packed, 1))
    r["ag_xq_xs"] = timed(lambda: (host.all_gather(xq.view(torch.uint8), 1), host.all_gather(xs.view(torch.uint8), 1)))
    # bitwise check: gathered xq/xs == full-K quant of the concatenated bf16 input
    xh_all = [torch.empty_like(xh) for _ in range(2)]
    dist.all_gather(xh_all, xh)
    xfull = torch.cat(xh_all, dim=1)
    fq, fs = T.quant_perm(xfull)
    gq = host.all_gather(xq.view(torch.uint8), 1).view(torch.float8_e4m3fn); gs = host.all_gather(xs.view(torch.uint8), 1).view(torch.float32)
    gp = host.all_gather(packed, 1).view(M, 2, 3168)
    r["bitwise_xq"] = bool((gq.view(torch.uint8) == fq.view(torch.uint8)).all().item())
    r["bitwise_xs"] = bool((gs == fs).all().item())
    r["bitwise_packed"] = bool((gp[:, :, :3072].reshape(M, 6144) == fq.view(torch.uint8)).all().item() and (gp[:, :, 3072:].reshape(M, 192).contiguous().view(torch.float32) == fs).all().item())
    # GEMMs (cold): half-K incl. quant (today) vs full-K from gathered bytes (+ own half quant is needed in both)
    outf = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    if M > 96:
        r["gemm_half_incl_quant"] = timed_cold(lambda: (lambda q: _ext.bs_gemm_marlin(q[0], WH, q[1], WS_H, outh, T.pick(N, KH, M), 1, *bsw))(T.quant_perm(xh)))
        r["quant_half"] = timed_cold(lambda: T.quant_perm(xh))
        r["gemm_half"] = timed_cold(lambda: _ext.bs_gemm_marlin(xq, WH, xs, WS_H, outh, T.pick(N, KH, M), 1, *bsw))
        r["gemm_full"] = timed_cold(lambda: _ext.bs_gemm_marlin(fq, W, fs, WS_F, outf, T.pick(N, KF, M), 1, *bsw))
        # exactness vs sum of halves: full-K output vs bf16(sum of the two fp32 halves) is the AR result; compare rel diff
        o2 = [torch.empty_like(outh) for _ in range(2)]; dist.all_gather(o2, outh)
        ar = (o2[0].float() + o2[1].float()).to(torch.bfloat16)
        r["rel_diff_full_vs_AR"] = ((outf.float() - ar.float()).norm() / ar.float().norm()).item()
        r["maxabs_full_vs_AR"] = (outf.float() - ar.float()).abs().max().item()
    else:
        for nm, (K, Wt, Ws, o, q, s) in {"half": (KH, WH, WS_H, outh, xq, xs), "full": (KF, W, WS_F, outf, fq, fs)}.items():
            ws = torch.zeros(max(max(t31_cfg.ws_numel(N, K, M, c) for c in {t31_cfg.pick(N, K, M, rg) for rg in ("cold", "mid", "r48", "warm")}), 1), device=dev, dtype=torch.float32)
            cnt = torch.zeros(4096, device=dev, dtype=torch.int32)
            try:
                r[f"t31_{nm}"] = timed_cold(lambda: ops.t31_gemm_e4m3(q, s, Wt, Ws, ws, cnt, M, N, K, t31_cfg.pick(N, K, M, "cold")))
            except Exception as e: r[f"t31_{nm}"] = f"err {e!r}"[:80]
        r["quant_half"] = timed_cold(lambda: T.quant_perm(xh))
    res[M] = r
    if rank == 0: print(M, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    dist.barrier()
if rank == 0: json.dump(res, open("res.json", "w"), indent=1)
