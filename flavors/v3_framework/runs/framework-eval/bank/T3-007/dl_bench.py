# Ceiling for DeltaLog L=2: GDN decode kernel (16 req x 3 rows, HV=24) with log_mode 0 (full writes) / 1 (T2 log, row-0 write) /
# 2 (row-0 write skipped: upper bound of a non-checkpoint step), followed by out_proj GEMM, ~48us spin (AR window), gate_up GEMM
# (with 40 MB prefetch of gate_up in the window, as in T3-003).
import torch, triton, triton.language as tl, random
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
dev = "cuda"; bf = torch.bfloat16
N, H, HV = 16, 8, 24
g = torch.Generator(device=dev); g.manual_seed(0)
A_log = (torch.randn(HV, device=dev, generator=g) * 0.5).float(); dt_bias = (torch.randn(HV, device=dev, generator=g) * 0.3).float()
norm_w = torch.randn(128, device=dev, generator=g).to(bf)
slots = (1 + torch.arange(N * 3, device=dev, dtype=torch.int32)).view(N, 3).contiguous()
nslots = 1 + 3 * N
# several layers' worth of state so the state is not L2-resident (48 GDN layers)
L = 48
S = [torch.randn(nslots, HV, 128, 128, device=dev, generator=g) * 0.05 for _ in range(L)]
n = torch.full((N,), 3, dtype=torch.int32, device=dev); cu = torch.zeros(N + 1, dtype=torch.int32, device=dev); cu[1:] = torch.cumsum(n, 0)
a_prev = torch.tensor([random.Random(1).choice([1, 2, 3]) for _ in range(N)], dtype=torch.int32, device=dev)
T = 48
mixed = torch.randn(T, 2 * H * 128 + HV * 128, device=dev, generator=g).to(bf)
a = torch.randn(T, HV, device=dev, generator=g).to(bf); b = torch.randn(T, HV, device=dev, generator=g).to(bf)
gate = torch.randn(T, HV, 128, device=dev, generator=g).to(bf); out = torch.zeros_like(gate)
def gdn(l, mode):
    ops.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, slots, cu, a_prev, S[l], gate, norm_w, out, 128 ** -0.5, 1e-5, "silu", mode)
M = 48
def mk(Nn, K):
    w = (torch.randn(Nn, K, device=dev) * 0.05).to(torch.float8_e4m3fn); ws = torch.rand(Nn // 128, K // 128, device=dev) * 0.01 + 0.001
    x = torch.randn(M, K, device=dev, dtype=bf); xq, xs = per_token_group_quant_fp8(x, 128); return w, ws, xq, xs
gus = [mk(17408, 5120) for _ in range(4)]; op = mk(5120, 3072)
gemm = lambda t: w8a8_triton_block_scaled_mm(t[2], t[0], t[3], t[1], [128, 128], bf)
@triton.jit
def _spin(o, iters):
    pid = tl.program_id(0); x = pid * 1.0
    for i in range(iters): x = x * 0.999999 + 1.0
    tl.store(o + pid, x)
@triton.jit
def _prefetch(p, o, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); nprog = tl.num_programs(0); acc = tl.zeros((BLOCK,), tl.int32)
    for off in range(pid * BLOCK, n, nprog * BLOCK):
        oo = off + tl.arange(0, BLOCK); acc += tl.load(p + oo, mask=oo < n, other=0, eviction_policy="evict_last")
    tl.store(o + pid, tl.sum(acc))
sout = torch.empty(64, device=dev); pout = torch.empty(4096, dtype=torch.int32, device=dev)
ITERS = 4000
for it in [2000, 4000, 8000, 16000]:
    _spin[(32,)](sout, it); torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record(); _spin[(32,)](sout, it); e1.record(); torch.cuda.synchronize()
    t = e0.elapsed_time(e1) * 1000
    if t > 45: ITERS = int(it * 48 / t); break
    ITERS = it
side = torch.cuda.Stream()
def seq(i, mode, pf):
    w = gus[i % 4]; l = i % L
    gdn(l, mode); gemm(op)
    ev = torch.cuda.Event(); ev.record()
    _spin[(32,)](sout, ITERS)
    if pf:
        with torch.cuda.stream(side):
            side.wait_event(ev); v = w[0].view(torch.int32).view(-1); _prefetch[(96,)](v, pout, int(v.numel() * 40 / 89.1), BLOCK=1024)
        torch.cuda.current_stream().wait_stream(side)
    gemm(w)
res = {}
# bitwise: modes 3/4 must leave the same state/out as mode 1
S0 = S[0].clone(); outs = {}
for mode in (1, 3, 4):
    S[0].copy_(S0); gdn(0, mode); torch.cuda.synchronize(); outs[mode] = (S[0].clone(), out.clone())
for mode in (3, 4):
    print("bitwise equal mode", mode, "vs 1:", torch.equal(outs[mode][0].view(torch.int32), outs[1][0].view(torch.int32)) and torch.equal(outs[mode][1], outs[1][1]), flush=True)
for pf in (True,):
    for mode in (1, 3, 4, 1, 3, 4):
        for _ in range(10): seq(0, mode, pf)
        torch.cuda.synchronize(); ts = []
        for i in range(120):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record(); seq(i, mode, pf); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
        ts = sorted(ts)[: 100]
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        # kernel alone
        for _ in range(5): gdn(0, mode)
        torch.cuda.synchronize(); e0.record()
        for i in range(50): gdn(i % L, mode)
        e1.record(); torch.cuda.synchronize()
        print(f"prefetch={pf} log_mode={mode}: sequence(gdn+out_proj+spin+gate_up) mean {sum(ts)/len(ts):.1f} us (trimmed), gdn kernel alone {e0.elapsed_time(e1)*1000/50:.1f} us", flush=True)
