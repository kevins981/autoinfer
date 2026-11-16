# Microbench: FP8 e4m3 Triton dot with fp32 vs fp16 accumulator on sm_89, prefill-size shapes.
# Per 128-K block: acc_blk = dot(a_blk, b_blk) (fp16 or fp32 acc), then acc32 += acc_blk * sa * sb (fp32), as the block-scaled scheme.
import torch, triton, triton.language as tl, sys

@triton.jit
def k(A, B, C, SA, SB, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F16: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, K // BK):
        a = tl.load(A + rm[:, None] * K + (kb * BK + rk)[None, :])
        b = tl.load(B + rn[None, :] * K + (kb * BK + rk)[:, None])
        sa = tl.load(SA + rm * (K // 128) + kb)
        sb = tl.load(SB + (rn // 128) * (K // 128) + kb)
        if F16:
            p = tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
        else:
            p = tl.dot(a, b)
        acc += p * sa[:, None] * sb[None, :]
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

def run(M, N, K, f16, BM=128, BN=128, BK=128, warps=8, stages=3):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device='cuda') * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device='cuda') * 2).to(torch.float8_e4m3fn)
    sa = torch.rand(M, K // 128, device='cuda') * 0.01; sb = torch.rand(N // 128, K // 128, device='cuda') * 0.01
    c = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    g = (M // BM, N // BN)
    f = lambda: k[g](a, b, c, sa, sb, M, N, K, BM, BN, BK, f16, num_warps=warps, num_stages=stages)
    f(); torch.cuda.synchronize()
    ms = triton.testing.do_bench(f, warmup=20, rep=200)
    tf = 2 * M * N * K / ms / 1e9
    return ms, tf, c.float()

for (M, N, K) in [(512, 17408, 5120), (512, 5120, 8704), (2048, 17408, 5120)]:
    for cfg in [dict(BM=128, BN=128, BK=128, warps=8, stages=3), dict(BM=64, BN=128, BK=128, warps=4, stages=3), dict(BM=128, BN=256, BK=128, warps=8, stages=2)]:
        try:
            m32, t32, c32 = run(M, N, K, False, **cfg)
            m16, t16, c16 = run(M, N, K, True, **cfg)
            d = (c16 - c32).abs().max().item(); rel = d / c32.abs().max().item()
            print(f"M={M} N={N} K={K} {cfg}: f32acc {m32*1e3:.0f} us {t32:.0f} TF | f16acc {m16*1e3:.0f} us {t16:.0f} TF | speedup {m32/m16:.2f}x | maxabs {d:.3g} rel {rel:.2e} inf {torch.isinf(c16).any().item()}", flush=True)
        except Exception as e:
            print(f"M={M} N={N} K={K} {cfg}: ERR {type(e).__name__} {str(e)[:200]}", flush=True)
