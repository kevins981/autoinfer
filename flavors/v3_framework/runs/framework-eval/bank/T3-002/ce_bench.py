# 2-rank: copy-engine AR vs hostmem (32 CTAs) vs NCCL, alone and concurrent with W8A8 GEMMs (gate_up+down at M rows)
# on another stream (models the mixed-step row-split overlap). Prints per-iteration us.
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
from pathlib import Path
from torch.utils.cpp_extension import load
ROWS = [96, 128, 192, 256, 512]
HERE = Path(__file__).parent
def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29541")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    from vllm.distributed.device_communicators.hostmem_all_reduce import HostMemAllReduce, _load_ext
    from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
    ce = load("t3_ce_ar", [str(HERE / "ce_ext/ce_ar.cu")], extra_ldflags=["-lcuda"], verbose=False)
    hx = _load_ext()
    maxb = 512 * 5120 * 2
    slot = (maxb + 4095) // 4096 * 4096
    name = [f"/dev/shm/t3_ce_{os.getpid()}"]; dist.broadcast_object_list(name, src=0, group=cpu)
    if rank == 0: _, dp = hx.map_shm(name[0], 4 * slot + 4096, True)
    dist.barrier(cpu)
    if rank == 1: _, dp = hx.map_shm(name[0], 4 * slot + 4096, False)
    dist.barrier(cpu)
    if rank == 0: os.unlink(name[0])
    ep = [0]
    def ce_ar(x, o, t):
        ep[0] += 1
        ce.ce_allreduce(x, o, t, dp, slot, rank, ep[0], 64)
        return o
    hm = HostMemAllReduce(cpu, torch.device("cuda", rank), maxb)
    def hm_ar(x, o, t):
        hm.ext.allreduce(x, o, hm.dp, hm.flags, hm.ctr.data_ptr(), hm.rank, 32, 256, hm.slot, 1); return o
    def nc_ar(x, o, t):
        dist.all_reduce(x); return x
    # GEMM weights (TP-sharded shapes): gate_up N=17408 K=5120; down N=5120 K=8704
    def w(N, K):
        B = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        Bs = torch.rand(N // 128, K // 128, device="cuda") * 0.01 + 0.001
        return B, Bs
    W1, W2 = w(17408, 5120), w(5120, 8704)
    side = torch.cuda.Stream()
    def tm(fn, it=40):
        for _ in range(4): fn()
        torch.cuda.synchronize(); dist.barrier(cpu)
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / it * 1000
    out = []
    for R in ROWS:
        torch.manual_seed(rank + R)
        x = torch.randn(R, 5120, device="cuda", dtype=torch.bfloat16)
        o = torch.empty_like(x); t = torch.empty_like(x)
        a1 = per_token_group_quant_fp8(torch.randn(R, 5120, device="cuda", dtype=torch.bfloat16), 128)
        a2 = per_token_group_quant_fp8(torch.randn(R, 8704, device="cuda", dtype=torch.bfloat16), 128)
        def gemm():
            w8a8_triton_block_scaled_mm(a1[0], W1[0], a1[1], W1[1], [128, 128], torch.bfloat16)
            w8a8_triton_block_scaled_mm(a2[0], W2[0], a2[1], W2[1], [128, 128], torch.bfloat16)
        # correctness: CE == hostmem bitwise
        r_ce = ce_ar(x, o, t).clone(); torch.cuda.synchronize()
        r_hm = hm_ar(x, torch.empty_like(x), None).clone(); torch.cuda.synchronize()
        eq = torch.equal(r_ce, r_hm)
        row = {"rows": R, "MB": R * 10240 / 1e6, "ce==hm": eq, "gemm": tm(gemm)}
        for nm, f in [("ce", ce_ar), ("hm", hm_ar), ("nccl", nc_ar)]:
            xx = x.clone()
            row[nm] = tm(lambda: f(xx, o, t))
            def both():
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side): f(xx, o, t)
                gemm()
                torch.cuda.current_stream().wait_stream(side)
            row[nm + "||gemm"] = tm(both)
        out.append(row)
    for r in out:
        print(f"rank{rank} " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    dist.barrier(cpu)
if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
