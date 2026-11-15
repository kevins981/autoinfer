# 2-rank: mixed-step row-split pipeline model. L layers; per layer, per chunk: [wait chunk's AR from prev layer] ->
# GEMMs (gate_up + down, W8A8 Triton, M=R/N) -> NCCL AR of the chunk on a side stream. "barrier" variant: before each
# layer, wait for all chunks' ARs (models the token mixer needing all rows). N=1 = no overlap (serial).
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
L = 8
def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29553")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
    def w(N, K):
        return ((torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn),
                torch.rand(N // 128, K // 128, device="cuda") * 0.01 + 0.001)
    W1, W2 = w(17408, 5120), w(5120, 8704)
    side = torch.cuda.Stream()
    def run(R, N, barrier):
        rows = [R // N + (1 if i < R % N else 0) for i in range(N)]
        xs = [torch.randn(r, 5120, device="cuda", dtype=torch.bfloat16) for r in rows]
        evs = [None] * N
        main = torch.cuda.current_stream()
        def step():
            nonlocal evs
            for l in range(L):
                if barrier:
                    for e in evs:
                        if e is not None: main.wait_event(e)
                new = []
                for i in range(N):
                    if evs[i] is not None: main.wait_event(evs[i])
                    q, s = per_token_group_quant_fp8(xs[i], 128)
                    h = w8a8_triton_block_scaled_mm(q, W1[0], s, W1[1], [128, 128], torch.bfloat16)
                    q2, s2 = per_token_group_quant_fp8(h[:, :8704].contiguous(), 128)
                    o = w8a8_triton_block_scaled_mm(q2, W2[0], s2, W2[1], [128, 128], torch.bfloat16)
                    if N == 1:
                        dist.all_reduce(o); xs[i] = o; new.append(None)
                    else:
                        ready = torch.cuda.Event(); ready.record(main)
                        side.wait_event(ready)
                        with torch.cuda.stream(side):
                            dist.all_reduce(o)
                            done = torch.cuda.Event(); done.record(side)
                        o.record_stream(side)
                        xs[i] = o; new.append(done)
                evs = new
            for e in evs:
                if e is not None: main.wait_event(e)
        for _ in range(2): step()
        torch.cuda.synchronize(); dist.barrier(cpu)
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record()
        for _ in range(5): step()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 5 / L * 1000  # us per layer
    res = []
    for R in (256, 512, 1024, 1536):
        for N in (1, 2, 3, 4):
            for b in ((False, True) if N > 1 else (False,)):
                res.append((R, N, b, run(R, N, b)))
    if rank == 0:
        for R, N, b, t in res:
            print(f"R={R:5d} N={N} barrier={int(b)} us/layer={t:7.1f}", flush=True)
    dist.barrier(cpu)
if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
