# AR microbenchmark: NCCL all_reduce of bf16 [M,5120] (2 procs), plus raw pinned D2H/H2D bandwidth.
import os, sys, torch, torch.distributed as dist, time
def main():
    rank=int(os.environ["RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    res=[]
    for M in (48, 450, 2048):
        x=torch.randn(M,5120,device="cuda",dtype=torch.bfloat16)
        for _ in range(20): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        # graph-captured loop of 50 ARs (like vLLM)
        s=torch.cuda.Stream(); g=torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            for _ in range(3): dist.all_reduce(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(50): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        for _ in range(2): g.replay()
        torch.cuda.synchronize(); dist.barrier()
        t=time.perf_counter(); 
        for _ in range(4): g.replay()
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/200
        res.append((M, dt*1e6, M*5120*2/dt/1e9))
    if rank==0:
        print("CFG", os.environ.get("CFG"), " ".join(f"M={m}: {u:.1f}us algBW={b:.1f}GB/s" for m,u,b in res), flush=True)
    # raw copy-engine bandwidth: concurrent D2H and H2D of 4.6MB pinned on both GPUs
    if os.environ.get("CFG")=="default":
        n=450*5120*2
        h1=torch.empty(n,dtype=torch.uint8,pin_memory=True); h2=torch.empty(n,dtype=torch.uint8,pin_memory=True)
        d1=torch.empty(n,dtype=torch.uint8,device="cuda"); d2=torch.empty(n,dtype=torch.uint8,device="cuda")
        s1=torch.cuda.Stream(); s2=torch.cuda.Stream()
        for mode in ("d2h","h2d","both"):
            dist.barrier(); torch.cuda.synchronize(); t=time.perf_counter()
            for _ in range(50):
                if mode in("d2h","both"):
                    with torch.cuda.stream(s1): h1.copy_(d1,non_blocking=True)
                if mode in("h2d","both"):
                    with torch.cuda.stream(s2): d2.copy_(h2,non_blocking=True)
            torch.cuda.synchronize(); dt=(time.perf_counter()-t)/50
            print(f"rank{rank} copy {mode} 4.6MB: {dt*1e6:.1f}us  {n/dt/1e9:.1f}GB/s per direction", flush=True)
    dist.destroy_process_group()
if __name__=="__main__": main()
