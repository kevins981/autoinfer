# W-DBO feasibility: does an NCCL all-reduce (half of a mixed step's rows) overlap with a
# compute-bound GEMM of the other half on a second stream? bf16 matmul as proxy for Marlin at M~130.
import os, torch, torch.distributed as dist, json, statistics as st
dist.init_process_group("nccl"); r=dist.get_rank(); torch.cuda.set_device(r)
dev=torch.device("cuda",r)
res={}
for M in (130, 260):
    x=torch.randn(M,5120,device=dev,dtype=torch.bfloat16)
    W=torch.randn(5120,17408,device=dev,dtype=torch.bfloat16)   # gate_up-sized per GPU (2x8704)
    a=torch.randn(M,5120,device=dev,dtype=torch.bfloat16)
    s_c=torch.cuda.Stream(); s_n=torch.cuda.Stream()
    def gemm(n=4):
        for _ in range(n): y=x@W
    def ar(n=4):
        for _ in range(n): dist.all_reduce(a)
    def t(fn):
        ts=[]
        for _ in range(15):
            dist.barrier(); torch.cuda.synchronize()
            e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1)*1000)
        return st.median(ts[3:])
    def both():
        cur=torch.cuda.current_stream()
        s_c.wait_stream(cur); s_n.wait_stream(cur)
        with torch.cuda.stream(s_n): ar()
        with torch.cuda.stream(s_c): gemm()
        cur.wait_stream(s_c); cur.wait_stream(s_n)
    tg=t(gemm); ta=t(ar); tb=t(both)
    res[M]=dict(gemm_us=tg, ar_us=ta, both_us=tb, serial=tg+ta, hidden_frac=(tg+ta-tb)/min(tg,ta))
if r==0: print(json.dumps(res,indent=1))
dist.destroy_process_group()
