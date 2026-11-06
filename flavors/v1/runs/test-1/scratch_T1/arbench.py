import os, sys, time, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
n=int(sys.argv[1]) if len(sys.argv)>1 else 48
x=torch.randn(n,5120,dtype=torch.bfloat16,device="cuda")
for _ in range(20): dist.all_reduce(x)
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(64): dist.all_reduce(x)
for _ in range(3): g.replay()
torch.cuda.synchronize(); dist.barrier()
t=time.time(); R=20
for _ in range(R): g.replay()
torch.cuda.synchronize(); dt=(time.time()-t)/(R*64)*1e6
if rank==0: print(f"RESULT n={n} {dt:.1f} us/AR", {k:v for k,v in os.environ.items() if k.startswith('NCCL')}, flush=True)
dist.destroy_process_group()
