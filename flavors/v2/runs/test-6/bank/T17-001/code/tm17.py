import os,sys,torch,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import build17
VARS=[int(v) for v in os.environ.get("VARS","0,16,32,48").split(",")]
exts={a:build17.get(abl=a%100, nst=(a//100 or 4)) for a in VARS}
dev=torch.device('cuda:0'); N,K=17408,5120
Bw=torch.randint(0,0x77,(N*K//4,),device=dev,dtype=torch.int32)
Sw=(torch.rand(K//128,N,device=dev)*0.01).bfloat16()
ws=torch.zeros(256*32*512,device=dev); fl=torch.zeros(256,dtype=torch.int32,device=dev)
flush=torch.empty(256<<20,dtype=torch.uint8,device=dev)
for M in (64,):
  x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); y=torch.zeros(M,N,device=dev,dtype=torch.bfloat16)
  for a in VARS:
    e=exts[a]
    for it in range(4):
      flush.view(torch.int32).max(); torch.cuda.synchronize()
      e.gemm(x,Bw,Sw,y,ws,fl,128); torch.cuda.synchronize()
    t=e.get_tm().cpu().numpy().reshape(256,4)[:128].astype(np.int64)
    t0=t[:,0].min(); b=(t[:,1]-t0)/1000; c=(t[:,2]-t0)/1000
    print(f"TM M={M} abl={a}: compute-done min/med/max {b.min():.1f}/{np.median(b):.1f}/{b.max():.1f}; end min/med/max {c.min():.1f}/{np.median(c):.1f}/{c.max():.1f}",flush=True)
