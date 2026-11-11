import os,sys,statistics,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import build17
exts={a:build17.get(abl=a) for a in (0,1,2,3)}
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
N,K=17408,5120
Bw=torch.randint(0,0x77,(N*K//4*4,),device=dev,dtype=torch.uint8).view(torch.int32)[:N*K//4]
Sw=(torch.rand(K//128,N,device=dev)*0.01).bfloat16()
ws=torch.zeros(256*32*512,device=dev); fl=torch.zeros(256,dtype=torch.int32,device=dev)
flush=torch.empty(256<<20,dtype=torch.uint8,device=dev)
for M in (64,48):
  x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); y=torch.zeros(M,N,device=dev,dtype=torch.bfloat16)
  for a,e in exts.items():
    def body(): flush.view(torch.int32).max(); e.gemm(x,Bw,Sw,y,ws,fl,128)
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): body()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s): body()
    ev=[]
    for _ in range(15):
        a0=torch.cuda.Event(True);a1=torch.cuda.Event(True);a0.record();g.replay();a1.record();a1.synchronize();ev.append(a0.elapsed_time(a1)*1000)
    gf=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf,stream=s): flush.view(torch.int32).max()
    ef=[]
    for _ in range(15):
        a0=torch.cuda.Event(True);a1=torch.cuda.Event(True);a0.record();gf.replay();a1.record();a1.synchronize();ef.append(a0.elapsed_time(a1)*1000)
    print(f"ABL M={M} abl={a} (1=noMMA,2=noLoad,3=neither): cold gemm ~ {statistics.median(ev)-statistics.median(ef):.1f} us",flush=True)
