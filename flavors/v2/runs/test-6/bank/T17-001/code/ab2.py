import os,sys,statistics as st,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import build17
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
dev=torch.device('cuda:0'); torch.cuda.set_device(0)
VARS=[0,4,8,12,10]
exts={a:build17.get(abl=a) for a in VARS}
N,K=17408,5120
m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
prepare_fp8_layer_for_marlin(m,size_k_first=False)
Bw=m.weight.data; Sw=m.weight_scale_inv.data
wsp=marlin_make_workspace_new(dev,max_blocks_per_sm=4)
ws=torch.zeros(256*32*512,device=dev); fl=torch.zeros(256,dtype=torch.int32,device=dev)
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32)
R=8
def graph(fn):
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); s.synchronize(); g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=s):
            for _ in range(R): fn()
    s.synchronize(); return g
def tm(g,n=25):
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b))
    return st.median(ts)
tf=tm(graph(lambda: flush.zero_()))
print(f"flush-only per rep {tf/R*1000:.1f} us",flush=True)
for M in (48,64):
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); y=torch.zeros(M,N,device=dev,dtype=torch.bfloat16)
    t=(tm(graph(lambda:(flush.zero_(),apply_fp8_marlin_linear(x,m.weight,m.weight_scale_inv,wsp,N,K,None))))-tf)/R*1000
    print(f"T5METHOD M={M} marlin cold {t:.1f} us",flush=True)
    for a in VARS:
        e=exts[a]
        t=(tm(graph(lambda:(flush.zero_(),e.gemm(x,Bw,Sw,y,ws,fl,128))))-tf)/R*1000
        print(f"T5METHOD M={M} mine abl={a} (4=noFixup 8=noCompute 12=both 10=noCompute+noLoad) cold {t:.1f} us",flush=True)
