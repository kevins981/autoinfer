import os, sys, json, statistics as st, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
import vllm._C_stable_libtorch as C
g_id=int(sys.argv[1]); dev=torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES selects the physical GPU
print('KB61 _C =',C.__file__,'phys gpu',g_id,flush=True)
FS=[0.88,0.92,0.96,1.00,1.04,1.08,1.12]; MS=[48,128,256,384,640]; NH=8704; K0=5120
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
ws=marlin_make_workspace_new(dev,4)
def call(w,s,N,K,x): return ops.marlin_gemm(x,None,w,None,s,None,None,ws,scalar_types.float8_e4m3fn,x.size(0),N,K,False,True)
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32); R=8
def graph(fn,r=R):
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); s.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=s):
            for _ in range(r): fn()
    s.synchronize(); return g
def tm(g,n=25):
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b))
    return st.median(ts)
tf=tm(graph(lambda: flush.zero_()))
res={}
for f in FS:
    nh=round(NH*f/128)*128
    for name,(N,K) in {'gate_up':(2*nh,K0),'down':(K0,nh)}.items():
        L=mk(N,K)
        for M in MS:
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
            cold=(tm(graph(lambda: (flush.zero_(), call(L.weight,L.weight_scale_inv,N,K,x))))-tf)/R*1000
            warm=tm(graph(lambda: call(L.weight,L.weight_scale_inv,N,K,x)))/R*1000
            res[f'{name}|{M}|{f}']=dict(nh=nh,cold=cold,warm=warm)
        del L; torch.cuda.empty_cache()
    print('done f',f,flush=True)
# subrange test: f=1 gate_up M=48, weight as offset view of bigger buffer
N,K=2*NH,K0; L=mk(N,K); x=torch.randn(48,K,device=dev,dtype=torch.bfloat16)
w=L.weight; buf=torch.empty(w.numel()+8192*3,device=dev,dtype=w.dtype)
buf2=buf.view(torch.uint8)
off=16384  # bytes offset (16B aligned, not 256B-page aligned multiples of the original)
wv=buf.view(torch.uint8)[off:off+w.numel()*w.element_size()].view(w.dtype) if w.dtype!=torch.int32 else None
if wv is None:
    bi=buf.view(torch.uint8); n=w.numel()*4
    wv=bi[off:off+n].view(torch.int32)
wv=wv.view(w.shape); wv.copy_(w)
assert wv.data_ptr()!=w.data_ptr()
sub={}
for tag,ww in (('own',w),('view',wv)):
    sub[tag]=dict(cold=[],warm=[])
    for rep in range(3):
        sub[tag]['cold'].append((tm(graph(lambda: (flush.zero_(), call(ww,L.weight_scale_inv,N,K,x))))-tf)/R*1000)
        sub[tag]['warm'].append(tm(graph(lambda: call(ww,L.weight_scale_inv,N,K,x)))/R*1000)
sub['equal_out']=bool(torch.equal(call(w,L.weight_scale_inv,N,K,x),call(wv,L.weight_scale_inv,N,K,x)))
sub['dtype']=str(w.dtype); sub['shape']=list(w.shape)
json.dump(dict(res=res,sub=sub),open(f'res_gpu{g_id}.json','w'),indent=1)
print('SUB',sub)
