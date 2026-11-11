# Task 2: stock vs single-launch parallel slabs, same build (env AI_SLAB / AI_SLAB_S49). Adapted from mb1.py.
import os, sys, statistics as st, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
D=0; dev=torch.device('cuda:0'); torch.cuda.set_device(0)
SH={'qkvz':(8192,5120),'gdn_out':(5120,3072),'attn_qkv':(7168,5120),'attn_o':(5120,3072),'gate_up':(17408,5120),'down':(5120,8704)}
CNT={'qkvz':48,'gdn_out':48,'attn_qkv':16,'attn_o':16,'gate_up':64,'down':64}
MS=[48,52,56,64,72,80,96,112,120,128,144,160,176,192]
MODES={'stock':{'AI_SLAB':'0','AI_SLAB_S49':'1','AI_SLAB_INNER':'0'},'new':{'AI_SLAB':'1','AI_SLAB_S49':'1','AI_SLAB_INNER':'0'},'inner':{'AI_SLAB':'1','AI_SLAB_S49':'1','AI_SLAB_INNER':'1'},'inner2x32':{'AI_SLAB':'1','AI_SLAB_S49':'2','AI_SLAB_INNER':'1'}}
def setmode(m): os.environ.update(MODES[m])
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32)
ws=marlin_make_workspace_new(dev); R=8
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
peak=133e12; bw=955e9
wbytes=sum(CNT[n]*SH[n][0]*SH[n][1] for n in SH); flops1=sum(CNT[n]*2*SH[n][0]*SH[n][1] for n in SH)
res={}
Ls={n:mk(*SH[n]) for n in SH}

def callc(L,N,K,x,c=None):
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    return ops.marlin_gemm(x,c,L.weight,None,L.weight_scale_inv,None,None,ws,scalar_types.float8_e4m3fn,x.size(0),N,K,False,True)
torch.manual_seed(2); nb=0; nt=0; mx=0.0; gbad=0; first=[]
for name in ('gdn_out','down','gate_up'):
    N,K=SH[name]; L=Ls[name]
    for M in range(1,257):
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        setmode('stock'); ref=callc(L,N,K,x)
        setmode('new'); o1=callc(L,N,K,x)
        for m in (['inner']+(['inner2x32'] if 48<M<=64 else [])):
            setmode(m)
            big=torch.full((M+80,N),float('nan'),device=dev,dtype=torch.bfloat16); c=big[:M]
            o=callc(L,N,K,x,c); nt+=1
            gbad+=int(not torch.isnan(big[M:]).all())
            if torch.equal(o,ref): nb+=1
            else:
                mx=max(mx,(o.float()-ref.float()).abs().max().item())
                if not torch.isfinite(o.float()).all(): first.append((name,M,m,'NONFINITE'))
            if m=='inner' and not torch.equal(o,o1): mx=max(mx,(o.float()-o1.float()).abs().max().item())
print(f'EXACT inner vs stock: {nb}/{nt} bitwise; max abs diff {mx}; guard violations {gbad}; nonfinite {first[:5]}',flush=True)
def pf(L):
    w=L.weight.view(-1); n=min(w.numel(), (40*2**20)//w.element_size(), w.numel()//2)
    return w[:n].max()
for name,(N,K) in SH.items():
    L=Ls[name]
    tfw=tm(graph(lambda: (flush.zero_(), pf(L))))
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        for mode in MODES:
            if mode=='inner2x32' and not (48<M<=64): continue
            setmode(mode)
            f=lambda: (flush.zero_(), apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None))
            res[(name,M,mode,'cold')]=(tm(graph(f))-tf)/R*1000
            fw=lambda: (flush.zero_(), pf(L), apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None))
            res[(name,M,mode,'warm')]=(tm(graph(fw))-tfw)/R*1000
    print(name,flush=True)
mem=wbytes/bw*1e3
print(f'mem roof {mem:.2f} ms')
print('M | cold: stock new inner inner2x32 | warm: stock new inner inner2x32 | roof | inner cold/roof')
for M in MS:
    def tt(m,k): 
        return sum(CNT[n]*res[(n,M,m,k)] for n in SH)/1000 if (list(SH)[0],M,m,k) in res else float('nan')
    roof=max(mem,flops1*M/peak*1e3)
    print(f"{M} | "+" ".join(f"{tt(m,'cold'):.2f}" for m in MODES)+" | "+" ".join(f"{tt(m,'warm'):.2f}" for m in MODES)+f" | {roof:.2f} | {tt('inner','cold')/roof:.2f}")
print('per-shape cold us stock/new/inner (gate_up, down, qkvz, gdn_out):')
for M in (64,72,96,112,128,160):
    print(M,{n:'/'.join(f"{res[(n,M,m,'cold')]:.0f}" for m in ('stock','new','inner')) for n in ('gate_up','down','qkvz','gdn_out')})
print('M=64 2x32 inner per shape cold:',{n:round(res[(n,64,'inner2x32','cold')],1) for n in SH})
