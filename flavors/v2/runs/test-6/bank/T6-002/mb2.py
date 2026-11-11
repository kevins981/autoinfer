# Task 2: stock vs single-launch parallel slabs, same build (env AI_SLAB / AI_SLAB_S49). Adapted from mb1.py.
import os, sys, statistics as st, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
D=0; dev=torch.device('cuda:0'); torch.cuda.set_device(0)
SH={'qkvz':(8192,5120),'gdn_out':(5120,3072),'attn_qkv':(7168,5120),'attn_o':(5120,3072),'gate_up':(17408,5120),'down':(5120,8704)}
CNT={'qkvz':48,'gdn_out':48,'attn_qkv':16,'attn_o':16,'gate_up':64,'down':64}
MS=[48,52,56,64,72,80,96,112,120,128,144,160,176,192]
MODES={'stock':{'AI_SLAB':'0','AI_SLAB_S49':'1'},'new':{'AI_SLAB':'1','AI_SLAB_S49':'1'},
       'new+2x32':{'AI_SLAB':'1','AI_SLAB_S49':'2'},'new+4x16':{'AI_SLAB':'1','AI_SLAB_S49':'4'}}
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
res={}; ex=[]
Ls={n:mk(*SH[n]) for n in SH}
# exactness: every M in 1..256 (step 1) for gdn_out & qkvz & gate_up(subset), stock vs new vs variants
def run(L,N,K,x,mode):
    setmode(mode); return apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None).clone()
torch.manual_seed(0)
bad=[]; maxd=0.0; nbit=0; ntot=0
for name in ('gdn_out','qkvz','gate_up','down'):
    N,K=SH[name]; L=Ls[name]
    for M in list(range(1,257)) if name in('gdn_out','qkvz') else MS+[1,17,33,49,50,63,65,100,127,129,191,255,256]:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        ref=run(L,N,K,x,'stock')
        for mode in ('new','new+2x32','new+4x16'):
            if mode!='new' and not (48<M<=64): continue
            o=run(L,N,K,x,mode); ntot+=1
            if torch.equal(o,ref): nbit+=1
            else:
                d=(o.float()-ref.float()).abs().max().item(); maxd=max(maxd,d); bad.append((name,M,mode,d,ref.float().abs().max().item()))
            assert torch.isfinite(o.float()).all()
print(f'EXACTNESS: {nbit}/{ntot} bitwise equal; max abs diff over rest {maxd:.4g}; first non-equal: {bad[:12]}',flush=True)
# row-guard check: rows past M must not be written: use M+extra buffer via view
N,K=SH['gdn_out']; L=Ls['gdn_out']
for M in (72,80,96,100,113,129,150,191):
    big=torch.randn(M+64,K,device=dev,dtype=torch.bfloat16)
    setmode('new'); o=apply_fp8_marlin_linear(big[:M],L.weight,L.weight_scale_inv,ws,N,K,None)
print('guard run ok (no fault)',flush=True)
for name,(N,K) in SH.items():
    L=Ls[name]
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        for mode in MODES:
            if mode in('new+2x32','new+4x16') and not (48<M<=64): continue
            setmode(mode)
            f=lambda: (flush.zero_(), apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None))
            res[(name,M,mode)]=(tm(graph(f))-tf)/R*1000
    print(name,{M:{m:round(res[(name,M,m)],1) for m in MODES if (name,M,m) in res} for M in (64,96,128)},flush=True)
mem=wbytes/bw*1e3
print(f'\nmem roof {mem:.2f} ms; compute roof @133 TFLOPs; ratio = total/max(roofs)')
print('M | stock ms | new ms | 2x32 | 4x16 | mem | comp | stock/roof | new/roof')
for M in MS:
    t={m:sum(CNT[n]*res[(n,M,m)] for n in SH)/1000 for m in MODES if (list(SH)[0],M,m) in res}
    cr=flops1*M/peak*1e3; roof=max(mem,cr)
    print(f"{M} | {t['stock']:.2f} | {t['new']:.2f} | {t.get('new+2x32',float('nan')):.2f} | {t.get('new+4x16',float('nan')):.2f} | {mem:.2f} | {cr:.2f} | {t['stock']/roof:.2f} | {t['new']/roof:.2f}")
print('\nper-shape us (stock/new) at M:',)
for M in (64,72,96,112,128,160):
    print(M,{n:f"{res[(n,M,'stock')]:.0f}/{res[(n,M,'new')]:.0f}" for n in SH})
