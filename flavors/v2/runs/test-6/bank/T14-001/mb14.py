# T14-001: Marlin policy exactness + cold per-forward timing, stock vs policy vs extensions (manual knobs). GPU0 only.
import os, json, statistics as st, torch, math
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
import vllm._C_stable_libtorch as C
print('KB61 build check: _C =',C.__file__,flush=True)
dev=torch.device('cuda:0'); torch.cuda.set_device(0)
SH={'qkvz':(8192,5120),'gdn_out':(5120,3072),'attn_qkv':(7168,5120),'attn_o':(5120,3072),'gate_up':(17408,5120),'down':(5120,8704)}
CNT={'qkvz':48,'gdn_out':48,'attn_qkv':16,'attn_o':16,'gate_up':64,'down':64}
MS=[48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,1024]
EXM=list(range(1,257))+list(range(288,1025,32))
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
ws=marlin_make_workspace_new(dev,4)
def call(L,N,K,x,c=None): return ops.marlin_gemm(x,c,L.weight,None,L.weight_scale_inv,None,None,ws,scalar_types.float8_e4m3fn,x.size(0),N,K,False,True)
MODES={'stock':dict(AI_POLICY='0'),'policy':{},
 'slabAll':dict(AI_POLICY='manual',AI_SLAB='1'),
 'slabInner':dict(AI_POLICY='manual',AI_SLAB='1',AI_SLAB_INNER='1')}
def mode(m):
    for k in ('AI_POLICY','AI_SLAB','AI_SLAB_S49','AI_SLAB_INNER','AI_FORCE'): os.environ.pop(k,None)
    os.environ.update(MODES[m])
Ls={n:mk(*SH[n]) for n in SH}
def ulps(o,ref):
    r=ref.float(); d=(o.float()-r).abs()
    e=torch.floor(torch.log2(r.abs().clamp(min=1e-30))).clamp(min=-126)
    u=torch.pow(2.0,e-7)
    return d.max().item(), (d/u).max().item()
torch.manual_seed(3)
for var in ('policy','slabAll','slabInner'):
    for n,(N,K) in SH.items():
        L=Ls[n]; eq=0; tot=0; mxa=0.0; mxu=0.0; gbad=0; nonfin=0; neq=[]; bw48=True
        for M in EXM:
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
            mode('stock'); ref=call(L,N,K,x)
            mode(var); big=torch.full((M+80,N),float('nan'),device=dev,dtype=torch.bfloat16); o=call(L,N,K,x,big[:M])
            tot+=1; gbad+=int(not torch.isnan(big[M:]).all()); nonfin+=int(not torch.isfinite(o.float()).all())
            if torch.equal(o,ref): eq+=1
            else:
                a,u=ulps(o,ref); mxa=max(mxa,a); mxu=max(mxu,u); neq.append(M)
                if M<=48: bw48=False
        print(f'EXACT {var} {n}: {eq}/{tot} bitwise; max abs {mxa}; max ulp-of-ref {mxu:.2f}; guard viol {gbad}; nonfinite {nonfin}; bitwise M<=48 {bw48}; non-bitwise M count {len(neq)} first {neq[:6]}',flush=True)
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32); R=8
def graph(fn,r=R):
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); s.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=s):
            for _ in range(r): fn()
    s.synchronize(); return g
def tm(g,n=15):
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b))
    return st.median(ts)
tf=tm(graph(lambda: flush.zero_()))
res={}
for n,(N,K) in SH.items():
    L=Ls[n]
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        for m in MODES:
            mode(m)
            res[(n,M,m)]=(tm(graph(lambda: (flush.zero_(), call(L,N,K,x))))-tf)/R*1000
json.dump({'|'.join(map(str,k)):v for k,v in res.items()},open('mb14_res.json','w'))
mem=sum(CNT[n]*SH[n][0]*SH[n][1] for n in SH)/955e9*1e3; fl=sum(CNT[n]*2*SH[n][0]*SH[n][1] for n in SH)
print(f'\nmem roof {mem:.2f} ms; compute roof at 0.183 ms/row')
print('M | stock ms | policy | slabAll | slabInner | roof(0.183*M or HBM) | stock/row | policy/row | best/row  (cold, per-forward sum over layers, GPU0)')
for M in MS:
    t={m:sum(CNT[n]*res[(n,M,m)] for n in SH)/1000 for m in MODES}
    roof=max(mem,0.183*M)
    b=min(t.values())
    print(f"{M} | {t['stock']:.2f} | {t['policy']:.2f} | {t['slabAll']:.2f} | {t['slabInner']:.2f} | {roof:.2f} | {t['stock']/M:.3f} | {t['policy']/M:.3f} | {b/M:.3f}")
print('per-shape us (stock/policy/slabAll/slabInner):')
for n in SH:
    print(n,' '.join(f"{M}:"+'/'.join(f"{res[(n,M,m)]:.0f}" for m in MODES) for M in MS))
