# exactness/guard detail for t6-slab: stock vs new (per mode), ulp-level diffs, and sentinel check that rows >= M of C are never written
import os, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new, marlin_repacked_nk
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
dev=torch.device('cuda:0'); torch.cuda.set_device(0); ws=marlin_make_workspace_new(dev)
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
def call(L,N,K,x,c=None):
    return ops.marlin_gemm(x,c,L.weight,None,L.weight_scale_inv,None,None,ws,scalar_types.float8_e4m3fn,x.size(0),N,K,False,True)
def mode(m):
    os.environ['AI_SLAB']='0' if m=='stock' else '1'; os.environ['AI_SLAB_S49']={'2x32':'2','4x16':'4'}.get(m,'1')
torch.manual_seed(1)
for name,(N,K) in {'gdn_out':(5120,3072),'down':(5120,8704),'gate_up':(17408,5120)}.items():
    L=mk(N,K); stats={}
    for M in range(1,257):
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        mode('stock'); ref=call(L,N,K,x); ref2=call(L,N,K,x)
        det=torch.equal(ref,ref2)
        for m in (['new'] + (['2x32','4x16'] if 48<M<=64 else [])):
            mode(m)
            big=torch.full((M+80,N),float('nan'),device=dev,dtype=torch.bfloat16); c=big[:M]
            o=call(L,N,K,x,c)
            guard=bool(torch.isnan(big[M:]).all())
            d=(o.float()-ref.float()).abs()
            ulp=(2.0**torch.floor(torch.log2(ref.float().abs().clamp(min=1e-30)))*2**-7)
            s=stats.setdefault(m,dict(n=0,eq=0,maxulp=0.0,maxabs=0.0,guard_bad=0,nondet=0,diffM=[]))
            s['n']+=1; s['eq']+=int(torch.equal(o,ref)); s['maxabs']=max(s['maxabs'],d.max().item())
            s['maxulp']=max(s['maxulp'],(d/ulp).max().item()); s['guard_bad']+=int(not guard); s['nondet']+=int(not det)
            if not torch.equal(o,ref) and len(s['diffM'])<40: s['diffM'].append(M)
    print(name,{k:{a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()} for k,v in stats.items()},flush=True)
