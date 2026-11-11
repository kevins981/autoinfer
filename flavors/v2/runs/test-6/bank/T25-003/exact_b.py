# S1(b): decode path bitwise vs champion layout; mixed path vs champion (<=2 ulp class). Offline, GPU0 only.
import torch, torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
import vllm._C_stable_libtorch as C
print('KB61 _C =',C.__file__)
dev=torch.device('cuda:0'); I=17408; H=5120; HALF=I//2; N1=9344; N0=I-N1; torch.manual_seed(0)
def mk(w,s):
    N,K=w.shape; m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(w.contiguous(),requires_grad=False); m.weight_scale_inv=torch.nn.Parameter(s.contiguous(),requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
def rnd(N,K): return torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn)
wg,wu,wd=rnd(I,H),rnd(I,H),rnd(H,I)
sg,su=(torch.rand(I//128,H//128,device=dev)*0.0004+0.0001 for _ in range(2)); sd=torch.rand(H//128,I//128,device=dev)*0.0004+0.0001
def lin(W,S,x,N,K): return apply_fp8_marlin_linear(input=x,weight=W,weight_scale=S,workspace=None,size_n=N,size_k=K,bias=None)
def champ(x,lo,hi):  # champion rank: natural [gate|up], K=hi-lo
    gu=mk(torch.cat([wg[lo:hi],wu[lo:hi]]),torch.cat([sg[lo//128:hi//128],su[lo//128:hi//128]])); dn=mk(wd[:,lo:hi],sd[:,lo//128:hi//128])
    g=lin(gu.weight,gu.weight_scale_inv,x,2*(hi-lo),H); h=torch.empty(x.shape[0],hi-lo,dtype=x.dtype,device=dev); torch.ops._C.silu_and_mul(h,g)
    return lin(dn.weight,dn.weight_scale_inv,h,H,hi-lo)
def bl(lo,hi,a):  # b-layout tensors for a rank holding global [lo,hi)
    n=hi-lo; idx=torch.cat([torch.arange(0,a),n+torch.arange(0,a),torch.arange(a,n),n+torch.arange(a,n)]).to(dev)
    w=torch.cat([wg[lo:hi],wu[lo:hi]]); s=torch.cat([sg[lo//128:hi//128],su[lo//128:hi//128]])
    w=w.view(torch.uint8).index_select(0,idx).view(torch.float8_e4m3fn); s=s.index_select(0,idx.view(-1,128)[:,0]//128)
    return mk(w,s), mk(wd[:,lo:hi],sd[:,lo//128:hi//128])
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import USE_FP32_REDUCE_DEFAULT, get_marlin_workspace, should_use_atomic_add_reduce
from vllm.scalar_type import scalar_types
def split_b(x,gu,dn,n_r,a,rank0,mixed):
    half=HALF; W,S=gu.weight,gu.weight_scale_inv; Wd,Sd=dn.weight,dn.weight_scale_inv; M=x.shape[0]
    if rank0 and not mixed:  # natural layout, exactly the champion
        g=lin(W,S,x,2*half,H); h=torch.empty(M,half,dtype=g.dtype,device=dev); torch.ops._C.silu_and_mul(h,g); return lin(Wd,Sd,h,H,half)
    if rank0:
        buf=torch.empty((2,M,a),dtype=x.dtype,device=dev); ws=get_marlin_workspace(dev)
        ua=should_use_atomic_add_reduce(m=M,n=a,k=H,device=dev,dtype=x.dtype)
        for part,c0 in enumerate((0,half)):
            ops.marlin_gemm(x,buf[part],W[:,c0*4:(c0+a)*4],None,S[:,c0:c0+a],None,None,ws,scalar_types.float8_e4m3fn,M,a,H,ua,USE_FP32_REDUCE_DEFAULT,False)
        h=torch.empty((M,a),dtype=x.dtype,device=dev); torch.ops._C.silu_and_mul_split(h,buf,0,M*a)
        return lin(Wd[:a//16],Sd[:a//128],h,H,a)
    if mixed:
        g=lin(W,S,x,2*n_r,H); h=torch.empty(M,n_r,dtype=g.dtype,device=dev); torch.ops._C.silu_and_mul_split(h,g,a,0); return lin(Wd,Sd,h,H,n_r)
    off=n_r-half; W,S=W[:,2*off*4:],S[:,2*off:]; Wd,Sd=Wd[off//16:],Sd[off//128:]
    g=lin(W,S,x,2*half,H); h=torch.empty(M,half,dtype=g.dtype,device=dev); torch.ops._C.silu_and_mul(h,g); return lin(Wd,Sd,h,H,half)
gu0,dn0=bl(0,HALF,0); gu1,dn1=bl(N0,I,N1-HALF)   # rank0 natural layout (perm a=0 = identity)
gu1,dn1=bl(N0,I,N1-HALF)
for M in (8,16,24,48,96,128,256,640):
    x=torch.randn(M,H,device=dev,dtype=torch.bfloat16)
    c0,c1=champ(x,0,HALF),champ(x,HALF,I); ref=(c0.float()+c1.float()).to(torch.bfloat16)
    d0=split_b(x,gu0,dn0,HALF,N0,True,False); d1=split_b(x,gu1,dn1,N1,N1-HALF,False,False)
    m0=split_b(x,gu0,dn0,HALF,N0,True,True); m1=split_b(x,gu1,dn1,N1,N1-HALF,False,True)
    mix=(m0.float()+m1.float()).to(torch.bfloat16)
    dif=(mix.float()-ref.float()).abs(); big=ref.float().abs()>1e-2*ref.float().abs().max()
    ulp=(dif/(torch.pow(2.0,torch.floor(torch.log2(ref.float().abs().clamp(min=1e-30)))-7)))[big].max().item()
    print(f'M={M}: decode rank0 bitwise={torch.equal(d0,c0)} rank1 bitwise={torch.equal(d1,c1)} | mixed-sum vs champion max|d|={dif.max().item():.4g} (|ref|max {ref.float().abs().max().item():.3g}) max ulp(|ref|>1%max)={ulp:.2f} bitwise frac={(mix==ref).float().mean().item():.3f}')
