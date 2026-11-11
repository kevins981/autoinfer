# even vs uneven (AI_MLP_SPLIT) MLP output: same Marlin FP8 path, same weights, only the K-grouping of down differs
import torch, torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
dev=torch.device('cuda:0'); I=17408; H=5120; torch.manual_seed(0)
def mk(w,s):
    N,K=w.shape; m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(w.contiguous(),requires_grad=False); m.weight_scale_inv=torch.nn.Parameter(s.contiguous(),requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
ws=marlin_make_workspace_new(dev,4)
def lin(L,x): return ops.marlin_gemm(x,None,L.weight,None,L.weight_scale_inv,None,None,ws,scalar_types.float8_e4m3fn,x.size(0),L.output_size_per_partition,L.input_size_per_partition,False,True)
def rnd(N,K): return torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn)
wg,wu,wd=rnd(I,H),rnd(I,H),rnd(H,I)
sg,su=(torch.rand(I//128,H//128,device=dev)*0.0004+0.0001 for _ in range(2)); sd=torch.rand(H//128,I//128,device=dev)*0.0004+0.0001
def rank_out(x,a,b):
    gu=mk(torch.cat([wg[a:b],wu[a:b]]),torch.cat([sg[a//128:b//128],su[a//128:b//128]]))
    dn=mk(wd[:,a:b],sd[:,a//128:b//128])
    g=lin(gu,x); h=F.silu(g[:,:b-a])*g[:,b-a:]
    return lin(dn,h.contiguous())
def red(parts): return sum(p.float() for p in parts).to(torch.bfloat16)  # bf16 allreduce-style sum (fp32 accum then round)
n0=8064
for M in (48,256,640):
    x=torch.randn(M,H,device=dev,dtype=torch.bfloat16)
    ev=red([rank_out(x,0,8704),rank_out(x,8704,I)]); un=red([rank_out(x,0,n0),rank_out(x,n0,I)])
    d=(ev.float()-un.float()).abs(); r=d.max()/ev.float().abs().max()
    big=ev.float().abs()>1e-2*ev.float().abs().max(); ulp=(d/(torch.pow(2.0,torch.floor(torch.log2(ev.float().abs().clamp(min=1e-30)))-7)))[big].max()
    print(f'M={M} |out|max={ev.float().abs().max():.3f} max-abs diff={d.max():.4g} rel-to-max={r:.3e} max ulp(bf16 of ref)={ulp:.2f} mean-abs diff={d.mean():.3g} bitwise-equal-frac={(ev==un).float().mean():.3f}')
