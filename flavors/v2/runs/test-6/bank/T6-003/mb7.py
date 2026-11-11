import os, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
dev=torch.device('cuda:0'); torch.cuda.set_device(0); ws=marlin_make_workspace_new(dev)
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
N,K=17408,5120; L=mk(N,K)
# order of profiled launches (ncu --kernel-name Marlin): label list
seq=[('new',96),('inner',96),('new',128),('inner',128)]
for mode,M in seq:
    os.environ['AI_SLAB']='0' if mode=='stock' else '1'; os.environ['AI_SLAB_INNER']='1' if mode=='inner' else '0'
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None)
torch.cuda.synchronize()
