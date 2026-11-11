# adapted from agents/leader-5.w1/mb/mb_marlin.py (T5-002); items 1-2 of Task 1
import os, sys, statistics as st, torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
D=int(sys.argv[1]); dev=torch.device(f'cuda:{D}'); torch.cuda.set_device(D)
SH={'qkvz(8192x5120)':(8192,5120),'gdn_out(5120x3072)':(5120,3072),'attn_qkv(7168x5120)':(7168,5120),
    'attn_o(5120x3072)':(5120,3072),'gate_up(17408x5120)':(17408,5120),'down(5120x8704)':(5120,8704)}
CNT={'qkvz(8192x5120)':48,'gdn_out(5120x3072)':48,'attn_qkv(7168x5120)':16,'attn_o(5120x3072)':16,'gate_up(17408x5120)':64,'down(5120x8704)':64}
MS=[3,16,32,48,52,56,64,72,80,96,112,128,160,192,224,256]
def mk(N,K):
    m=torch.nn.Module(); m.output_size_per_partition=N; m.input_size_per_partition=K; m.weight_block_size=[128,128]; m.orig_dtype=torch.bfloat16
    m.weight=torch.nn.Parameter(torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn),requires_grad=False)
    m.weight_scale_inv=torch.nn.Parameter(torch.rand(N//128,K//128,device=dev)*0.02+0.005,requires_grad=False)
    prepare_fp8_layer_for_marlin(m,size_k_first=False); return m
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32)
ws=marlin_make_workspace_new(dev)
R=8
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
print(f'=== GPU {D} {torch.cuda.get_device_name(D)}; flush-only per rep {tf/R*1000:.1f} us',flush=True)
# roofs
big=torch.empty(2*2**30//4,device=dev,dtype=torch.float32).normal_()
g=graph(lambda: big.sum(),4); t=tm(g)/4/1000
bw_read=big.numel()*4/t/1e9; print(f'HBM read (sum over 2GiB fp32): {bw_read:.0f} GB/s',flush=True)
big2=torch.empty_like(big); g=graph(lambda: big2.copy_(big),2); t=tm(g)/2/1000
print(f'HBM copy: {2*big.numel()*4/t/1e9:.0f} GB/s (r+w)',flush=True)
del big,big2
a=torch.randn(4096,4096,device=dev,dtype=torch.bfloat16); b=torch.randn(4096,4096,device=dev,dtype=torch.bfloat16)
g=graph(lambda: a@b,8); t=tm(g)/8/1000; peak=2*4096**3/t/1e12; print(f'bf16 matmul 4096^3: {peak:.1f} TFLOPs',flush=True)
for M in (128,256):
    a=torch.randn(M,5120,device=dev,dtype=torch.bfloat16); b=torch.randn(5120,17408,device=dev,dtype=torch.bfloat16)
    g=graph(lambda: a@b,8); t=tm(g)/8/1000; print(f'bf16 hot matmul M={M} K=5120 N=17408: {t*1e6:.0f} us = {2*M*5120*17408/t/1e12:.1f} TFLOPs',flush=True)
del a,b
bw=bw_read*1e9
res={}; w16={}
for name,(N,K) in SH.items():
    L=mk(N,K)
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        f=lambda: (flush.zero_(), apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None))
        res[(name,M)]=(tm(graph(f))-tf)/R*1000
    print(name,[round(res[(name,M)],1) for M in MS],flush=True)
    del L
# bf16-weight GEMM reference for gate_up (cold)
N,K=17408,5120
W=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
for M in MS:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    f=lambda: (flush.zero_(), x@W)
    w16[M]=(tm(graph(f))-tf)/R*1000
print('bf16-weight gate_up cold us',[round(w16[M]) for M in MS],flush=True)
wbytes=sum(CNT[n]*SH[n][0]*SH[n][1] for n in SH)  # fp8 bytes per step (per rank)
flops1=sum(CNT[n]*2*SH[n][0]*SH[n][1] for n in SH)
print(f'\nFP8 weight bytes/step/rank {wbytes/1e9:.2f} GB; mem roof {wbytes/bw*1e3:.2f} ms @ {bw_read:.0f} GB/s')
print('M | marlin_total_ms | mem_roof_ms | compute_roof_ms | marlin/max(roofs) | per-shape us: '+' '.join(SH))
for M in MS:
    tot=sum(CNT[n]*res[(n,M)] for n in SH)/1000
    mem=wbytes/bw*1e3; cr=flops1*M/(peak*1e12)*1e3
    print(f'{M} | {tot:.2f} | {mem:.2f} | {cr:.2f} | {tot/max(mem,cr):.2f} | '+' '.join(f'{res[(n,M)]:.0f}' for n in SH))
