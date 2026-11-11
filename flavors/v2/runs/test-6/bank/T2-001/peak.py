import torch, time, sys
def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it  # ms
for dev in (0,1):
    torch.cuda.set_device(dev); d=f'cuda:{dev}'
    print(f'=== GPU {dev}', torch.cuda.get_device_properties(dev).multi_processor_count, 'SMs', flush=True)
    n=8192
    a=torch.randn(n,n,device=d,dtype=torch.bfloat16); b=torch.randn(n,n,device=d,dtype=torch.bfloat16)
    t=bench(lambda: a@b); print(f'bf16 8192^3: {2*n**3/t/1e9:.1f} TFLOPS')
    a16=a.half(); b16=b.half()
    t=bench(lambda: a16@b16); print(f'fp16 8192^3: {2*n**3/t/1e9:.1f} TFLOPS')
    ai=torch.randint(-100,100,(n,n),device=d,dtype=torch.int8); bi=torch.randint(-100,100,(n,n),device=d,dtype=torch.int8).t().contiguous().t()
    try:
        t=bench(lambda: torch._int_mm(ai,bi)); print(f'int8 8192^3: {2*n**3/t/1e9:.1f} TOPS')
    except Exception as ex: print('int8 fail', ex)
    try:
        af=a.to(torch.float8_e4m3fn); bf=b.to(torch.float8_e4m3fn).t().contiguous().t(); one=torch.ones((),device=d)
        t=bench(lambda: torch._scaled_mm(af,bf,one,one,out_dtype=torch.bfloat16)); print(f'fp8 8192^3: {2*n**3/t/1e9:.1f} TFLOPS')
    except Exception as ex: print('fp8 fail', ex)
    # model shapes per rank (TP=2): (name, N, K)
    shapes=[('qkvz',8192,5120),('gate_up',17408,5120),('down',5120,8704),('out',5120,3072),('qkv_attn',7168,5120)]
    for M in (48,128,256,320,512,1024):
        tot=0; fl=0; row=[]
        for nm,N,K in shapes:
            x=torch.randn(M,K,device=d,dtype=torch.bfloat16); w=torch.randn(N,K,device=d,dtype=torch.bfloat16)
            t=bench(lambda: x@w.t(),50); row.append(f'{nm} {t*1e3:.0f}us {2*M*N*K/t/1e9:.0f}TF')
        print(f'M={M}: '+' | '.join(row), flush=True)
    del a,b,a16,b16,ai,bi
    torch.cuda.empty_cache()
# green contexts: does SM partitioning work? gemm on 32/64/128 SMs
torch.cuda.set_device(0)
from torch.cuda.green_contexts import GreenContext
x=torch.randn(4096,5120,device='cuda:0',dtype=torch.bfloat16); w=torch.randn(17408,5120,device='cuda:0',dtype=torch.bfloat16)
for nsm in (32,64,96):
    try:
        g=GreenContext(num_sms=nsm); st=g.Stream()
        with torch.cuda.stream(st):
            t=bench(lambda: x@w.t())
        print(f'green ctx {nsm} SMs: gate_up M=4096 {2*4096*17408*5120/t/1e9:.1f} TF')
    except Exception as ex: print('green fail', nsm, repr(ex))
# concurrency: memory-bound stream (big copy) + compute stream on disjoint partitions
try:
    gA=GreenContext(num_sms=80); gB=GreenContext(num_sms=48); sA=gA.Stream(); sB=gB.Stream()
    src=torch.empty(1<<30,dtype=torch.uint8,device='cuda:0'); dst=torch.empty_like(src)
    def memop(): dst.copy_(src)   # copy kernel, 2 GB traffic
    with torch.cuda.stream(sA): tm=bench(memop,10)
    with torch.cuda.stream(sB): tc=bench(lambda: x@w.t(),10)
    print(f'alone: copy(80SM) {2/tm*1e3:.0f} GB/s; gemm(48SM) {2*4096*17408*5120/tc/1e9:.1f} TF')
    torch.cuda.synchronize(); s=time.time()
    for _ in range(10):
        with torch.cuda.stream(sA): memop(); memop()
        with torch.cuda.stream(sB): x@w.t()
    torch.cuda.synchronize(); el=(time.time()-s)
    print(f'together 10x(2copy+1gemm): {el*1e3:.1f} ms vs serial {10*(2*tm+tc):.1f} ms')
except Exception as ex: print('concurrency fail', repr(ex))
