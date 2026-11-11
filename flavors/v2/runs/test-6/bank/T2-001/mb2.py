import os, torch, threading, time, sys, statistics as st, types
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from torch.cuda.green_contexts import GreenContext
SMOKE = len(sys.argv)>1 and sys.argv[1]=='smoke'
NL = 4 if SMOKE else 16
dev = torch.device('cuda:0'); torch.cuda.set_device(0)
SH = [(8192,5120),(17408,5120),(5120,8704),(5120,3072)]
def mk(N,K):
    l = types.SimpleNamespace(output_size_per_partition=N,input_size_per_partition=K,weight_block_size=[128,128],orig_dtype=torch.bfloat16)
    l.weight = torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).view(torch.float8_e4m3fn)
    l.weight = torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn)
    l.weight_scale_inv = (torch.rand(N//128,K//128,device=dev)*0.02+0.005)
    # replace_parameter wants nn.Module
    m = torch.nn.Module(); 
    for k,v in vars(l).items(): setattr(m,k,v)
    m.weight = torch.nn.Parameter(l.weight, requires_grad=False); m.weight_scale_inv=torch.nn.Parameter(l.weight_scale_inv,requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    return m
def build_chain(M):
    layers=[[mk(N,K) for N,K in SH] for _ in range(NL)]
    ws = marlin_make_workspace_new(dev)
    xs = [torch.randn(M,K,device=dev,dtype=torch.bfloat16) for N,K in SH]
    outs=[None]*4
    def run():
        for L in layers:
            for i,(N,K) in enumerate(SH):
                outs[i]=apply_fp8_marlin_linear(xs[i],L[i].weight,L[i].weight_scale_inv,ws,N,K,None)
    flops = NL*sum(2*M*N*K for N,K in SH)
    byts = NL*sum(N*K for N,K in SH)  # fp8 bytes
    return run, flops, byts
def capture(run, stream, sms=None):
    if sms is None: os.environ.pop('MARLIN_SMS',None)
    else: os.environ['MARLIN_SMS']=str(sms)
    with torch.cuda.stream(stream):
        for _ in range(2): run()
        stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream): run()
    stream.synchronize(); return g
def timeit(g, stream, n=20):
    with torch.cuda.stream(stream):
        for _ in range(3): g.replay()
        stream.synchronize(); ts=[]
        for _ in range(n):
            a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(stream); g.replay(); b.record(stream); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)
dec_run, dflops, dbytes = build_chain(48)
pre_run, pflops, pbytes = build_chain(256)
print(f'decode chain weights {dbytes/1e9:.2f} GB; prefill chain {pflops/1e12:.3f} TFLOP', flush=True)
default = torch.cuda.Stream()
# eager correctness-free sanity: time eager once
print('=== 1. decode chain alone (graph)')
res={}
g = capture(dec_run, default); t=timeit(g, default); print(f'full GPU (default stream): {t:.2f} ms  {dbytes/t/1e6:.0f} GB/s'); 
sms_list = [128,96,88,80,72,64] if not SMOKE else [128,64]
gcs={}
for n in sms_list:
    gc=GreenContext(num_sms=n); s=gc.Stream(); gcs[n]=(gc,s)
    g=capture(dec_run,s); t=timeit(g,s); print(f'green {n} SMs, default sms=128: {t:.2f} ms  {dbytes/t/1e6:.0f} GB/s', flush=True)
    g=capture(dec_run,s,n); t=timeit(g,s); print(f'green {n} SMs, sms={n}: {t:.2f} ms  {dbytes/t/1e6:.0f} GB/s', flush=True)
print('=== 2. prefill chain alone')
g=capture(pre_run,default); t=timeit(g,default); print(f'full: {t:.2f} ms {pflops/t/1e9:.0f} TFLOPS')
for n in ([32,40,48,64] if not SMOKE else [32]):
    gc=GreenContext(num_sms=n); s=gc.Stream(); gcs[('p',n)]=(gc,s)
    g=capture(pre_run,s); t=timeit(g,s); print(f'green {n} SMs default sms: {t:.2f} ms {pflops/t/1e9:.0f} TFLOPS', flush=True)
    g=capture(pre_run,s,n); t=timeit(g,s); print(f'green {n} SMs sms={n}: {t:.2f} ms {pflops/t/1e9:.0f} TFLOPS', flush=True)

def loop(g, stream, T, out, ref):
    ev=[]; t0=time.time()
    with torch.cuda.stream(stream):
        while time.time()-t0<T:
            a=torch.cuda.Event(True); b=torch.cuda.Event(True)
            a.record(stream); g.replay(); b.record(stream); ev.append((a,b))
            if len(ev)>=4: ev[-4][1].synchronize()
        stream.synchronize()
    out.extend((ref.elapsed_time(a),ref.elapsed_time(b)) for a,b in ev)
def corun(gd, sd, gp, sp, T):
    torch.cuda.synchronize(); ref=torch.cuda.Event(True); ref.record(); ref.synchronize()
    d=[];p=[]
    th=[threading.Thread(target=loop,args=(gd,sd,T,d,ref)),threading.Thread(target=loop,args=(gp,sp,T,p,ref))]
    [t.start() for t in th]; [t.join() for t in th]
    w0=max(d[2][0],p[2][0]); w1=min(d[-2][1],p[-2][1])
    dd=[b-a for a,b in d if a>=w0 and b<=w1]; pp=[b-a for a,b in p if a>=w0 and b<=w1]
    return st.median(dd), st.median(pp), len(dd), len(pp), w1-w0
T = 0.6 if SMOKE else 2.0
print('=== 3. concurrent: decode on A=128-f, prefill on B=f')
for f in ([32] if SMOKE else [32,40,48,64]):
    gA=GreenContext(num_sms=128-f); gB=GreenContext(num_sms=f); sA=gA.Stream(); sB=gB.Stream()
    for label,sd_,sp_,dst_,dsms,psms in [('green/green sms=default',sA,sB,sA,None,None),('green/green sms=partition',sA,sB,sA,128-f,f),
                                   ('decode plain-full-stream(sms=128-f)/prefill green(sms=f)',None,sB,None,128-f,f),
                                   ('decode plain-full-stream(sms=128-f)/prefill green(default)',None,sB,None,128-f,None)]:
        sdd = sd_ if sd_ is not None else torch.cuda.Stream()
        gd=capture(dec_run,sdd,dsms); gp=capture(pre_run,sp_,psms)
        ad=timeit(gd,sdd); ap=timeit(gp,sp_)
        d,p,nd,np_,w=corun(gd,sdd,gp,sp_,T)
        print(f'f={f} [{label}]: decode alone {ad:.2f} -> co-run {d:.2f} ms (n={nd}); prefill alone {ap:.2f} ms {pflops/ap/1e9:.0f}TF -> co-run {p:.2f} ms {pflops/p/1e9:.0f}TF (n={np_}); window {w:.0f} ms', flush=True)
print('=== 4. plain streams: decode high prio, prefill low prio')
sH=torch.cuda.Stream(priority=-1); sL=torch.cuda.Stream(priority=0)
gd=capture(dec_run,sH); gp=capture(pre_run,sL)
ad=timeit(gd,sH); ap=timeit(gp,sL)
d,p,nd,np_,w=corun(gd,sH,gp,sL,T)
print(f'prio: decode alone {ad:.2f} -> co-run {d:.2f} ms (n={nd}); prefill alone {ap:.2f} ms {pflops/ap/1e9:.0f}TF -> co-run {p:.2f} ms {pflops/p/1e9:.0f}TF (n={np_}); window {w:.0f} ms')
sH2=torch.cuda.Stream(priority=0); sL2=torch.cuda.Stream(priority=0)
gd=capture(dec_run,sH2); gp=capture(pre_run,sL2)
d,p,nd,np_,w=corun(gd,sH2,gp,sL2,T)
print(f'equal-prio plain streams: decode co-run {d:.2f} ms; prefill co-run {p:.2f} ms {pflops/p/1e9:.0f}TF')
