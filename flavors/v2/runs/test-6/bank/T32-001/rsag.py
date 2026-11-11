# Transport microbench: host-staged DMA all-reduce (bf16) vs reduce-scatter(bf16)+all-gather(fp8 e4m3 + fp32 group scales),
# token-pieced and pipelined on copy engines, 2 GPUs, one process, cross-device events. Timing: both devices blocked by a
# spin kernel while the host enqueues, then per-device elapsed from the spin's end to the last op; report max over devices.
import os, sys, time, statistics, torch
os.sched_setaffinity(0, set(list(range(0,12))+list(range(24,36))))  # NUMA node 0 (both GPUs' node)
H=5120; G=128
dev=[torch.device('cuda',0),torch.device('cuda',1)]
def streams(d):
    with torch.cuda.device(d): return dict(d2h=torch.cuda.Stream(d), h2d=torch.cuda.Stream(d), comp=torch.cuda.Stream(d))
S=[streams(d) for d in dev]
GATE=[torch.cuda.Stream(d) for d in dev]
def quant(x):  # per-token-group-128 e4m3 (stand-in for norm+quant; same bytes/work order of magnitude)
    xg=x.view(x.shape[0],-1,G).float()
    s=xg.abs().amax(-1,keepdim=True).clamp(min=1e-6)/448.0
    return (xg/s).to(torch.float8_e4m3fn).view(x.shape[0],H), s.view(x.shape[0],-1)
def setup(R,npieces):
    st={}
    for g in (0,1):
        with torch.cuda.device(dev[g]):
            st[g]=dict(P=torch.randn(R,H,device=dev[g],dtype=torch.bfloat16),
                       recv=torch.empty(R,H,device=dev[g],dtype=torch.bfloat16),
                       q=torch.empty(R,H,device=dev[g],dtype=torch.uint8), sc=torch.empty(R,H//G,device=dev[g],dtype=torch.float32),
                       hb=torch.empty(R,H,dtype=torch.bfloat16).pin_memory(),
                       hq=torch.empty(R,H,dtype=torch.uint8).pin_memory(), hs=torch.empty(R,H//G,dtype=torch.float32).pin_memory())
    return st
def ev(d,s):
    e=torch.cuda.Event(); e.record(s); return e
def run(scheme,R,npieces,st,spin_cycles=int(2e7)):
    starts=[];ends=[]
    for g in (0,1):
        with torch.cuda.device(dev[g]):
            gs=GATE[g]
            with torch.cuda.stream(gs): torch.cuda._sleep(spin_cycles)
            e=torch.cuda.Event(enable_timing=True); e.record(gs); starts.append(e)
            for k in S[g].values(): k.wait_event(e)
    # make every stream of a device wait for the spin end marker (spins are equal length per stream)
    last={0:[],1:[]}
    if scheme=='AR':
        bnd=[(i*R//npieces,(i+1)*R//npieces) for i in range(npieces)]
        for (a,b) in bnd:
            evs={}
            for g in (0,1):
                with torch.cuda.device(dev[g]), torch.cuda.stream(S[g]['d2h']):
                    st[g]['hb'][a:b].copy_(st[g]['P'][a:b],non_blocking=True); evs[g]=ev(dev[g],S[g]['d2h'])
            for g in (0,1):
                o=1-g
                with torch.cuda.device(dev[o]):
                    S[o]['h2d'].wait_event(evs[g])
                    with torch.cuda.stream(S[o]['h2d']): st[o]['recv'][a:b].copy_(st[g]['hb'][a:b],non_blocking=True)
                    e2=ev(dev[o],S[o]['h2d']); S[o]['comp'].wait_event(e2)
                    with torch.cuda.stream(S[o]['comp']): st[o]['P'][a:b].add_(st[o]['recv'][a:b])
    else:  # RSAG: GPU g owns tokens [g*R/2,(g+1)*R/2)
        half=R//2; own=[(0,half),(half,R)]
        n=npieces
        for i in range(n):
            evs={}
            for g in (0,1):  # send piece of peer-owned half
                o=1-g; a0,b0=own[o]; a=a0+i*(b0-a0)//n; b=a0+(i+1)*(b0-a0)//n
                with torch.cuda.device(dev[g]), torch.cuda.stream(S[g]['d2h']):
                    st[g]['hb'][a:b].copy_(st[g]['P'][a:b],non_blocking=True); evs[g]=(ev(dev[g],S[g]['d2h']),a,b)
            evq={}
            for g in (0,1):  # owner o receives, reduces, quantizes, posts fp8
                o=1-g; e,a,b=evs[g]
                with torch.cuda.device(dev[o]):
                    S[o]['h2d'].wait_event(e)
                    with torch.cuda.stream(S[o]['h2d']): st[o]['recv'][a:b].copy_(st[g]['hb'][a:b],non_blocking=True)
                    S[o]['comp'].wait_event(ev(dev[o],S[o]['h2d']))
                    with torch.cuda.stream(S[o]['comp']):
                        st[o]['P'][a:b].add_(st[o]['recv'][a:b]); qq,ss=quant(st[o]['P'][a:b])
                        st[o]['q'][a:b].copy_(qq.view(torch.uint8)); st[o]['sc'][a:b].copy_(ss)
                    S[o]['d2h'].wait_event(ev(dev[o],S[o]['comp']))
                    with torch.cuda.stream(S[o]['d2h']):
                        st[o]['hq'][a:b].copy_(st[o]['q'][a:b],non_blocking=True); st[o]['hs'][a:b].copy_(st[o]['sc'][a:b],non_blocking=True)
                    evq[o]=(ev(dev[o],S[o]['d2h']),a,b)
            for o in (0,1):  # the non-owner fetches fp8
                g=1-o; e,a,b=evq[o]
                with torch.cuda.device(dev[g]):
                    S[g]['h2d'].wait_event(e)
                    with torch.cuda.stream(S[g]['h2d']):
                        st[g]['q'][a:b].copy_(st[o]['hq'][a:b],non_blocking=True); st[g]['sc'][a:b].copy_(st[o]['hs'][a:b],non_blocking=True)
    for g in (0,1):
        with torch.cuda.device(dev[g]):
            for k in ('d2h','comp'): S[g]['h2d'].wait_event(ev(dev[g],S[g][k]))
            e=torch.cuda.Event(enable_timing=True); e.record(S[g]['h2d']); ends.append(e)
    for g in (0,1): torch.cuda.synchronize(dev[g])
    return max(starts[g].elapsed_time(ends[g]) for g in (0,1))*1e3  # us
torch.cuda.set_device(0)
# sanity: spin must outlast host enqueue
for R in [int(x) for x in sys.argv[1:]] or [128,256,512,734,1024]:
    st=setup(R,8)
    for scheme in ('AR','RSAG'):
        for np_ in (4,8,16):
            ts=[run(scheme,R,np_,st) for _ in range(12)][2:]
            print(f'R={R:5d} {scheme:4s} pieces={np_:2d}  median {statistics.median(ts):8.1f} us  min {min(ts):8.1f}  bytes/dir/GPU {"%.2f"%((R*H*2 if scheme=="AR" else R//2*H*2+R//2*(H+H//G*4))/1e6)} MB', flush=True)
# spin length check
t0=time.perf_counter(); torch.cuda._sleep(int(2e7)); torch.cuda.synchronize(); print('spin ms',(time.perf_counter()-t0)*1e3)
