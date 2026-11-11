# Closed-loop c=16 simulator: TP event model (T45-calibrated) vs a 2-stage pipeline with ONE decode
# microbatch and prefill chunks filling the other stage ("PP-bubble"). CPU only; all costs from records.
import numpy as np, itertools
rng=np.random.default_rng(1)
N=400; C=16
P=np.clip(rng.lognormal(np.log(180),0.85,N),20,2000).astype(int); P=(P*262/P.mean()).astype(int)
O=np.clip(rng.lognormal(np.log(150),0.95,N),5,1500).astype(int); O=np.maximum((O*227/O.mean()).astype(int),2)
ACC=[0.774,0.552,0.400]
# --- TP event model (T45 sim, KB111): calibrated 80.5 s vs official 79.7
def tp(fixed=16.9,rowcost=0.10,draft=0.83,k=3,ev_fixed=15.0,ev_tok=0.166):
    t=0;nxt=0;slots=[]
    while nxt<N or slots:
        while len(slots)<C and nxt<N: slots.append(dict(p=P[nxt],o=O[nxt])); nxt+=1
        dec=[s for s in slots if s['p']==0]; pre=[s for s in slots if s['p']>0]
        cost=fixed+rowcost*(k+1)*len(dec)+draft*k
        if pre:
            cost+=ev_fixed+ev_tok*sum(s['p'] for s in pre)
            for s in pre: s['p']=0
        g=1+sum(ACC[:k])
        for s in dec: s['o']-=g
        slots=[s for s in slots if s['o']>0]; t+=cost
    return t/1e3
# --- PP stage GEMM time vs rows (T38-003/005, best exact W8A8 kernels, 32 full-width layers)
GM=[(0,13.3),(64,13.4),(96,14.49),(128,15.24),(160,18.88),(192,19.27),(256,25.9),(512,52.0),(1024,104.0)]
def g(m):
    xs,ys=zip(*GM); return float(np.interp(m,xs,ys))
def pp(nodes_rows=4, tok=2.73, levels=3, lvl=0.85, nong=3.5, head=2.1, pre_fixed=3.0, Bp=128, tree_ovh=0.0, verbose=False):
    t=0;nxt=0;slots=[];inflight=None  # inflight: list of (slot, ntok) chunk that finished stage0, waits for stage1
    periods=0; busy0=busy1=0
    while nxt<N or slots:
        while len(slots)<C and nxt<N: slots.append(dict(p=P[nxt],o=O[nxt],q=0)); nxt+=1
        dec=[s for s in slots if s['p']==0 and s['q']==0]
        Md=nodes_rows*len(dec)
        A0=(g(Md)+nong+tree_ovh) if dec else 0.0
        A1=(g(Md)+nong+tree_ovh+head+levels*lvl) if dec else 0.0
        # pick new prefill chunk for stage0 (FIFO over waiting prompts not yet in flight)
        budget=Bp if dec else 2048; chunk=[]
        for s in slots:
            if s['p']>0 and budget>0:
                take=min(s['p'],budget); chunk.append((s,take)); s['p']-=take; budget-=take
                if s['p']==0: s['q']=1   # fully issued; becomes decodable after stage1 completes
        m_new=sum(x for _,x in chunk); m_old=sum(x for _,x in inflight) if inflight else 0
        B0=(g(m_new)+pre_fixed) if m_new else 0.0
        B1=(g(m_old)+pre_fixed) if m_old else 0.0
        ph1=max(A0,B1); ph2=max(A1,B0)
        t+=ph1+ph2; periods+=1; busy0+=A0+B0; busy1+=A1+B1
        if inflight:
            for s,_ in inflight:
                if s['p']==0: s['q']=0
        inflight=chunk if chunk else None
        for s in dec: s['o']-=tok
        slots=[s for s in slots if s['o']>0]
    if verbose: print('  periods %d  util GPU0 %.2f GPU1 %.2f'%(periods,busy0/t,busy1/t))
    return t/1e3
base=tp(); tot=O.sum()
print('TP event model: %.1f s %.0f tok/s'%(base,tot/base))
print('TP, prefill free (ev 0,0): %.1f s'%tp(ev_fixed=0,ev_tok=0))
cfgs={'chain k=3 (64 rows, 2.73)':dict(nodes_rows=4,tok=2.73,levels=3),
      'tree 6 nodes (96 rows, ~3.5?)':dict(nodes_rows=6,tok=3.5,levels=5),
      'tree 8 nodes (128 rows, ~3.85?)':dict(nodes_rows=8,tok=3.85,levels=6),
      'tree 11 nodes (176 rows, 4.21)':dict(nodes_rows=11,tok=4.21,levels=6)}
for name,c in cfgs.items():
    for Bp in (64,128,192):
        for lvl,nong in ((0.75,3.0),(0.95,4.0)):
            r=pp(Bp=Bp,lvl=lvl,nong=nong,**c)
            print('%-34s Bp=%3d lvl=%.2f nong=%.1f: %.1f s %+.1f%%'%(name,Bp,lvl,nong,r,100*(base/r-1)))
print('verbose one:'); pp(Bp=128,verbose=True,**cfgs['tree 8 nodes (128 rows, ~3.85?)'])
