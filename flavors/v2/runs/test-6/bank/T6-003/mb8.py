import os, statistics as st, torch
exec(open('mb7.py').read().split("N,K=17408")[0])
N,K=17408,5120; L=mk(N,K)
flush=torch.empty(256*2**20//4,device=dev,dtype=torch.float32)
MODES={'stock':('0','0'),'new':('1','0'),'inner':('1','1')}
for M in (64,72,96,112,128):
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); out={}
    for m,(a,i) in MODES.items():
        os.environ['AI_SLAB']=a; os.environ['AI_SLAB_INNER']=i
        for _ in range(3): apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None)
        ts=[]
        for _ in range(40):
            flush.zero_(); torch.cuda.synchronize()
            e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
            e0.record(); apply_fp8_marlin_linear(x,L.weight,L.weight_scale_inv,ws,N,K,None); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1)*1000)
        out[m]=round(st.median(ts),1)
    print('single-call events, cold gate_up M=',M,out,flush=True)
