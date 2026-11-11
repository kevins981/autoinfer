import os, torch
exec(open('mb10.py').read().split("# proof the new kernels")[0])
def setenv(**kw):
    for k in ('AI_POLICY','AI_SLAB','AI_SLAB_S49','AI_SLAB_INNER','AI_FORCE'): os.environ.pop(k,None)
    os.environ.update(kw)
torch.manual_seed(5)
for n,(N,K) in SH.items():
    L=Ls[n]; bad=[]
    for M in range(49,257):
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        setenv(AI_POLICY='0'); ref=call(L,N,K,x)
        setenv(); o=call(L,N,K,x)
        d=(o.float()-ref.float()).abs(); mx=d.max().item() if torch.isfinite(o.float()).all() else float('inf')
        if mx>16: bad.append((M,round(mx,1)))
    print(n,'BIG-DIFF M (policy vs stock):',bad[:40],'count',len(bad),flush=True)
    # which knob? test on first bad M
    if bad:
        M=bad[0][0]; x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        setenv(AI_POLICY='0'); ref=call(L,N,K,x)
        for name,kw in {'slab':dict(AI_POLICY='manual',AI_SLAB='1'),'slab+inner':dict(AI_POLICY='manual',AI_SLAB='1',AI_SLAB_INNER='1'),
                        'force c128_s3':dict(AI_POLICY='manual',AI_SLAB='0',AI_FORCE='64,128,128,3,1'),
                        'force bps2_s3':dict(AI_POLICY='manual',AI_SLAB='0',AI_FORCE='64,128,128,3,2'),
                        'slab2x32':dict(AI_POLICY='manual',AI_SLAB='1',AI_SLAB_S49='2'),
                        'slab2x32+bps2s3':dict(AI_POLICY='manual',AI_SLAB='1',AI_SLAB_S49='2',AI_FORCE='64,128,128,3,2')}.items():
            setenv(**kw); o=call(L,N,K,x); torch.cuda.synchronize()
            print('   ',n,'M',M,name,'maxdiff',round((o.float()-ref.float()).abs().max().item(),1),'finite',bool(torch.isfinite(o.float()).all()),flush=True)
