# per-step breakdown of the champion T1-005 nsys, steps from NVTX execute_ ranges (rank 0 thread, dev 0)
import sqlite3,re,bisect,collections,statistics,sys
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsysB/profile.sqlite")
T=334368626187075; DEV=0
nv=c.execute("select start,end,text from NVTX_EVENTS where text like 'execute_%' and globalTid=? order by start",(T,)).fetchall()
rt=c.execute("""select r.start,k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k
 on k.correlationId=r.correlationId and k.deviceId=? and k.globalPid=(r.globalTid/16777216)*16777216
 join StringIds s on s.id=k.shortName where r.globalTid=? order by r.start""",(DEV,T)).fetchall()
print("kernels joined",len(rt))
starts=[n[0] for n in nv]
cats=[('mar','Marlin'),('ar','hostar'),('lmh','Kernel2'),('lmh','gemm'),('lmh','gemv'),('lmh','splitK'),('gdn','gdn'),('gdn','chunk'),('gdn','delta'),('gdn','conv1d'),('gdn','merge_16'),('gdn','recompute_w'),('gdn','solve'),('attn','Prefill'),('attn','Decode'),('attn','MergeStates')]
agg=[]
for n in nv:
    m=re.match(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)',n[2]); agg.append(dict(nctx=int(m[1]),ctx=int(m[2]),ngen=int(m[3]),gen=int(m[4]),k0=None,k1=0,tot=0,cpu=n[1]-n[0],**{k:0 for k,_ in cats},oth=0))
for r in rt:
    i=bisect.bisect_right(starts,r[0])-1
    if i<0: continue
    a=agg[i]; d=r[2]-r[1]; a['tot']+=d
    a['k0']=r[1] if a['k0'] is None else min(a['k0'],r[1]); a['k1']=max(a['k1'],r[2])
    for k,p in cats:
        if p in r[3]: a[k]+=d; break
    else: a['oth']+=d
ks=[a for a in agg if a['k0'] and (a['ctx'] or a['gen'])]
for i in range(len(ks)-1): ks[i]['wall']=ks[i+1]['k0']-ks[i]['k0']
ks=ks[:-1]
pre=[a for a in ks if a['nctx']>0]; dec=[a for a in ks if a['nctx']==0]
W=sum(a['wall'] for a in ks)/1e6
print("total wall ms %.0f"%W)
keys=['tot','mar','ar','lmh','gdn','attn','oth']
for nm,l in (('decode',dec),('prefill',pre)):
    w=sum(a['wall'] for a in l)/1e6
    print(nm,"steps",len(l),"wall %.0f ms (%.0f%%)"%(w,100*w/W),"ctx tok",sum(a['ctx'] for a in l),"gen tok",sum(a['gen'] for a in l))
    print("   "+"  ".join("%s %.0f"%(k,sum(a[k] for a in l)/1e6) for k in keys))
xs=[a['ctx'] for a in pre]; ys=[a['wall']/1e6 for a in pre]
mx,my=statistics.mean(xs),statistics.mean(ys)
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); print("prefill wall: slope ms/token %.3f intercept %.1f"%(b,my-b*mx))
print("prefill steps sorted by ctx (nctx ctx ngen wall mar ar lmh gdn attn oth gpu_busy)")
for a in sorted(pre,key=lambda a:a['ctx'])[::max(1,len(pre)//15)]: print(a['nctx'],a['ctx'],a['ngen'],*["%.1f"%(a[k]/1e6) for k in ['wall','mar','ar','lmh','gdn','attn','oth','tot']])
by=collections.defaultdict(list)
for a in dec: by[a['ngen']].append(a)
for n in sorted(by):
    l=by[n]; w=sorted(x['wall'] for x in l)
    print("reqs",n,"steps",len(l),"med wall %.1f"%(w[len(w)//2]/1e6),"  ".join("%s %.1f"%(k,sum(x[k] for x in l)/len(l)/1e6) for k in keys))
