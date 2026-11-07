import sqlite3,re,bisect,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
T=333147580333815
nv=c.execute("select start,end,text from NVTX_EVENTS where text like 'execute_%' and globalTid=? order by start",(T,)).fetchall()
rt=c.execute("select r.start,k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=r.correlationId and k.deviceId=0 join StringIds s on s.id=k.shortName where r.globalTid=? order by r.start",(T,)).fetchall()
starts=[n[0] for n in nv]
agg=[]
for n in nv:
    m=re.match(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)',n[2]); agg.append(dict(nctx=int(m[1]),ctx=int(m[2]),ngen=int(m[3]),gen=int(m[4]),k0=None,k1=0,mar=0,nccl=0,tot=0,ag=0,gdn=0,attn=0))
for r in rt:
    i=bisect.bisect_right(starts,r[0])-1
    if i<0: continue
    a=agg[i]
    d=r[2]-r[1]; a['tot']+=d
    a['k0']=r[1] if a['k0'] is None else min(a['k0'],r[1]); a['k1']=max(a['k1'],r[2])
    nm=r[3]
    if 'Marlin' in nm: a['mar']+=d
    elif 'AllReduce' in nm: a['nccl']+=d
    elif 'AllGather' in nm: a['ag']+=d
    elif 'chunk' in nm or 'gdn' in nm or 'delta' in nm or 'conv1d' in nm or 'merge_16' in nm or 'recompute_w' in nm or 'solve' in nm: a['gdn']+=d
    elif 'Prefill' in nm or 'Decode' in nm or 'MergeStates' in nm: a['attn']+=d
# wall per step = next step first kernel - this first kernel
ks=[a for a in agg if a['k0']]
for i in range(len(ks)-1): ks[i]['wall']=ks[i+1]['k0']-ks[i]['k0']
ks=ks[:-1]
pre=[a for a in ks if a['nctx']>0]; dec=[a for a in ks if a['nctx']==0]
def s(l,k): return sum(a[k] for a in l)/1e6
print("decode steps",len(dec),"wall ms",s(dec,'wall'),"gen tokens",sum(a['gen'] for a in dec))
print("prefill steps",len(pre),"wall ms",s(pre,'wall'),"ctx tokens",sum(a['ctx'] for a in pre),"gen tok",sum(a['gen'] for a in pre))
for k in ('tot','mar','nccl','ag','gdn','attn'): print(k, "dec %.0f  pre %.0f"%(s(dec,k),s(pre,k)))
# regression: prefill step wall vs ctx tokens
import statistics
xs=[a['ctx'] for a in pre]; ys=[a['wall']/1e6 for a in pre]
mx,my=statistics.mean(xs),statistics.mean(ys)
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); print("slope ms/token",b,"intercept",my-b*mx)
for a in sorted(pre,key=lambda a:a['ctx'])[::6]: print(a['nctx'],a['ctx'],a['ngen'],round(a['wall']/1e6,1),round(a['mar']/1e6,1),round(a['nccl']/1e6,1),round(a['gdn']/1e6,1))
# full decode steps with 16 reqs
f=[a for a in dec if a['ngen']==16]; print("full-16 decode steps",len(f),"median wall", sorted(a['wall'] for a in f)[len(f)//2]/1e6)
by=collections.defaultdict(list)
for a in dec: by[a['ngen']].append(a)
for n in sorted(by):
    l=by[n]; w=sorted(x['wall'] for x in l)
    print("reqs",n,"steps",len(l),"median wall ms %.1f"%(w[len(w)//2]/1e6),"marlin %.1f nccl %.1f"%(sum(x['mar'] for x in l)/len(l)/1e6,sum(x['nccl'] for x in l)/len(l)/1e6))
