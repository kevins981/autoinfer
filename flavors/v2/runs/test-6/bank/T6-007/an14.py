import sqlite3, bisect, re, sys, statistics as st
from collections import defaultdict
sys.path.insert(0,'../../task/scripts')
from step_anatomy import RX, cat
c=sqlite3.connect('nsys_C/profile.sqlite'); gpu=0
work=defaultdict(list)
for s,e,corr,name in c.execute("select k.start,k.end,k.correlationId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=?",(gpu,)):
    work[corr].append((s,e,name))
# memcpy on gpu too
steps=defaultdict(list)
for s,e,tid,text in c.execute("select n.start,n.end,n.globalTid,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value) like 'execute_context%'"):
    m=RX.search(text or '')
    if m: steps[tid].append((s,e,tuple(int(x) for x in m.groups())))
rt=defaultdict(list)
for s,tid,corr in c.execute("select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    if corr in work: rt[tid].append((s,corr))
tid=max(steps,key=lambda t:len(rt.get(t,[])))
calls=sorted(rt[tid]); starts=[x[0] for x in calls]; rng=sorted(steps[tid])
def merged(iv):
    out=[]
    for s,e in sorted(iv):
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
rows=[]
for (s,e,(p,pt,d,dt)),nxt in zip(rng,rng[1:]+[None]):
    if nxt is None: continue
    end=nxt[0]; i,j=bisect.bisect_left(starts,s),bisect.bisect_left(starts,end)
    items=[w for _,corr in calls[i:j] for w in work[corr]]
    if not items: continue
    mg=merged([(x[0],x[1]) for x in items])
    span=mg[-1][1]-mg[0][0]; busy=sum(b-a for a,b in mg)
    gaps=[mg[k+1][0]-mg[k][1] for k in range(len(mg)-1)]
    by=defaultdict(int); byc=defaultdict(int); nk=defaultdict(int)
    for a,b,n in items: by[n]+=b-a; byc[cat(n)]+=b-a; nk[n]+=1
    rows.append(dict(p=p,pt=pt,d=d,dt=dt,period=(end-s)/1e6,span=span/1e6,busy=busy/1e6,idle_span=(span-busy)/1e6,idle_win=(end-s-busy)/1e6,
        maxgap=(max(gaps) if gaps else 0)/1e6,ngaps_gt100us=sum(1 for g in gaps if g>100e3),nk=len(items),by=by,byc=byc,nkd=nk))
def sel(f): return [r for r in rows if f(r)]
groups={'mixed M=96 (p>0, pt+dt in 90..96)':sel(lambda r:r['p']>0 and 90<=r['pt']+r['dt']<=96),
        'mixed other M 49-89':sel(lambda r:r['p']>0 and 49<=r['pt']+r['dt']<90),
        'decode M=48 (dt=48)':sel(lambda r:r['p']==0 and r['dt']==48),
        'decode dt 36-45':sel(lambda r:r['p']==0 and 36<=r['dt']<=45)}
for g,L in groups.items():
    print(f'\n##### {g}: n={len(L)}')
    if not L: continue
    md=lambda k: st.median(r[k] for r in L)
    print(f"period {md('period'):.1f} | GPU span {md('span'):.1f} | busy(union) {md('busy'):.1f} | idle inside span {md('idle_span'):.1f} | idle in window {md('idle_win'):.1f} | largest gap {md('maxgap'):.2f} (max {max(r['maxgap'] for r in L):.2f}) | gaps>100us {md('ngaps_gt100us'):.0f} | kernels/step {md('nk'):.0f}")
    cs=defaultdict(list)
    for r in L:
        for k in set(r['byc'])|set(cs): pass
    allc=set(k for r in L for k in r['byc'])
    print('category ms (median):',{k:round(st.median(r['byc'].get(k,0)/1e6 for r in L),2) for k in sorted(allc,key=lambda k:-st.median(r['byc'].get(k,0) for r in L))})
    alln=set(k for r in L for k in r['by'])
    top=sorted(alln,key=lambda k:-st.median(r['by'].get(k,0) for r in L))[:12]
    for k in top:
        print(f"  {st.median(r['by'].get(k,0)/1e6 for r in L):7.2f} ms  x{st.median(r['nkd'].get(k,0) for r in L):.0f}  {k[:110]}")
