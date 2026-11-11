# per-step Marlin stats from an nsys sqlite (rank 0 = deviceId 0). usage: steps14.py <profile.sqlite> <out.json>
import sqlite3, re, bisect, sys, json
from collections import defaultdict
RX=re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
c=sqlite3.connect(sys.argv[1])
work=defaultdict(list)
for s,e,corr,name,gx,gy,gz in c.execute("select k.start,k.end,k.correlationId,sid.value,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sid on sid.id=k.shortName where k.deviceId=0"):
    work[corr].append((s,e,name,gx,gy))
for s,e,corr in c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0"):
    work[corr].append((s,e,'MEMCPY',0,0))
steps=defaultdict(list)
for s,e,tid,text in c.execute("select n.start,n.end,n.globalTid,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value) like 'execute_context%'"):
    m=RX.search(text or '')
    if m: steps[tid].append((s,e,tuple(int(x) for x in m.groups())))
rt=defaultdict(list)
for s,tid,corr in c.execute("select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    if corr in work: rt[tid].append((s,corr))
tid=max(steps,key=lambda t:len(rt.get(t,[])))
calls=sorted(rt[tid]); starts=[x[0] for x in calls]; rng=sorted(steps[tid])
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for a,b in iv:
        if ce is None or a>ce:
            if ce is not None: tot+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    if ce is not None: tot+=ce-cs
    return tot
def isar(n): return n=='MEMCPY' or 'hostar' in n or 'nccl' in n.lower()
out=[]
for (s,_e,(p,pt,d,dt)),nxt in zip(rng,rng[1:]+[None]):
    if nxt is None: continue
    i,j=bisect.bisect_left(starts,s),bisect.bisect_left(starts,nxt[0])
    items=[w for _,corr in calls[i:j] for w in work[corr]]
    if not items: continue
    mar=[w for w in items if 'marlin' in w[2].lower()]
    if not mar: continue
    s0=min(w[0] for w in items); e0=max(w[1] for w in items)
    mi=[(w[0],w[1]) for w in mar]; ari=[(w[0],w[1]) for w in items if isar(w[2])]
    others=[(w[0],w[1]) for w in items if 'marlin' not in w[2].lower()]
    m_sum=sum(b-a for a,b in mi); m_un=union(mi)
    # Marlin time with no AR running at the same time (exposed Marlin) and Marlin wall time overlapped with AR
    both=union(mi+ari)-union(ari)  # time covered by marlin but not by AR
    # critical-path Marlin: time where marlin active and no other non-AR kernel active
    # (marlin-priority attribution as cp_2.py = m_un)
    # launches per distinct big launch: kernels with grid >= 100 CTAs counted separately
    big=[w for w in mar if (w[3]*w[4])>=64]
    out.append(dict(p=p,pt=pt,d=d,dt=dt,R=pt+dt,n=len(mar),nbig=len(big),m_sum=m_sum/1e6,m_un=m_un/1e6,m_noar=both/1e6,
                    span=(e0-s0)/1e6,period=(nxt[0]-s)/1e6,ar_un=union(ari)/1e6,busy=union([(w[0],w[1]) for w in items])/1e6))
json.dump(out,open(sys.argv[2],'w'))
dec=[o for o in out if o['p']==0 and o['dt']>=40]
mx=[o for o in out if o['p']]
import statistics as st
print('decode steps',len(dec),'median marlin sum ms',st.median(o['m_sum'] for o in dec) if dec else None,'launches',st.median(o['n'] for o in dec) if dec else None)
print('mixed steps',len(mx))
