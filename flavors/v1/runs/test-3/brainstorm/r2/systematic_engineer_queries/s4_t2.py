# per-rank duration of the same kernel (i-th instance of each name) dev0 vs dev1: ratio by kernel class and by time
import sqlite3,collections,statistics
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsys/profile.sqlite")
q="""select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=? order by k.start"""
K={d:collections.defaultdict(list) for d in (0,1)}
for d in (0,1):
    for r in c.execute(q,(d,)): K[d][r[2]].append((r[0],r[1]-r[0]))
t0=min(v[0][0] for v in K[0].values())
res=[]
for nm in K[0]:
    a,b=K[0][nm],K[1].get(nm,[])
    if len(a)!=len(b) or len(a)<500: continue
    short=nm[:40] if 'Marlin<' not in nm else 'Marlin'+",".join(nm[nm.index('<')+1:].split(',')[4:8])
    for (s0,d0),(s1,d1) in zip(a,b): res.append((short,s0-t0,d0,d1))
by=collections.defaultdict(lambda:[0,0,0])
for nm,t,d0,d1 in res:
    big = d0>100000
    k=(nm,'long' if big else 'short'); by[k][0]+=d0; by[k][1]+=d1; by[k][2]+=1
print("kernel  class  n  sum0 ms  sum1 ms  ratio0/1")
for k,v in sorted(by.items(),key=lambda x:-x[1][0])[:16]: print("%-45s %-5s %7d %8.1f %8.1f %.3f"%(k[0],k[1],v[2],v[0]/1e6,v[1]/1e6,v[0]/v[1]))
# Marlin ratio over time (5 s buckets)
tb=collections.defaultdict(lambda:[0,0])
for nm,t,d0,d1 in res:
    if nm.startswith('Marlin'): tb[int(t/5e9)][0]+=d0; tb[int(t/5e9)][1]+=d1
print("Marlin dev0/dev1 by 5s bucket:",[ "%d:%.3f"%(k,v[0]/v[1]) for k,v in sorted(tb.items())])
