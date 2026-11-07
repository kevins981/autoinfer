# rank skew at host-AR kernels in champion T1-005 nsys: pair the i-th hostar on dev0 and dev1 (same sequence),
# skew = start1-start0; attribute to the segment between consecutive ARs: busy time + idle gap per rank
import sqlite3,collections,statistics
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsysB/profile.sqlite")
q="""select k.start,k.end,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=? order by k.start"""
K={d:c.execute(q,(d,)).fetchall() for d in (0,1)}
AR={d:[i for i,r in enumerate(K[d]) if r[2].startswith('hostar')] for d in (0,1)}
print("hostar counts",len(AR[0]),len(AR[1]))
n=min(len(AR[0]),len(AR[1]))
rows=[]
for j in range(1,n):
    seg={}
    for d in (0,1):
        a,b=AR[d][j-1],AR[d][j]
        ks=K[d][a+1:b]
        busy=sum(r[1]-r[0] for r in ks)
        span=K[d][b][0]-K[d][a][1]   # from end of prev AR to start of this AR
        seg[d]=(busy,span-busy,K[d][b][0],K[d][b][1]-K[d][b][0],len(ks),ks)
    skew=seg[1][2]-seg[0][2]
    rows.append((j,skew,seg))
# classify: large AR (prefill) by duration of the later-arriving AR's post-arrival time
big=[r for r in rows if max(r[2][0][3],r[2][1][3])>300000]
small=[r for r in rows if max(r[2][0][3],r[2][1][3])<=300000]
for nm,l in (("big(>300us)",big),("small",small)):
    sk=[abs(r[1]) for r in l]
    late1=sum(1 for r in l if r[1]>0)
    print(nm,"n",len(l),"mean|skew| us %.1f"%(statistics.mean(sk)/1e3),"median %.1f"%(statistics.median(sk)/1e3),"rank1 later frac %.2f"%(late1/len(l)))
    db=[ (r[2][1][0]-r[2][0][0]) for r in l]; dg=[(r[2][1][1]-r[2][0][1]) for r in l]
    print("   mean(busy1-busy0) us %.1f  mean(idle1-idle0) us %.1f  mean idle0 %.1f idle1 %.1f"%(statistics.mean(db)/1e3,statistics.mean(dg)/1e3,statistics.mean(r[2][0][1] for r in l)/1e3,statistics.mean(r[2][1][1] for r in l)/1e3))
# kernels with largest per-name time difference between ranks inside big segments
diff=collections.defaultdict(lambda:[0,0,0])
for r in big:
    for d in (0,1):
        for k in r[2][d][5]:
            diff[k[2][:50]][d]+=k[1]-k[0]
            if d==0: diff[k[2][:50]][2]+=1
print("per-kernel total (ms) rank0 rank1 diff, in big-AR segments")
for k,v in sorted(diff.items(),key=lambda x:-abs(x[1][1]-x[1][0]))[:12]: print("  %-50s %8.1f %8.1f %7.1f n=%d"%(k,v[0]/1e6,v[1]/1e6,(v[1]-v[0])/1e6,v[2]))
