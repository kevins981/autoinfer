import sqlite3,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
t0=rows[0][0]
A=t0+int(7587.5e6)+int(1.5e6); B=t0+int(7627.34e6)+int(1.5e6)
seg=[r for r in rows if A<=r[0]<B]
print("n",len(seg),"span ms",(seg[-1][1]-seg[0][0])/1e6,"busy",sum(r[1]-r[0] for r in seg)/1e6)
agg=collections.defaultdict(lambda:[0,0])
for r in seg:
    k=r[2][:60]+(" g=%d,%d"%(r[3],r[4]) if ('Marlin' in r[2] or 'Kernel2' in r[2] or 'nccl' in r[2]) else "")
    agg[k][0]+=1; agg[k][1]+=r[1]-r[0]
for k,v in sorted(agg.items(),key=lambda x:-x[1][1])[:40]: print("%5d %8.3f ms  %s"%(v[0],v[1]/1e6,k))
# compressed sequence (first 80 kernels)
prev=None
out=[]
for r in seg[:400]:
    n=r[2][:25]
    out.append("%s(%d)"%(n,(r[1]-r[0])//1000))
print(" | ".join(out[:150]))
