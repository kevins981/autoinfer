import sqlite3,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
lm=[r for r in rows if 's161616gemm_bf16_16x16_128x2' in r[2]]
d=collections.Counter(( round((r[1]-r[0])/1e5)/10, r[3],r[4],r[5]) for r in lm)
print(d.most_common(15))
# Step detection: find the long wmma (>1ms) as lm_head marker
lmh=[r for r in lm if r[1]-r[0]>1e6]
print(len(lmh))
ts=[r[0] for r in lmh]
dt=[(ts[i+1]-ts[i])/1e6 for i in range(len(ts)-1)]
dt.sort(); print("period ms pctl",[dt[int(len(dt)*p)] for p in (0.1,0.25,0.5,0.75,0.9)])
