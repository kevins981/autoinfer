import sqlite3,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
dm=dict(c.execute("select k.start, s.value from CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.deviceId=0").fetchall())
lmh=[i for i,r in enumerate(rows) if 'Kernel2' in r[2] and r[4]==970]
# print sequence of lm_head times with gaps to identify step pattern
t0=rows[0][0]
for a in lmh[200:215]:
    r=rows[a]; print(round((r[0]-t0)/1e6,2), round((r[1]-r[0])/1e3), r[3])
