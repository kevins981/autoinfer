import sqlite3,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
t0=rows[0][0]
A=t0+int(7620e6); B=t0+int(7630e6)
seg=[r for r in rows if A<=r[0]<B]
last=None
for r in seg:
    gap=(r[0]-last)/1e3 if last else 0
    last=r[1]
    if 'Marlin' in r[2] and (r[1]-r[0])>40000: continue
    print("%9.3f +%6.1f %7.1fus %s g=%d,%d s=%d"%((r[0]-t0)/1e6,gap,(r[1]-r[0])/1e3,r[2][:50],r[3],r[4],r[6]))
