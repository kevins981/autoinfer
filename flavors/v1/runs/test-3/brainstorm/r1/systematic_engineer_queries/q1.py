import sqlite3,sys,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value,k.streamId,k.gridX,k.gridY FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
print(len(rows), (rows[-1][1]-rows[0][0])/1e9)
busy=sum(r[1]-r[0] for r in rows)/1e9; print("busy s",busy)
# gaps
gaps=[rows[i+1][0]-rows[i][1] for i in range(len(rows)-1)]
big=[g for g in gaps if g>0]
print("sum positive gaps s",sum(big)/1e9, "gaps>100us",sum(g for g in gaps if g>1e5)/1e9, "n",sum(1 for g in gaps if g>1e5))
# identify lm_head-like: gemvx / wmma 16x16_128x2 long
# find steps: split at gaps>300us
