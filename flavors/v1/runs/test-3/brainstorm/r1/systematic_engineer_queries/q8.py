import sqlite3,re,bisect,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
nv=c.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'execute_%' order by start").fetchall()
tid0=nv[0][3]
# determine which tid is device 0: use runtime pid
rt=c.execute("select r.start,r.globalTid,k.start,k.end,s.value,k.deviceId from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=r.correlationId and k.globalPid/0x1000000=r.globalTid/0x1000000 join StringIds s on s.id=k.shortName").fetchall()
print(len(rt))
by=collections.defaultdict(list)
for x in rt: by[x[1]].append(x)
# map tid->device
for t,l in by.items(): print(t, collections.Counter(x[5] for x in l).most_common(2), len(l))
