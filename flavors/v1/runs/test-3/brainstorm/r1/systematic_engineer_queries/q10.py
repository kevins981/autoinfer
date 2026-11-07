import sqlite3,re,bisect,collections
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
T=333147580333815
nv=c.execute("select start,end,text from NVTX_EVENTS where text like 'execute_%' and globalTid=? order by start",(T,)).fetchall()
rt=c.execute("select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.globalTid=? order by r.start",(T,)).fetchall()
starts=[n[0] for n in nv]
cnt=collections.defaultdict(collections.Counter)
for r in rt:
    i=bisect.bisect_right(starts,r[0])-1
    if i<0 or r[0]>nv[i][1]: continue
    cnt[i][r[2].split('_v')[0]]+=1
for i in [k for k,n in enumerate(nv) if 'context_1(57)' in n[2] or 'context_1(836)' in n[2]][:2]+[100,101]:
    print(nv[i][2], "cpu range ms %.1f"%((nv[i][1]-nv[i][0])/1e6), cnt[i].most_common(5))
