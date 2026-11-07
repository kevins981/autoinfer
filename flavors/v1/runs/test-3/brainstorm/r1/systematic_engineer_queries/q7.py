import sqlite3,re,bisect
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
nv=c.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'execute_%' order by start").fetchall()
tids=set(n[3] for n in nv); print("nvtx",len(nv),"tids",len(tids))
# take one tid (rank0)
from collections import Counter
print(Counter(n[3] for n in nv))
