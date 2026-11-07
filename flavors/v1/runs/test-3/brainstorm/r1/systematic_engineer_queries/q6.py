import sqlite3,collections,bisect
c=sqlite3.connect("task/baseline/nsys_c16/profile.sqlite")
rows=c.execute("""SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
marks=[r[0] for r in rows if r[2]=='_rejection_kernel']
print("steps",len(marks))
steps=[]
j=0
cur=None
for r in rows:
    i=bisect.bisect_right(marks,r[0])
    if i==0 or i>=len(marks): continue
    if cur is None or cur['i']!=i:
        cur={'i':i,'pre':False,'busy':0,'mar':0,'nccl':0,'big':0}; steps.append(cur)
    cur['busy']+=r[1]-r[0]
    if r[2].startswith('chunk_fwd_kernel_o'): cur['pre']=True
    if 'Marlin' in r[2]: cur['mar']+=r[1]-r[0]
    if 'nccl' in r[2]: cur['nccl']+=r[1]-r[0]
for s in steps: s['dur']=marks[s['i']]-marks[s['i']-1]
tot=sum(s['dur'] for s in steps)
for pre in (False,True):
    ss=[s for s in steps if s['pre']==pre]
    d=sorted(s['dur'] for s in ss)
    print("prefill" if pre else "decode", len(ss), "time s %.2f (%.0f%%)"%(sum(d)/1e9,100*sum(d)/tot), "median ms %.1f p10 %.1f p90 %.1f"%(d[len(d)//2]/1e6,d[len(d)//10]/1e6,d[9*len(d)//10]/1e6),
      "marlin %.0f%% nccl %.0f%%"%(100*sum(s['mar'] for s in ss)/sum(d),100*sum(s['nccl'] for s in ss)/sum(d)))
