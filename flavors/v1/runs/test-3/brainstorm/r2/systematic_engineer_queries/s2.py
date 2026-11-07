# kernels inside full (16-req) decode steps of champion T1-005, dev0: per name/grid avg duration and count per step
import sqlite3,re,bisect,collections
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsysB/profile.sqlite")
T=334368626187075
nv=c.execute("select start,end,text from NVTX_EVENTS where text like 'execute_%' and globalTid=? order by start",(T,)).fetchall()
rt=c.execute("""select r.start,k.start,k.end,s.value,k.gridX,k.gridY,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k
 on k.correlationId=r.correlationId and k.deviceId=0 and k.globalPid=(r.globalTid/16777216)*16777216
 join StringIds s on s.id=k.demangledName where r.globalTid=? order by k.start""",(T,)).fetchall()
starts=[n[0] for n in nv]
sel=set(i for i,n in enumerate(nv) if re.match(r'execute_context_0\(0\)_generation_16\(48\)',n[2]))
agg=collections.defaultdict(list); steps=set()
for r in rt:
    i=bisect.bisect_right(starts,r[0])-1
    if i in sel:
        steps.add(i)
        nm=r[3]
        if 'Marlin<' in nm: nm='Marlin<'+",".join(nm[nm.index('<')+1:].split(',')[4:8])+'>'
        agg[(nm[:70],r[4],r[5],r[6])].append(r[2]-r[1])
n=len(steps); print("steps",n)
tot=0
for k,v in sorted(agg.items(),key=lambda x:-sum(x[1])):
    s=sum(v)/n/1e3; tot+=s
    if s>50: print("%7.0f us/step %5.1f calls/step avg %6.1f us  %s grid=%d,%d blk=%d"%(s,len(v)/n,sum(v)/len(v)/1e3,k[0],k[1],k[2],k[3]))
print("total us/step",tot)
