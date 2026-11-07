# In-server decode Marlin (T1-005 nsys, full 16-req decode steps, GPU 0): per call duration by grid (shape) and preceding kernel
import sqlite3,re,bisect,collections
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsysB/profile.sqlite")
T=334368626187075
nv=c.execute("select start,end,text from NVTX_EVENTS where text like 'execute_%' and globalTid=? order by start",(T,)).fetchall()
rt=c.execute("""select r.start,k.start,k.end,s.value,k.gridX,k.gridY,k.blockX from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k
 on k.correlationId=r.correlationId and k.deviceId=0 and k.globalPid=(r.globalTid/16777216)*16777216
 join StringIds s on s.id=k.demangledName where r.globalTid=? order by k.start""",(T,)).fetchall()
starts=[n[0] for n in nv]
sel=set(i for i,n in enumerate(nv) if re.match(r'execute_context_0\(0\)_generation_16\(48\)',n[2]))
def cls(nm):
    for k in ['Marlin','chunk_gated','fused_recurrent','recurrent','gdn','allreduce','all_reduce','host','flash','BatchDecode','BatchPrefill','rms','norm','act_and_mul','causal_conv','conv']:
        if k.lower() in nm.lower(): return k
    return nm[:40]
agg=collections.defaultdict(list); prev=None; steps=set()
for r in rt:
    i=bisect.bisect_right(starts,r[0])-1
    if i in sel:
        steps.add(i)
        if 'Marlin' in r[3]:
            # find previous non-trivial kernel class (skip tiny ones? keep immediate)
            agg[(r[4],r[5],r[6],prev)].append((r[2]-r[1])/1e3)
        prev=cls(r[3])
    else: prev=None
n=len(steps); print("steps",n)
for k,v in sorted(agg.items(),key=lambda x:-sum(x[1])):
    if len(v)/n>=0.9: print("grid=%s,%s blk=%s prev=%-22s calls/step %5.1f avg %6.1f us med %6.1f"%(k[0],k[1],k[2],k[3],len(v)/n,sum(v)/len(v),sorted(v)[len(v)//2]))
