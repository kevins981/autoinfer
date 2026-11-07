# does anything run concurrently with the large-M Marlin kernels on each device? (other-stream kernels, memcpys)
import sqlite3,bisect
c=sqlite3.connect("workspaces/leader-1.hostar2/out/nsysB/profile.sqlite")
for d in (0,1):
    ks=c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=? order by k.start",(d,)).fetchall()
    mc=c.execute("select start,end,bytes,copyKind,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? order by start",(d,)).fetchall()
    big=[k for k in ks if 'Marlin<' in k[3] and '(int)256, (int)4, (int)16' in k[3] and k[1]-k[0]>100000]
    streams={}
    for k in ks: streams[k[2]]=streams.get(k[2],0)+1
    ov=0; ovt=0
    ms=[m[0] for m in mc]
    for b in big:
        i=bisect.bisect_left(ms,b[0]-10**7)
        while i<len(mc) and mc[i][0]<b[1]:
            if mc[i][1]>b[0]: ov+=1; ovt+=min(mc[i][1],b[1])-max(mc[i][0],b[0])
            i+=1
    kov=0
    starts=[k[0] for k in ks]
    for b in big[:3000]:
        i=bisect.bisect_left(starts,b[0]-5*10**6)
        while i<len(ks) and ks[i][0]<b[1]:
            if ks[i][2]!=b[2] and ks[i][1]>b[0]: kov+=1
            i+=1
    print("dev",d,"streams",streams,"memcpys",len(mc),"big marlin",len(big),"memcpy overlaps",ov,"ovl ms %.1f"%(ovt/1e6),"other-stream kernel overlaps (first 3000)",kov)
    import collections
    agg=collections.Counter((m[3],m[4]) for m in mc); print("  memcpy kinds/streams",agg.most_common(8))
