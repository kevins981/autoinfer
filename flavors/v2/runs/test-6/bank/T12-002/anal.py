import sys, glob, json, collections, statistics as st
k=2
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d+"/timers.*"))[:1]:
        rows=[json.loads(l) for l in open(f)]
        steps=[];cur=None
        for name,meta,t in rows:
            if name=="fwd0": cur=dict(meta=meta,fwd0=t); steps.append(cur)
            elif cur is not None: cur[name]=t
        by=collections.defaultdict(list)
        for s in steps:
            if "prop1" not in s or "fwd1" not in s: continue
            n,r,cg=s["meta"]
            if n==r*(1+k) and r>=8: by[r].append((s["fwd1"]-s["fwd0"],s["prop1"]-s["prop0"]))
        allp=[x[1] for v in by.values() for x in v]
        print(d, "pure decode steps:", len(allp), "propose median %.3f ms (per draft step %.3f)"%(st.median(allp),st.median(allp)/k) if allp else "")
        for r,v in sorted(by.items()):
            if len(v)>=5: print("   reqs=%2d n=%4d fwd %.2f propose %.3f -> %.3f/draft step"%(r,len(v),st.median(x[0] for x in v),st.median(x[1] for x in v),st.median(x[1] for x in v)/k))
