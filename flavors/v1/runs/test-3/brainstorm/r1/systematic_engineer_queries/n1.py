import sys,os; sys.path.insert(0, os.environ["NCU_PYTHON"])
import ncu_report
ctx=ncu_report.load_report("task/baseline/ncu_marlin/marlin_c16.ncu-rep")
r=ctx.range_by_idx(0)
a=r.action_by_idx(0)
names=a.metric_names()
want=[n for n in names if any(s in n for s in ("pipe_tensor","dram__throughput","sm__cycles_elapsed.avg.per_second","dram__cycles_elapsed.avg.per_second","smsp__average_warp","warp_issue_stalled","smsp__pcsamp_warps_issue_stalled")) and ("pct" in n or "per_second" in n or "ratio" in n or "stalled" in n)]
for i in range(4):
    a=r.action_by_idx(i)
    print("---",i,a.name()[:40], a.metric_by_name("gpu__time_duration.sum").as_double())
    for n in want:
        try:
            v=a.metric_by_name(n).as_double()
            if v>0.5 or 'per_second' in n: print("  %-90s %.2f"%(n,v))
        except: pass
