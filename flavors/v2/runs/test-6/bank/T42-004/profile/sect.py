# Print the kernel sequence of the target lm_head + sample section for a few decode steps (GPU 0), with gaps.
# usage: python sect.py <profile.sqlite> <start-kernel-regex> [n_steps_to_print] [after=40]
import sqlite3, sys, re, statistics as st
db, rx = sys.argv[1], re.compile(sys.argv[2]); npr = int(sys.argv[3]) if len(sys.argv) > 3 else 2
after = int(sys.argv[4]) if len(sys.argv) > 4 else 40
c = sqlite3.connect(db).cursor()
ks = c.execute("""SELECT k.start, k.end, s.value, k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
                  WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
rt = dict((r[0], (r[1], r[2])) for r in c.execute("SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall())
starts = [i for i, k in enumerate(ks) if rx.search(k[2])]
print(len(starts), 'section starts')
for si in starts[len(starts)//2: len(starts)//2 + npr]:
    t0 = ks[si][0]; prev = t0
    for k in ks[si: si + after]:
        api = rt.get(k[3]); lag = (k[0] - api[0]) / 1e3 if api else float('nan')
        print(f'{(k[0]-t0)/1e3:9.1f} us dur {(k[1]-k[0])/1e3:8.1f} gap {(k[0]-prev)/1e3:7.1f} launch->start {lag:8.1f}  {k[2][:70]}')
        prev = k[1]
    print('---')
