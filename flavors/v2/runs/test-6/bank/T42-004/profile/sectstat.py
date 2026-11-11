# Median critical path of the lm_head+sample section: start kernel -> _greedy_ids_rejection_kernel start (GPU 0),
# plus median duration of each kernel name inside it.
import sqlite3, sys, re, statistics as st
from collections import defaultdict
db, rx = sys.argv[1], re.compile(sys.argv[2])
c = sqlite3.connect(db).cursor()
ks = c.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
                  WHERE k.deviceId=0 ORDER BY k.start""").fetchall()
tot, per, busy = [], defaultdict(list), []
for i, k in enumerate(ks):
    if not rx.search(k[2]): continue
    j = i; d = defaultdict(float); b = 0
    while j < len(ks) and j < i + 60 and '_greedy_ids_rejection' not in ks[j][2]:
        d[ks[j][2][:40]] += (ks[j][1] - ks[j][0]) / 1e3; b += (ks[j][1] - ks[j][0]) / 1e3; j += 1
    if j >= len(ks) or '_greedy_ids_rejection' not in ks[j][2]: continue
    tot.append((ks[j][0] - k[0]) / 1e3); busy.append(b)
    for n, v in d.items(): per[n].append(v)
print(f'n={len(tot)} section median {st.median(tot):.1f} us (p10 {sorted(tot)[len(tot)//10]:.1f} p90 {sorted(tot)[9*len(tot)//10]:.1f}); busy median {st.median(busy):.1f}')
for n, v in sorted(per.items(), key=lambda x: -st.median(x[1])):
    print(f'  {st.median(v):8.1f} us  x{len(v):4d}  {n}')
