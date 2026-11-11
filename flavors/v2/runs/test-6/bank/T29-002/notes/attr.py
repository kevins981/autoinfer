"""T29 task 3: per-step kernel attribution A (Marlin) vs B (W8A8) from two nsys sqlite files (GPU0 by default).
usage: python attr.py profA.sqlite profB.sqlite [--gpu 0]
"""
import sys, re, sqlite3, bisect, statistics as st, argparse, json
from collections import defaultdict

RX = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
GDN = [i % 4 != 3 for i in range(64)]
SEQ = []
for i in range(64):
    SEQ += (['qkvz', 'gdn_out'] if GDN[i] else ['attn_qkv', 'attn_o']) + ['gate_up', 'down']


def short(n):
    if 'marlin' in n.lower() or 'Marlin' in n: return 'GEMM:marlin'
    if '_w8a8_triton_block_scaled_mm' in n: return 'GEMM:w8a8'
    return n[:90]


def is_gemm(n): return n in ('GEMM:marlin', 'GEMM:w8a8')


def load(db, gpu):
    c = sqlite3.connect(db)
    work = defaultdict(list)
    for s, e, corr, name, gx, gy, gz in c.execute(
            "select k.start, k.end, k.correlationId, s.value, k.gridX, k.gridY, k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id = k.shortName where k.deviceId = ?", (gpu,)):
        work[corr].append((s, e, short(name), (gx, gy, gz)))
    steps = defaultdict(list)
    for s, e, tid, text in c.execute(
            "select n.start, n.end, n.globalTid, coalesce(n.text, s.value) from NVTX_EVENTS n "
            "left join StringIds s on s.id = n.textId where coalesce(n.text, s.value) like 'execute_context%'"):
        m = RX.search(text or "")
        if m: steps[tid].append((s, e, tuple(int(x) for x in m.groups())))
    rt = defaultdict(list)
    for s, tid, corr in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if corr in work: rt[tid].append((s, corr))
    tid = max(steps, key=lambda t: len(rt.get(t, [])))
    calls = sorted(rt[tid]); starts = [x[0] for x in calls]; rng = sorted(steps[tid])
    out = []
    for (s, _e, (p, pt, d, dt)), nxt in zip(rng, rng[1:] + [None]):
        if nxt is None: continue
        i, j = bisect.bisect_left(starts, s), bisect.bisect_left(starts, nxt[0])
        items = sorted(w for _, corr in calls[i:j] for w in work[corr])
        if items: out.append(dict(p=p, pt=pt, d=d, dt=dt, period=(nxt[0] - s) / 1e6, k=items))
    return out


def per_name(steps):
    agg = defaultdict(list); cnt = defaultdict(list)
    names = set(x[2] for s in steps for x in s['k'])
    for s in steps:
        t = defaultdict(float); n = defaultdict(int)
        for x in s['k']: t[x[2]] += (x[1] - x[0]) / 1e3; n[x[2]] += 1
        for nm in names: agg[nm].append(t.get(nm, 0.0)); cnt[nm].append(n.get(nm, 0))
    return {nm: (st.median(agg[nm]), st.median(cnt[nm])) for nm in names}


def gemm_by_shape(steps):
    """decode steps: first 256 GEMMs = target forward in layer order; rest = drafter."""
    res = defaultdict(list); dr = []; ngem = []
    for s in steps:
        g = [x for x in s['k'] if is_gemm(x[2])]
        ngem.append(len(g))
        if len(g) < 256: continue
        t = defaultdict(float)
        for name, x in zip(SEQ, g[:256]): t[name] += (x[1] - x[0]) / 1e3
        for k, v in t.items(): res[k].append(v)
        dr.append(sum((x[1] - x[0]) / 1e3 for x in g[256:]))
    return {k: st.median(v) for k, v in res.items()}, (st.median(dr) if dr else 0), st.median(ngem) if ngem else 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('A'); ap.add_argument('B'); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    S = {'A': load(a.A, a.gpu), 'B': load(a.B, a.gpu)}
    print(f'GPU{a.gpu}: steps A {len(S["A"])} B {len(S["B"])}')
    dec = {k: [s for s in v if s['p'] == 0] for k, v in S.items()}
    mix = {k: [s for s in v if s['p'] > 0] for k, v in S.items()}
    for k in 'AB':
        print(f'{k}: decode steps {len(dec[k])} median period {st.median(s["period"] for s in dec[k]):.2f} ms, '
              f'GPU busy-sum {st.median(sum((x[1]-x[0])/1e6 for x in s["k"]) for s in dec[k]):.2f} ms, decode tokens {st.median(s["dt"] for s in dec[k])}')
    # (a) per-kernel decode diff
    pa, pb = per_name(dec['A']), per_name(dec['B'])
    rows = []
    for nm in set(pa) | set(pb):
        ta, ca = pa.get(nm, (0, 0)); tb, cb = pb.get(nm, (0, 0))
        rows.append((tb - ta, nm, ta, ca, tb, cb))
    rows.sort(key=lambda r: -abs(r[0]))
    print('\n== decode step: per-kernel median us/step, A vs B (top 30 by |diff|)')
    print(f'{"diff":>9} {"A us":>9} {"#A":>5} {"B us":>9} {"#B":>5}  kernel')
    for d, nm, ta, ca, tb, cb in rows[:30]:
        print(f'{d:9.1f} {ta:9.1f} {ca:5.0f} {tb:9.1f} {cb:5.0f}  {nm}')
    print(f'sum of all per-name median diffs: {sum(r[0] for r in rows):.1f} us')
    for k in 'AB':
        g, dr, ng = gemm_by_shape(dec[k])
        print(f'\n{k} decode GEMMs per step (median): #GEMM {ng}; target by shape (us): ' + ', '.join(f'{n} {v:.0f}' for n, v in g.items())
              + f'; target total {sum(g.values()):.0f}; drafter+other GEMMs {dr:.0f}')
    # (b) mixed steps: linear time vs tokens
    print('\n== mixed steps: tokens bucket -> median (period ms, GEMM ms, #GEMM, quant-ish ms) A | B')
    def bucket(s): return 32 * -(-(s['pt'] + s['dt']) // 32)
    for k in 'AB':
        by = defaultdict(list)
        for s in mix[k]: by[bucket(s)].append(s)
        print(f'-- {k}: {len(mix[k])} mixed steps; total mixed period {sum(s["period"] for s in mix[k]):.0f} ms, '
              f'total mixed GEMM {sum(sum((x[1]-x[0])/1e6 for x in s["k"] if is_gemm(x[2])) for s in mix[k]):.0f} ms')
        for b in sorted(by):
            ss = by[b]
            gm = st.median(sum((x[1]-x[0])/1e6 for x in s['k'] if is_gemm(x[2])) for s in ss)
            ng = st.median(sum(1 for x in s['k'] if is_gemm(x[2])) for s in ss)
            busy = st.median(sum((x[1]-x[0])/1e6 for x in s['k']) for s in ss)
            print(f'   M~{b:5d} n={len(ss):3d} period {st.median(s["period"] for s in ss):7.2f}  kernel-sum {busy:7.2f}  GEMM {gm:7.2f} (#{ng:.0f})')
    pa, pb = per_name(mix['A']), per_name(mix['B'])
    rows = []
    for nm in set(pa) | set(pb):
        ta, ca = pa.get(nm, (0, 0)); tb, cb = pb.get(nm, (0, 0))
        rows.append((tb - ta, nm, ta, ca, tb, cb))
    rows.sort(key=lambda r: -abs(r[0]))
    print('\n== mixed step (all M pooled): per-kernel median us/step, A vs B (top 20 by |diff|)')
    for d, nm, ta, ca, tb, cb in rows[:20]:
        print(f'{d:9.1f} {ta:9.1f} {ca:5.0f} {tb:9.1f} {cb:5.0f}  {nm}')


if __name__ == '__main__':
    main()
