"""Per-step anatomy of an nsys profile (from profile_nsys.sh): what one engine step costs, on the GPU.
usage: python step_anatomy.py <profile.sqlite> [--gpu 0] [--csv steps.csv]

A step = one `execute_model` call of a TP worker: vLLM marks each with an NVTX range named
execute_context_<P>(<prefill tokens>)_generation_<D>(<decode tokens>) (P = requests with prefill
work, D = decoding requests). The step's window runs from the start of its range to the start of
the worker's next one, so it includes sampling, drafting and bookkeeping. Its kernels are the ones
launched (CUDA runtime calls, matched by correlation id) by that worker thread in that window.
Steps with P > 0 are "prefill" (mixed prefill + decode), else "decode".

Per kind it prints medians of: period (window length = time between consecutive steps),
GPU span (first kernel start -> last kernel end), GPU busy (union of kernel time), GPU idle in the
window (period - busy), and kernel time by category (categories from kernel_breakdown.py).
With a graph-level trace (TRACE=graph in profile_nsys.sh), graph replays count as one item each
in category "CUDA graph (graph-level trace)": busy/idle are exact, categories inside graphs are not.
"""
import argparse, bisect, csv, os, re, sqlite3, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_breakdown import CATS  # noqa: E402

RX = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def cat(name):
    return next((c for c, rx in CATS if re.search(rx, name, re.I)), "other")


def union(iv):
    tot, ce = 0, -1
    for s, e in sorted(iv):
        if s > ce:
            tot += e - s; ce = e
        elif e > ce:
            tot += e - ce; ce = e
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("--gpu", type=int, default=0, help="device id (TP rank's GPU), default 0")
    ap.add_argument("--csv", help="write one row per step here")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}

    # GPU work of this device, by correlation id
    work = defaultdict(list)                     # corr -> [(start, end, category)]
    for s, e, corr, name in c.execute(
            "select k.start, k.end, k.correlationId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id = k.shortName where k.deviceId = ?", (a.gpu,)):
        work[corr].append((s, e, cat(name)))
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        for s, e, corr in c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                                    "where deviceId = ?", (a.gpu,)):
            work[corr].append((s, e, "CUDA graph (graph-level trace)"))
    if not work:
        sys.exit(f"no kernels on device {a.gpu}")

    # step ranges per worker thread; the thread that launched this device's work is ours
    steps = defaultdict(list)
    for s, e, tid, text in c.execute(
            "select n.start, n.end, n.globalTid, coalesce(n.text, s.value) from NVTX_EVENTS n "
            "left join StringIds s on s.id = n.textId where coalesce(n.text, s.value) like 'execute_context%'"):
        m = RX.search(text or "")
        if m:
            steps[tid].append((s, e, tuple(int(x) for x in m.groups())))
    if not steps:
        sys.exit("no execute_context NVTX ranges in this profile (was it made with --trace=cuda,nvtx?)")
    rt = defaultdict(list)                       # tid -> sorted [(start, corr)]
    for s, tid, corr in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if corr in work:
            rt[tid].append((s, corr))
    tid = max(steps, key=lambda t: len(rt.get(t, [])))
    calls = sorted(rt[tid]); starts = [x[0] for x in calls]
    rng = sorted(steps[tid])

    rows = []
    for (s, _e, (p, pt, d, dt)), nxt in zip(rng, rng[1:] + [None]):
        end = nxt[0] if nxt else s + 10**9
        i, j = bisect.bisect_left(starts, s), bisect.bisect_left(starts, end)
        items = [w for _, corr in calls[i:j] for w in work[corr]]
        if not items or nxt is None:
            continue
        busy = union([(x[0], x[1]) for x in items])
        bycat = defaultdict(int)
        for x in items:
            bycat[x[2]] += x[1] - x[0]
        rows.append({"kind": "prefill" if p else "decode", "prefill_reqs": p, "prefill_tokens": pt,
                     "decode_reqs": d, "decode_tokens": dt, "period_ms": (end - s) / 1e6,
                     "gpu_span_ms": (max(x[1] for x in items) - min(x[0] for x in items)) / 1e6,
                     "gpu_busy_ms": busy / 1e6, "gpu_idle_ms": (end - s - busy) / 1e6,
                     **{f"cat:{k}": v / 1e6 for k, v in bycat.items()}})

    cats = sorted({k for r in rows for k in r if k.startswith("cat:")})
    print(f"{a.db}: GPU {a.gpu}, {len(rows)} steps")
    for kind in ("decode", "prefill"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        med = lambda k: st.median(r.get(k, 0.0) for r in rs)
        print(f"\n== {kind} steps: {len(rs)} (median prefill tokens {med('prefill_tokens'):.0f}, "
              f"decode tokens {med('decode_tokens'):.0f})")
        print(f"  period {med('period_ms'):.2f} ms   GPU span {med('gpu_span_ms'):.2f}   "
              f"GPU busy {med('gpu_busy_ms'):.2f}   GPU idle in window {med('gpu_idle_ms'):.2f}")
        for k in sorted(cats, key=lambda k: -med(k)):
            if med(k) >= 0.01:
                print(f"  {med(k):8.2f} ms  {k[4:]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())[:9] + cats, restval=0.0)
            w.writeheader(); w.writerows(rows)
        print(f"\nper-step rows: {a.csv}")


if __name__ == "__main__":
    main()
