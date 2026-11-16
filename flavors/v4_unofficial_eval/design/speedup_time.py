#!/usr/bin/env python3
"""Time to a speedup: how much GPU time (and wall time) each champion promotion costs, compared with
the cost of one full measurement. Ideal = every measurement is a speedup (ratio 1.0).

usage: python speedup_time.py <run dir> [<run dir> ...] [--plot out.png]

Uses only what runs already keep:
  speedups     champion.md history lines "<id> promoted" (minus ids later marked REVERTED) that
               raised the best tok/s measured so far
  when         the end of the GPU job that measured it: "job N" in its champion.md line, else in its
               record's performance line; else the earliest results.json under
               bank/<id>/{eval,unofficial}/ (approximate: a copy can land later)
  GPU time     gpu/jobs/*/job.json (gpu_minutes, ended): GPU-min spent up to that moment
  measurement  median GPU-min of the run's promotion-grade measurement jobs (labels eval:* in the
               review-eval design, unofficial:* in the earlier v4 design), ignoring jobs < 2 min
               (startup failures)
"""
import json, re, statistics, sys
from datetime import datetime, timezone
from pathlib import Path


def jobs_of(run):
    out = []
    for f in (run / "gpu" / "jobs").glob("*/job.json"):
        j = json.loads(f.read_text())
        if j.get("gpu_minutes") and j.get("ended"):
            j["_end"] = datetime.fromisoformat(j["ended"])
            out.append(j)
    return sorted(out, key=lambda j: j["_end"])


def promotions(run):
    """[(id, job number or None)] in promotion order."""
    hist = (run / "champion.md").read_text()
    reverted = set(re.findall(r"^- \S+ (\S+) REVERTED", hist, re.M))
    out = []
    for m in re.finditer(r"^- \S+ (\S+) promoted(.*)$", hist, re.M):
        rid = m.group(1)
        if rid in reverted or rid in [o[0] for o in out]:
            continue
        job = re.search(r"\bjob (\d+)", m.group(2))
        if not job:
            rec = run / "bank" / rid / "record.md"
            perf = re.search(r"^performance:(.*(?:\n\s+.*)*)", rec.read_text(), re.M) if rec.exists() else None
            job = re.search(r"\bjob (\d+)", perf.group(1)) if perf else None
        out.append((rid, int(job.group(1)) if job else None))
    return out


def measured_at(run, rid, job, jobs):
    for j in jobs:
        if job is not None and int(j["id"]) == job:
            return j["_end"]
    files = [p for k in ("eval", "unofficial") for p in (run / "bank" / rid / k).glob("**/results.json")]
    if not files:
        return None
    return datetime.fromtimestamp(min(p.stat().st_mtime for p in files), timezone.utc)


def tps(run, rid):
    for k in ("eval", "unofficial"):
        for p in sorted((run / "bank" / rid / k).glob("*/results.json")) + sorted((run / "bank" / rid / k).glob("*/*/results.json")):
            v = json.loads(p.read_text()).get("output_throughput")
            if v:
                return v
    return None


def analyse(run):
    jobs = jobs_of(run)
    t0 = min(datetime.fromisoformat(j["started"]) for j in jobs if j.get("started"))
    labels = ("eval:",) if any((j.get("label") or "").startswith("eval:") for j in jobs) else ("unofficial",)
    meas = [j["gpu_minutes"] for j in jobs if (j.get("label") or "").startswith(labels) and j["gpu_minutes"] >= 2]
    m_cost = statistics.median(meas) if meas else float("nan")
    base = tps(run, "BASE") or 635.9
    pts = [(0.0, 0.0, "BASE", base)]
    for rid, job in promotions(run):
        t = measured_at(run, rid, job, jobs)
        if t is None:
            continue
        gpu = sum(j["gpu_minutes"] for j in jobs if j["_end"] <= t)
        pts.append((gpu, (t - t0).total_seconds() / 60, rid, tps(run, rid)))
    pts.sort()
    # a speedup = a promotion that raised the best result measured so far (drops e.g. a record
    # promoted first but measured after a faster one, which the next promotion replaced)
    front, best = [pts[0]], pts[0][3]
    for p in pts[1:]:
        if p[3] and p[3] > best:
            front.append(p); best = p[3]
    return front, m_cost, len(meas)


def main():
    args = sys.argv[1:]
    plot = None
    if "--plot" in args:
        i = args.index("--plot"); plot = args[i + 1]; del args[i:i + 2]
    series = []
    for run in map(Path, args):
        pts, m_cost, n = analyse(run)
        k = len(pts) - 1
        print(f"== {run.name}: {k} speedups (promotions); full measurement = {m_cost:.1f} GPU-min (median of {n})")
        for gpu, wall, rid, v in pts[1:]:
            print(f"   {gpu:6.1f} GPU-min  {wall:6.1f} wall-min  {rid:8s} {v or float('nan'):7.1f} tok/s")
        if k:
            g, w = pts[-1][0] / k, pts[-1][1] / k
            print(f"   per speedup: {g:.1f} GPU-min, {w:.1f} wall-min; measurement/speedup = {m_cost / g:.0%}")
        series.append((run.name, pts, m_cost))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#2a6fdb", "#e0782b", "#3a9a5b", "#9b59b6"]
        for (name, pts, m), c in zip(series, colors):
            xs = [p[0] for p in pts]; ys = [p[3] or float("nan") for p in pts]
            ax.step(xs, ys, where="post", color=c, lw=2, label=f"{name} ({m / (pts[-1][0] / max(1, len(pts) - 1)):.0%} of ideal)")
            ax.plot(xs, ys, "o", color=c, ms=5)
            for x, y, p in zip(xs[1:], ys[1:], pts[1:]):
                ax.annotate(p[2], (x, y), textcoords="offset points", xytext=(3, 4), fontsize=8, color=c)
        ax.set_xlabel("GPU-minutes spent")
        ax.set_ylabel("champion tok/s (c=16)")
        ax.set_title("Time to a speedup: champion promotions vs GPU time")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(plot, dpi=130)
        print("plot:", plot)


if __name__ == "__main__":
    main()
