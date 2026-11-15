#!/usr/bin/env python3
"""Compare GPU efficiency of runs: where GPU time goes, and champion tok/s vs GPU-minutes spent.
usage: python track_gpu.py <run dir> [<run dir> ...]
Categories (from each job's command, label and the scripts it runs):
  official   harness/eval.sh with default settings (official_eval)
  unofficial unofficial_eval jobs (v4)
  abba       A/B jobs (abba.sh, directly or inside the job's script)
  profile    nsys / ncu
  other      everything else (microbenchmarks, smoke tests, debugging)
Champion curve: v3 = time of each champion's official eval job; v4 = champion.json history
(HEAD promotions and confirmations)."""
import json, re, sys
from datetime import datetime
from pathlib import Path


def script_text(cmd, cwd):
    out = " ".join(cmd)
    for tok in cmd:
        for p in (Path(tok), Path(cwd) / tok):
            try:
                if p.is_file() and p.stat().st_size < 200_000:
                    out += "\n" + p.read_text(errors="replace")
            except OSError:
                pass
    return out


def category(j):
    label = j.get("label") or ""
    if label.startswith("unofficial"):
        return "unofficial"
    t = script_text(j["command"], j.get("cwd", "."))
    if "abba.sh" in t:
        return "abba"
    if "harness/eval.sh" in " ".join(j["command"]):
        return "official"
    if re.search(r"\bnsys\b|\bncu\b|profile_n(sys|cu)", t):
        return "profile"
    return "other"


def ts(s):
    return datetime.fromisoformat(s)


def main():
    for run in map(Path, sys.argv[1:]):
        jobs = []
        for f in (run / "gpu" / "jobs").glob("*/job.json"):
            j = json.loads(f.read_text())
            if j.get("gpu_minutes") and j.get("ended"):
                jobs.append(j)
        jobs.sort(key=lambda j: j["ended"])
        tot = sum(j["gpu_minutes"] for j in jobs)
        print(f"== {run.name}: {len(jobs)} jobs, {tot:.0f} GPU-min")
        cats = {}
        for j in jobs:
            c = category(j)
            n, m = cats.get(c, (0, 0.0))
            cats[c] = (n + 1, m + j["gpu_minutes"])
        for c, (n, m) in sorted(cats.items(), key=lambda x: -x[1][1]):
            print(f"  {c:10s} {n:4d} jobs {m:7.1f} GPU-min ({100 * m / tot:4.1f}%), {m / n:5.1f} per job")

        def cum_at(t):
            return sum(j["gpu_minutes"] for j in jobs if ts(j["ended"]) <= t)
        print("  champion curve (GPU-min spent when it became champion -> tok/s):")
        cj = run / "champion.json"
        if cj.exists():   # v4
            c = json.loads(cj.read_text())
            for h in c["history"]:
                m = re.match(r"(\S+T\S+|\d{4}-\d\d-\d\dT\d\d:\d\d): (.*)", h)
                if m:
                    t = ts(m.group(1) + (":00" if len(m.group(1)) == 16 else ""))
                    if t.tzinfo is None:
                        t = t.astimezone()
                    print(f"    {cum_at(t):6.0f}  {m.group(2)[:150]}")
        else:             # v3: champions from champion.md, time = their official eval job's end
            ids = re.findall(r"^- \S+: (\S+) \(([\d.]+) tok/s", (run / "champion.md").read_text(), re.M)
            for rid, tps in reversed(ids):
                ends = [j for j in jobs if f"bank/{rid}/official" in " ".join(j["command"])]
                if ends:
                    print(f"    {cum_at(ts(ends[-1]['ended'])):6.0f}  {rid} {tps} tok/s (official)")


if __name__ == "__main__":
    main()
