"""Resources a run actually used: tokens per agent/role, wall clock, GPU-minutes.

usage: python3 resources.py <run dir> [--transcripts DIR]   (default DIR: ~/.claude/projects)
Reads only. Tokens come from each agent's Claude Code transcript (<session_id>.jsonl, found via
agents/*/meta.json); transcripts in the run's project folders that no meta.json names are listed as
"unregistered"; a hand-started manager runs in the flavor folder, so its transcript
is the flavor folder's transcript that starts within 10 min of agents/manager/meta.json "started".
Each agent's tokens and hours count only its meta.json started..stopped window. Assistant messages are deduplicated by message id.
"""
import argparse, collections, glob, json, os, re
from datetime import datetime


def usage_of(path, t0=None, t1=None):
    """Sum token usage over unique assistant messages in one transcript (+ its subagents),
    only messages within [t0, t1] when given (ISO timestamps)."""
    files = [path] + sorted(glob.glob(os.path.join(path[:-len(".jsonl")], "subagents", "*.jsonl")))
    msgs, first, last = {}, None, None
    for f in files:
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if ts and ((t0 and iso(ts) < iso(t0)) or (t1 and iso(ts) > iso(t1))):
                continue
            if ts:
                first = min(first or ts, ts); last = max(last or ts, ts)
            m = d.get("message") if isinstance(d.get("message"), dict) else None
            if d.get("type") == "assistant" and m and m.get("usage"):
                msgs[m.get("id") or id(d)] = (m.get("model"), m["usage"])
    tot = collections.Counter()
    models = collections.Counter()
    for model, u in msgs.values():
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            tot[k] += u.get(k) or 0
        models[model] += 1
    return tot, first, last, models


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours(a, b):
    if not a or not b:
        return None
    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (f(b) - f(a)).total_seconds() / 3600


def fmt(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--transcripts", default=os.path.expanduser("~/.claude/projects"))
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    enc = re.sub(r"[^A-Za-z0-9]", "-", run)
    proj_dirs = [d for d in glob.glob(os.path.join(a.transcripts, "*"))
                 if os.path.basename(d) == enc or os.path.basename(d).startswith(enc + "-")]

    rows, known = [], set()
    for mf in sorted(glob.glob(os.path.join(run, "agents", "*", "meta.json"))):
        m = json.load(open(mf))
        # an agent restarted with new sessions (baseline flavor: new_run --resume) lists them all
        sids = [s["session_id"] for s in m.get("sessions", [])] or [m.get("session_id")]
        for i, sid in enumerate(sids):
            hits = glob.glob(os.path.join(a.transcripts, "*", f"{sid}*.jsonl")) if sid else []
            if sid:
                known.add(sid)
            name = m["name"] if len(sids) == 1 else f"{m['name']}#{i + 1}"
            rows.append((name, m.get("role", "?"), m.get("started"), m.get("stopped"), hits[0] if hits else None))
    for d in proj_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            sid = os.path.basename(f)[:-len(".jsonl")]
            if sid not in known:
                rows.append((f"unregistered:{sid[:8]}", "unregistered", None, None, f))
    # a hand-started manager runs in the flavor folder (flavors/<f>/), not the run folder: take the
    # flavor folder's transcripts that mention this run's path
    flavor_dir = os.path.dirname(os.path.dirname(run))
    fenc = os.path.join(a.transcripts, re.sub(r"[^A-Za-z0-9]", "-", flavor_dir))
    mgr = next((r for r in rows if r[1] == "manager" and not r[4]), None)
    ms = os.path.join(run, "manager_session.json")      # v1 new_run records the manager's session id
    if mgr and os.path.exists(ms):
        sid = json.load(open(ms))["session_id"]
        hits = glob.glob(os.path.join(a.transcripts, "*", f"{sid}*.jsonl"))
        if hits:
            known.add(sid)
            rows = [r for r in rows if r[4] != hits[0]]
            rows[rows.index(mgr)] = mgr = (mgr[0], "manager", mgr[2], mgr[3], hits[0])
    if mgr and mgr[2]:
        for f in sorted(glob.glob(os.path.join(fenc, "*.jsonl"))):
            sid = os.path.basename(f)[:-len(".jsonl")]
            _, first, _, _ = usage_of(f)
            # the manager's transcript starts within minutes of its registration (meta "started")
            if sid not in known and first and abs((iso(first) - iso(mgr[2])).total_seconds()) < 600:
                rows[rows.index(mgr)] = mgr = (mgr[0], "manager", mgr[2], mgr[3], f)
                break

    print(f"# Resources used: {run}\n")
    print("## Tokens (unique assistant messages; cache reads are cheap but counted separately)")
    print(f"{'agent':32} {'role':13} {'msgs':>5} {'input':>8} {'cache_wr':>9} {'cache_rd':>9} {'output':>8}  hours  model")
    by_role = collections.defaultdict(collections.Counter)
    t_first, t_last = None, None
    for name, role, started, stopped, f in rows:
        if not f:
            print(f"{name:32} {role:13} (no transcript found)")
            continue
        tot, first, last, models = usage_of(f, started, stopped)   # e.g. a manager resumed after the run is not counted
        by_role[role].update(tot); by_role["ALL"].update(tot)
        by_role[role]["msgs"] += sum(models.values()); by_role["ALL"]["msgs"] += sum(models.values())
        if role != "unregistered":
            t_first = min(t_first or started or first, started or first)
            t_last = max(t_last or stopped or last, stopped or last)
        h = hours(started or first, stopped or last)
        print(f"{name:32} {role:13} {sum(models.values()):5} {fmt(tot['input_tokens']):>8} "
              f"{fmt(tot['cache_creation_input_tokens']):>9} {fmt(tot['cache_read_input_tokens']):>9} "
              f"{fmt(tot['output_tokens']):>8}  {h if h is None else round(h, 2)}  "
              f"{','.join(k or '?' for k in models)}")
    print("\nPer role:")
    for role, c in sorted(by_role.items()):
        print(f"  {role:13} msgs {c['msgs']:5}  input {fmt(c['input_tokens'])}  cache_wr {fmt(c['cache_creation_input_tokens'])}"
              f"  cache_rd {fmt(c['cache_read_input_tokens'])}  output {fmt(c['output_tokens'])}")

    print(f"\n## Wall clock (registered agents' start/stop): {t_first} -> {t_last}, "
          f"{hours(t_first, t_last) and round(hours(t_first, t_last), 2)} h")

    led = os.path.join(run, "gpu", "ledger.jsonl")
    print("\n## GPU-minutes (gpu/ledger.jsonl)")
    if not os.path.exists(led):
        print("  no ledger")
        return
    g = collections.Counter()
    for line in open(led):
        d = json.loads(line)
        kind = ("official" if str(d.get("label", "")).startswith("official") else
                "brainstorm/uncharged" if not d.get("charged") else "unofficial")
        g[(d.get("round"), d.get("team") or "-", kind)] += d.get("gpu_minutes") or 0
        g[("ALL", "", kind)] += d.get("gpu_minutes") or 0
    for (r, team, kind), v in sorted(g.items(), key=lambda x: str(x[0])):
        print(f"  round {r!s:4} team {team:3} {kind:22} {v:7.1f}")
    print(f"  total {sum(v for (r, _, _), v in g.items() if r == 'ALL'):.1f}")


if __name__ == "__main__":
    main()
