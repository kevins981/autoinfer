"""Per-agent activity intervals from Claude transcripts + the GPU job queue.
usage: python activity.py <run dir> <out.json> [agent=session_id ...]
Session ids come from agents/*/meta.json (and runs/<run>/manager_session.json for the v1 manager);
pass agent=<id> to fill in or override one. Prints any agent whose transcript it can't find.
States (one per moment per agent, while the agent's session is alive):
  llm      the model is generating (request sent -> response)
  tools    a tool call is running (reading, editing, shell commands, messages)
  build    a shell command that builds code
  gpu_q    waiting for the GPU: own job submitted but not started (queue)
  gpu_run  waiting on its own GPU job (idle or blocked while it runs)
  idle     turn ended, waiting for a message / notification / teammates
"""
import glob, json, os, sys
from datetime import datetime

run = sys.argv[1].rstrip("/")
out = sys.argv[2]
override = dict(a.split("=", 1) for a in sys.argv[3:])
P = os.path.expanduser("~/.claude/projects")
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def transcript(sid):
    hits = glob.glob(f"{P}/*/{sid}*.jsonl")
    return hits[0] if hits else None


def intervals(path):
    """(start, end, state) from one transcript."""
    ev = []
    for line in open(path):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        t = m.get("timestamp")
        if not t:
            continue
        typ, msg = m.get("type"), m.get("message") or {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        kinds = [c.get("type") for c in content]
        if typ == "assistant":
            cmd = ""
            for c in content:
                if c.get("type") == "tool_use":
                    i = c.get("input", {})
                    cmd = f"{c.get('name')} {i.get('command', '')}"
            ev.append((ts(t), "A", msg.get("stop_reason"), cmd))
        elif typ == "user":
            ev.append((ts(t), "U", "tool_result" in kinds, ""))
        elif typ == "api-request":
            ev.append((ts(t), "R", None, ""))
    ev.sort(key=lambda e: e[0])
    res, state, since = [], "llm", ev[0][0] if ev else 0
    for t, kind, x, cmd in ev:
        if t > since:
            res.append((since, t, state))
        since = t
        if kind in ("U", "R"):
            state = "llm"
        elif kind == "A":
            if x == "tool_use":
                state = "build" if (" build" in cmd or "ws build" in cmd or "cmake" in cmd) else "tools"
            elif x == "end_turn":
                state = "idle"
            else:
                state = "llm"
    return res, (ev[0][0], ev[-1][0]) if ev else (0, 0)


agents = {}
ms = f"{run}/manager_session.json"
if os.path.exists(ms) and "manager" not in override:
    d = json.load(open(ms))
    override["manager"] = d.get("session_id") or d.get("id") or ""
for meta in sorted(glob.glob(f"{run}/agents/*/meta.json")):
    name = os.path.basename(os.path.dirname(meta))
    sid = override.get(name) or json.load(open(meta)).get("session_id")
    path = transcript(sid) if sid else None
    if not path:
        print(f"{name}: no transcript ({sid})")
        continue
    ivs, span = intervals(path)
    agents[name] = {"span": span, "intervals": ivs}

# GPU jobs per agent: queue + run windows (job.json has the submit time; else the ledger, no queue)
jobs = []
if not glob.glob(f"{run}/gpu/jobs/*/job.json") and os.path.exists(f"{run}/gpu/ledger.jsonl"):
    for line in open(f"{run}/gpu/ledger.jsonl"):
        e = json.loads(line)
        jobs.append({"agent": e["agent"], "label": e.get("label"), "id": e["job"],
                     "submitted": ts(e["start"]), "started": ts(e["start"]), "ended": ts(e["end"])})
for jf in glob.glob(f"{run}/gpu/jobs/*/job.json"):
    j = json.load(open(jf))
    if j.get("started") and j.get("ended"):
        jobs.append({"agent": j["agent"], "label": j.get("label"), "id": j["id"],
                     "submitted": ts(j["submitted"]), "started": ts(j["started"]), "ended": ts(j["ended"])})


def overlay(name, a, b, state):
    """Split an idle/tools interval by the agent's own queued/running GPU jobs."""
    if state not in ("idle", "tools"):
        return [(a, b, state)]
    cuts = []
    for j in jobs:
        if j["agent"] != name:
            continue
        cuts += [(j["submitted"], j["started"], "gpu_q"), (j["started"], j["ended"], "gpu_run")]
    out, cur = [], a
    pts = sorted({a, b, *[x for c in cuts for x in c[:2] if a < x < b]})
    for s, e in zip(pts, pts[1:]):
        mid = (s + e) / 2
        st = next((c[2] for c in cuts if c[0] <= mid < c[1]), state)
        out.append((s, e, st))
    return out


for name, d in agents.items():
    d["intervals"] = [x for a, b, s in d["intervals"] for x in overlay(name, a, b, s)]
json.dump({"agents": agents, "jobs": jobs}, open(out, "w"))
for name, d in agents.items():
    tot = {}
    for a, b, s in d["intervals"]:
        tot[s] = tot.get(s, 0) + (b - a) / 60
    print(f"{name:28} " + "  ".join(f"{k} {v:5.1f}" for k, v in sorted(tot.items())))
