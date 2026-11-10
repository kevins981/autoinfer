#!/usr/bin/env python3
"""dashboard: a read-only web view of AutoInfer runs, for the human.

Everything is read live from the run folders (no agent writes anything for it):
  teams      gpu/accounts.jsonl (open/close/grants), gpu/ledger.jsonl (GPU used), teams/<T>.md,
             team_slots.json, proposals/<T>-*.md, reports/<T>-*.md, inspiration/<T>-*.md, bank/<T>-*/
  agents     agents/<name>/meta.json, events.jsonl (their activity), inbox.jsonl, local_kb/<name>.md,
             agents/<name>/ (their files); `agent list --json --all` for live state and cost
  manager    its Claude Code transcript (~/.claude/projects/<run>/*.jsonl, found by its first message)
  run        champion.md, journal.md, shared_kb.md, announcements.md, config.md, `gpu status`
Any other file in the run folder: the Files view.

usage: python3 flavors/v2/tools/dashboard.py [--port 8765] [--host 127.0.0.1]
From your own computer:  ssh -N -L 8765:localhost:8765 <user>@<server>   then open http://localhost:8765
"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # autoinfer/
PROJECTS = Path.home() / ".claude" / "projects"
MAX_FILE = 2_000_000
_cache = {}


def cached(key, ttl, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def runs():
    out = []
    for r in sorted(glob.glob(str(ROOT / "flavors" / "*" / "runs" / "*"))):
        p = Path(r)
        if p.is_dir() and (p / "agents").is_dir():
            out.append({"id": f"{p.parents[1].name}/{p.name}", "mtime": p.stat().st_mtime})
    out.sort(key=lambda x: -x["mtime"])
    return out


def run_path(run_id):
    flavor, name = run_id.split("/", 1)
    p = (ROOT / "flavors" / flavor / "runs" / name).resolve()
    if not str(p).startswith(str(ROOT / "flavors")) or not p.is_dir():
        raise ValueError("no such run")
    return p


def read(p, limit=MAX_FILE):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            data = f.read()
        txt = data.decode("utf-8", "replace")
        if size > limit:
            txt = f"[... showing the last {limit // 1000} KB of {size // 1000} KB ...]\n" + txt
        return txt
    except OSError:
        return None


def jsonl(p):
    out = []
    try:
        for line in open(p, errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def tool(run, *args):
    t = run.parents[1] / "tools" / args[0]
    if not t.exists():
        return ""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTOINFER_")}
    env["AUTOINFER_RUN"] = str(run)
    try:
        r = subprocess.run([str(t), *args[1:]], env=env, capture_output=True, text=True, timeout=20)
        return r.stdout
    except Exception as e:  # noqa
        return f"({e})"


def agent_list(run):
    def f():
        try:
            return json.loads(tool(run, "agent", "list", "--json", "--all") or "[]")
        except ValueError:
            return []
    return cached(("agents", str(run)), 5, f)


def team_of(name):
    m = re.match(r"(T\d+|MGR)", name)
    return m.group(1) if m else None


def teams(run):
    acc = jsonl(run / "gpu" / "accounts.jsonl")
    led = jsonl(run / "gpu" / "ledger.jsonl")
    slots = {}
    try:
        slots = json.loads((run / "team_slots.json").read_text())
    except (OSError, ValueError):
        pass
    T = {}
    for e in acc:
        t = e.get("team")
        if not t or t == "MGR":
            continue
        d = T.setdefault(t, {"team": t, "type": None, "status": "open", "opened": None, "closed": None,
                             "granted": 0.0, "used": 0.0, "jobs": 0, "note": None})
        if e["event"] == "open":
            d["type"], d["opened"] = e.get("type"), e["t"]
        elif e["event"] == "close":
            d["status"], d["closed"], d["note"] = "closed", e["t"], e.get("note")
        elif e["event"] == "grant":
            d["granted"] += e.get("minutes", 0)
    for r in led:
        d = T.get(r.get("team"))
        if d and r.get("charged", True):
            d["used"] += r.get("gpu_minutes", 0)
            d["jobs"] += 1
    ag = agent_list(run)
    for d in T.values():
        t = d["team"]
        d["slot"] = slots.get(t)
        d["agents"] = [a for a in ag if a.get("team") == t]
        d["cost"] = round(sum(a.get("cost_usd") or 0 for a in d["agents"]), 2)
        d["records"] = sorted(os.path.basename(p) for p in glob.glob(str(run / "bank" / f"{t}-*")))
        d["proposals"] = sorted(os.path.basename(p) for p in glob.glob(str(run / "proposals" / f"{t}-*")))
        lead = [a for a in d["agents"] if a.get("role", "").startswith("leader")]
        d["leader"] = lead[-1]["name"] if lead else None
        d["models"] = sorted({meta_model(run, a["name"]) for a in d["agents"]} - {None})
        d["alive"] = sum(a.get("state") == "alive" for a in d["agents"])
    return sorted(T.values(), key=lambda d: -int(d["team"][1:]))


def meta_model(run, name):
    try:
        return json.loads((run / "agents" / name / "meta.json").read_text()).get("model")
    except (OSError, ValueError):
        return None


def performance(rec):
    txt = read(rec / "record.md", 20000) or ""
    m = re.search(r'^performance:\s*"?(.*?)"?\s*$', txt, re.M)
    return m.group(1) if m else ""


def official(rec):
    best = None
    for p in glob.glob(str(rec / "official" / "*" / "results.json")):
        try:
            r = json.loads(open(p).read())
            if r.get("output_throughput"):
                best = max(best or 0, r["output_throughput"])
        except (OSError, ValueError):
            pass
    return best


def team_detail(run, t):
    d = next((x for x in teams(run) if x["team"] == t), {"team": t})
    files = {}
    for kind, pat in (("track", f"teams/{t}.md"), ("proposals", f"proposals/{t}-*.md"),
                      ("reports", f"reports/{t}-*.md"), ("inspiration", f"inspiration/{t}-*.md"),
                      ("scout", f"scout/{t}-*/*.md")):
        files[kind] = [str(Path(p).relative_to(run)) for p in sorted(glob.glob(str(run / pat)))]
    recs = []
    for p in sorted(glob.glob(str(run / "bank" / f"{t}-*"))):
        rp = Path(p)
        recs.append({"id": rp.name, "performance": performance(rp), "official": official(rp),
                     "path": str(rp.relative_to(run) / "record.md")})
    draws = [x for x in jsonl(run / "inspiration" / "draws.jsonl") if x.get("team") == t]
    d.update({"files": files, "bank": recs, "draws": draws})
    return d


def enc(path):
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def transcripts(run, name):
    """Claude Code transcripts of an agent, oldest first: the manager's sessions (manager_session*.json
    record their ids), or a headless agent's session (meta.json: cwd + session_id)."""
    def f():
        out = []
        if name == "manager":
            for sf in sorted(glob.glob(str(run / "manager_session*.json"))):
                try:
                    sid = json.loads(open(sf).read())["session_id"]
                except (OSError, ValueError, KeyError):
                    continue
                out += glob.glob(str(PROJECTS / enc(run) / f"{sid}*.jsonl"))
        else:
            try:
                m = json.loads((run / "agents" / name / "meta.json").read_text())
                p = PROJECTS / enc(m["cwd"]) / f"{m['session_id']}.jsonl"
                if p.exists():
                    out.append(str(p))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        out = sorted(set(out), key=os.path.getmtime)
        return out
    return cached(("tr", str(run), name), 30, f)


def transcript_cost(path):
    """The session's own running cost (cost-state lines), if it has written one."""
    cost = None
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 3_000_000))
            for line in fh:
                if b'"cost-state"' in line:
                    try:
                        cost = json.loads(line).get("totalCostUSD")
                    except ValueError:
                        pass
    except OSError:
        pass
    return cost


def agent_detail(run, name):
    a = next((x for x in agent_list(run) if x["name"] == name), {"name": name})
    meta = {}
    try:
        meta = json.loads((run / "agents" / name / "meta.json").read_text())
    except (OSError, ValueError):
        pass
    notes = run / "local_kb" / f"{name}.md"
    d = run / "agents" / name
    files = []
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(d)
            if p.is_file() and len(rel.parts) <= 4 and p.name not in ("events.jsonl", "meta.json"):
                files.append({"path": str(p.relative_to(run)), "size": p.stat().st_size,
                              "mtime": p.stat().st_mtime})
            if len(files) >= 400:
                break
    files.sort(key=lambda x: -x["mtime"])
    sources = []
    tr = transcripts(run, name)
    for i in range(len(tr) - 1, -1, -1):
        sources.append({"id": f"tr{i}", "label": "transcript" if len(tr) == 1 else f"session {i + 1}"})
    if (d / "events.jsonl").exists():
        sources.append({"id": "events", "label": "raw stream"})
    if a.get("cost_usd") is None and tr:
        a["cost_usd"] = sum(c for c in (transcript_cost(p) for p in tr) if c) or None
    return {"agent": a, "meta": meta, "notes": str(notes.relative_to(run)) if notes.exists() else None,
            "files": files, "sources": sources,
            "inbox": str((d / "inbox.jsonl").relative_to(run)) if (d / "inbox.jsonl").exists() else None}


def clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, indent=1)
    return s if len(s) <= n else s[:n] + f"\n[... {len(s) - n} more characters]"


def result_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for x in c:
            if isinstance(x, dict):
                if x.get("type") == "text":
                    parts.append(x.get("text", ""))
                elif x.get("type") == "tool_reference":
                    parts.append(f"(loaded tool {x.get('tool_name')})")
                elif x.get("type") == "image":
                    parts.append("(image)")
        return "\n".join(parts)
    return json.dumps(c)


XS = re.compile(r'<cross-session-message[^>]*?from-name="([^"]*)"[^>]*>(.*?)</cross-session-message>', re.S)


def incoming(text):
    """(sender, body) of an incoming message; cross-session wrappers unpacked."""
    m = XS.search(text)
    if m:
        return m.group(1), m.group(2).strip()
    m = re.match(r"\[(?:from|announcement from) ([^;\]]+)", text)
    return (m.group(1) if m else None), text


def parse_events(lines):
    """Raw stream-json / transcript lines -> readable items."""
    items = []
    seen_in = set()

    def add_in(ln, ts, text):
        text = text.strip()
        if not text or text.startswith("<system-reminder>") or text.startswith("<local-command") \
                or text.startswith("<command-") or text.startswith("<task-notification>"):
            return
        key = hash(text)
        if key in seen_in:
            return
        seen_in.add(key)
        who, body = incoming(text)
        items.append({"k": "in", "ln": ln, "ts": ts, "from": who, "text": clip(body, 30000)})

    for ln, e in lines:
        t = e.get("type")
        ts = e.get("timestamp")
        m = e.get("message") if isinstance(e.get("message"), dict) else {}
        if t == "assistant":
            for b in m.get("content") or []:
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    items.append({"k": "text", "ln": ln, "ts": ts, "text": b["text"]})
                elif bt == "tool_use":
                    inp = b.get("input") or {}
                    items.append({"k": "tool", "ln": ln, "ts": ts, "id": b.get("id"), "name": b.get("name"),
                                  "input": clip(inp, 20000)})
        elif t == "user":
            if e.get("isMeta") and (e.get("origin") or {}).get("kind") != "peer":
                continue
            c = m.get("content")
            if isinstance(c, str):
                add_in(ln, ts, c)
            elif isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result":
                        items.append({"k": "res", "ln": ln, "id": b.get("tool_use_id"),
                                      "err": bool(b.get("is_error")), "text": clip(result_text(b.get("content")), 8000)})
                    elif b.get("type") == "text":
                        add_in(ln, ts, b.get("text", ""))
        elif t == "attachment":
            at = e.get("attachment") or {}
            if at.get("type") == "queued_command" and isinstance(at.get("prompt"), str):
                add_in(ln, ts, at["prompt"])
        elif t == "result":
            items.append({"k": "turn", "ln": ln, "cost": e.get("total_cost_usd"),
                          "dur": round((e.get("duration_ms") or 0) / 1000), "err": bool(e.get("is_error")),
                          "text": clip(str(e.get("result") or ""), 300) if e.get("is_error") else ""})
        elif t == "system":
            st = e.get("subtype")
            if st == "init":
                items.append({"k": "sys", "ln": ln, "text": f"session started (model {e.get('model')})"})
            elif st == "task_notification":
                items.append({"k": "sys", "ln": ln, "text": f"background command {e.get('status')}: {clip(e.get('summary') or '', 200)}"})
            elif st == "compact_boundary":
                items.append({"k": "sys", "ln": ln, "text": "context compacted"})
        elif t == "rate_limit_event":
            info = e.get("rate_limit_info") or {}
            if info.get("status") not in (None, "allowed", "allowed_warning"):
                items.append({"k": "sys", "ln": ln, "text": f"rate limit: {info.get('status')}"})
    return items


def events(run, name, src, start, limit):
    if src == "events":
        path = run / "agents" / name / "events.jsonl"
    elif src.startswith("tr"):
        path = Path(transcripts(run, name)[int(src[2:])])
    else:
        raise ValueError("bad source")
    lines = []
    try:
        with open(path, errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError:
        all_lines = []
    n = len(all_lines)
    if start is None or start < 0:
        start = max(0, n - limit)
    for i in range(start, min(n, start + limit)):
        try:
            lines.append((i, json.loads(all_lines[i])))
        except ValueError:
            pass
    return {"items": parse_events(lines), "start": start, "end": min(n, start + limit), "total": n}


def safe(run, rel):
    p = (run / rel).resolve()
    if not str(p).startswith(str(run.resolve())):
        raise ValueError("outside the run")
    return p


def overview(run):
    champ = read(run / "champion.md", 20000) or ""
    led = jsonl(run / "gpu" / "ledger.jsonl")
    return {"champion": champ, "gpu": tool(run, "gpu", "status"), "teams": teams(run),
            "config": read(run / "config.md", 20000), "jobs": len(led),
            "manager": next((a for a in agent_list(run) if a["name"] == "manager"), None),
            "scouts": scout_decks(run)}


def gpu_jobs(run):
    led = jsonl(run / "gpu" / "ledger.jsonl")[-300:]
    led.reverse()
    return {"status": tool(run, "gpu", "status"), "jobs": led}


def tree(run, rel):
    p = safe(run, rel or ".")
    out = []
    for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
        try:
            st = c.stat()
        except OSError:
            continue
        out.append({"name": c.name, "dir": c.is_dir(), "size": st.st_size, "mtime": st.st_mtime,
                    "path": str(c.relative_to(run))})
    return {"path": str(p.relative_to(run)) if p != run else "", "entries": out[:2000]}



# ---------------- timeline (champion steps, GPU use, per-agent activity) ----------------
from datetime import datetime

_iv_cache = {}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _intervals(path):
    """(start, end, state) from one Claude transcript (same states as the eval skill's activity.py):
    llm = model generating, tools = a tool call running, build = a build command, idle = turn ended.
    Cached per file (size, mtime): only growing transcripts are re-read."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_size, st.st_mtime)
    hit = _iv_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    ev = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"timestamp"' not in line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue
            t, typ = m.get("timestamp"), m.get("type")
            if not t or typ not in ("assistant", "user", "api-request"):
                continue
            msg = m.get("message") if isinstance(m.get("message"), dict) else {}
            content = msg.get("content") if isinstance(msg.get("content"), list) else []
            if typ == "assistant":
                cmd = ""
                for c in content:
                    if c.get("type") == "tool_use":
                        cmd = f"{c.get('name')} {(c.get('input') or {}).get('command', '')}"
                ev.append((_ts(t), "A", "error" if m.get("isApiErrorMessage") else msg.get("stop_reason"), cmd))
            else:
                ev.append((_ts(t), "U", None, ""))
    ev.sort(key=lambda e: e[0])
    res, state, since = [], "llm", ev[0][0] if ev else 0
    for t, kind, x, cmd in ev:
        if t > since:
            res.append([since, t, state])
        since = t
        if kind == "U":
            state = "llm"
        elif x == "tool_use":
            state = "build" if (" build" in cmd or "ws build" in cmd or "cmake" in cmd) else "tools"
        elif x == "end_turn":
            state = "idle"
        elif x == "error":
            state = "stalled"
        else:
            state = "llm"
    _iv_cache[path] = (key, res)
    return res


def _split_by_jobs(ivs, jobs):
    """Idle/tools time while the agent's own GPU job is queued or running -> gpu_q / gpu_run."""
    if not jobs:
        return ivs
    out = []
    for a, b, st in ivs:
        if st not in ("idle", "tools"):
            out.append([a, b, st])
            continue
        cuts = [(j[0], j[1], "gpu_q") for j in jobs] + [(j[1], j[2], "gpu_run") for j in jobs]
        pts = sorted({a, b, *[x for c in cuts for x in c[:2] if a < x < b]})
        for s0, e0 in zip(pts, pts[1:]):
            mid = (s0 + e0) / 2
            out.append([s0, e0, next((c[2] for c in cuts if c[0] <= mid < c[1]), st)])
    return out


def _merge(ivs, t0):
    out = []
    for a, b, st in ivs:
        if b <= t0:
            continue
        a = max(a, t0)
        if out and out[-1][2] == st and a - out[-1][1] < 1:
            out[-1][1] = b
        else:
            out.append([a, b, st])
    return [[round(a), round(b), st] for a, b, st in out if b - a >= 1]


def timeline(run, since_h):
    now = time.time()
    acc = jsonl(run / "gpu" / "accounts.jsonl")
    start = _ts(acc[0]["t"]) if acc else now - 3600
    t0 = max(start, now - since_h * 3600) if since_h else start
    # GPU jobs
    jobs = {}
    led_end = {r["job"]: _ts(r["end"]) for r in jsonl(run / "gpu" / "ledger.jsonl")}
    live = {int(x) for x in re.findall(r"RUNNING\s+job (\d+)", tool(run, "gpu", "status"))}
    for jf in glob.glob(str(run / "gpu" / "jobs" / "*" / "job.json")):
        try:
            j = json.loads(open(jf).read())
        except (OSError, ValueError):
            continue
        if not j.get("started"):
            continue
        if j.get("ended"):
            end = _ts(j["ended"])
        elif j.get("id") in led_end:
            end = led_end[j["id"]]
        elif j.get("id") in live:
            end = now
        else:
            continue                 # left marked running by a killed process: no end known
        jobs.setdefault(j.get("agent"), []).append((_ts(j["submitted"]), _ts(j["started"]), end, j.get("id"), j.get("team")))
    led = sorted(jsonl(run / "gpu" / "ledger.jsonl"), key=lambda r: r["end"])
    gpu, tot = [[round(start), 0]], 0.0
    for r in led:
        tot += r.get("gpu_minutes", 0)
        gpu.append([round(_ts(r["end"])), round(tot, 1)])
    # champion steps from champion.md's history line + the current record
    champ = read(run / "champion.md", 20000) or ""
    base = None
    try:
        base = json.loads((run / "bank" / "BASE" / "official" / "1" / "results.json").read_text())["output_throughput"]
    except (OSError, ValueError, KeyError):
        pass
    steps = []
    hist = re.search(r"^History:(.*)$", champ, re.M)
    cur = re.search(r"^record:\s*(\S+)", champ, re.M)
    chain = re.findall(r"\b([A-Z]+\d*-\d+)\s+([\d.]+)", hist.group(1)) if hist else []
    if cur and cur.group(1) not in [c[0] for c in chain]:
        m = re.search(r"official[^:]*:\s*([\d.]+)\s*tok/s", champ)
        if m:
            chain.append((cur.group(1), m.group(1)))
    for rid, val in chain:
        best = None
        for rf in glob.glob(str(run / "bank" / rid / "official" / "*" / "results.json")):
            t = os.path.getmtime(rf)
            best = t if best is None else min(best, t)
        steps.append({"id": rid, "tps": float(val), "t": round(max(best or start, start))})
    # agents
    agents = []
    for meta in sorted(glob.glob(str(run / "agents" / "*" / "meta.json"))):
        name = Path(meta).parent.name
        try:
            m = json.loads(open(meta).read())
        except (OSError, ValueError):
            continue
        ivs = []
        for path in transcripts(run, name):
            ivs += _intervals(path)
        if not ivs:
            continue
        stopped = _ts(m["stopped"]) if m.get("stopped") else None
        if (stopped or ivs[-1][1]) < t0:
            continue
        own = [(a, b, c) for a, b, c, _, _ in jobs.get(name, [])]
        ivs = _merge(_split_by_jobs(sorted(ivs), own), t0)
        if stopped is None and ivs and ivs[-1][2] == "idle":
            ivs[-1][1] = round(now)
        runs_ = [[round(max(b, t0)), round(c), jid] for a, b, c, jid, _ in jobs.get(name, []) if c > t0]
        agents.append({"name": name, "team": m.get("team"), "role": m.get("role"), "model": m.get("model"),
                       "intervals": ivs, "jobs": runs_})
    def order(a):
        t = a["team"] or ""
        return (0 if a["name"] == "manager" else 1, int(t[1:]) if t[1:].isdigit() else 0, a["name"])
    agents.sort(key=order)
    return {"t0": round(t0), "now": round(now), "start": round(start), "base": base, "steps": steps,
            "gpu": gpu, "agents": agents}



# ---------------- scouts (external teams' Codex card writers) ----------------
def scout_decks(run):
    out = []
    for f in sorted(glob.glob(str(run / "scout" / "*.json"))):
        try:
            d = json.loads(open(f).read())
        except (OSError, ValueError):
            continue
        if d.get("state") == "running":
            d["minutes"] = round((time.time() - _ts(d["started"])) / 60, 1)
            d["cards_so_far"] = len(glob.glob(str(Path(d["work"]) / "cards" / "*.md")))
        out.append({k: d.get(k) for k in ("deck", "team", "state", "model", "effort", "cards", "cards_wanted",
                                          "cards_so_far", "minutes", "window", "started", "ended", "cards_dir")})
    return out


def parse_codex(lines):
    """Codex `exec --json` events -> the same item kinds as parse_events (text/tool/res/turn/sys)."""
    items = []
    for ln, line in enumerate(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        typ, it = e.get("type"), e.get("item") or {}
        kind = it.get("type")
        iid = f"c{it.get('id')}"
        if typ == "item.completed" and kind == "agent_message":
            items.append({"k": "text", "text": it.get("text", "")})
        elif typ == "item.completed" and kind == "reasoning":
            t = (it.get("text") or "").strip()
            if t:
                items.append({"k": "sys", "text": "thinking: " + t[:600]})
        elif kind == "web_search" and typ == "item.completed":
            act = it.get("action") or {}
            if act.get("type") == "search":
                q = act.get("queries") or [it.get("query", "")]
                items.append({"k": "tool", "id": iid, "name": "web search", "input": json.dumps({"query": " | ".join(q)})})
            elif act.get("url"):
                name = {"open_page": "open page", "find_in_page": "find in page"}.get(act.get("type"), act.get("type") or "web")
                inp = {"url": act["url"]}
                if act.get("pattern"):
                    inp["query"] = f"{act['pattern']}  ({act['url']})"
                items.append({"k": "tool", "id": iid, "name": name, "input": json.dumps(inp)})
            else:
                doms = sorted({r.get("domain", "") for r in it.get("results") or [] if r.get("domain")})
                items.append({"k": "tool", "id": iid, "name": "web" if not it.get("query") else "web search",
                              "input": json.dumps({"query": it.get("query") or ("(page view) " + ", ".join(doms))})})
            res = [f"{r.get('domain', '')}: {(r.get('title') or r.get('snippet') or '')[:160]}" + (f"  {r['url']}" if r.get("url") else "")
                   for r in (it.get("results") or [])[:12]]
            if res:
                items.append({"k": "res", "id": iid, "text": "\n".join(res)})
        elif kind == "command_execution" and typ == "item.completed":
            items.append({"k": "tool", "id": iid, "name": "shell", "input": json.dumps({"command": it.get("command", "")})})
            items.append({"k": "res", "id": iid, "text": (it.get("aggregated_output") or "")[-4000:],
                          "err": it.get("exit_code") not in (0, None)})
        elif kind == "mcp_tool_call" and typ == "item.completed":
            items.append({"k": "tool", "id": iid, "name": it.get("tool") or "tool", "input": json.dumps(it.get("arguments") or {})})
            err = it.get("error")
            items.append({"k": "res", "id": iid, "text": (err or {}).get("message") if err else str(it.get("result"))[:4000],
                          "err": bool(err)})
        elif kind == "file_change" and typ == "item.completed":
            ch = it.get("changes") or []
            paths = ", ".join(Path(c.get("path", "")).name + f" ({c.get('kind', '')})" for c in ch)
            items.append({"k": "tool", "id": iid, "name": "write file", "input": json.dumps({"file_path": paths})})
        elif typ == "turn.completed":
            u = e.get("usage") or {}
            items.append({"k": "turn", "text": f"tokens in {u.get('input_tokens', '?')} (cached {u.get('cached_input_tokens', '?')}), out {u.get('output_tokens', '?')}"})
        elif typ in ("error", "turn.failed"):
            items.append({"k": "sys", "text": "error: " + json.dumps(e)[:500]})
    return items


def scout_detail(run, deck):
    f = run / "scout" / f"{deck}.json"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", deck) or not f.exists():
        return {"error": "no such deck"}
    d = json.loads(f.read_text())
    ev = Path(d["work"]) / "events.jsonl"
    if not ev.exists():
        ev = run / "scout" / f"{deck}.events.jsonl"
    lines = ev.read_text(errors="replace").splitlines() if ev.exists() else []
    if d.get("state") == "running":
        cards = sorted(glob.glob(str(Path(d["work"]) / "cards" / "*.md")))
        card_list = [{"name": Path(c).name, "path": None} for c in cards]
    else:
        cards = sorted(glob.glob(str(run / "scout" / deck / "*.md")))
        card_list = [{"name": Path(c).name, "path": str(Path(c).relative_to(run))} for c in cards]
    info = next((x for x in scout_decks(run) if x["deck"] == deck), {})
    return {"deck": info, "items": parse_codex(lines), "cards": card_list}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self.send(200, PAGE, "text/html")
            if u.path == "/api/runs":
                return self.send(200, json.dumps(runs()))
            run = run_path(q["run"])
            if u.path == "/api/overview":
                r = overview(run)
            elif u.path == "/api/team":
                r = team_detail(run, q["team"])
            elif u.path == "/api/agent":
                r = agent_detail(run, q["name"])
            elif u.path == "/api/events":
                st = q.get("start")
                r = events(run, q["name"], q.get("src", "events"), int(st) if st not in (None, "") else None,
                           min(int(q.get("limit", 400)), 3000))
            elif u.path == "/api/file":
                p = safe(run, q["path"])
                r = {"path": q["path"], "text": read(p) if p.is_file() else None}
            elif u.path == "/api/tree":
                r = tree(run, q.get("path", ""))
            elif u.path == "/api/gpu":
                r = gpu_jobs(run)
            elif u.path == "/api/scout":
                r = scout_detail(run, q["deck"])
            elif u.path == "/api/timeline":
                r = timeline(run, float(q.get("hours") or 0))
            else:
                return self.send(404, '{"error":"not found"}')
            self.send(200, json.dumps(r, default=str))
        except Exception as e:  # noqa
            self.send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoInfer runs</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--accent:#2f5d8a;--accent-bg:#eaf1f8;
--ok:#2d7a46;--warn:#9a6a00;--bad:#b3261e;--code:#f4f3f0;--tool:#f6f5ff;--in:#fff8e8}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1e1e1c;--ink:#e8e7e3;--muted:#9a9993;--line:#33322f;
--accent:#8db7e0;--accent-bg:#22303d;--ok:#6cc58a;--warn:#e0b44f;--bad:#f08a80;--code:#262624;--tool:#23223a;--in:#2e2a1e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600}header select{font:inherit;padding:3px 6px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px}
header .sp{flex:1}header label{color:var(--muted);font-size:13px}
.wrap{display:flex;height:calc(100vh - 45px)}
nav{width:270px;min-width:270px;overflow:auto;border-right:1px solid var(--line);padding:8px 0;background:var(--panel)}
nav .sec{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);padding:12px 14px 4px}
nav a{display:flex;gap:6px;align-items:center;padding:4px 14px;color:var(--ink);text-decoration:none;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
nav a:hover{background:var(--code)}nav a.on{background:var(--accent-bg);color:var(--accent)}
nav a.sub{padding-left:30px;font-size:13px}nav .meta{color:var(--muted);font-size:12px;margin-left:auto}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none;background:var(--line)}
.dot.busy{background:var(--ok)}.dot.idle{background:var(--warn)}.dot.stopped{background:var(--line)}
main{flex:1;overflow:auto;padding:16px 24px 60px}
h2{font-size:18px;margin:4px 0 12px}h3{font-size:14px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px 16px}.kv div span{display:block;color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:12px}tr.click{cursor:pointer}tr.click:hover{background:var(--code)}
pre{background:var(--code);padding:8px 10px;border-radius:6px;overflow:auto;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;margin:6px 0}
code{font:12px ui-monospace,Menlo,monospace;background:var(--code);padding:1px 4px;border-radius:4px}
.md{overflow-wrap:anywhere}.md table{width:auto}.md pre code{background:none;padding:0}.md h1{font-size:17px}.md h2{font-size:15px}.md h3{font-size:14px;text-transform:none;color:var(--ink)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 12px}.tabs a{padding:6px 12px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent}
.tabs a.on{color:var(--accent);border-color:var(--accent)}
.ev{margin:8px 0}.ev .who{font-size:12px;color:var(--muted)}
.ev.text .body{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.ev.in .body{background:var(--in);border-radius:8px;padding:8px 12px;border:1px solid var(--line)}
.ev.tool{background:var(--tool);border-radius:6px;padding:4px 10px;font-size:13px}
.ev.tool summary{cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ev.tool summary b{font-weight:600}
.ev.tool .err{color:var(--bad)}.ev.sys,.ev.turn{font-size:12px;color:var(--muted);text-align:center}
.ev.turn.bad{color:var(--bad)}
.pill{font-size:11px;padding:1px 7px;border-radius:10px;background:var(--code);color:var(--muted);white-space:nowrap}
.pill.open{background:var(--accent-bg);color:var(--accent)}.pill.win{color:var(--ok)}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
.muted{color:var(--muted)}.filters{display:flex;gap:12px;align-items:center;font-size:13px;color:var(--muted);flex-wrap:wrap}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:1100px){.grid2{grid-template-columns:1fr}}
a.lnk{color:var(--accent);cursor:pointer;text-decoration:none}a.lnk:hover{text-decoration:underline}
</style></head><body>
<header><h1>AutoInfer</h1><select id="run"></select><span class="sp"></span>
<label><input type="checkbox" id="auto" checked> auto-refresh</label><span id="stamp" class="muted" style="font-size:12px"></span></header>
<div class="wrap"><nav id="nav"></nav><main id="main"><p class="muted">Loading...</p></main></div>
<script>
const $=s=>document.querySelector(s);let RUN=null,VIEW={k:'overview'},OV=null,TIMER=null,EVSTATE=null;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function md(s){if(window.marked){try{marked.use({renderer:{html(t){return esc(typeof t==='string'?t:t.text)}}});return '<div class="md">'+marked.parse(String(s??''))+'</div>'}catch(e){}}return '<pre>'+esc(s)+'</pre>'}
async function api(p,q={}){q.run=RUN;const r=await fetch(p+'?'+new URLSearchParams(q));const j=await r.json();if(j&&j.error)throw new Error(j.error);return j}
const ago=t=>{if(!t)return'';const s=(Date.now()-new Date(t).getTime())/1000;return s<90?Math.round(s)+'s ago':s<5400?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago'};
const ageS=s=>s==null?'':s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':Math.round(s/3600)+'h';
const short=m=>(m||'').replace('claude-','').replace(/-(\d)-(\d)$/,' $1.$2');
function state(a){return a.state!=='alive'?'stopped':a.busy?'busy':'idle'}
function go(v){VIEW={...v,run:RUN};location.hash=encodeURIComponent(JSON.stringify(VIEW));render()}
async function init(){const rs=await (await fetch('/api/runs')).json();$('#run').innerHTML=rs.map(r=>`<option>${esc(r.id)}</option>`).join('');
 let h={};try{h=JSON.parse(decodeURIComponent(location.hash.slice(1)))}catch(e){}
 RUN=h.run&&rs.find(r=>r.id===h.run)?h.run:(rs[0]||{}).id;$('#run').value=RUN;if(h.k)VIEW=h;
 $('#run').onchange=()=>{RUN=$('#run').value;go({k:'overview',run:RUN})};VIEW.run=RUN;await refresh(true);
 setInterval(()=>{if($('#auto').checked)refresh(false)},8000)}
async function refresh(full){try{OV=await api('/api/overview');navRender();$('#stamp').textContent='updated '+new Date().toLocaleTimeString();
 if(full)render();else if(VIEW.k==='agent')pollEvents();else if(['overview','gpu','timeline'].includes(VIEW.k))render(true);else if(VIEW.k==='scout')pollScout()}catch(e){$('#main').innerHTML='<p>'+esc(e.message)+'</p>'}}
function navRender(){const T=OV.teams;const open=T.filter(t=>t.status==='open'),closed=T.filter(t=>t.status!=='open');
 const a=(v,label,meta='',cls='')=>{const on=JSON.stringify({...v,run:RUN})===JSON.stringify({...VIEW,run:RUN});return `<a class="${cls} ${on?'on':''}" onclick='go(${JSON.stringify({...v,run:RUN})})'>${label}${meta?`<span class="meta">${meta}</span>`:''}</a>`};
 const team=t=>{let s=a({k:'team',team:t.team},`<b>${t.team}</b> <span class="muted">${esc(t.type||'')}${t.slot?' · slot '+t.slot:''}</span>`,t.used.toFixed(0)+'/'+t.granted.toFixed(0)+' min');
   if(t.status==='open')for(const ag of t.agents.filter(x=>x.state==='alive'||true).sort((x,y)=>x.name.localeCompare(y.name)))s+=a({k:'agent',name:ag.name},`<span class="dot ${state(ag)}"></span>${esc(ag.name)}`,ag.cost_usd?'$'+ag.cost_usd.toFixed(0):'','sub');
   for(const sc of (OV.scouts||[]).filter(x=>x.team===t.team))s+=a({k:'scout',deck:sc.deck},`<span class="dot ${sc.state==='running'?'busy':sc.state==='done'?'stopped':'idle'}"></span>scout ${esc(sc.deck)}`,sc.state==='running'?(sc.cards_so_far||0)+' files':(sc.cards||0)+' cards','sub');return s};
 const m=OV.manager;
 $('#nav').innerHTML=`<div class="sec">Run</div>${a({k:'overview'},'Overview')}${a({k:'timeline'},'Timeline')}${a({k:'gpu'},'GPU jobs')}${a({k:'file',path:'journal.md',title:'Journal'},'Journal')}
 ${a({k:'file',path:'shared_kb.md',title:'Shared KB'},'Shared KB')}${a({k:'file',path:'announcements.md',title:'Announcements'},'Announcements')}${a({k:'tree',path:''},'All files')}
 <div class="sec">Manager</div>${a({k:'agent',name:'manager'},`<span class="dot ${m?state(m):''}"></span>manager`)}
 <div class="sec">Open teams</div>${open.map(team).join('')||'<p class="muted" style="padding:0 14px">none</p>'}
 <div class="sec">Closed teams (${closed.length})</div>${closed.map(team).join('')}`}
async function render(soft){clearTimeout(TIMER);const v=VIEW;const M=$('#main');try{
 if(v.k==='overview'){const T=OV.teams;M.innerHTML=`<h2>${esc(RUN)}</h2><div class="grid2"><div class="card">${md(OV.champion)}</div><div class="card"><pre>${esc(OV.gpu)}</pre></div></div>
 <h3>Teams</h3><div class="card"><table><tr><th>team</th><th>type</th><th>status</th><th>models</th><th>GPU min used / granted</th><th>jobs</th><th>records</th><th>agent cost</th><th>opened</th></tr>
 ${T.map(t=>`<tr class="click" onclick='go({k:"team",team:"${t.team}"})'><td><b>${t.team}</b></td><td>${esc(t.type)}${t.slot?' · slot '+t.slot:''}</td>
 <td><span class="pill ${t.status==='open'?'open':''}">${t.status}</span></td><td>${t.models.map(short).join(', ')}</td><td>${t.used.toFixed(1)} / ${t.granted.toFixed(0)}</td><td>${t.jobs}</td><td>${t.records.length}</td><td>$${t.cost.toFixed(2)}</td><td class="muted">${ago(t.opened)}</td></tr>`).join('')}</table></div>
 <h3>Config</h3><div class="card">${md(OV.config)}</div>`}
 else if(v.k==='scout'){M.innerHTML=`<h2>scout ${esc(v.deck)}</h2><div id="schead"></div><div class="tabs"><a class="on">activity</a></div><div class="filters"><label><input type="checkbox" id="fT" checked> messages</label><label><input type="checkbox" id="fU" checked> searches / pages / commands</label><label><input type="checkbox" id="fS" checked> thinking / turn</label><button id="bottom">jump to latest</button></div><div id="evs"></div>`;
  $('#bottom').onclick=()=>M.scrollTop=M.scrollHeight;await pollScout(true)}
 else if(v.k==='timeline'){const h=v.hours??6;const d=await api('/api/timeline',{hours:h});
 const btn=x=>`<button style="${x==h?'border-color:var(--accent);color:var(--accent)':''}" onclick='go({k:"timeline",hours:${x}})'>${x?x+' h':'whole run'}</button>`;
 M.innerHTML=`<h2>Timeline</h2><div class="filters">${[3,6,12,24,0].map(btn).join(' ')}<span>hover a bar for details; click a name to open the agent</span></div><div id="tl"></div>`;drawTimeline(d)}
 else if(v.k==='gpu'){const g=await api('/api/gpu');M.innerHTML=`<h2>GPU jobs</h2><div class="card"><pre>${esc(g.status)}</pre></div><div class="card"><table><tr><th>job</th><th>team</th><th>agent</th><th>label</th><th>GPU min</th><th>exit</th><th>ended</th><th></th></tr>
 ${g.jobs.map(j=>`<tr><td>${j.job}</td><td>${esc(j.team)}</td><td>${esc(j.agent)}</td><td>${esc(j.label||'')}</td><td>${(j.gpu_minutes||0).toFixed(1)}</td><td>${j.timed_out?'<span style="color:var(--bad)">timeout</span>':j.exit_code}</td><td class="muted">${ago(j.end)}</td><td><a class="lnk" onclick='go({k:"file",path:"gpu/jobs/${j.job}/output.log",title:"job ${j.job} output"})'>output</a></td></tr>`).join('')}</table></div>`}
 else if(v.k==='file'){const f=await api('/api/file',{path:v.path});const isMd=/\.md$/.test(v.path);
 M.innerHTML=`<h2>${esc(v.title||v.path)}</h2><p class="muted"><code>${esc(v.path)}</code></p><div class="card">${f.text==null?'<p class="muted">(not found)</p>':isMd?md(f.text):'<pre>'+esc(f.text)+'</pre>'}</div>`;
 if(v.path==='journal.md'||v.path==='announcements.md')M.scrollTop=M.scrollHeight}
 else if(v.k==='tree'){const t=await api('/api/tree',{path:v.path});const up=t.path?t.path.split('/').slice(0,-1).join('/'):null;
 M.innerHTML=`<h2>Files: /${esc(t.path)}</h2><div class="card"><table>${up!==null?`<tr class="click" onclick='go({k:"tree",path:${JSON.stringify(up)}})'><td>..</td><td></td><td></td></tr>`:''}
 ${t.entries.map(e=>`<tr class="click" onclick='go(${JSON.stringify(e.dir?{k:'tree',path:e.path}:{k:'file',path:e.path})})'><td>${e.dir?'📁 ':''}${esc(e.name)}</td><td class="muted">${e.dir?'':(e.size/1000).toFixed(1)+' KB'}</td><td class="muted">${ago(e.mtime*1000)}</td></tr>`).join('')}</table></div>`}
 else if(v.k==='team'){const t=await api('/api/team',{team:v.team});
 const fl=(arr)=>arr.map(p=>`<a class="lnk" onclick='go({k:"file",path:${JSON.stringify(p)}})'>${esc(p.split('/').pop())}</a>`).join(' · ')||'<span class="muted">none</span>';
 M.innerHTML=`<h2>${esc(t.team)} <span class="pill ${t.status==='open'?'open':''}">${esc(t.status)}</span></h2>
 <div class="card kv"><div><span>type</span>${esc(t.type)}</div><div><span>slot / models</span>${esc(t.slot||'-')} · ${(t.models||[]).map(short).join(', ')}</div>
 <div><span>GPU min used / granted</span>${(t.used||0).toFixed(1)} / ${(t.granted||0).toFixed(0)}</div><div><span>jobs</span>${t.jobs}</div><div><span>agent cost</span>$${(t.cost||0).toFixed(2)}</div>
 <div><span>opened</span>${ago(t.opened)}</div>${t.closed?`<div><span>closed</span>${ago(t.closed)}</div>`:''}</div>
 ${t.note?`<div class="card"><b>Closed:</b> ${esc(t.note)}</div>`:''}
 <h3>Agents</h3><div class="card"><table><tr><th></th><th>agent</th><th>role</th><th>model</th><th>cost</th><th>last event</th><th>current task</th></tr>
 ${(t.agents||[]).map(a=>`<tr class="click" onclick='go({k:"agent",name:"${a.name}"})'><td><span class="dot ${state(a)}"></span></td><td>${esc(a.name)}</td><td>${esc(a.role)}</td><td></td><td>$${(a.cost_usd||0).toFixed(2)}</td><td class="muted">${ageS(a.last_event_age_s)}</td><td class="muted">${esc((a.task||'').slice(0,120))}</td></tr>`).join('')}</table></div>
 <h3>Documents</h3><div class="card"><p><b>Track record</b> (manager): ${fl(t.files.track)}</p><p><b>Proposals</b>: ${fl(t.files.proposals)}</p><p><b>Resync reports / write-up</b>: ${fl(t.files.reports)}</p><p><b>Inspiration draws</b>: ${fl(t.files.inspiration)}</p>${t.files.scout&&t.files.scout.length?`<p><b>Scout cards</b>: ${fl(t.files.scout)}</p>`:''}</div>
 <h3>Bank records (${t.bank.length})</h3><div class="card"><table><tr><th>record</th><th>official</th><th>performance</th></tr>
 ${t.bank.map(r=>`<tr class="click" onclick='go({k:"file",path:${JSON.stringify(r.path)},title:"${r.id}"})'><td><b>${r.id}</b></td><td>${r.official?'<span class="pill win">'+r.official.toFixed(1)+' tok/s</span>':''}</td><td>${esc(r.performance)}</td></tr>`).join('')||'<tr><td class="muted">none yet</td></tr>'}</table></div>`;
 const rows=M.querySelectorAll('table')[0].querySelectorAll('tr.click');(t.agents||[]).forEach((a,i)=>{api('/api/agent',{name:a.name}).then(d=>{rows[i].children[3].textContent=short(d.meta.model)})})}
 else if(v.k==='agent'){const d=await api('/api/agent',{name:v.name});const a=d.agent,mt=d.meta;const tab=v.tab||'activity';
 const src=v.src||(d.sources[0]||{}).id;
 const tabs=['activity','notes','inbox','files'].map(x=>`<a class="${x===tab?'on':''}" onclick='go(${JSON.stringify({...v,tab:x})})'>${x}</a>`).join('');
 M.innerHTML=`<h2><span class="dot ${state(a)}"></span> ${esc(v.name)}</h2>
 <div class="card kv"><div><span>role</span>${esc(a.role||mt.role)}</div><div><span>team</span>${a.team?`<a class="lnk" onclick='go({k:"team",team:"${a.team}"})'>${esc(a.team)}</a>`:'-'}</div>
 <div><span>model / effort</span>${esc(short(mt.model))} ${esc(mt.effort||'')}</div><div><span>state</span>${state(a)}${a.last_event_age_s!=null?' · last event '+ageS(a.last_event_age_s)+' ago':''}</div>
 <div><span>cost</span>${a.cost_usd!=null?'$'+a.cost_usd.toFixed(2):'n/a'}</div><div><span>parent</span>${esc(a.parent||mt.parent||'')}</div><div><span>started</span>${ago(mt.started||a.started)}</div>
 <div><span>workspace</span><code>${esc((mt.cwd||'').split('/runs/').pop())}</code></div></div>
 <div class="tabs">${tabs}</div><div id="tab"></div>`;const T=$('#tab');
 if(tab==='activity'){T.innerHTML=`<div class="filters">${d.sources.length>1?'<span>source: '+d.sources.map(s=>`<a class="lnk" style="${s.id===src?'font-weight:600':''}" onclick='go(${JSON.stringify({...v,src:s.id})})'>${s.label}</a>`).join(' · ')+'</span>':''}
 <label><input type="checkbox" id="fT" checked> messages</label><label><input type="checkbox" id="fU" checked> tool calls</label><label><input type="checkbox" id="fS" checked> turn markers</label>
 <button id="earlier">load earlier</button><button id="bottom">jump to latest</button></div><div id="evs"></div>`;
  if(!src){$('#evs').innerHTML='<p class="muted">No activity stream found.</p>';return}
  EVSTATE={name:v.name,src,start:null,end:0,items:[]};for(const id of ['fT','fU','fS'])$('#'+id).onchange=drawEvents;
  $('#earlier').onclick=async()=>{const s=Math.max(0,EVSTATE.start-500);const r=await api('/api/events',{name:v.name,src,start:s,limit:EVSTATE.start-s});EVSTATE.items=r.items.concat(EVSTATE.items);EVSTATE.start=s;drawEvents()};
  $('#bottom').onclick=()=>M.scrollTop=M.scrollHeight;
  const r=await api('/api/events',{name:v.name,src,limit:500});EVSTATE.items=r.items;EVSTATE.start=r.start;EVSTATE.end=r.end;drawEvents();M.scrollTop=M.scrollHeight}
 else if(tab==='notes'){if(!d.notes){T.innerHTML='<p class="muted">No notes file (local_kb/'+esc(v.name)+'.md).</p>';return}const f=await api('/api/file',{path:d.notes});T.innerHTML=`<p class="muted"><code>${esc(d.notes)}</code></p><div class="card">${md(f.text)}</div>`}
 else if(tab==='inbox'){if(!d.inbox){T.innerHTML='<p class="muted">No inbox (interactive sessions receive messages directly; see activity).</p>';return}
  const f=await api('/api/file',{path:d.inbox});const msgs=(f.text||'').split('\n').filter(Boolean).map(l=>{try{return JSON.parse(l).message.content}catch(e){return l}});
  T.innerHTML=msgs.map((m,i)=>`<div class="card"><div class="muted" style="font-size:12px">message ${i+1}${i===0?' (first task)':''}</div>${md(m)}</div>`).join('')||'<p class="muted">empty</p>'}
 else if(tab==='files'){T.innerHTML=`<p class="muted">Files in <code>agents/${esc(v.name)}/</code>, newest first. Workspace: <a class="lnk" onclick='go({k:"tree",path:${JSON.stringify((mt.cwd||'').includes('/runs/')?(mt.cwd.split(/\/runs\/[^/]+\//)[1]||''):'')}})'>browse</a></p>
  <div class="card"><table>${d.files.map(f=>`<tr class="click" onclick='go({k:"file",path:${JSON.stringify(f.path)}})'><td>${esc(f.path.split('/').slice(2).join('/'))}</td><td class="muted">${(f.size/1000).toFixed(1)} KB</td><td class="muted">${ago(f.mtime*1000)}</td></tr>`).join('')}</table></div>`}}
 }catch(e){M.innerHTML='<p>'+esc(e.message)+'</p>'}}

const TLC={llm:['#2f5fb3','model generating'],tools:['#9cc0ec','running tools (read / edit / shell / message)'],build:['#7b5cc4','building'],
 gpu_q:['#d0453a','waiting for the GPU (queued)'],gpu_run:['#eda24a','waiting on its own GPU job'],idle:['#e3e1db','idle: waiting for messages'],
 stalled:['#8f8c86','stopped by an API error (e.g. usage limit)']};
function drawTimeline(d){const box=$('#tl');const W=Math.max(700,box.clientWidth||900),L=170,R=20,PW=W-L-R;const t0=d.t0,t1=d.now;
 const x=t=>L+(Math.max(t0,Math.min(t1,t))-t0)/(t1-t0)*PW;const fmt=t=>new Date(t*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
 const dur=s=>s<90?Math.round(s)+' s':s<5400?(s/60).toFixed(1)+' min':(s/3600).toFixed(1)+' h';
 let svg='',y=10;const ink='var(--ink)',mut='var(--muted)',grid='var(--line)';
 // time ticks
 const span=t1-t0,step=[600,1800,3600,7200,14400,21600,43200].find(s=>span/s<=10)||86400;const ticks=[];for(let t=Math.ceil(t0/step)*step;t<=t1;t+=step)ticks.push(t);
 const tickLines=(y0,y1)=>ticks.map(t=>`<line x1="${x(t)}" x2="${x(t)}" y1="${y0}" y2="${y1}" stroke="${grid}"/>`).join('');
 // panel 1: champion speedup
 const H1=120,base=d.base||1;const pts=d.steps.map(s=>({...s,v:(s.tps/base-1)*100}));
 const before=pts.filter(p=>p.t<=t0),inside=pts.filter(p=>p.t>t0);const v0=before.length?before[before.length-1].v:0;
 const vs=[v0,...inside.map(p=>p.v)];const vmin=Math.min(...vs)-2,vmax=Math.max(...vs)+3;const yv=v=>y+H1-(v-vmin)/(vmax-vmin)*H1;
 svg+=`<text x="${L}" y="${y-0}" font-size="12" fill="${mut}">champion: speedup vs BASE ${base.toFixed(1)} tok/s (%)</text>`;y+=8;
 svg+=tickLines(y,y+H1);for(const g of [vmin,(vmin+vmax)/2,vmax])svg+=`<text x="${L-6}" y="${yv(g)+4}" font-size="11" fill="${mut}" text-anchor="end">${g.toFixed(0)}</text>`;
 let path=`M${x(t0)},${yv(v0)}`,cur=v0;for(const p of inside){path+=` L${x(p.t)},${yv(cur)} L${x(p.t)},${yv(p.v)}`;cur=p.v}path+=` L${x(t1)},${yv(cur)}`;
 svg+=`<path d="${path}" fill="none" stroke="${ink}" stroke-width="2"/>`;
 for(const p of inside)svg+=`<circle cx="${x(p.t)}" cy="${yv(p.v)}" r="4.5" fill="${ink}"><title>${p.id}: ${p.tps} tok/s (+${p.v.toFixed(1)}%) at ${fmt(p.t)}</title></circle><text x="${x(p.t)+6}" y="${yv(p.v)-6}" font-size="11" fill="${ink}">${p.id} +${p.v.toFixed(1)}%</text>`;
 if(!inside.length)svg+=`<text x="${x(t1)-4}" y="${yv(cur)-6}" font-size="11" fill="${mut}" text-anchor="end">no new champion in this window (+${cur.toFixed(1)}%)</text>`;
 y+=H1+30;
 // panel 2: GPU use
 const H2=80;const g=d.gpu;let g0=0;for(const p of g)if(p[0]<=t0)g0=p[1];const gin=g.filter(p=>p[0]>t0);const used=(gin.length?gin[gin.length-1][1]:g0)-g0;
 const cap=(t1-t0)/60,gmax=Math.max(cap,used,1);const yg=v=>y+H2-v/gmax*H2;
 svg+=`<text x="${L}" y="${y}" font-size="12" fill="${mut}">GPU-minutes used in this window: ${used.toFixed(0)} of ${cap.toFixed(0)} wall minutes (${(100*used/cap).toFixed(0)}% busy)</text>`;y+=8;
 svg+=tickLines(y,y+H2);const yg2=v=>y+H2-v/gmax*H2;let gp=`M${x(t0)},${yg2(0)}`;for(const p of gin)gp+=` L${x(p[0])},${yg2(p[1]-g0)}`;gp+=` L${x(t1)},${yg2(used)}`;
 svg+=`<line x1="${x(t0)}" y1="${yg2(0)}" x2="${x(t1)}" y2="${yg2(cap)}" stroke="${mut}" stroke-dasharray="4 4"/><text x="${x(t1)-4}" y="${yg2(cap)+14}" font-size="11" fill="${mut}" text-anchor="end">GPU always busy</text>`;
 svg+=`<path d="${gp}" fill="none" stroke="${ink}" stroke-width="2"/>`;for(const v of [0,gmax])svg+=`<text x="${L-6}" y="${yg2(v)+4}" font-size="11" fill="${mut}" text-anchor="end">${v.toFixed(0)}</text>`;
 y+=H2+30;
 // panel 3: agents
 const RH=22;svg+=`<text x="${L}" y="${y}" font-size="12" fill="${mut}">what each agent is doing (thin black bar: its GPU job running)</text>`;y+=8;const y3=y;
 let lastTeam=null;for(const a of d.agents){if(a.team!==lastTeam&&lastTeam!==null)svg+=`<line x1="${L}" x2="${L+PW}" y1="${y+1}" y2="${y+1}" stroke="${grid}"/>`;lastTeam=a.team;
  svg+=`<text x="${L-8}" y="${y+RH/2+4}" font-size="12" fill="var(--accent)" text-anchor="end" style="cursor:pointer" onclick='go({k:"agent",name:"${a.name}"})'>${esc(a.name)}<title>${esc(a.role)} · ${esc(short(a.model))}</title></text>`;
  for(const [s,e,st] of a.intervals){const c=TLC[st]||['#999',st];const w=Math.max(0.6,x(e)-x(s));svg+=`<rect x="${x(s)}" y="${y+3}" width="${w}" height="${RH-9}" fill="${c[0]}"><title>${esc(a.name)}: ${c[1]}, ${dur(e-s)} (${fmt(s)})</title></rect>`}
  for(const [s,e,id] of a.jobs)svg+=`<rect x="${x(s)}" y="${y+RH-5}" width="${Math.max(1,x(e)-x(s))}" height="3" fill="${ink}"><title>GPU job ${id}: ${dur(e-s)}</title></rect>`;
  y+=RH}
 svg=tickLines(y3,y).concat(svg);
 // axis
 y+=6;for(const t of ticks)svg+=`<text x="${x(t)}" y="${y+10}" font-size="11" fill="${mut}" text-anchor="middle">${fmt(t)}</text>`;y+=24;
 const leg=Object.values(TLC).map(([c,l])=>`<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px"><span style="width:12px;height:12px;background:${c};display:inline-block;border-radius:2px"></span>${l}</span>`).join('');
 box.innerHTML=`<div class="card" style="overflow-x:auto"><svg width="${W}" height="${y}" style="font-family:inherit">${svg}</svg><div style="font-size:12px;color:var(--muted);margin-top:6px">${leg}</div></div>`}

async function pollScout(first){const v=VIEW;if(v.k!=='scout')return;const M=$('#main');const atBottom=first||M.scrollHeight-M.scrollTop-M.clientHeight<80;
 const d=await api('/api/scout',{deck:v.deck});if(d.error){M.innerHTML='<p>'+esc(d.error)+'</p>';return}const s=d.deck;
 $('#schead').innerHTML=`<div class="card kv"><div><span>team</span><a class="lnk" onclick='go({k:"team",team:"${s.team}"})'>${esc(s.team)}</a></div><div><span>state</span>${esc(s.state)}</div><div><span>agent</span>codex ${esc(s.model)} / ${esc(s.effort)}</div>
 <div><span>time</span>${s.minutes!=null?s.minutes+' min':''}</div><div><span>cards</span>${s.state==='running'?(s.cards_so_far||0)+' files so far':(s.cards||0)}${' of '+(s.cards_wanted||'?')}</div><div><span>window</span>${(s.window||[]).join(' to ')}</div></div>
 <div class="card"><b>Cards</b> ${s.state==='running'?'<span class="muted">(readable here when the scout finishes)</span>':''}<br>${d.cards.map(c=>c.path?`<a class="lnk" onclick='go({k:"file",path:${JSON.stringify(c.path)},title:${JSON.stringify(c.name)}})'>${esc(c.name)}</a>`:esc(c.name)).join(' · ')||'<span class="muted">none yet</span>'}</div>`;
 const res={};for(const it of d.items)if(it.k==='res')res[it.id]=it;const fT=$('#fT').checked,fU=$('#fU').checked,fS=$('#fS').checked;
 const open=new Set([...document.querySelectorAll('#evs details')].map((x,i)=>x.open?i:-1).filter(i=>i>=0));
 $('#evs').innerHTML=d.items.filter(it=>it.k!=='res'&&((it.k==='text')?fT:it.k==='tool'?fU:fS)).map(it=>evHTML(it,res)).join('')||'<p class="muted">no events yet</p>';
 document.querySelectorAll('#evs details').forEach((x,i)=>{if(open.has(i))x.open=true});for(const id of ['fT','fU','fS'])$('#'+id).onchange=()=>pollScout();
 if(atBottom)M.scrollTop=M.scrollHeight}
function evHTML(it,res){if(it.k==='text')return `<div class="ev text">${it.ts?'<div class="who">'+new Date(it.ts).toLocaleTimeString()+'</div>':''}<div class="body">${md(it.text)}</div></div>`;
 if(it.k==='in')return `<div class="ev in"><div class="who">message${it.from?' from <b>'+esc(it.from)+'</b>':''}${it.ts?' · '+new Date(it.ts).toLocaleString():''}</div><div class="body">${md(it.text)}</div></div>`;
 if(it.k==='tool'){let inp=it.input;let head=inp;try{const o=JSON.parse(inp);head=o.command||o.file_path||o.message||o.query||o.url||o.description||inp;inp=o.command?o.command:(o.message&&o.to?'to: '+o.to+'\n\n'+o.message:JSON.stringify(o,null,1))}catch(e){}
  const r=res[it.id];return `<details class="ev tool"><summary><b>${esc(it.name)}</b> ${esc(String(head).split('\n')[0].slice(0,200))} ${r&&r.err?'<span class="err">error</span>':''}</summary>
  <pre>${esc(inp)}</pre>${r?`<div class="muted" style="font-size:12px">result</div><pre${r.err?' style="color:var(--bad)"':''}>${esc(r.text)}</pre>`:'<div class="muted" style="font-size:12px">(no result yet)</div>'}</details>`}
 if(it.k==='turn')return `<div class="ev turn ${it.err?'bad':''}">— turn done${it.dur?' in '+it.dur+' s':''}${it.cost!=null?', session cost $'+it.cost.toFixed(2):''}${it.text?': '+esc(it.text):''} —</div>`;
 if(it.k==='sys')return `<div class="ev sys">${esc(it.text)}</div>`;return ''}
function drawEvents(){const res={};for(const it of EVSTATE.items)if(it.k==='res')res[it.id]=it;
 const fT=$('#fT').checked,fU=$('#fU').checked,fS=$('#fS').checked;
 const html=EVSTATE.items.filter(it=>it.k!=='res'&&((it.k==='text'||it.k==='in')?fT:it.k==='tool'?fU:fS)).map(it=>evHTML(it,res)).join('');
 const open=new Set([...document.querySelectorAll('#evs details[open]')].map(x=>x.dataset.i));
 $('#evs').innerHTML=(EVSTATE.start>0?`<p class="muted">${EVSTATE.start} earlier lines not loaded.</p>`:'')+html}
async function pollEvents(){if(!EVSTATE||VIEW.k!=='agent'||(VIEW.tab&&VIEW.tab!=='activity'))return;const M=$('#main');
 const atBottom=M.scrollHeight-M.scrollTop-M.clientHeight<80;
 const r=await api('/api/events',{name:EVSTATE.name,src:EVSTATE.src,start:EVSTATE.end,limit:2000});if(!r.items.length){EVSTATE.end=r.end;return}
 EVSTATE.items=EVSTATE.items.concat(r.items);EVSTATE.end=r.end;const opened=[...document.querySelectorAll('#evs details')].map(d=>d.open);drawEvents();
 const ds=document.querySelectorAll('#evs details');opened.forEach((o,i)=>{if(o&&ds[i])ds[i].open=true});if(atBottom)M.scrollTop=M.scrollHeight}
window.onhashchange=()=>{try{const h=JSON.parse(decodeURIComponent(location.hash.slice(1)));if(JSON.stringify(h)!==JSON.stringify(VIEW)){VIEW=h;if(h.run&&h.run!==RUN){RUN=h.run;$('#run').value=RUN;refresh(true)}else{navRender();render()}}}catch(e){}};
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"dashboard on http://{a.host}:{a.port}  (runs under {ROOT / 'flavors'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
