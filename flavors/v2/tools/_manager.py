"""The manager's background session (`claude --bg`): find it, and wake it when it has stopped.

The background service that hosts the manager can stop (taking the session with it), and a
session resumed with flags gets a new name. Agents message `<run>-manager`, so both break delivery.
_msg_guard.py uses this module to send such a message to the session's current name, or, when
the session has stopped, to wake it with the message as its prompt (`claude --bg --resume <id>`,
no flags: the session keeps its saved model, settings, prompt and name).
"""
import json, os, subprocess
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def session_prefix(run):
    try:
        return json.loads((Path(run) / "manager_session.json").read_text())["session_id"]
    except (OSError, ValueError, KeyError):
        return None


def entry(run):
    """The manager's entry in `claude agents --json --all` (None if not found)."""
    prefix = session_prefix(run)
    if not prefix:
        return None
    try:
        out = subprocess.run(["claude", "agents", "--json", "--all"], capture_output=True, text=True,
                             timeout=30).stdout
        rows = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    for r in rows if isinstance(rows, list) else []:
        if str(r.get("sessionId", "")).startswith(prefix):
            return r
    return None


def alive(e):
    pid = e.get("pid") if e else None    # listed only while the session's process runs ("state"
    if not pid:                             # reads "done" for a live session between turns)
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def full_id(run, e):
    if e and e.get("sessionId"):
        return e["sessionId"]
    prefix = session_prefix(run)
    proj = Path.home() / ".claude" / "projects" / str(Path(run).resolve()).replace("/", "-")
    hits = sorted(proj.glob(f"{prefix}*.jsonl")) if prefix else []
    return hits[0].stem if hits else None


def wake(run, text):
    """Resume the stopped manager session with `text` as its next prompt. Returns an error or None."""
    run = Path(run).resolve()
    sid = full_id(run, entry(run))
    if not sid:
        return "manager session not found (manager_session.json)"
    env = dict(os.environ, AUTOINFER_RUN=str(run), AUTOINFER_AGENT="manager", AUTOINFER_ROLE="manager",
               PATH=f"{TOOLS}:{os.environ.get('PATH', '')}")
    env.pop("AUTOINFER_TEAM", None)
    try:
        r = subprocess.run(["claude", "--bg", "--resume", sid, text], cwd=run, env=env, capture_output=True,
                           text=True, timeout=120, start_new_session=True)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return f"could not wake the manager: {ex}"
    if r.returncode != 0:
        return f"could not wake the manager: {(r.stderr or r.stdout).strip()[:300]}"
    from datetime import datetime
    with open(run / "manager_wakes.log", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} session {sid[:8]} had stopped; woken by a "
                f"message from {os.environ.get('AUTOINFER_AGENT', 'human')}\n")
    return None
