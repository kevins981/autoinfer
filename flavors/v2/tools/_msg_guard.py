#!/usr/bin/env python3
"""PreToolUse hook (SendMessage) for run agents: messages may only go to this run's agents.
usage (in the run's settings): python3 _msg_guard.py <run name>
Allowed targets: names starting with "<run>-" (run-scoped agent names, see `agent list`), "main"
and in-process subagent ids. Everything else (other runs, the designer, judges, socket addresses)
is blocked with exit code 2, which Claude Code shows the agent as the reason.

Messages to "<run>-manager" (the manager is a `claude --bg` session, see _manager.py): if the
session is running under another name, the message is sent to that name; if it has stopped, it
is woken with the message as its prompt and the SendMessage call is blocked (exit 2) with a
note that the message was delivered."""
import json, os, re, sys

run = sys.argv[1]
try:
    tool_input = json.load(sys.stdin).get("tool_input", {})
except ValueError:
    sys.exit(0)
to = str(tool_input.get("to", "")).strip()
name = to.split(" [")[0]
if not (name.startswith(run + "-") or name == "main" or re.fullmatch(r"a[0-9a-f]{8,}", name)):
    print(f"Blocked: you may only message agents of your own run ({run}). Use the run-scoped name "
          f"from `agent list` (SENDMESSAGE TO column, e.g. {run}-manager); to reply to a message, use "
          f"the sender's run-scoped name, not its socket address.", file=sys.stderr)
    sys.exit(2)

run_dir = os.environ.get("AUTOINFER_RUN")
if name != f"{run}-manager" or not run_dir:
    sys.exit(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _manager

if _manager.session_prefix(run_dir) is None:
    sys.exit(0)                                   # no recorded manager session: leave it to Claude Code
e = _manager.entry(run_dir)
if _manager.alive(e):
    current = e.get("name")
    if current and current != name:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                                 "updatedInput": dict(tool_input, to=current)}}))
    sys.exit(0)
sender = os.environ.get("AUTOINFER_AGENT", "unknown")
text = f"[from {sender}; reply with SendMessage to {run}-{sender}]\n{tool_input.get('message', '')}"
err = _manager.wake(run_dir, text)
if err:
    print(f"The manager's session has stopped and could not be woken ({err}). Tell the human.",
          file=sys.stderr)
else:
    print("Delivered: the manager's session had stopped and was woken with your message. "
          "Don't send it again.", file=sys.stderr)
sys.exit(2)
