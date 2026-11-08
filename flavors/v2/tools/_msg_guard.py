#!/usr/bin/env python3
"""PreToolUse hook (SendMessage) for run agents: messages may only go to this run's agents.
usage (in the run's settings): python3 _msg_guard.py <run name>
Allowed targets: names starting with "<run>-" (run-scoped agent names, see `agent list`), "main"
and in-process subagent ids. Everything else (other runs, the designer, judges, socket addresses)
is blocked with exit code 2, which Claude Code shows the agent as the reason."""
import json, re, sys

run = sys.argv[1]
try:
    to = str(json.load(sys.stdin).get("tool_input", {}).get("to", "")).strip()
except ValueError:
    sys.exit(0)
name = to.split(" [")[0]
if name.startswith(run + "-") or name == "main" or re.fullmatch(r"a[0-9a-f]{8,}", name):
    sys.exit(0)
print(f"Blocked: you may only message agents of your own run ({run}). Use the run-scoped name "
      f"from `agent list` (SENDMESSAGE TO column, e.g. {run}-manager); to reply to a message, use "
      f"the sender's run-scoped name, not its socket address.", file=sys.stderr)
sys.exit(2)
