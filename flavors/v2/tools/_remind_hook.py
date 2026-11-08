#!/usr/bin/env python3
"""SessionStart hook (matcher "compact") for run agents: after a context compaction, re-inject the
agent's role reminder (the flavor's agents/reminders/<AUTOINFER_ROLE>.md) as additional context.
Spec: flavors/v2/design/infra/announcements_and_reminders.md. Prints nothing if there is no file."""
import json, os, sys
from pathlib import Path

role = os.environ.get("AUTOINFER_ROLE") or ""
f = Path(__file__).resolve().parent.parent / "agents" / "reminders" / f"{role}.md"
if not role or not f.is_file():
    sys.exit(0)
text = ("<system message>: your context was just compacted. A reminder of your role:\n"
        + f.read_text().strip())
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))
