#!/usr/bin/env python3
"""Follow a headless run agent live: its text, tool calls and incoming messages, one line each.
usage: python3 agent_tail.py <run folder> <agent name> [--all]   (--all: from the start, then follow)"""
import json, sys, time, os
run, name = sys.argv[1], sys.argv[2]
p = os.path.join(run, "agents", name, "events.jsonl")
f = open(p, errors="replace")
if "--all" not in sys.argv:
    f.seek(0, 2)
def show(e):
    t = e.get("type")
    m = e.get("message") or {}
    content = m.get("content") if isinstance(m, dict) else None
    if t == "assistant" and isinstance(content, list):
        for b in content:
            if b.get("type") == "text" and b["text"].strip():
                print(f"\033[1m{name}:\033[0m {b['text'].strip()[:600]}")
            elif b.get("type") == "tool_use":
                i = b.get("input", {})
                arg = i.get("command") or i.get("file_path") or i.get("message") or json.dumps(i)
                print(f"  \033[36m[{b['name']}]\033[0m {str(arg).replace(chr(10), ' ')[:300]}")
    elif t == "user" and isinstance(content, str):
        print(f"\033[33m<< {content.strip()[:400]}\033[0m")
    elif t == "result":
        print(f"\033[2m-- turn done ({e.get('subtype')}), cost so far ${e.get('total_cost_usd', 0):.2f}\033[0m")
    sys.stdout.flush()
while True:
    line = f.readline()
    if not line:
        time.sleep(1); continue
    try: show(json.loads(line))
    except ValueError: pass
