# Agent runtime: launching agents and messaging between them

How AutoInfer's agents run on top of the `claude` CLI. This applies to all flavors. Tested
2026-11-06 with claude 2.1.319, using haiku for the cheap tests.

## Tested: what works
**1. A persistent headless agent with an inbox file and an event log.**
```bash
tail -n +1 -f inbox.jsonl | claude -p --input-format stream-json --output-format stream-json \
    --verbose [--name leader-1] [...] > events.jsonl
```
- The process stays alive between tasks. To give it a task or a follow-up, append one line to
  `inbox.jsonl`:
  `{"type":"user","message":{"role":"user","content":"..."}}`.
- **Messages sent while it is working are picked up mid-turn**, after its next tool call. They are
  not held until the turn ends. Test: an agent ran 4 x `sleep 15`; a message was appended during
  the first sleep. It followed the message in that same turn.
- A message sent while it is idle starts a new turn. This is how persistent leaders and workers
  get follow-ups.
- `events.jsonl` gets one JSON event per line: init (session id, tools), each assistant
  message and tool call, tool results, and one `result` event per finished turn. The `result`
  event holds the final text (= the report), `total_cost_usd` and token usage. Anyone can
  `tail` it to see what the agent is doing.

**2. Built-in messaging (`SendMessage` / `ListAgents`) between headless agents.**
Test: two headless agents launched with `--name leader-x` / `--name bs-x`, both in
`--permission-mode dontAsk` with `--allowedTools SendMessage ListAgents`. The brainstormer found
the leader with `ListAgents`, asked it two questions with `SendMessage` (ending its turn after
each), and wrote an idea from the answers. Two complete exchanges.
- **A finished agent waiting for its next instruction still receives messages.** The process is
  alive (its stdin stays open), and an incoming message starts a new turn. Both agents were idle
  between every exchange. A busy agent reads the message after its next tool call.
- **The sender doesn't wait or poll.** It ends its turn, and the reply wakes it up as a new turn.
- `--name` sets the address. Replies go back automatically: the receiver sees a `from` attribute
  and sends to it.
- Limits:
  - The receiving **process must be alive**. Our agents stay alive until we kill them, so this
    only excludes stopped or crashed agents.
  - **Same permission mode** for every agent; otherwise messages are held for approval.
  - `ListAgents` lists **every** Claude session on the machine, the human's own included, with
    names and busy/idle only: no roles, teams or tasks, and not scoped to the run.
  - Machine-local only.
  - Incoming messages don't appear in the receiver's `events.jsonl` (the sender's log has them,
    as its `SendMessage` call). `--replay-user-messages` may fix this (untested).
  - A first cross-session message makes an agent cautious ("this came from another session");
    the role prompt must say peer messages are part of the job.

**3. A multi-turn chat through a small `ask` script** (prototype in `.cache/msgtest/ask.sh`;
superseded by native messaging, kept as a fallback).
`ask <agent> <message>` appends the message to that agent's inbox, waits for its next `result`
event and prints the final text. A headless "brainstormer" asked an idle headless "leader" three
questions in a row, each building on the last answer; the leader kept its context, and when it
didn't know something it said so instead of guessing. The sender sees exactly what the receiver
said, because it is that turn's `result`.

## Spec (v1)
**Messaging: native `SendMessage`.** It covers everything we need: busy or idle receivers,
replies, multi-turn conversations, and no waiting or polling. Agents are launched with
`SendMessage` allowed, **native `ListAgents` disallowed** (it lists every Claude session on the
machine and knows nothing about the run), and the same permission mode. To find other agents,
they use **our `agent list`**, and send to the bare name it shows (untested: bare names without
the `[ref]` suffix; the docs say an exact name match delivers).

**Common instructions.** Every agent's prompt = `common.md` (shared by all roles) + its role
prompt. `common.md` holds:
- the hard rules (no sudo, no writes to `$HOME`, ...);
- the run's tools: `agent list` to see who is who, `SendMessage` to talk, the GPU queue;
- messaging norms: messages from the run's agents and from the human are part of the job;
  answer questions with `SendMessage`; send your report to your parent when you finish a task.
- identify yourself in every `SendMessage`: start it with your name and role, e.g.
  `[leader-1, leader of team 1]`.

**The `agent` script: launching and who-is-who only.** Native tools can't launch an agent wired
up our way, and `ListAgents` doesn't know roles, teams, tasks or which run an agent belongs to.
No daemon: just files and processes.

Files: the run folder is `$AUTOINFER_RUN`. Each agent has `$AUTOINFER_RUN/agents/<name>/` with
`meta.json` (name, role, team, parent, cwd, pid, start time), `prompt.md` (`common.md` + its role prompt),
`inbox.jsonl` (its stdin), `events.jsonl` (its stdout) and `err.log`. Names are readable and
unique in the run, e.g. `manager`, `leader-1`, `leader-1.w2`, `bs-r3-systematic`, and are also
the `SendMessage` address (`--name`).

| Command | What it does |
|---|---|
| `agent start <name> --role R [--team T] [--cwd DIR] [--prompt-file F] [--model M] [-f first_task.md]` | Launch a persistent headless agent: `--name <name>`, stdin = its inbox (`tail -f`), stdout = its event log, the run's shared permission settings. The parent is the caller. Optionally gives it its first task. |
| `agent list` | The run's agents: name, role, team, parent, alive or stopped, busy or idle, the first line of its current task, time since its last event. How brainstormers find the leaders, and how the manager spots stragglers. |
| `agent stop <name>` | Kill the process (an agent never exits on its own). Its folder stays. |

**Typical flows**
- Manager → leader: `agent start leader-1 ... -f task.md`, or later a `SendMessage` with the next
  task. When the leader is done, it sends its report to `manager`. A message to an idle manager
  starts a new turn, so the manager reacts to reports as they arrive. The round ends when every
  leader has reported (the barrier).
- Leader → worker: the same pattern.
- Brainstormer ↔ leader: `agent list` to see the leaders, then `SendMessage`, back and forth.
  (Never native `ListAgents`.)
- Human: `agent list`, `tail` any `events.jsonl`, and `SendMessage` from their own Claude session
  (or append to an agent's inbox).

**Not in v1**: authentication, access rules, restart/resume after a crash (`list` shows the agent
as stopped; the manager starts a new one).

## Features we need, and their status
| Feature | Status |
|---|---|
| Launch persistent headless agents; give follow-up tasks | **works** (stream-json stdin) |
| Asynchronous messages, delivered mid-turn | **works** (native `SendMessage`; also the inbox file) |
| Agent ↔ agent multi-turn chat | **works** (native `SendMessage`, headless ↔ headless, receiver idle between messages) |
| Who is who, who is alive | `agent list` (spec above) |
| Watch what an agent does (structured log) | **works** (`events.jsonl`) |
| Know when an agent finished a task | it sends its report to its parent (`SendMessage`) |
| Role prompts, per-role CLAUDE.md scoping | `--append-system-prompt[-file]`, `--settings` with `claudeMdExcludes` (partly tested 2026-11-05) |
| Permissions without a human: no prompts, no sudo | `--permission-mode dontAsk` + allow/deny lists (deny `sudo`, writes outside the run). Same mode for every agent (see caveat). TBD: exact lists |
| Confine workers to their workspace | cwd + `--add-dir` + deny rules; trusted in v1 (no sandbox) |
| Long-lived agents (compaction) | auto-compaction; `--autocompact` sets the window. Untested over hours |
| Crash recovery | deferred: `list` shows dead agents; later `--resume <session_id>` |
| Kill or pause an agent (manager, human) | `agent stop`; pause = a message ("stop and wait") |
| Track token / dollar usage (stop when out of plan) | `total_cost_usd` and usage in each `result` event; `--max-budget-usd` per agent |
| Model and effort per role | `--model`, `--effort` |
| Session data in `$HOME` | **decided**: leave it in `~/.claude/` (a few MB per run); the human cleans it up |

## Still to test (while implementing)
- An idle manager reacting to leader reports over hours; messages to a busy agent stuck in a long
  Bash call (e.g. a 10-minute bench) wait until that call ends.
- `--replay-user-messages`, so incoming messages show up in the receiver's log.
- The shared permission settings (allow/deny lists) for all agents.
