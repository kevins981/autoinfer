# Run folder layout and run kickoff (spec)

## Where things live
```
autoinfer/
  tools/                      # shared scripts used by every flavor (on every agent's PATH):
    agent  ws  gpu  official_eval
  tasks/<task>/               # read-only to run agents (TASK.md, harness/, baseline profile, ...)
  flavors/<flavor>/
    agents/                   # role prompts: common.md, manager.md, leader.md, worker.md,
                              #   brainstormer.md
    runs/<run>/               # one folder per run (layout below)
  .cache/tasks/<task>/        # materialized baseline (vllm.git, prebuilt .so, ccache); not in git
```

## Inside a run: `flavors/<flavor>/runs/<run>/`
```
config.md                 # CONFIG (free text): leaders, workers per leader, max rounds,
                          #   GPU-min per team per round, brainstormer personas,
                          #   human_checkpoints, model/effort per role
focus.md                  # the human's focus for this run
task_definition.md        # agreed in initialize(): correctness, hard constraints, loopholes
assumption_ledger.md      # agreed in initialize(): frozen vs open
search_policy.md          # copied from the flavor at start; the human may edit it mid-run
champion.md               # current champion: record id + official numbers (manager)
tree.md  journal.md  shared_kb.md          # manager
local_kb/<leader>.md                       # each leader's notes
bank/<record id>/         # record.md, change.diff, serve_args.txt (optional), official/<n>/,
                          #   profiles, logs
brainstorm/r<k>/          # the round's briefing and each brainstormer's ideas
reports/r<k>.md           # the round summary for the human
agents/<name>/            # agent runtime files (meta, prompt, inbox, events)
gpu/                      # GPU queue: budget.json, ledger.jsonl, jobs/
workspaces/<name>/        # git worktrees (kept; not in git)
```
**In git**: everything except `workspaces/`, `gpu/jobs/` and logs (already in `.gitignore`). Run
records (bank records, tree, journal, KBs, reports) are small and are the main output for the
human, so they are tracked. The manager commits them at the end of each round, never pushes.
The bank belongs to the run (no sharing between runs in v1).

## Kickoff
1. The human starts the manager as a **background Claude session** in the flavor folder:
   `cd flavors/v1 && claude --bg --name manager <run settings>`, and talks to it with
   `claude attach manager` (detach and re-attach at any time; the run keeps going). The flavor's
   `CLAUDE.md` is the manager's prompt.
2. The human says: new run `<run>` on task `<task>`, focus `"..."`, plus any CONFIG overrides.
   The manager creates the run folder, copies defaults (`config.md`, `search_policy.md`) and
   registers itself with `agent` so `agent list` shows it.
3. `initialize()`: the task-definition conversation with the human (blocking), then the baseline
   official eval.
4. Rounds run as in `DESIGN.md`. The manager reports to the human as `human_checkpoints` says;
   the human can attach, ask, redirect or stop at any time.

Resuming a stopped run: start a new manager session in the flavor folder and point it at
`runs/<run>/`. It rebuilds its picture from the files (journal, tree, champion, bank). Agents
that were still alive keep running; stopped ones are started fresh.

## To check while implementing
- A `--bg` session is reachable by the other agents' `SendMessage`, and runs in the same
  permission mode as them.
- `claude attach` works well enough for the human's conversation with the manager (otherwise: an
  interactive session in tmux).
