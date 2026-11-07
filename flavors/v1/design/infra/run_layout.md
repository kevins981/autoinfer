# Run folder layout and run kickoff (spec)

## Where things live
```
autoinfer/
  tasks/<task>/               # read-only to run agents (TASK.md, harness/, baseline profile, ...)
  flavors/<flavor>/
    agents/                   # role prompts: common.md, manager.md, leader.md, worker.md,
                              #   brainstormer.md
    tools/                    # the flavor's tools (on every agent's PATH):
                              #   agent  ws  gpu  official_eval  new_run
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
graph.md  journal.md  shared_kb.md          # manager
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

## Kickoff (implemented: `tools/new_run`)
1. The human runs `flavors/v1/tools/new_run <run> --task <task>`. It creates the run folder (task link,
   `config.md` from `flavors/v1/config_default.md`, `search_policy.md`, empty records, folders,
   `agent_settings.json` + `manager_settings.json`), runs `ws setup-task`, and starts the manager
   as a background session in `flavors/v1/` (`claude --bg --name manager`, auto mode, the run's
   settings, `AUTOINFER_RUN` set; prompts `agents/common.md` + `agents/manager.md` via `--append-system-prompt`). First time only: run
   `claude` once in `flavors/v1/` and accept the trust prompt.
2. The human talks to it with `claude attach <id>` (detach and re-attach any time).
3. `initialize()`: the task-definition conversation (blocking), then `official_eval BASE`.
4. Rounds as in `DESIGN.md`; reports to the human as `config.md`'s human checkpoints say.

Resuming a stopped run: start a manager session in `flavors/v1/` with `AUTOINFER_RUN` pointed at
`runs/<run>/`; it rebuilds its picture from the files. Live agents keep running.

Tested: a `--bg` session inherits the environment; a headless agent and a `--bg` session reach
each other with `SendMessage` in both directions. Not yet tested: a full manager kickoff
(happens in the first real run).
