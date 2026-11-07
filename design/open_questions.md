# Open questions and TODOs shared by all flavors

Flavor-specific TODOs live in each flavor's `design/TODO.md` (e.g. `flavors/v1/design/TODO.md`).

## Agent launching and messaging
- See [../flavors/v1/design/infra/agent_runtime.md](../flavors/v1/design/infra/agent_runtime.md): tested 2026-11-06. Messaging is native
  `SendMessage` (works between headless agents, busy or idle). Our `agent` script only launches
  agents and lists them (start / list / stop). Open: implementing it, permission lists.
- Crash recovery: agent sessions will die (API errors, context limits, server restarts). The
  launcher should detect this and restart the agent with its last task; agents resume from state on
  disk (bank, KBs, tree, journal). The algorithm assumes this exists.
- CLAUDE.md scoping (tested 2026-11-05 with claude 2.1.318): `claudeMdExcludes` in
  `flavors/v1/.claude/settings.json` hides the top-level CLAUDE.md for sessions started in
  `flavors/v1/`. Relative `@../` imports of files outside the project are *not* loaded in `-p`
  mode, so the hard rules are duplicated in each CLAUDE.md. Sub-agents (leaders, workers, brainstormers)
  must exclude *both* CLAUDE.md files (top-level and flavor/manager) via `--settings` and get
  their role prompt explicitly. Untested: the exact launch command, and whether the role prompt
  should repeat the hard rules.

## Keeping run agents in their run (decided and implemented 2026-11-07)
Run agents (the manager or researcher included) must not see answer keys, evaluations, design
notes, other runs or qwen38_opt's findings: that would spoil the comparison. Claude Code has no
"confine to this folder" switch, and an OS sandbox (bubblewrap) is ruled out (GPU jobs; human).
What `new_run` does now (both flavors, `confine()` + `copy_task()`), tested with Haiku agents:
- **Task copied into the run** (`task/`, 455 MB) + `task_source.json` (name, autoinfer commit).
  Trusted tools (`ws`, `official_eval`, `experiment`) read the original `tasks/<task>/`, so an
  agent editing its copy can't change how it is measured. `task/` and `tasks/` are Edit-denied.
- **Deny rules** (Read + Edit) on: `answer_keys/`, `evals/`, `design/`, `.claude/`, `.git/`,
  `EVAL.md`, `CLAUDE.md`; in every flavor everything but `agents/`, `tools/`, `runs/`,
  `config_default.md`, `program.md` (catches `design/`, `observations.md`, future notes); every
  other run that exists when the run starts; qwen38_opt's `FINDINGS.md`, `TODO.md`, `CLAUDE.md`,
  `results/`, `logs/`, `ext/`, `vllm/`, `build/` (its `env/` and `.cache/` weights stay readable).
  Hard for Read/Grep and for shell commands that name these paths (`cat`, `ls`).
- **Auto-mode classifier rule** (`autoMode.soft_deny`, plain English, naming these paths and "any
  other run"): catches python, `find`, `git log/show` on the autoinfer repo. The list must start
  with `"$defaults"`, otherwise it *replaces* the 72 built-in rules. Soft: an explicit instruction
  from the user overrides it (fine: nothing in a run asks for this).
- **`includeGitInstructions: false`**: otherwise every session's system prompt carries the autoinfer
  repo's recent commits and git status (agents run inside the repo).
- Only `Edit(...)` deny rules work for file writes; `Write(...)`/`NotebookEdit(...)` are ignored
  (removed from the templates).
Residual gaps: a determined agent can still get around the soft rule (e.g. a script reading a path
built at run time); runs created after a run starts aren't in its deny list (the classifier rule
covers "any other run"); live runs keep the settings they started with. Backstop: the cheat audit
(EVAL.md TODO).
Found by the first audit (test-4, 2026-11-07):
- **Finished runs' agents stay reachable.** Agent names are reused across runs (`manager`,
  `leader-1`, ...), and an idle finished session keeps its name: in test-4 a brainstormer messaged
  "manager" and test-3's manager answered (it refused; nothing leaked). Fix options: run-scoped
  names (`<run>/manager`), or stop a run's sessions when it ends. Open.
- Under `task/env.sh`, a bare `import vllm` loads `qwen38_opt/vllm` (the env's editable install).
  It is clean at the baseline commit, so not a leak, but agents must point PYTHONPATH at their
  workspace to run their own code. Open: say so in the tools README, or fix the env.
**Pending, needs the human's go-ahead (not while runs are live):** move `flavors/<f>/runs/` out of
the autoinfer repo (e.g. `/scratch/kevinsong/learning/autoinfer_runs/<f>/`, `runs` a symlink to
it). Then git in a run folder no longer finds the autoinfer repo. Runs aren't meant to be tracked
anyway. At that point: set AUTOINFER_RUN to the real path, check tools that resolve paths, add a
`Bash(git -C <autoinfer> *)` deny (today it would also match workspaces inside runs), and consider
starting the v1 manager in its run folder instead of `flavors/v1/` (inside the repo).

## Infrastructure
- GPU job queue: a wrapper we write; all GPU work goes through it (agents trusted to use it in v1).
  Needs: jobs tagged with team/agent + timeout, exclusive GPUs, per-team budget accounting with a
  `status` command, official vs unofficial jobs (official = clean build + frozen harness). v1's
  requirements: `flavors/v1/design/budget_and_stopping.md`. Also must coexist with other users of
  the shared GPUs.
- Experiment bank: proposed as one vLLM git repo (attempt = branch/commit + record file; champion =
  a git ref; teams use `git worktree`). Per-worktree builds: editable install + C++ extension
  reuse, ccache.
- Layout of a run folder.

## Task: qwen38-27b-fp8-2x4090
- Done 2026-11-06: qwen task ported (TASK.md, harness, scripts/, baseline/ raw profiles, PROFILING.md).
- Cheap-funnel tools, built once: a microbenchmark harness with real shapes, and a "load once, run
  one batched forward pass, profile" script.
- The task requires FP8 KV, so the baseline must be re-measured with
  `--kv-cache-dtype fp8 --kv-cache-memory 5368709120`, and the GSM8K reference re-measured too.
- Write the correctness criteria (what a validity review checks) (forbidden: precision changes beyond the task's,
  routing/expert changes, sampling or output-length changes, benchmark changes).
