# Open questions and TODOs shared by all flavors

Flavor-specific TODOs live in each flavor's `design/TODO.md` (e.g. `flavors/v1/design/TODO.md`).

## Agent launching and messaging
- How agents are launched (claude CLI flags, auth) and how they message each other asynchronously
  in headless mode. Check `SendMessage`/`ListAgents` for local sessions. **Hard requirement**:
  brainstormers hold multi-turn chats with persistent leaders, and the manager talks to running
  leaders.
- CLAUDE.md scoping (tested 2026-11-05 with claude 2.1.318): `claudeMdExcludes` in
  `flavors/v1/.claude/settings.json` hides the top-level CLAUDE.md for sessions started in
  `flavors/v1/`. Relative `@../` imports of files outside the project are *not* loaded in `-p`
  mode, so the hard rules are duplicated in each CLAUDE.md. Sub-agents (leaders, workers, brainstormers)
  must exclude *both* CLAUDE.md files (top-level and flavor/manager) via `--settings` and get
  their role prompt explicitly. Untested: the exact launch command, and whether the role prompt
  should repeat the hard rules.

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

## Task: qwen38-27b-fp8-tp2-4090
- Port TASK.md, task.yaml and harness from `qwen38_opt/`.
- Cheap-funnel tools, built once: a microbenchmark harness with real shapes, and a "load once, run
  one batched forward pass, profile" script.
- The task requires FP8 KV, so the baseline must be re-measured with
  `--kv-cache-dtype fp8 --kv-cache-memory 5368709120`, and the GSM8K reference re-measured too.
- Write the correctness criteria (what a validity review checks) (forbidden: precision changes beyond the task's,
  routing/expert changes, sampling or output-length changes, benchmark changes).
