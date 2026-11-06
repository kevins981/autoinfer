# Open questions and TODOs

## Algorithm (v1 pseudocode), in planned order
- `evaluate()`: who calls it (the leader decides when a candidate deserves an official eval?), proxy
  vs official evals.
- Budget and stopping: budget units (official evals / wall time / tokens), when a leader's
  direction is done, when the run stops.
- Champion lifecycle: when it changes, comparing candidates built on different starting points,
  `integrate()`.
- **Async "new best version" notification**: v1 snapshots the champion at the start of each
  round. Later: how to tell agents about a new best mid-round without interrupting their work.
- Directions ↔ persistent leaders across rounds (continue vs new; affinity vs mixing), k vs number
  of leaders.
- `initialize()`: who does the profile, bound analysis and ledger; whether the ledger review is a
  blocking human gate.
- Human checkpoints (a config knob between human-in-the-loop and autonomous).
- Failure paths: judge rejection (fix/resubmit?), broken builds, stuck workers or leaders.

## Agent topology
- Barrier rounds vs. rolling (reassign each leader slot as soon as that leader finishes).
  Leaning: barrier for v1.
- Who runs `integrate()` (combining wins from several teams)? Leaning: a dedicated worker
  delegated by the manager.
- How agents are launched (claude CLI flags, auth) and how they message each other asynchronously
  in headless mode. Check `SendMessage`/`ListAgents` for local sessions. **Hard requirement**:
  brainstormers hold multi-turn chats with persistent leaders, and the manager talks to running
  leaders.
- Brainstorming details: how many brainstormers and how big the random sample is, whether to use
  other models, the exact briefing contents, and the default selection policy.
- CLAUDE.md scoping (tested 2026-11-05 with claude 2.1.318): `claudeMdExcludes` in
  `flavors/v1/.claude/settings.json` hides the top-level CLAUDE.md for sessions started in
  `flavors/v1/`. Relative `@../` imports of files outside the project are *not* loaded in `-p`
  mode, so the hard rules are duplicated in each CLAUDE.md. Sub-agents (leaders, workers, judge)
  must exclude *both* CLAUDE.md files (top-level and flavor/manager) via `--settings` and get
  their role prompt explicitly. Untested: the exact launch command, and whether the role prompt
  should repeat the hard rules.

## Infrastructure
- GPU job queue (fair sharing, exclusive benchmarking) and tiered evaluation: cheap proxy tests
  agents can run freely vs. rationed official evals.
- Experiment bank: proposed as one vLLM git repo (attempt = branch/commit + record file; champion =
  a git ref; teams use `git worktree`). Per-worktree builds: editable install + C++ extension
  reuse, ccache.
- Layout of a run folder.

## Task: qwen38-27b-fp8-tp2-4090
- Port TASK.md, task.yaml and harness from `qwen38_opt/`.
- The task requires FP8 KV, so the baseline must be re-measured with
  `--kv-cache-dtype fp8 --kv-cache-memory 5368709120`, and the GSM8K reference re-measured too.
- Write the correctness rubric for the judge (forbidden: precision changes beyond the task's,
  routing/expert changes, sampling or output-length changes, benchmark changes).
