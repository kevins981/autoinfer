# Working area: infrastructure for the first v1 run

Temporary. Specs of the subsystems we are building, plus progress. When a subsystem is
implemented, its spec moves into `design/` (or next to the code) and this file is updated.

Process (agreed with the human, 2026-11-06): spec each subsystem at a high level, one by one;
Claude makes the easy decisions and asks only about critical ones; then Claude implements them.

## Subsystems
| # | Subsystem | Spec | Status |
|---|---|---|---|
| 1 | Agent runtime: `agent start/list/stop`, native `SendMessage`, common instructions | [infra/agent_runtime.md](infra/agent_runtime.md) | **implemented + tested** (`tools/agent`) |
| 2 | Workspaces and builds: `ws create/build/diff/rm`, shared env, ccache | [infra/workspaces_and_builds.md](infra/workspaces_and_builds.md) | **implemented + tested** (`tools/ws`) |
| 3 | GPU queue: `gpu run/status/cancel`, batch jobs, FIFO, per-team budgets | [infra/gpu_queue.md](infra/gpu_queue.md) | **implemented + tested** (`tools/gpu`) |
| 4 | Official eval: `official_eval <record>` = fresh workspace + build (CPU) → one GPU job → append to bank | [infra/official_eval.md](infra/official_eval.md) | spec drafted |
| 5 | Run folder layout and kickoff: shared `tools/`, run folder, manager as a `--bg` session | [infra/run_layout.md](infra/run_layout.md) | spec drafted |
| 6 | Role prompts (`common.md`, manager, leader, worker, brainstormer) | (TBD) | |
| 7 | Porting the qwen task (TASK.md, harness, baseline profile) | (TBD) | |

## Decisions made along the way (not yet in design/)
- Permissions (human decision, 2026-11-06): agents run in **`--permission-mode auto`** (Claude's
  classifier judges each action) + a deny list (sudo/su/doas/pkexec, git push, writes to `$HOME`,
  qwen38_opt, /usr, /etc). The allow list holds only low-risk tools and our own tools (`agent`,
  `ws`, `gpu`, `official_eval`), so Bash/Edit/Write go through the classifier. Tested: a haiku
  worker did normal experiment work unblocked; dummy deny rules blocked. Tests may use short GPU jobs (check nvidia-smi first) and haiku
  agents. Commit + push at each tested milestone without asking.
- Tools are Python 3 (stdlib only) in top-level `tools/`.
- GPUs in use by someone outside the run: `gpu run` refuses (exit 75) and tells the agent to
  report to the manager; the manager stops the run's GPU work and reports to the human. Goes into
  the manager's role prompt too.
- Deny rules verified (2026-11-06, harmless dummy rules: a Bash prefix and an absolute-path Write
  rule) in both `dontAsk` and `auto` modes; `agent start --permission-mode` added.
- Workspaces reuse `qwen38_opt/env` and the model weights read-only; all other caches go to
  `autoinfer/.cache`.
- Shared scripts live in top-level `tools/` (agent, ws, gpu, official_eval); flavors keep role
  prompts in `flavors/<f>/agents/`.
- Official evals may use extra serve flags from the record's `serve_args.txt`, checked by the
  leader's validity review (needed because the parallelization scheme is open).
- `PYTHONPATH=<workspace>/vllm` overrides the editable vLLM install (tested), so workspaces need
  no `pip install`.

## Progress log
- 2026-11-06: `tools/ws` and `tools/gpu` implemented and tested: workspace create 1.6 s; cold
  build 10.7 min, the first build of another workspace 1.8 min (ccache 100% hits across paths),
  incremental rebuilds rebuild only the changed file; fake-job queue tests (FIFO, budget checks,
  timeout, cancel, stale tickets, leftovers); a real GPU job served Qwen from a workspace (answer
  correct, GPUs clean after); a GPU process outside the run makes `gpu run` refuse (exit 75).
  Agent-facing usage notes: `tools/README.md` (to be part of `common.md`).
- 2026-11-06: `tools/agent` implemented and tested with haiku agents.
- 2026-11-06: agent runtime tested and specced (pushed d31f472). Workspaces, GPU queue, official eval and run layout specs drafted.
