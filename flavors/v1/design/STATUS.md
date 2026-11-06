# v1 implementation status

Status of the infrastructure for the first v1 run, decisions made while building it, and a
progress log. Specs are in [infra/](infra/); paths below are relative to `flavors/v1/`.

Process (agreed with the human, 2026-11-06): spec each subsystem at a high level, one by one;
Claude makes the easy decisions and asks only about critical ones; then Claude implements them.

## Subsystems
| # | Subsystem | Spec | Status |
|---|---|---|---|
| 1 | Agent runtime: `agent start/list/stop`, native `SendMessage`, common instructions | [infra/agent_runtime.md](infra/agent_runtime.md) | **implemented + tested** (`tools/agent`) |
| 2 | Workspaces and builds: `ws create/build/diff/rm`, shared env, ccache | [infra/workspaces_and_builds.md](infra/workspaces_and_builds.md) | **implemented + tested** (`tools/ws`) |
| 3 | GPU queue: `gpu run/status/cancel`, batch jobs, FIFO, per-team budgets | [infra/gpu_queue.md](infra/gpu_queue.md) | **implemented + tested** (`tools/gpu`) |
| 4 | Official eval: `official_eval <record>` = fresh workspace + build (CPU) → one GPU job → append to bank | [infra/official_eval.md](infra/official_eval.md) | **implemented, tested** (`tools/official_eval`, `tasks/<task>/harness/`) |
| 5 | Run folder layout and kickoff: the flavor's `tools/`, run folder, manager as a `--bg` session | [infra/run_layout.md](infra/run_layout.md) | **implemented, tested** (`tools/new_run`, `flavors/v1/config_default.md`) |
| 6 | Role prompts (`common.md`, manager, leader, worker, brainstormer) | `flavors/v1/agents/*.md` (incl. `manager.md`) | **drafted**, not yet exercised in a real run |
| 7 | Porting the qwen task (TASK.md, harness, baseline profile) | `tasks/qwen38-27b-fp8-2x4090/` | harness done (eval.sh, serve_args.sh, gsm8k.py, summarize.py); TASK.md + baseline profile TBD |

## Decisions made along the way (not yet in design/)
- Permissions (human decision, 2026-11-06): agents run in **`--permission-mode auto`** (Claude's
  classifier judges each action) + a deny list (sudo/su/doas/pkexec, git push, writes to `$HOME`,
  qwen38_opt, /usr, /etc). The allow list holds only low-risk tools and our own tools (`agent`,
  `ws`, `gpu`, `official_eval`), so Bash/Edit/Write go through the classifier. Tested: a haiku
  worker did normal experiment work unblocked; dummy deny rules blocked.
- Tests may use short GPU jobs (check nvidia-smi first) and haiku agents. Commit + push at each
  tested milestone without asking.
- Tools are Python 3 (stdlib only) in the flavor's `tools/` (moved from top-level `tools/`, human decision
  2026-11-06: they are part of v1).
- GPUs in use by someone outside the run: `gpu run` refuses (exit 75) and tells the agent to
  report to the manager; the manager stops the run's GPU work and reports to the human. Goes into
  the manager's role prompt too.
- Deny rules verified (2026-11-06, harmless dummy rules: a Bash prefix and an absolute-path Write
  rule) in both `dontAsk` and `auto` modes; `agent start --permission-mode` added.
- Workspaces reuse `qwen38_opt/env` and the model weights read-only; all other caches go to
  `autoinfer/.cache`.
- Scripts live in `flavors/v1/tools/` (agent, ws, gpu, official_eval, new_run), role prompts in
  `flavors/v1/agents/`, specs in `flavors/v1/design/infra/` (moved from `wip/`).
- Official evals may use extra serve flags from the record's `serve_args.txt`, checked by the
  leader's validity review (needed because the parallelization scheme is open).
- `PYTHONPATH=<workspace>/vllm` overrides the editable vLLM install (tested), so workspaces need
  no `pip install`.

## Progress log
- 2026-11-06: role prompts rewritten around the DESIGN.md pseudocode (human request): manager.md
  has the whole algorithm, leader/worker/brainstormer their own functions, common.md the global
  state (mapped to files), interface and record format. Brainstormer personas are files in
  `agents/personas/` (`agent start --persona`). Defaults: 1 leader, 2 workers, 5 rounds,
  80 GPU-min per team per round (sized in config_default.md), full model ids. Tool help: every
  subcommand has a description and examples; `gpu run` budget-exhausted exit code is now 76
  (2 is argparse's usage error).
- 2026-11-06: per-role model/effort in `config.md` (human: manager + leaders Opus 5.5 medium,
  workers Sonnet 5.5 medium; brainstormers Opus medium, my choice), read by `agent start` and
  `new_run`, so it doesn't depend on agents passing flags. Explicit, because the user's
  `~/.claude/settings.json` default model (sonnet) otherwise applies. Auto-compact at 400K tokens
  for every agent (`--autocompact`, overridable with `$AUTOINFER_AUTOCOMPACT`). Tested.
- 2026-11-06: manager prompt moved to `flavors/v1/agents/manager.md` (consistent with the other
  roles); `flavors/v1/CLAUDE.md` only points to it. `new_run` passes common + manager prompts via
  `--append-system-prompt` (tested on a `--bg` session) and excludes both CLAUDE.md files.
  Human decisions: baseline uses FP8 KV cache; one manager name (`manager`) for all runs is fine
  for now (one run at a time).
- 2026-11-06: official eval, run kickoff and role prompts implemented. A real `official_eval BASE`
  took 6.2 min end to end: 630.6 tok/s at c=16, GSM8K 92.8%, MTP acceptance 2.34, GPUs clean
  after. A diff that doesn't apply is recorded as `FAILED at apply_diff`. `new_run` creates the run
  folder. A `--bg` session and headless agents reach each other with `SendMessage` (both ways).
  The bench and GSM8K clients run from a clean baseline tree (`.cache/tasks/<task>/baseline/`), so
  a record's diff can't change how it is measured. `agent start` defaults to the flavor's
  `agents/common.md` + `agents/<role>.md`.
- 2026-11-06: `tools/ws` and `tools/gpu` implemented and tested: workspace create 1.6 s; cold
  build 10.7 min, the first build of another workspace 1.8 min (ccache 100% hits across paths),
  incremental rebuilds rebuild only the changed file; fake-job queue tests (FIFO, budget checks,
  timeout, cancel, stale tickets, leftovers); a real GPU job served Qwen from a workspace (answer
  correct, GPUs clean after); a GPU process outside the run makes `gpu run` refuse (exit 75).
  Agent-facing usage notes: `tools/README.md` (to be part of `common.md`).
- 2026-11-06: `tools/agent` implemented and tested with haiku agents.
- 2026-11-06: agent runtime tested and specced (pushed d31f472). Workspaces, GPU queue, official eval and run layout specs drafted.
