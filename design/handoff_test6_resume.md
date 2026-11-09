# Handoff: resuming v2 test-6 after the measurement-cost changes (2026-11-09)

For the designer session that resumes test-6. Two changes made today should show up in the
resumed run; this says what they are, how test-6 gets them, and what to check. Background and
decisions: [measurement_cost.md](measurement_cost.md). Commits: `efd99c3`, `5d76701` (+ the sync
of test-6's task copy, uncommitted run state).

## Change 1: JIT extension compiles are cached across workspaces
- What: `tasks/qwen38-27b-fp8-2x4090/nvcc_jit.sh`, set as `PYTORCH_NVCC` by the task's `env.sh`.
  torch's JIT builds (`cpp_extension.load`, used by the champion's host_staged_ar, dma_staged_ar,
  lmh_codec) used to recompile in every new workspace path (~40 s each, inside the GPU job, both
  ranks waiting). Now identical sources hit the shared ccache.
- How test-6 gets it: every test-6 workspace's `env.sh` sources the repo's task `env.sh`
  (`source /scratch/.../autoinfer/tasks/qwen38-27b-fp8-2x4090/env.sh`), and `official_eval`
  goes through the workspace env. Nothing in the run had to change.
- Expected: a server start from a *new* workspace of known code takes ~60-80 s, not 160-230 s.
  Verified once: test-6-smoke job 3 (champion code, two new paths): 229 s then 74 s.
- Check in the resumed run:
  - `grep -h "server ready after" runs/test-6/gpu/jobs/<new ids>/output.log` and
    `bank/<new record>/official/*/trial1/startup.txt`: official trial 1 should be ~60-90 s like
    trial 2 (before: 160-210 s). A long start is now only legitimate for a *new* `.cu` source.
  - `source tasks/qwen38-27b-fp8-2x4090/env.sh; ccache -s`: hits should grow with each new
    workspace start; `ls runs/test-6/.cache/jit_src/` lists the content-addressed source copies.
  - No job should time out because of a cold compile (test-6's job 119 was the failure mode).
- If something looks wrong: `PYTORCH_NVCC` unset or pointing elsewhere in a job's env; a source
  with local `#include "..."` (passed through uncached, expected); ccache missing from the env.

## Change 2: same-job A/B with early stopping on a clear loss
- What: `tasks/qwen38-27b-fp8-2x4090/scripts/abba.sh` (interleaved fresh starts, same prompts,
  stops after the first A,B pair when B trails by >1.5%, prints a verdict line, per-start
  acceptance) and the rule in `TASK.md` (Noise): a stopped run is a *screen negative (1 pair)*,
  not a measured negative; wins get no shortcut. v2 `agents/common.md` tells all agents to use it
  and to record stopped runs with that label.
- How test-6 gets it: `new_run` copies the task into `runs/test-6/task/` at creation, so this had
  to be synced by hand (done 2026-11-09: `task/TASK.md`, `task/env.sh`, `task/nvcc_jit.sh`,
  `task/scripts/lib.sh`, `task/scripts/abba.sh`; the run copy had no run-specific edits).
  Agents started after the resume get the new `common.md` (`agent start` reads
  `flavors/v2/agents/*.md` at launch). The resumed **manager keeps its old context**: tell it in
  the resume message, e.g.:
  > Resume. The task copy was updated: `task/scripts/abba.sh` runs same-job A/Bs with early
  > stopping on a clear loss (rule in `task/TASK.md`, Noise); point new teams at it. JIT
  > extension compiles are cached across workspaces now, so expect ~70 s server starts.
- Check in the resumed run:
  - Workers/leaders use `abba.sh` (or at least stop after a clearly losing first pair) instead of
    hand-written 4-start loops: look at job scripts under `agents/<agent>/` and at job lengths in
    `gpu/ledger.jsonl` (a losing A/B should be ~5 GPU-min, not 10-12).
  - Records of stopped runs say `screen negative (1 pair, -x%)` in `performance:`, and the KB /
    journal does not turn them into "X doesn't work" facts.
  - Wins still come with a full interleaved set (all B above all A) before an official eval.
- Known gaps: `abba.sh` compares two *workspaces*; a flag-only or env-only change needs two
  workspaces or `--env-b`/`--args-b` (documented in the script header). It has not yet been used
  by an agent; the first use may surface rough edges (report them to the designer).

## Not in test-6 (by decision)
- The step-replay / `vllm bench forward` tier is being evaluated separately (measurement_cost.md,
  decision 4). Don't expect or introduce it in test-6.

## Housekeeping
- test-6-smoke jobs 3-7 (human, uncharged) are the designer's verification jobs for the above;
  ignore them when reading that run.
- Scratch for the forward-bench evaluation: `.cache/fwd_eval/` (not in git).
