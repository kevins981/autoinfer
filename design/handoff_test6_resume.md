# Handoff: resuming v2 test-6 after the measurement-cost changes (2026-11-09)

For the designer session that resumes test-6. Three changes made today should show up in the
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

## Change 3: inspiration cards (`inspire`)
- What: `flavors/v2/tools/inspire` prints 5 principle + 5 hardware + 5 software cards (about
  15,500 characters), with fixed framing text. Spec: `flavors/v2/design/inspiration.md`. The
  manager attaches one draw, unchanged, to each new team's first task; a leader can draw again at
  any time; the manager may offer a draw at a last chance. Each draw asks the leader for 2-3 lines
  in its notes.
- How test-6 gets it: done by hand 2026-11-09 (uncommitted run state): the three `inspiration`
  lines in `runs/test-6/config.md`, and `inspire` allowed in `agent_settings.json` and
  `manager_settings.json`. Leaders started after the resume get the new prompts. A manager
  session that is **resumed with its old context** has the old `open_teams()`: tell it in the
  resume message, e.g.:
  > When you open a team, first run `inspire --team <team>` and append the saved file
  > (`inspiration/<team>-<n>.md`) to the leader's first task, unchanged. Re-read
  > `flavors/v2/agents/manager.md` (`open_teams()` and the resync section).
- Tested: one throwaway run with one leader (Sonnet); all commands worked with test-6's settings.
  Not yet used in a real run.
- Check that it works as built:
  - `runs/test-6/inspiration/draws.jsonl`: one draw `by` the manager for every team opened after
    the resume, made before the leader started; later draws `by` leaders.
  - Each new leader's first task (first line of `agents/<leader>/inbox.jsonl`) ends with the
    saved draw, identical to `inspiration/<team>-1.md`: nothing picked, dropped, reordered or
    commented on by the manager, and no direction added.
  - Each draw has all three sections (a missing one means a config line is wrong; the tool
    warns on stderr). No `inspire` call blocked by permissions (`events.jsonl`, `err.log`).
- Check whether it helps (the real question):
  - **Notes:** does each leader write its 2-3 lines in `local_kb/<leader>.md`? Are they specific
    to this system and a named card, or generic, or "nothing"? Collect them per team.
  - **Use:** does any proposal or experiment trace back to a card (leader's reasoning in
    `events.jsonl`, the proposal text)? Which cards, and which kind (principle, hardware,
    software)? An idea counts only once it is measured.
  - **Verification:** when a leader builds on a hardware or software card, does it check the
    detail itself (web, docs, a probe) before relying on it?
  - **Order:** does the leader look at the system first (task, profiles), as the cards ask, or
    go straight from a card to a proposal (anchoring)?
  - **Voluntary draws:** does any leader run `inspire` again, and when (stuck, after a failed
    test, at a last chance)? Does it use `--list` / `--show`?
  - **Diversity:** are the new teams' directions more varied than those of test-6's earlier
    teams, or do they converge on the same few as before?
  - **Cost:** do leaders spend long on the cards without getting anywhere, or ignore them?
- Don't steer: if the cards go unused, record it; don't prompt the agents about them mid-run.

## Not in test-6 (by decision)
- The step-replay / `vllm bench forward` tier is being evaluated separately (measurement_cost.md,
  decision 4). Don't expect or introduce it in test-6.

## Housekeeping
- test-6-smoke jobs 3-9 (human, uncharged) are the designer's verification jobs for the above
  and for the forward-bench evaluation; ignore them when reading that run.
- Scratch for the forward-bench evaluation: `.cache/fwd_eval/` (not in git; paused 2026-11-09,
  state and resume steps in its `README.md`). It submits nothing to the GPU unless a designer
  resumes it by hand.
