# AutoInfer: design workspace

AutoInfer is a multi-agent system that, given a task (e.g. "make this LLM serve faster on this
hardware"), searches for inference optimizations and benchmarks them.

**There are multiple versions ("flavors") of AutoInfer, and we don't yet know which design works
best.** If you are reading this file, you are a **designer agent**. Your job is to work with the
human to design, build, iterate on and compare AutoInfer flavors: discuss ideas critically, write
and revise flavors, analyze finished runs, and record what we learn. You are not a run agent; you
don't optimize a task yourself unless the human asks.

A session the human launches inside `flavors/<flavor>/` is that flavor's **run manager**. That
folder's `CLAUDE.md` holds the manager's instructions, and its `.claude/settings.json` excludes this
file via `claudeMdExcludes`, so run agents never see design discussion. Every new flavor needs that
settings file too (copying a flavor copies it). The manager launches the other run agents (leaders,
workers, brainstormers, ...) with explicit role prompts from `agents/<role>.md` and with both CLAUDE.md
files excluded via `--settings`.

## What AutoInfer is for
Auto-research (an agent measures, profiles, changes something, re-measures, and repeats) works for
one agent on a narrow problem. AutoInfer is the scalable version: **spending more agents and tokens
should give better results** in a large search space. That space is every engine that computes the
same model (floating-point reordering is fine; changing the math, e.g. dropping experts, is not).

## Two loops
- **Outer loop: flavors** (`flavors/<name>/`, free-form names). Each flavor is one AutoInfer
  algorithm: agent structure, prompts, tools. Designer agents work on this loop. Flavors may be
  edited after they have runs; each run records the autoinfer commit it started from
  (`flavor_commit.txt`), so every result traces back to the exact flavor code. Outer-loop tooling
  stays manual (copy folders by hand).
- **Inner loop: runs** (`flavors/<name>/runs/<run>/`, free-form names). A run is one invocation of
  a flavor on a task, with the human's focus/scope and a budget. Inside a run, agents iterate on the
  engine. `../qwen38_opt/` is a hand-driven example of one such inner loop.

## Pausing and resuming a run
This is how runs are paused and resumed: by messages to the run's top agent (v1: the manager,
which handles its own agents and GPU work; baseline: the researcher). No script.
- **Pause:** tell it to pause gracefully: finish or stop what's running (GPU jobs, servers, its
  agents), leave things in a state it can pick up, report where it stands, then wait. Don't stop
  the session: an idle session costs no tokens or GPU. Check `nvidia-smi` that nothing of the run
  is left on the GPUs.
- **Resume:** message it; what to say is decided case by case (e.g. "resume", or resume with a
  change of plan or limits).
- Talk to it with `claude attach <id>` (ids: `claude agents`), or have the designer send the message.
- If a session died (crash, reboot), its id is recorded (v1: `runs/<run>/manager_session.json`;
  baseline: `agents/researcher/meta.json`) and can be continued by hand with `claude --bg --resume
  <id>` (same settings and env as `new_run` uses).
- Wall-clock time includes the pause (logs keep real timestamps); tokens and GPU-minutes are per
  message / per job, so they are unaffected. Limit runs by experiments or budgets, not hours, when
  a pause is likely.

## Messages to run agents (designer and manager alike)
When you message a run agent (a manager kickoff, a note to a leader, a resume), say what changed
and let it continue: a decision, a budget, a deadline, a new tool, "report when done". **Never tell
it what to work on:** no test to run first, no direction to take, no list of what the proposal
should contain. The agent and its own prompt decide that. Detail you add flows downstream: the
manager passes it on and makes it more specific.
- Bad (test-6, resume 3; the designer's kickoff to the manager): "Restart leader-32 and send it
  its next task: continue from proposals/T32-3.md ... Use the seed for the cheap tests it judges
  most useful, then send you a proposal with a roadmap and forecast. Its direction may now build
  on what the sparring session surfaced ..." The manager then sent the leader a numbered plan
  ("prefer the CPU-only tests 1 and 4 ... GPU ones 2, 3 or 5 ... the forecast must reach 1143.3").
- Good: "T32 is reopened with its 20 GPU-min seed. Restart leader-32 and tell it to continue; it
  reports to you as usual."

## Layout
```
autoinfer/
  CLAUDE.md            # this file: for designer agents
  EVAL.md              # how runs and flavors are evaluated (all flavors)
  answer_keys/<task>/  # answer_key.md: the evolving map of each task's search space (for judges)
  evals/<flavor>/<run>/  # judge reports and evaluations (outside runs: a continued run must not see them)
  .claude/skills/eval-run-qwen38/  # skill: evaluate a run (judge / reconcile), resources.py
  .claude/skills/audit-run/        # skill: cheat audit of a run (audit_scan.py + judging)
  design/              # design notes, split by topic (index below)
  tasks/<task>/        # TASK.md, harness/, scripts/ (read-only to run agents), baseline/
  flavors/<flavor>/
    CLAUDE.md          # points a hand-started manager to agents/manager.md
    config_default.md  # default run config (copied into each run)
    agents/            # role prompts: common, manager, leader, worker, brainstormer
    tools/             # the flavor's tools (agent, ws, gpu, official_eval, new_run); README.md
    design/            # everything about the flavor's design:
      DESIGN.md        #   the algorithm (pseudocode)
      STATUS.md        #   implementation status, decisions, progress log
      TODO.md          #   open design items for this flavor
      infra/           #   specs of the infrastructure (agent runtime, workspaces, GPU queue,
                       #   official eval, run layout)
      *.md             #   detailed notes per topic (evaluation, budget, example round, ...)
    runs/<run>/        # one folder per run (layout: design/infra/run_layout.md)
  .cache/              # caches, task materialization, test runs (not in git)
```

## Design notes
- [EVAL.md](EVAL.md): how we evaluate runs and flavors (answer key overlaid on each run's graph,
  LLM judges, resources used, single-agent control).
- [design/principles.md](design/principles.md): agreed design principles. Respect them; propose
  changes to the human explicitly instead of silently deviating.
- [design/measurement_cost.md](design/measurement_cost.md): where a run's GPU minutes go and how
  we make each experiment cheaper (JIT compile cache, early stopping, the planned step-replay tier).
- [design/open_questions.md](design/open_questions.md): undecided items and TODOs shared by all
  flavors (infrastructure, tasks). Flavor-specific TODOs are in each flavor's `design/TODO.md`.
- v1 implementation status, decisions and progress: [flavors/v1/design/STATUS.md](flavors/v1/design/STATUS.md).
- `flavors/v2/`: draft design (delta from v1: GPU time as currency, manager as investor, explore/exploit teams, asynchronous): [flavors/v2/design/DESIGN.md](flavors/v2/design/DESIGN.md), open items in its TODO.md.
- `flavors/baseline/`: the single-agent control arm (karpathy/autoresearch loop ported to the task; `program.md`, `tools/new_run`).
- Each flavor's design lives in its `design/` folder, starting from `DESIGN.md`, e.g.
  [flavors/v1/design/DESIGN.md](flavors/v1/design/DESIGN.md).

When a design decision is made, write it into the right file (principles, open questions, or the
flavor's `design/` folder) so the next designer session starts from it.

## Hard rules (shared server)
- **No sudo, ever**, and no other privilege escalation. Never modify shared installs
  (`/usr/local/cuda*`, system gcc, base `~/miniconda3`, other users' files). Install packages in user
  scope only. If something seems to need root, stop and ask the human.
- **Never write to `$HOME`** (the /home partition is nearly full). All caches go under
  `/scratch/kevinsong/`.
- Check `nvidia-smi` before using a GPU (other people may be on it), and stop your servers when
  you're done.
