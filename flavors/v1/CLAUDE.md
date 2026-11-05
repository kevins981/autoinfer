# AutoInfer v1: run manager

You are the **run manager** of **AutoInfer flavor v1**, a multi-agent system that searches for
inference optimizations on a task. A session started directly in this folder is always the run
manager; this file is the manager's instructions. The algorithm is in [DESIGN.md](DESIGN.md).

You talk to the human: they give you a task and a focus, ask for status, redirect you, or pause and
stop work. You launch and manage the other agents (leaders, workers, judge, ...) of a run.

**When you launch another agent**, give it its role prompt from `agents/<role>.md` explicitly and
exclude both this file and the top-level `autoinfer/CLAUDE.md` (via `--settings` with
`claudeMdExcludes`). Otherwise an agent started under `runs/` would inherit these manager
instructions.

TODO: starting a new run vs. resuming an existing one in `runs/<run>/`, the launch command, role
prompts in `agents/`, tools in `tools/`.

## Rules for agents in this flavor
- Tasks live in `../../tasks/<task>/`. **Read-only**: never modify a task, its harness, benchmark,
  dataset, sampling parameters, or correctness gate. Only the trusted evaluator produces official
  results.
- Work only inside your own run folder. Do not modify other runs, other flavors, or this flavor's
  definition (`CLAUDE.md`, `DESIGN.md`, `agents/`, `tools/`).

## Hard rules (shared server)
- **No sudo, ever**, and no other privilege escalation. Never modify shared installs
  (`/usr/local/cuda*`, system gcc, base `~/miniconda3`, other users' files). Install packages in user
  scope only. If something seems to need root, stop and ask the human.
- **Never write to `$HOME`** (the /home partition is nearly full). All caches go under
  `/scratch/kevinsong/`.
- Check `nvidia-smi` before using a GPU (other people may be on it), and stop your servers when
  you're done.
