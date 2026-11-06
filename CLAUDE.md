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
workers, judge, ...) with explicit role prompts from `agents/<role>.md` and with both CLAUDE.md
files excluded via `--settings`.

## What AutoInfer is for
Auto-research (an agent measures, profiles, changes something, re-measures, and repeats) works for
one agent on a narrow problem. AutoInfer is the scalable version: **spending more agents and tokens
should give better results** in a large search space. That space is every engine that computes the
same model (floating-point reordering is fine; changing the math, e.g. dropping experts, is not).

## Two loops
- **Outer loop: flavors** (`flavors/<name>/`, free-form names). Each flavor is one AutoInfer
  algorithm: agent structure, prompts, tools. Designer agents work on this loop. **A flavor is
  frozen once it has runs**; to change it, copy it under a new name, so every result stays tied to
  the exact flavor that produced it. Outer-loop tooling stays manual (copy folders by hand).
- **Inner loop: runs** (`flavors/<name>/runs/<run>/`, free-form names). A run is one invocation of
  a flavor on a task, with the human's focus/scope and a budget. Inside a run, agents iterate on the
  engine. `../qwen38_opt/` is a hand-driven example of one such inner loop.

## Layout
```
autoinfer/
  CLAUDE.md            # this file: for designer agents
  design/              # design notes, split by topic (index below)
  tasks/<task>/        # TASK.md, task.yaml, harness/ (read-only to run agents), baseline/
  flavors/<flavor>/
    CLAUDE.md          # for agents using this flavor (excludes the top-level CLAUDE.md)
    .claude/settings.json
    DESIGN.md          # the algorithm (pseudocode)
    TODO.md            # open design items for this flavor
    search_policy.md   # (v1) how the manager picks what to work on next
    agents/ tools/ runs/<run>/
```
The layout inside a run folder is still to be decided.

## Design notes
- [design/principles.md](design/principles.md): agreed design principles. Respect them; propose
  changes to the human explicitly instead of silently deviating.
- [design/open_questions.md](design/open_questions.md): undecided items and TODOs shared by all
  flavors (infrastructure, tasks). Flavor-specific TODOs are in each flavor's `TODO.md`.
- Each flavor's algorithm lives in its `DESIGN.md`, e.g. [flavors/v1/DESIGN.md](flavors/v1/DESIGN.md).

When a design decision is made, write it into the right file (principles, open questions, or the
flavor's DESIGN.md) so the next designer session starts from it.

## Hard rules (shared server)
- **No sudo, ever**, and no other privilege escalation. Never modify shared installs
  (`/usr/local/cuda*`, system gcc, base `~/miniconda3`, other users' files). Install packages in user
  scope only. If something seems to need root, stop and ask the human.
- **Never write to `$HOME`** (the /home partition is nearly full). All caches go under
  `/scratch/kevinsong/`.
- Check `nvidia-smi` before using a GPU (other people may be on it), and stop your servers when
  you're done.
