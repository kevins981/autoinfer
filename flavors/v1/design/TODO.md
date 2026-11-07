# Flavor v1: TODO

Open design items for this flavor's algorithm (`DESIGN.md`). Items shared by all flavors
(infrastructure, tasks) are in `../../../design/open_questions.md`.

## Algorithm, in planned order
- Evaluation and budget follow-ups (see `evaluation.md`, `budget_and_stopping.md`):
  - Move the test cost table to the task folder when the harness is ported.
  - Default budget size per team per round (CONFIG default) for the qwen task.

## Agent topology
- Barrier rounds vs. rolling (reassign each leader slot as soon as that leader finishes).
  Leaning: barrier for v1. If stuck leaders block rounds in practice, revisit a per-round time
  limit (deferred).
- Brainstorming details: the exact persona prompts, whether to use other models, the exact
  briefing contents, and tuning the default `search_policy.md`. (Decided: personas come from
  CONFIG, all brainstormers can access everything, round 1 starts from the task's baseline
  profile; see DESIGN.md.)

## Benchmark temperature (human, 2026-11-07)
- Future task versions: run the official benchmark at a non-zero temperature (e.g. 0.5), not 0, so
  gains don't rely on greedy-only paths (real traffic rarely uses T=0). Needs a new task version
  (re-measure BASE, the noise and the accuracy gate's reference; MTP acceptance changes with
  rejection sampling). Don't change it under a live run: runs read tasks/ live. test-3 stays at T=0.

## Untested correctness edge case: preemption (from test-3)
No gate checks a request that vLLM preempts (frees its KV/state under memory pressure) and later
resumes. A change to per-request state (e.g. T1-012's GDN replay log) could break only there. The
benchmark never preempts at c=16 with the current KV budget. Human: low priority (edge case).
Possible gate if needed: a short run with a tiny KV budget that forces preemption, outputs compared
with the reference one request at a time.

## Handoff packet as a skill (from test-3)
The human-readable packet (offline index.html: interactive graph (a DAG: records with several parents), per-record diffs, overview, KB,
untried ideas, rulings) was built ad hoc by test-3's manager (runs/test-3/reports/packet_src/).
Make it a reusable skill/tool for checkpoints and the end of a run.
