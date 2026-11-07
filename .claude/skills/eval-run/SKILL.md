---
name: eval-run
description: Evaluate a finished AutoInfer run (any flavor) against its task's answer key, as one LLM judge, or reconcile several judges' reports and propose answer-key updates. Use when asked to evaluate, grade, judge or compare AutoInfer runs.
---

# Evaluate an AutoInfer run

Read `EVAL.md` (repo root) first: it is the framework. This skill is the procedure.
Inputs: a run folder `flavors/<flavor>/runs/<run>/`, and the task's answer key at
`answer_keys/<task>/answer_key.md`. The task name is in the run's `task_definition.md` / `task/`.

Two modes. The person who asked tells you which; default is **judge**.

## Judge mode (one of 2-3 independent judges)
Work alone, on this one run: don't read other judges' reports or other runs' folders (the answer
key already carries what other runs found). Output: `flavors/<flavor>/runs/<run>/eval/judge-<your id>.md`
(the id is given to you, else a short random one).

1. **Resources.** Run `python3 .claude/skills/eval-run/resources.py flavors/<flavor>/runs/<run>`
   and paste its output. It reads the agents' transcripts (`~/.claude/projects`, read-only) and
   the GPU ledger. Note anything it couldn't find.
2. **Config and outcome.** From `config.md`, `focus.md`, `champion.md`, `reports/final.md`:
   flavor, run config (leaders, workers, rounds, budgets, models), BASE and champion (official),
   and the champion after each round (`reports/r<N>.md`) as a curve against GPU-minutes.
3. **Read the run's own artifacts, not just summaries:** the graph (`graph.md`, older runs
   `tree.md`), every `bank/*/record.md`, `shared_kb.md`, `ideas.md`, `brainstorm/r*/`,
   `journal.md`, `assumption_ledger.md`. Open diffs and logs when a claim depends on them.
4. **Overlay.** For each answer-key node (K, D, O), find the run's matching records/ideas by what
   the change *does* (use the node's "Recognize" line), not by name. Label it:
   - K nodes: `reached` / `partial` (weaker or narrower version) / `tried-dropped` /
     `idea-only` (surfaced in ideas or a brainstorm, never tried) / `never-considered`.
     For alternatives (e.g. K4a | K4b), reaching either counts; say which.
     A node reached without its prerequisites is `partial` unless the gain still showed.
   - D nodes: `not-tried` / `tried, dropped, right cause` / `tried, dropped, wrong or no cause` /
     `tried, kept` (a contradiction: flag it under key updates).
   - O nodes: `not-tried` / `idea-only` / `tried` (with the result: a key update).
   Then list **novel** run nodes that match nothing in the key, with results.
5. **Why missed.** For each K node not `reached`: the run's closest approach (nearest record or
   idea in the same direction) and where its path diverged, then what happened in plain words
   (e.g. dropped after one noisy A/B; never surfaced; out of budget; tried on the wrong base). Don't
   force a fixed category.
6. **Process notes** (short): diversity of directions per round; whether drops and promotions
   rest on measurements (and how strong: official / abab / ab / component / estimate); where GPU
   time or tokens went without producing information; anything that violated the flavor's rules.
7. **Key updates (proposals only; don't edit the key).** Novel wins, D nodes the run overturned or
   confirmed on a new base, O nodes it measured, new facts, corrections. Each with evidence.

**Citations are mandatory.** Every claim cites a record id, KB entry, idea id, or file path + line.
An uncited claim doesn't count. Every `never-considered` / `not-tried` says what you searched
(e.g. `grep -i "prefill_cudagraph"` over records, ideas, brainstorms, KB, journal). Prefer
`grep -rn` over memory. If unsure, say so and give the label you'd bet on.

Report layout:
```
# Judge <id>: <flavor>/<run>
## Resources        (resources.py output)
## Config and outcome
## Overlay          (table: node | label | run evidence | note)
## Novel nodes
## Why missed
## Process notes
## Key updates (proposed)
```

## Reconcile mode (after the judges)
Read all `flavors/<flavor>/runs/<run>/eval/judge-*.md`. Write `flavors/<flavor>/runs/<run>/eval/eval.md`:
- The overlay with one label per node. Where judges agree, keep it. Where they disagree, check
  the cited evidence yourself and decide; list each disagreement and its resolution. Drop claims
  whose citations don't hold up.
- Merged why-missed notes and process notes, and the resources and outcome.
- Proposed key updates, merged and checked against the rules in `EVAL.md` ("Updating the key").
Then show the human the proposed key updates. Apply them to the answer key only with the human's
OK, and add a line to its History.
