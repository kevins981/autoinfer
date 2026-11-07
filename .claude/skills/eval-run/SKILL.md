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
key already carries what other runs found). Output: `evals/<flavor>/<run>/judge-<your id>.md`
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
4. **Overlay.** For each answer-key node, find the run's matching records/ideas by what the change
   *does* (the node's "recognize" line), not by name. Record:
   - **contact:** `reached` (tried and kept in the run's stack) / `tried` (tried, not kept) /
     `idea-only` (in ideas or a brainstorm, never tried) / `not-considered`. For a node with
     `alternative_to`, say which alternative the run reached.
   - **base:** the stack the run tried it on, in key ids (e.g. "host-ar + draft-head-diet").
     `reached` without the node's `builds_on` is still `reached` if the gain showed; note it.
   - **run's result and stated cause**, with strength (official / abab / ab / component / estimate).
   - **verdict vs the key:**
     - `as expected`: matches what the key knows on that base, incl. dropping a dead end for the
       right cause.
     - `missed win`: a node with a win on some base, not reached (including tried only on a base
       where it doesn't pay, e.g. `prefill-graphs` on BASE).
     - `wrong cause`: dropped with a cause the key's facts contradict, or with none.
     - `new info`: a result on a base the key doesn't have, or one that contradicts it (a key
       update).
     - `-`: nothing to grade (e.g. a dead end the run never considered).
   **Only measurements count.** A run gets credit for what it tested, never for what it only
   thought: an `idea-only` node is never `as expected` (dropping a win on reasoning alone is a
   `missed win`; an unmeasured dead end gets `-`), and an estimate is not a result.
   Then list **novel** run nodes: changes the run **measured** (tried or reached) that match no
   key node: what, builds_on (key ids), result. Every key node has a measured result; untested
   ideas and estimate-only ones are in `answer_keys/<task>/ideas.md`, which you don't grade.
   A run's own unmeasured ideas are not novel nodes; list new ones briefly under Key updates
   (for the key's `ideas.md`), and say if the run measured one of its ideas (a key update).
5. **Why missed.** For each `missed win` and `wrong cause`: the run's closest approach (nearest record or
   idea in the same direction) and where its path diverged, then what happened in plain words
   (e.g. dropped after one noisy A/B; never surfaced; out of budget; tried on the wrong base). Don't
   force a fixed category.
6. **Process notes** (short): diversity of directions per round; whether drops and promotions
   rest on measurements (and how strong: official / abab / ab / component / estimate); where GPU
   time or tokens went without producing information; anything that violated the flavor's rules.
7. **Key updates (proposals only; don't edit the key).** New nodes (with `builds_on`),
   results on new bases, overturned or confirmed dead ends, measured open nodes, new facts,
   corrections. Each with evidence.
8. **Picture.** Write `evals/<flavor>/<run>/overlay.json` from your overlay, then render it:
   ```
   MPLCONFIGDIR=.cache/mpl .cache/venv/bin/python .claude/skills/eval-run/overlay_plot.py map \
     answer_keys/<task>/answer_key.md evals/<flavor>/<run>/overlay.json evals/<flavor>/<run>/overlay.png
   ```
   Look at the PNG (it is the key as a tree, each node colored by contact, missed wins / wrong
   causes in red, novel nodes in purple under their parent) and fix the JSON if it misreads.
   `novel` holds measured changes only (contact `reached` or `tried`).
   Schema (all key node ids must appear in `nodes`):
   ```
   {"run": "<flavor>/<run>", "source": "judge-<id>.md", "base": <BASE tok/s>,
    "champion": {"record": "T1-003", "tok_s": 691.3, "pct": 8.7, "stack": ["host-ar", ...]},
    "best_measured": {"record": ..., "tok_s": ..., "pct": ..., "adds": [key ids]},   # optional: a
                                                       # better measured stack never promoted
    "resources": {"wall_h": 3.26, "gpu_min": 164.5, "output_tokens": "411K"},
    "nodes": {"<key id>": {"contact": "reached|tried|idea-only|not-considered",
                           "verdict": "as expected|missed win|wrong cause|new info|-",
                           "records": "T2-002, T2-005"}},
    "novel": [{"id": "<short-id>", "title": "...", "builds_on": ["<key id>"],
               "contact": "reached|tried", "result": "+1.0% ab"}]}
   ```

**Grade the agents against the run's own rules and conditions.**
- Separate the human's decisions (stopping the run, rulings, budget changes; see `journal.md`,
  `config.md`, `reports/`) from the agents' own choices. Something the human decided is not an
  agent miss: report it as context.
- Judge each decision by the rules in force during the run (the flavor's prompts at that time,
  e.g. a promotion bar), not by today's rules or today's key. A correct move under the run's rules
  is not a miss, even if the key now knows better.
- Read every record (`bank/*/record.md`), not a sample. Open a record's diff or worker report when
  a label depends on it.

**Citations are mandatory.** Every claim cites a record id, KB entry, idea id, or file path + line.
An uncited claim doesn't count. Every `never-considered` / `not-tried` says what you searched
(e.g. `grep -i "prefill_cudagraph"` over records, ideas, brainstorms, KB, journal). Prefer
`grep -rn` over memory. If unsure, say so and give the label you'd bet on.

Report layout:
```
# Judge <id>: <flavor>/<run>
## Resources        (resources.py output)
## Config and outcome
## Overlay graph    (![overlay](overlay.png), then the key's outline annotated with contact / verdict)
## Overlay          (table: node | contact | base | run's result and cause | verdict | evidence)
## Novel nodes
## Why missed
## Process notes
## Key updates (proposed)
```

## Reconcile mode (after the judges)
Read all `evals/<flavor>/<run>/judge-*.md`. Write `evals/<flavor>/<run>/eval.md`:
- The overlay with one label per node. Where judges agree, keep it. Where they disagree, check
  the cited evidence yourself and decide; list each disagreement and its resolution. Drop claims
  whose citations don't hold up.
- Merged why-missed notes and process notes, and the resources and outcome.
- Proposed key updates, merged and checked against the rules in `EVAL.md` ("Updating the key").
With one judge per run, still write `eval.md`: the judge's overlay plus your corrections (claims
whose citations don't hold, or that blame agents for the human's decisions). Then show the human
the proposed key updates. Apply them to the answer key only with the human's
OK, and add a line to its History.
