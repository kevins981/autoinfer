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
4. **Overlay.** Grade every answer-key node with the definitions in `EVAL.md`, "Grading a run
   against the key"; apply them exactly (they are the contract, not a guideline):
   - **win nodes:** `idea` / `implementation` / `result`, each yes or no, each yes with its
     evidence; the base (key ids); the run's result with strength (official / abab / ab /
     component); `covered_by_alt` if its alternative passed `result`; `wrong_cause` if the run's
     stated cause for dropping it contradicts the key.
   - **dead-end nodes:** `reason` right / wrong / not-considered, `how` measured / reasoned, and the
     run's stated reason vs the key's cause.
   - A doubtful case gets the stricter answer and a sentence why.
   Then list **novel** run nodes: changes the run **measured** (passed `implementation` with a
   measured outcome) that match no key node: what, builds_on (key ids), result. Untested ideas and
   estimate-only ones are in `answer_keys/<task>/ideas.md`, which you don't grade. A run's own
   unmeasured ideas are not novel nodes; list new ones briefly under Key updates (for the key's
   `ideas.md`), and say if the run measured one of its ideas (a key update).
5. **Why missed.** For each missed win, wrong cause and wrong reason: the stage where it stopped, then the run's closest approach (nearest record or
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
   MPLCONFIGDIR=.cache/mpl .cache/venv/bin/python .claude/skills/eval-run/overlay_plot.py \
     answer_keys/<task>/answer_key.md evals/<flavor>/<run>/overlay.json evals/<flavor>/<run>/overlay.png
   ```
   Look at the PNG (the key as a tree: win nodes colored by the last stage passed, missed wins in
   red, dead ends dashed with their reason grade, novel nodes in purple under their parent) and fix
   the JSON if it misreads. Schema (every key node id must appear in `nodes`):
   ```
   {"run": "<flavor>/<run>", "source": "judge-<id>.md", "base": <BASE tok/s>,
    "champion": {"record": "T1-003", "tok_s": 691.3, "pct": 8.7, "stack": [key ids]},
    "best_measured": {"record": ..., "tok_s": ..., "pct": ..., "adds": [key ids]},  # optional: a
                                                     # better measured stack that wasn't promoted
    "resources": {"wall_h": 3.26, "gpu_min": 164.5, "output_tokens": "411K"},
    "nodes": {
      "<win id>": {"type": "win", "idea": true, "implementation": true, "result": false,
                   "covered_by_alt": null, "wrong_cause": false, "base": "BASE",
                   "evidence": "T1-004 (official +0.3%), KB:22"},
      "<dead-end id>": {"type": "dead-end", "reason": "right|wrong|not-considered",
                        "how": "measured|reasoned|-", "evidence": "T1-002 (component)"}},
    "novel": [{"id": "<short-id>", "title": "...", "builds_on": [key ids], "result": "+1.0% ab"}]}
   ```
   Keep `evidence` short (it is printed beside the node; details go in the report).

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
An uncited claim doesn't count. Every `no` (and every `not-considered`) says what you searched
(e.g. `grep -i "prefill_cudagraph"` over records, ideas, brainstorms, KB, journal). Prefer
`grep -rn` over memory. If unsure, say so and give the label you'd bet on.

Report layout:
```
# Judge <id>: <flavor>/<run>
## Resources        (resources.py output)
## Config and outcome
## Overlay graph    (![overlay](overlay.png))
## Win nodes        (table: node | idea | impl | result | base | run's result (strength) | missed? / wrong cause | evidence)
## Dead-end nodes   (table: node | reason | how | run's stated reason vs the key's cause | evidence)
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
