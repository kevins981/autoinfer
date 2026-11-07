---
name: eval-run-qwen38
description: Evaluate a finished AutoInfer run (any flavor) of the qwen38-27b-fp8-2x4090 task against its answer key, as one LLM judge, or reconcile several judges' reports and propose answer-key updates. Use when asked to evaluate, grade, judge or compare AutoInfer runs.
---

# Evaluate an AutoInfer run (task: qwen38-27b-fp8-2x4090)
Written for this task; most of it is general, and should be split from the task-specific parts
(metric, noise tolerance, answer key) when a second task arrives.

Read `EVAL.md` (repo root) first: it is the framework, and its "Grading a run against the key" is
the contract for every label. Inputs: a run folder `flavors/<flavor>/runs/<run>/` and the task's
answer key `answer_keys/<task>/answer_key.md` (`ideas.md` next to it is not graded). Two modes;
default is **judge**.

## Judge mode
Work alone on this one run: don't read other judges' reports or other runs (the key already
carries what other runs found). Write `evals/<flavor>/<run>/judge-<id>.md` (id given to you, else
a short random one), `overlay.json` and `overlay.png` there.

**What to find out**
- **Resources:** `python3 .claude/skills/eval-run-qwen38/resources.py <run dir>` (tokens per agent from the
  transcripts, wall clock, GPU-minutes from the run's GPU ledger). Note anything it missed.
- **Config and outcome:** the flavor and run config, BASE, the champion, and how the best result
  grew against GPU-minutes (and wall time).
- **What the run did:** learn from the flavor's `CLAUDE.md` / docs where its runs keep their
  records, then read all of them (every experiment or record, its ideas, notes, reports), not just
  the summaries. Open diffs and logs when a label depends on them.
- **The overlay:** grade every key node by `EVAL.md` (win nodes: idea / implementation / result;
  dead ends: the reason). For each win node that passed `result`, also record **when**: the
  GPU-minutes (and wall time) into the run at which that measurement finished. Then the **novel**
  nodes: measured changes that match no key node.
- **Why missed:** for each missed win and wrong cause or reason: the stage where it stopped, the
  run's closest approach, and what happened, in plain words.
- **Process notes** (short): diversity of directions, how strong the evidence behind drops and
  promotions was, where GPU time or tokens produced no information, any rule violations.
- **Key updates** (proposals only): new nodes, results on new bases, corrections, new unmeasured
  ideas for `ideas.md`; each with evidence.

**How to judge**
- Cite evidence for every label (record id, file path + line); for every "no", say what you
  searched. Uncited claims don't count. If unsure, say so and give the label you'd bet on.
- Grade the agents by the run's own rules at the time (e.g. its promotion bar), and separate the
  human's decisions (stopping the run, rulings) from the agents' choices.

**Picture.** Write `overlay.json` and render it, then look at the PNG and fix what misreads:
```
MPLCONFIGDIR=.cache/mpl .cache/venv/bin/python .claude/skills/eval-run-qwen38/overlay_plot.py \
  answer_keys/<task>/answer_key.md evals/<flavor>/<run>/overlay.json evals/<flavor>/<run>/overlay.png
```
```
{"run": "<flavor>/<run>", "source": "judge-<id>.md", "base": <BASE tok/s>,
 "champion": {"record": "...", "tok_s": 691.3, "pct": 8.7, "stack": [key ids]},
 "best_measured": {"record": ..., "tok_s": ..., "pct": ..., "adds": [key ids]},   # optional
 "resources": {"wall_h": 3.26, "gpu_min": 164.5, "output_tokens": "411K"},
 "nodes": {"<win id>": {"type": "win", "idea": true, "implementation": true, "result": true,
                        "result_at_gpu_min": 42.0, "covered_by_alt": null, "wrong_cause": false,
                        "evidence": "short"},
           "<dead-end id>": {"type": "dead-end", "reason": "right|wrong|not-considered",
                             "how": "measured|reasoned|-", "evidence": "short"}},
 "novel": [{"id": "...", "title": "...", "builds_on": [key ids], "result": "+1.0% ab"}]}
```
The report is yours to lay out; include the picture, a table per node type, novel nodes, why
missed, process notes and key updates.

## Reconcile mode (after the judges)
Read the run's `judge-*.md` and write `evals/<flavor>/<run>/eval.md`: one label per node (where
judges disagree, check the evidence yourself and say how you decided; drop claims whose citations
don't hold), merged notes, and key updates checked against `EVAL.md` ("Updating the key"). Show the
human the key updates; apply them only with their OK, and log them in the key's History.
