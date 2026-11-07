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
a short random one), and the two pictures there (below).

**What to find out**
- **Resources:** `python3 .claude/skills/eval-run-qwen38/resources.py <run dir>` (tokens per agent from the
  transcripts, wall clock, GPU-minutes from the run's GPU ledger). Note anything it missed.
- **Config and outcome:** the flavor and run config, BASE, the champion, and how the best result
  grew against wall clock (and GPU-minutes).
- **What the run did:** learn from the flavor's `CLAUDE.md` / docs where its runs keep their
  records, then read all of them (every experiment or record, its ideas, notes, reports), not just
  the summaries. Open diffs and logs when a label depends on them.
- **The overlay:** grade every key node by `EVAL.md` (win nodes: idea / implementation / result;
  dead ends: the reason), and **when** each stage happened (wall clock, for the timeline). Then the
  **novel** nodes: measured changes that match no key node.
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

**Pictures** (two per run, in `evals/<flavor>/<run>/`; look at each PNG and fix what misreads).
Run from the repo root with `PY="MPLCONFIGDIR=.cache/mpl .cache/venv/bin/python"` and
`K=.claude/skills/eval-run-qwen38`.

1. `overlay.png`: the answer key as a tree, colored by the stage each node reached.
   `$PY $K/overlay_plot.py answer_keys/<task>/answer_key.md <eval dir>/overlay.json <eval dir>/overlay.png`
   ```
   {"run": "<flavor>/<run>", "source": "judge-<id>.md", "base": <BASE tok/s>,
    "champion": {"record": "...", "tok_s": 691.3, "pct": 8.7, "stack": [key ids]},
    "best_measured": {"record": ..., "tok_s": ..., "pct": ..., "adds": [key ids]},   # optional
    "resources": {"wall_h": 3.26, "gpu_min": 164.5, "output_tokens": "411K"},
    "nodes": {"<win id>": {"type": "win", "idea": true, "implementation": true, "result": true,
                           "covered_by_alt": null, "wrong_cause": false, "evidence": "short"},
              "<dead-end id>": {"type": "dead-end", "reason": "right|wrong|not-considered",
                                "how": "measured|reasoned|-", "evidence": "short"}},
    "novel": [{"id": "...", "title": "...", "builds_on": [key ids], "result": "+1.0% ab"}]}
   ```
2. `timeline.png`: against wall clock, the best result, GPU-minutes used, what each agent is doing
   (model, tools, building, queued for the GPU, waiting on its GPU job, idle), and when each key
   node was proposed / first run / reached result (or a dead end settled).
   ```
   python3 $K/activity.py <run dir> <eval dir>/activity.json      # per-agent states from transcripts
   $PY $K/timeline_plot.py <eval dir>/timeline.json <run dir>/gpu/ledger.jsonl <eval dir>/timeline.png <eval dir>/activity.json
   ```
   `timeline.json` (minutes since `start`; times from the records, journal, ledger, transcripts):
   ```
   {"run": "<flavor>/<run>", "start": "<ISO run start>", "end_min": 194.5, "base": <BASE tok/s>,
    "phases": [[0.7, 29.0, "brainstorm r1"], [26.5, 49.9, "waiting for human"]],   # shaded bands
    "rounds": [[0.7, "round 1"], [119.6, "round 2 (no brainstorm)"]],             # [] if none
    "official": [[62.7, 653.9, "T2-003", "promoted"], [164.8, 699.5, "T2-006", "rejected: why"]],
    "nodes": {"<win id>": {"type": "win", "idea": 29.0, "impl": 31.6, "result": 62.7},
              "<win id>": {"type": "win"},                                  # missed: never proposed
              "<win id>": {"type": "win", "covered": "<alt id>"},
              "<dead id>": {"type": "dead-end", "first": 40.5, "settled": 129.3, "reason": "wrong"},
              "<dead id>": {"type": "dead-end"}}}                           # not considered
   ```
   "official" holds every official eval (status "promoted" draws the champion line); a flavor
   without promotions marks each kept result "promoted".

The report is yours to lay out; include both pictures, a table per node type, novel nodes, why
missed, process notes and key updates.

## Reconcile mode (after the judges)
Read the run's `judge-*.md` and write `evals/<flavor>/<run>/eval.md`: one label per node (where
judges disagree, check the evidence yourself and say how you decided; drop claims whose citations
don't hold), merged notes, and key updates checked against `EVAL.md` ("Updating the key"). Show the
human the key updates; apply them only with their OK, and log them in the key's History.
