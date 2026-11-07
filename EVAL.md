# Evaluating AutoInfer (all flavors)

How we judge a run, and through runs a flavor. Framework first. Lessons from specific runs come
later, once we have enough of them.

The final speedup is the metric that ultimately matters, but it isn't enough on its own:
- One run is one noisy sample.
- Runs often differ in budget and config.
- The number doesn't say *why* a run did well or badly, which is what we need in order to improve
  the design.

## The picture: the run's graph overlaid on an answer key
- **Run graph.** Every run produces a DAG of records (`graph.md`; edges are each record's
  `builds_on`). This is what the run actually did.
- **Answer key** (per task). Known optimizations for the task: the target nodes a good run should
  eventually reach. Each node has:
  - what it is;
  - its measured gain and the base it was measured on;
  - its prerequisites (key nodes can build on each other, so the key is a small DAG too);
  - evidence (where it was measured);
  - known dead ends near it, with their causes.

  The key is a lower bound on what's possible, not the optimum. It grows as runs find new things.
- **Overlay.** For each key node, did the run's graph reach it? Labels:
  - **reached**: found and kept.
  - **partial**: a weaker or narrower version.
  - **tried and dropped**: record whether the stated reason was right.
  - **never considered**.

  Run nodes that match no key node are **novel**. Novel nodes that work get added to the key.
- **Why it was missed (qualitative).** For each missed node, find the run's closest approach (the
  nearest record or idea in the same direction) and where its path diverged. Then say what
  happened, e.g.:
  - branched off too early;
  - dropped after one noisy measurement;
  - never surfaced in a brainstorm;
  - ran out of budget.

  We expect to build a **taxonomy of failure causes** from these notes once there are enough runs.
  Don't fix the categories in advance.

## What every evaluation records
1. **Outcome:** champion speedup vs BASE (official eval), plus the speedup after each round /
   official eval, as a curve against resources used.
2. **Resources actually used** (not the budget):
   - tokens per agent role;
   - wall-clock time;
   - GPU-minutes (unofficial jobs, official evals, profiling).
3. **Config:** flavor, run config (leaders, workers, rounds, budgets, models), task.
4. **Overlay** against the answer key, with the qualitative notes above.
5. **Process notes** (qualitative), e.g. how diverse the brainstorm ideas were; whether drops and
   promotions were backed by measurements; and where tokens or GPU time went without producing
   information.

## Comparing runs and flavors
- Compare at **matched config and resources**. Otherwise, say what differs and don't attribute the
  gap to the flavor.
- One run per side is anecdotal. Repeat runs when a comparison matters.
- **Control arm:** a single-agent auto-research loop (one agent measures, profiles, changes,
  re-measures, repeats) on the same task with the same resources. AutoInfer's claim is that more
  agents and tokens give better results, and this control is the reference for that claim.

## LLM as judge
Most of the grading (matching records to key nodes, judging reasons, "why missed") needs an LLM
judge. Rules:
- **2-3 independent judges** per evaluation, same inputs, no shared context. A designer, or the
  human, reconciles where they disagree.
- **Every claim cites evidence**: a record id, a KB entry, a brainstorm idea id, or a file path and
  line. Uncited claims don't count. Absence claims ("never considered") must say what was searched
  (graph, records, ideas, brainstorms, KB).
- Judges read the run's artifacts, not summaries of them.

## TODO
- [ ] Single-agent control harness: a very basic auto-research loop (one agent, the task's
      harness and GPU queue, same budget), and its run layout.
- [ ] Answer key for `qwen38-27b-fp8-2x4090`, from `../qwen38_opt` and the finished runs, every
      node with evidence. Decide where keys live so that run agents can't read them (a task ships
      no history).
- [ ] Record resources used in every run: tokens per role, wall clock, GPU-minutes. Check what
      the flavors already log.
- [ ] Judge prompt and output format (overlay table + notes with citations), and how to reconcile
      the judges.
- [ ] Evaluate test-3 and test-4 with it, as the first trial of the framework.
