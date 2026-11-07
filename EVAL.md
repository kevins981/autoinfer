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
- **Answer key** (per task, `answer_keys/<task>/answer_key.md`). A map of the task's search
  space as far as we know it, merged from all runs of all flavors (and hand studies). Details in
  "The answer key" below. It is a lower bound on what's possible, and it evolves.
- **Overlay.** For each key node, where did the run get? Wins: reached / partial / tried and
  dropped / idea only / never considered. Dead ends: not tried / dropped for the right cause /
  dropped for a wrong or no cause / kept (a contradiction worth checking). Run nodes that match
  no key node are **novel**; they feed the key.
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

## The answer key
A DAG of the search space, curated (not a copy of any run's graph: run graphs have intermediate
and duplicate steps; key nodes are canonical changes). It holds what we know, not only what works:
- **Wins (K):** changes with a measured gain on a stated base.
- **Alternatives:** wins that substitute for each other (K4a | K4b). Reaching either counts; the
  better one is recorded, the other isn't wrong.
- **Dead ends (D):** no gain or harmful, **on a stated base, with the cause**. A dead end is a
  conditional claim, not a verdict: a change can be dead on BASE and a win on top of another one.
  Each says what would make it worth revisiting.
- **Open (O):** promising, not measured. Not counted as misses; noted when a run reaches them.
- **Facts (F):** diagnoses and environment facts that explain nodes, used to judge whether a
  run's reasons were right.
- **Edges:** prerequisites ("needs K1"): the base on which the gain was measured or exists.
- **Frontier:** the best known stack and its official number.

Each node needs enough detail for a judge to recognize it in a run by what the change does
(flags, code locations, mechanism), what it is not (nearby look-alikes), its gain with the
measurement's strength and base, and evidence ids (`<flavor>/<run>:<record or KB>`).

### Updating the key
After each evaluation, the reconciled report proposes updates; the human approves; the History
section records them. Rules:
- **Evidence strength**, strongest first: official eval (fresh-server trials) > interleaved
  same-job A/B (abab) > one same-job A/B > component measurement (microbench, nsys) > estimate.
- **A result holds on its base.** A positive result on base Y doesn't contradict a negative on
  base X: keep both, each with its base (the node becomes "win on Y, no gain on X").
- **Same base, conflicting results:** the stronger evidence wins. Also weigh exactness and gate
  results: a gain that breaks exactness or a gate is not a win. If the evidence is equally strong,
  mark the node disputed with both results, until a run settles it.
- **Never delete;** change the status and log why in History. Wrong causes get corrected, with the
  evidence that corrected them.
- **Merge, don't append:** a run's finding that matches an existing node adds evidence to it.
- Box state (e.g. a locked GPU clock) is a fact with a date; nodes that depend on it name it.

## Comparing runs and flavors
- Compare at **matched config and resources**. Otherwise, say what differs and don't attribute the
  gap to the flavor.
- One run per side is anecdotal. Repeat runs when a comparison matters.
- **Control arm:** a single-agent auto-research loop (one agent measures, profiles, changes,
  re-measures, repeats) on the same task with the same resources. AutoInfer's claim is that more
  agents and tokens give better results, and this control is the reference for that claim.

## LLM as judge (procedure: `.claude/skills/eval-run/SKILL.md`; reports in `evals/<flavor>/<run>/`, outside the run folder
so a continued run's agents never see its grades)
Most of the grading (matching records to key nodes, judging reasons, "why missed") needs an LLM
judge. Rules:
- **Each run is judged on its own**: a judge sees one run (plus the key), never other runs or
  other judges' reports. Cross-run comparisons read the finished evaluations afterwards.
- **1-3 independent judges** per run, same inputs, no shared context. A designer, or the
  human, reconciles where they disagree.
- **Every claim cites evidence**: a record id, a KB entry, a brainstorm idea id, or a file path and
  line. Uncited claims don't count. Absence claims ("never considered") must say what was searched
  (graph, records, ideas, brainstorms, KB).
- Judges read the run's artifacts, not summaries of them.

## TODO
- [ ] Single-agent control harness: a very basic auto-research loop (one agent, the task's
      harness and GPU queue, same budget), and its run layout.
- [x] Answer key for `qwen38-27b-fp8-2x4090`: `answer_keys/qwen38-27b-fp8-2x4090/answer_key.md`
      (from v1 test-3, test-4 and qwen38_opt).
- [x] Judge and reconcile procedure: the `eval-run` skill (`.claude/skills/eval-run/`), with
      `resources.py` (tokens per agent/role, wall clock, GPU-minutes from transcripts + ledger).
- [ ] The flavor should record the manager's session id (v1's agents/manager/meta.json has none;
      resources.py finds the transcript by start time).
- [ ] Keep run agents away from `answer_keys/` and `evals/` (part of the general "agents stay in
      their run" TODO).
- [ ] Evaluate test-3 and test-4 with the skill, as the first trial of the framework.
