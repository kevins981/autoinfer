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
- **Overlay.** Each key node graded against the run, by its type (win or dead end), as defined
  in "Grading a run against the key" below. Changes the run **measured** that match no key node
  are **novel**; they feed the key. (Its unmeasured ideas are not nodes; new ones may feed
  `ideas.md`.) **Only measurements count**, as in real research: an idea a run thought of but never
  tested claims no gain, earns no credit and is never novel.
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
   official eval, as a curve against wall clock (and GPU-minutes); and when each key node was
   proposed, first run and reached result, so runs that get to the same place can be compared on
   speed. Wall clock is the main axis: it is the real deadline, and it shows what parallel agents
   buy; GPU-minutes show the physical cost.
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
A DAG of the task's search space (`answer_keys/<task>/answer_key.md`), curated: not a copy of any
run's graph (run graphs have intermediate and duplicate steps; key nodes are canonical changes). It
holds what we know, not only what works, and **only what was measured**: a node needs a measured
result of the change itself (official / abab / ab / component). Untested ideas and changes ruled
out only by an estimate go to `answer_keys/<task>/ideas.md`, which is not graded and not drawn; an
idea moves into the key once a run measures it.
- **Node = one change**, with a status-neutral id (e.g. `host-ar`), because status can change and
  can differ per base.
- **Edges:**
  - `builds_on`: the changes a node needs (its result depends on them). This is not the stack it
    happened to be measured on; that goes in its results.
  - `alternative_to`: substitutes (same effect, don't stack). Reaching either counts; the better
    one is recorded, the other isn't wrong.
- **Results per base:** one line per stack it was measured on, with strength and evidence. So one
  node can be "no gain on BASE, win on `host-ar`".
- **Status** (summary): win / win on <id> / no gain / harmful. A dead end is a conditional claim
  (on its bases, with its cause), not a verdict; it says what would make it worth revisiting. An
  idea that would revive a dead end goes to `ideas.md` with its edges (e.g.
  `draft-depth-3-small-state` builds on `small-spec-state` + `draft-depth-3`) until it is measured.
- **Facts (F):** diagnoses and environment facts that explain nodes, used to judge whether a run's
  reasons were right.
- **Frontier:** the best known stack and its official number.
- **Outline:** generated from the edges by `.claude/skills/eval-run-qwen38/keygraph.py` (each node under
  its first parent, "+ X" for other parents). Regenerate it after every edit.

Each node needs enough detail for a judge to recognize it in a run by what the change does
(flags, code locations, mechanism) and what it is not (nearby look-alikes); evidence ids are
`<flavor>/<run>:<record, KB or path:line>`.

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
- **Never delete;** change the status and log why in History. (Unmeasured entries live in
  `ideas.md`, never in the key.) Wrong causes get corrected, with the
  evidence that corrected them.
- **Merge, don't append:** a run's finding that matches an existing node adds evidence to it.
- Box state (e.g. a locked GPU clock) is a fact with a date; nodes that depend on it name it.

## Grading a run against the key
Precise definitions; judges apply them exactly and cite evidence for every yes. Where evidence
sits between two answers, give the stricter one and say why.

**Matching.** A run's work matches a node if it contains the node's core mechanism (the node's
"what"; recognize it by its "recognize" line), whatever the run called it. Anything the node's
"Not this" excludes doesn't match. Work by any of the run's agents counts (leader, worker,
brainstormer); anything the human did is reported as the human's, not the agents'.

**Node type**, from the key: a **win node** has a win on at least one base; a **dead-end node** has
no gain or harm on every base the key knows.

**Win nodes: three stages, each yes / no** (a later stage implies the earlier ones):
1. **idea**: some agent wrote this change down as a proposal of its own: an entry (idea id,
   bullet, paragraph) in ideas, brainstorms, records, KB, journal or reports whose subject is this
   change and that says what to change. A passing mention inside another idea's entry is not
   enough (cite it as context).
2. **implementation**: the change (code, flag or env var) was written **and executed** on this
   task: a server run, a microbenchmark or a kernel test of the changed code. Not enough: an
   estimate, back-of-envelope math, reading code, or code written but never run.
3. **result**: the run measured a gain from this change at least as large as the key's best known
   gain for this node minus **1.6 percentage points** (the spread between repeat measurements of
   the same code across jobs, TASK.md), on a base where the key says it pays. E.g. key best +3.0%,
   run +2.4%: yes. A node measured only as a swap for its `alternative_to` passes if it measured
   equal to or better than that alternative (within the same 1.6 points) and the alternative
   passed **result** (its gain then follows from the alternative's). Being kept or
   promoted is not required (that's shown separately: ★ champion, ☆ best measured). Not enough: a
   microbenchmark alone (it can't show the end-to-end gain); a test only on a base where it doesn't
   pay (e.g. `prefill-graphs` on BASE). A gain beyond the key's best is a yes and a key update.

A win node that doesn't pass **result** is a **missed win**, unless its `alternative_to` passed
**result** ("covered by its alternative"; not a miss). Where the run stated a cause for dropping a
win node and the key's facts contradict it, flag **wrong cause** too.

**Dead-end nodes: one grade, the run's reason**:
- **right**: the run dropped or ruled it out for a reason the key's cause and facts support.
- **wrong**: dropped or ruled out for a reason the key contradicts, or with no reason; or kept it
  in the stack claiming it gave a gain. (Keeping it as a stated neutral building block, e.g. "no
  gain, exact, kept for later", is not wrong.)
- **not considered**: the run never proposed it (stage-1 test above). No penalty, no credit.
If the run revised its reason over time, grade the **final** one, and note in the process notes
how long the wrong or missing reason stood (e.g. "cause undiagnosed in round 2, found in round 3").
Also record **how**: `measured` or `reasoned` (estimate, math, code reading). Not graded: a run that
correctly rules out a dead end without spending GPU time loses nothing; time spent measuring it
shows up in resources and process notes. (Ruling out by reasoning is fine here because the key
already holds the measurement; claims of gains always need a measurement.)

**Base.** The stage tests don't depend on the stack the run used, except **result** (it must be a
base where the node pays). Record the base in key ids either way.

## Comparing runs and flavors
- Compare at **matched config and resources**. Otherwise, say what differs and don't attribute the
  gap to the flavor.
- One run per side is anecdotal. Repeat runs when a comparison matters.
- **Control arm:** a single-agent auto-research loop (one agent measures, profiles, changes,
  re-measures, repeats) on the same task with the same resources. AutoInfer's claim is that more
  agents and tokens give better results, and this control is the reference for that claim.

## LLM as judge (procedure: `.claude/skills/eval-run-qwen38/SKILL.md`; reports in `evals/<flavor>/<run>/`, outside the run folder
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
- [x] Single-agent control harness (`flavors/baseline/`): a very basic auto-research loop (one agent, the task's
      harness and GPU queue, same budget), and its run layout.
- [x] Answer key for `qwen38-27b-fp8-2x4090`: `answer_keys/qwen38-27b-fp8-2x4090/answer_key.md`
      (from v1 test-3, test-4 and qwen38_opt).
- [x] Judge and reconcile procedure: the `eval-run` skill (`.claude/skills/eval-run-qwen38/`), with
      `resources.py` (tokens per agent/role, wall clock, GPU-minutes from transcripts + ledger).
- [ ] The flavor should record the manager's session id (v1's agents/manager/meta.json has none;
      resources.py finds the transcript by start time).
- [x] Keep run agents away from `answer_keys/` and `evals/`: deny rules + a classifier rule, from
      `new_run` (design/open_questions.md, "Keeping run agents in their run").
- [ ] **Cheat audit** (human, 2026-11-07; built as the `audit-run` skill + `audit_scan.py`, scan
      tested on all runs, judge steps not yet tested): a skill in the judging process that checks a
      run's transcripts for any access to answer keys, evals, design notes, other runs or
      qwen38_opt's findings (including attempts that were blocked), as a second check on top of the
      permissions.
- [ ] Evaluate test-3 and test-4 with the skill, as the first trial of the framework.
- [ ] **Check in the next real run that the confinement and today's tooling fixes work** (human,
      2026-11-07). Tested only with Haiku probe agents so far (design/open_questions.md, "Keeping run
      agents in their run"), so fairly confident, not proven. After the run, check transcripts and files:
      - nothing read or listed outside the run: answer keys, evals, design, other runs, other sessions'
        `~/.claude/projects/` and `/tmp/claude-<uid>/` folders, the shared runtime caches, the repo via GitHub
        (run the cheat audit; note blocked attempts too);
      - no SendMessage outside the run got through, and in-run messaging still worked: run-scoped names
        (`<run>-<agent>`), replies after the guard hook's "use the run-scoped name" block;
      - the v1 manager started in its run folder and could read its own background-job output;
      - per-run caches: `$AUTOINFER_RUN/.cache/` filled, first inference server start compiled cold (how
        much slower), later starts warm;
      - `import vllm` under `task/env.sh` loaded the task's baseline tree;
      - legitimate work was not blocked (look for denials of normal commands);
      - the other fixes from test-4's feedback: `gpu wait` / background `gpu run` used instead of polling,
        `ws create --from-ws` (workers copying workspaces), `TRACE=graph` and `step_anatomy.py`;
      - the noise wording: agents kept measuring small gains instead of stopping early.
