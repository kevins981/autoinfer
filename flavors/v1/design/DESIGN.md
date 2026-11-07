# Flavor v1

The baseline AutoInfer algorithm: one run manager, round-based brainstorming, leaders that each own
one direction per round, workers that implement, and a trusted evaluator.
**The role prompts in [../agents/](../agents/) carry this pseudocode, split by role** (the manager
gets all of it, each other role its own functions, `common.md` the global state and interface),
with comments on how each step is done. Keep them in sync when the algorithm changes.
Bodies marked `TBD` are not designed yet. See [TODO.md](TODO.md). Detailed notes per topic are in
this folder (e.g. [evaluation.md](evaluation.md)). For a visual walkthrough of one round with a
concrete example, see [example_round.md](example_round.md).

## Agents
- **Manager**: **persistent** (one session for the whole run, with compaction). Owns the search
  graph, the shared KB (the only writer), the journal, the champion pointer and assignments; scores
  and selects ideas; talks to the human.
- **Leaders** (the brain): **persistent**: one long-running session each, carrying its context
  across rounds (with compaction). Each drives one direction per round: plans experiments, picks
  starting points, sets up worker workspaces, files experiment records into the bank, keeps a local
  KB, and reports findings back. In v1 leaders don't propose next directions (brainstormers do),
  but brainstormers may chat with them. **Responsible for the validity
  of everything it submits**: it reviews the raw diff against the task's constraints itself.
- **Workers** (the arms): implement and test what their leader asks, **confined to the workspace**
  the leader set up. **Persistent by default** (with compaction): the leader keeps giving follow-ups
  to the same workers, and decides when to start a fresh one instead (e.g. a new, unrelated
  experiment, or a worker stuck in a rut).
- **Brainstormers**: **temporary**: new sessions every round (fresh context is the point). The
  only source of ideas in v1. Each is defined by a **persona**: a prompt file in
  `agents/personas/`, listed in `CONFIG.brainstormers`. All of them can access everything (see [Brainstorming](#brainstorming));
  variety comes from the personas, not from what each one may see.

## What is recorded
| Store | Contents | Written by |
|---|---|---|
| Experiment records (bank) | One per experiment: id, team, performance, short summary (results = **Observed**, conclusion = **Interpretation**, kept separate), validity review, and a diff against the task baseline. See [Experiment record](#experiment-record) | the leader, from its workers' results; the evaluator appends official numbers |
| Search graph | What was tried: directions -> experiments (records) that build on each other (a DAG: a record that combines several has several parents), with results and conclusions, failures included, the champion's path marked. Untried ideas are in a separate idea archive (`ideas.md`: scores, who proposed it, round) | manager |
| Shared KB | Understanding of the system not tied to a single idea; each entry cites the experiments behind it | **manager only** (leaders propose insights in their reports) |
| Local KB | A leader's working notes, hypotheses, plans | that leader |
| Journal | The order of decisions and why | manager |

**Observations vs. insights.** Insights close questions; observations leave them open, so someone
new can draw a different conclusion from the same data. That is why results and conclusions are
kept separate. (v1 does not hide insights from brainstormers; a persona's prompt may tell it to
start from the observations.)

## Experiment record
One folder per experiment in the bank. Every field is free-form text.
```
bank/<id>/                # id by team, e.g. T1-003 (the manager may rename)
  record.md
    team:         T1
    builds_on:    BASE, one record, or several ("T1-001 + T2-003")
    performance:  e.g. "official: 689 tok/s at c=16 (+5% vs E0), GSM8K 93.6%",
                  or "unofficial: ~+1% (within noise)", or "build failed"
    summary:      short, for a quick scan:
                    hypothesis
                    change         (what was done)
                    results        = Observed: what was measured or seen
                    conclusion     = Interpretation: what we think it means
    validity:     the leader's review of the diff against TASK.md
  change.diff             # against the task baseline, so it applies to the baseline directly
  (optional) profiles, logs
```

## Global state
These are global variables in the pseudocode: available to every agent that needs them and never
passed in assignments. (Workers are confined to their workspace, so their leader copies what they
need into it.)
```python
TASK       # task folder: TASK.md (objective, metric, hard constraints, correctness criteria), harness,
           # and the baseline's profiling results (provided with the task: what round 1 starts from)
FOCUS      # the human's natural-language scope for this run
CONFIG     # number of leaders, max workers per leader (a limit, not a quota), max rounds,
           # gpu_minutes_per_team_per_round, brainstormers (one persona file each, in agents/personas/)
           # human_checkpoints: free text, e.g. "a short summary after every round" (default),
           #   "check in every hour", "wait for my OK on each round's directions"
           # (all set by the human per run). Knobs that steer an agent's behavior are text placed
           # into that agent's prompt (human_checkpoints -> the manager's prompt), not code paths.
SEARCH_POLICY  # design/search_policy.md: plain text; how the manager picks what to work on next
               # (portfolio, continuing a winning leader's line, fallback). Isolated variable:
               # a run can override it, and the human can edit it mid-run.
ASSUMPTION_LEDGER  # frozen vs open assumptions; agreed with the human in initialize()
CHAMPION   # current best version: a BANK record id + its official numbers + profile.
           # A PERFORMANCE REFERENCE, NOT A REQUIRED STARTING POINT: agents may start from any bank
           # entry. v1: synchronous: set between rounds and passed to leaders in each round's task.
           # (No mid-round "new best" notifications.)
BANK       # experiment bank: every attempt (record + diff vs the task baseline), pass or fail
SHARED_KB  # confirmed understanding of the system (written by the manager only)
GRAPH      # search graph, a DAG (manager)
JOURNAL    # decisions and why (manager)
```

## Interface
Every persistent agent (leader, worker) has the same interface:
```python
agent.run(task: str) -> report: str
```
`task` is free text of any length. It is the same argument for the first assignment and every
follow-up: a persistent agent just gets another `task`. `report` is free text, with links to bank
records rather than copies. What differs between roles is the role prompt, which global state the
agent can see, and where it runs.

## Algorithm
```python
AutoInfer():
    initialize()
    for round in 1..CONFIG.max_rounds:
        CHAMPION_snapshot = CHAMPION                    # synchronous: fixed for this round
        directions = brainstorm()                       # what to work on next
        if directions is empty: break                   # manager thinks nothing is worth running
                                                        # (the human can also stop the run any time)
        tasks = manager.assign(directions, leaders)     # one task string per leader
        reports = execute(tasks)                        # leaders work in parallel; barrier
        collect_and_update(reports)
    return finalize()                                   # final report: champion, graph, journal
```

```python
initialize():                                   # run by the manager, WITH the human
    # 1. Define the task together: a multi-turn conversation, BLOCKING (round 1 waits for the
    #    human's OK). See "Defining the task" below.
    TASK.definition, ASSUMPTION_LEDGER = manager.define_task_with_human(TASK, FOCUS)
    # 2. Baseline
    CHAMPION  = evaluate(TASK.baseline)                 # official eval (record BASE, empty diff);
                                                        # also checks the harness works
    CHAMPION.profile = TASK.baseline_profile            # provided with the task (no profiling here)
    GRAPH     = new_graph(root = FOCUS)
    SHARED_KB = []; BANK = [CHAMPION]; JOURNAL = []
```
No separate bound analysis: speed-of-light estimates are napkin math that leaders, workers and
brainstormers do themselves when they need them.

### Defining the task
The most important step of the run. Whatever the task leaves loose, leaders will eventually find
and exploit (reward hacking), and the validity review can only check what is written down. So the
manager and the human build the task definition **together, in a multi-turn conversation**, and
the manager's job is to dig for what is missing, not to accept the first draft:
- **Correctness**: what "correct" means, spelled out. E.g. mathematically equivalent to the
  baseline (is fp reordering OK? where?), precision (FP8 weights, FP8 KV; nothing lower), same
  sampling settings, GSM8K must hold.
- **Hard constraints**: what must not change (hardware, model, workload, serving features such
  as the 128K max context).
- **Edge cases and loopholes**: the manager actively looks for ways to "win" without a real
  speedup, and asks the human about each. For the qwen task, e.g.: caching outputs for the fixed
  benchmark prompts; producing fewer or shorter outputs; changing the number of MTP draft tokens,
  the temperature, or thinking on/off; shrinking the max context to free memory; anything tuned to
  seed 0 or to these 400 prompts only.
- **Assumption ledger**: which assumptions are frozen and which are open for this run (given
  the focus).

Output: the agreed task definition (correctness criteria, hard constraints, known loopholes) and
ASSUMPTION_LEDGER, saved with the run; the validity review checks against them. The human decides
whether improvements go back into `tasks/<task>/` for future runs. Likely packaged as a manager
skill (a checklist of what to ask about) once we have run it a few times.

```python
brainstorm() -> directions:
    # out: one direction per leader (k = number of leaders), each = (graph node, rationale,
    #      bottleneck addressed)
    # Skipped when the search policy's continuing leaders already take every slot.

    # 1. Manager writes the round briefing: what you'd tell a colleague in an adjacent field.
    briefing = manager.write_briefing()     # problem, facts (profile/bound numbers, not just
                                            # interpretation), high-level summary of what was
                                            # tried and observed. Also a status report for the human.
                                            # Round 1: the task, the focus and the baseline profile.

    # 2. Brainstormers: the only idea source in v1. Fresh sessions, run in parallel.
    for persona in CONFIG.brainstormers:    # a persona = a prompt file in agents/personas/
        ideas += brainstormer(persona, briefing,
                              access = everything: TASK (incl. the baseline profile), BANK,
                                       SHARED_KB, GRAPH, JOURNAL, multi-turn chat with any leader,
                                       profile any record, e.g. a new CHAMPION
                                         (GPU queue, 15 GPU-min each per round; saved into that
                                         record's folder so others reuse it)).propose()

    # 3. Central scoring: one scorer with full context, on one scale.
    ideas = manager.dedupe(ideas)           # many proposals of the same idea = weak signal only
    for idea in ideas:
        idea.novelty, idea.upside = manager.score(idea)   # each 1-5
        idea.proposed_by = persona          # to learn which personas win

    # 4. Selection: a portfolio, not a ranking.
    directions = manager.select(ideas, len(leaders), SEARCH_POLICY)   # e.g. "if a leader improved the champion,
                                                            # continue that line; 1 slot most novel"
    GRAPH.add(directions, status = assigned)
    IDEAS.add(ideas - directions, round = r)
        # backlog = an archive, not a queue: visible to the human and to brainstormers, never
        # selected directly. The manager selects only from this round's ideas; an old idea comes
        # back only if a brainstormer proposes it again (updated for what is known now).
    JOURNAL.append(why these directions)
```

### Brainstorming
- **Defined in CONFIG.** `CONFIG.brainstormers` is a list of persona prompts, one brainstormer
  each. A persona is a short prompt file (`agents/personas/<name>.md`): what kind of thinker this is
  and how it works. `agent start --persona <name>` appends it to the brainstormer's prompt.
- **Everyone can see everything.** Technically each brainstormer can access all global state,
  chat with any leader, look at past experiments and profile any record. Variety comes from the
  personas. If a persona should ignore something (e.g. start from observations only), its prompt
  says so; nothing enforces it.
- **Bootstrap (round 1).** There are no experiments, leaders' findings or KB yet, so all
  brainstormers start from the same inputs: the task, the focus and the **baseline profile that
  comes with the task**. They differ only by persona. From round 2 on they also have the bank,
  the KB, the graph and the leaders to talk to.
- **Default personas** (v1):
  - **Systematic engineer**: careful and methodical. Works from the profile and the bank: where
    the time goes, what has been tried, what the numbers imply, what hasn't been covered. Also
    spots combinations of past wins.
  - **Smart colleague from an adjacent field**: very capable, not an inference specialist.
    Explores widely and questions assumptions; aims for surprising ideas that the specialists
    would not think of.
- Other personas to try later: first principles / assumption challenger, physicist
  (speed-of-light bounds), systems / scheduling, cross-domain transfer, literature scout (web
  search).

```python
manager.assign(directions, leaders) -> tasks:
    # One task string per leader:
    #   what to work on: the direction + rationale
    #   GPU budget for this round (CONFIG default; the manager may adjust it per direction)
    #   the current CHAMPION (record id + official numbers): the reference to beat
    #   optional: scope constraints, pointers to related experiments in BANK
    # NOT included (global, read on demand): correctness criteria, metric, BANK, SHARED_KB.
    # Which leader continues which line is decided by SEARCH_POLICY.
```

```python
execute(tasks) -> reports:                      # all leaders in parallel; barrier at the end
                                                # while waiting: the manager checks that running
                                                # leaders make progress (see Decisions: stragglers)
    return parallel([leader.run(task) for leader, task in tasks])

leader.run(task) -> report:
    # The direction is high level: the leader first thinks about how to break it into experiments,
    # in what order, and what can run in parallel (up to CONFIG.max_workers_per_leader; fewer is fine).
    # Everything below is the leader's judgment; see "Leader behavior" for its role prompt.
    loop:
        plan = leader.plan(task, local_kb)              # next experiments; may revise the plan
        for exp in plan (in parallel, one worker each):
            start = leader.pick_start(exp)              # CHAMPION or any BANK entry
            ws = leader.setup_workspace(worker, start)  # baseline + start's diff; copy in what the
                                                        # worker needs (task brief, metric, scripts,
                                                        # relevant excerpts) as files in the workspace
            wreport = worker.run(exp.task)              # first task says what to read first and
                                                        # gives a GPU budget; follow-ups: another task
                                                        # cheap tests before expensive ones
                                                        # (guidance; see Leader behavior)
            record = leader.file_record(wreport, ws)    # record.md + diff vs baseline into BANK
            record.validity = leader.review(record.diff, TASK)   # the leader signs off: raw diff vs
                                                                 # hard constraints / correctness
            if record.validity.ok and leader.wants_official_eval(record):   # typically: the
                                                # unofficial result beats CHAMPION by > noise
                evaluate(record.candidate)              # appends official numbers
            local_kb.update(observations)
        if leader reached a useful checkpoint or manager says stop: break
            # out of budget = no new GPU jobs, but GPU-free work may continue.
            # Need not finish the direction: the report says what was left open.
            # See budget_and_stopping.md
    return report:
        # summary of findings
        # + experiments run, each linked to its BANK record id (+ performance)
        # + proposed insights (the manager decides what enters SHARED_KB)
        # + what was left open (v1: no "what I'd do next"; brainstormers propose directions)

worker.run(task) -> report:                     # confined to its workspace
    # task:   the experiment: what to try, what to measure, when it's done, GPU budget
    # may submit unofficial GPU jobs (smoke tests, microbenchmarks, short benches, profiles)
    # through the GPU queue, within its budget
    # report: summary (Observed / Interpretation) + numbers; the change stays in its workspace
```

### Leader behavior
v1 leaves almost everything to the leader. Its role prompt says:
- **The direction is intentionally high level and somewhat vague.** It is up to you to decide how
  to explore it: which experiments, in what order, what runs in parallel. The goal is everyone's
  goal: the best result on the task's metric.
- **You may go outside the direction, if the evidence says so.** The direction is the
  brainstorm's best guess, and you will learn things it didn't know. Rule of thumb:
  - Reshaping it (narrowing, widening, reinterpreting it in its spirit): free.
  - Pivoting to something outside it: OK when **your own evidence from this round** (a profile, a
    measurement, a failed experiment) says it is more promising. Say so in the report: what you
    changed, why, and the evidence.
  - A new idea with no evidence behind it: don't chase it. Put it in the report as left open; the
    next brainstorm can pick it up. (Generating ideas is the brainstormers' job.)
- **What to read first is up to you**: related BANK records, SHARED_KB, your local KB, the
  CHAMPION's profile, whatever helps.
- **Be aware of your GPU budget** and allocate it to whatever looks most promising; shift it as
  results come in.
- **Up to `CONFIG.max_workers_per_leader` workers in parallel.** A limit, not a quota: use fewer if
  that is better.
- **Starting points are your call**: the CHAMPION by default, or any BANK record the experiment
  builds on.
- **Workers**: give follow-ups to the same worker by default; start a fresh one for an unrelated
  experiment or when a worker is stuck in a rut.
- **Cheap before expensive** (guidance, not a gate): napkin math (Amdahl, roofline) →
  microbenchmark with real shapes → one batched forward pass + profile → short unofficial bench →
  official eval. Skip steps when they don't fit; proxies can mislead and napkin math can be wrong
  about a novel idea. **Record the napkin-math estimate next to the measurement**: a mismatch is
  itself an observation. (Costs: [evaluation.md](evaluation.md).) Workers get the same guidance.
- **Official evals cost ~7 GPU-minutes.** Request one after your validity review, typically when
  the unofficial result beats the CHAMPION by more than the noise (~3%). It is your call.

```python
evaluate(candidate) -> record:          # OFFICIAL; requested by a leader only, after its review.
                                        # Trusted: frozen harness agents can't modify.
    job = gpu_queue.submit(official, candidate)   # exclusive GPUs; charged to the team's budget
    build(TASK.baseline + candidate.diff)         # clean build: the baseline + the record's diff
    numbers = bench(c=16, 400 prompts) + gsm8k(n=250)   # one server start; bench also gives
                                                        # MTP acceptance
    return append_only_record(candidate, numbers) # only official numbers count
```
Unofficial GPU jobs (any agent, within budget) go through the same queue and never count. Details,
costs and an end-to-end example: [evaluation.md](evaluation.md).

```python
collect_and_update(reports):
    GRAPH.update(reports)                   # manager turns reports into directions + experiments
                                            # (edges = builds_on); may reorganize the outline
    best = the BANK record from this round with the best official numbers
    if best beats CHAMPION consistently (no minimum %) and the gates pass:
        CHAMPION = best                           # manager promotes; starting point doesn't matter
        # No re-profiling here: the bottlenecks have changed, and next round's brainstormers
        # profile the new champion themselves (see brainstorm()); the manager may turn what they
        # find into SHARED_KB entries.
    # No integrate() step: any leader may build on any record, another team's included (a
    # combination is a record with several builds_on), and combining is also a brainstorm idea.
    SHARED_KB.update(accepted insights from reports)   # manager is the only writer
    JOURNAL.append(round summary)
    report_to_human()                       # as CONFIG.human_checkpoints says (default: a short
                                            # round summary; no blocking checkpoint)
```

These run throughout, outside the loop:
- **Human ↔ manager at any time**: status, redirect, change the focus or the assumption ledger, pause or kill a
  leader. The manager can do the same to leaders in the middle of `execute`.
- **Asynchronous messages**: manager ↔ leader, leader ↔ worker.

## Decisions specific to v1
- Barrier rounds (tentative; rolling is a candidate for a future flavor).
- The search graph is one free-form markdown file of what was tried (directions -> records, a
  DAG via each record's builds_on, written as an indented outline);
  untried ideas live in `ideas.md`.
- Agent lifetimes: manager and leaders are persistent (compaction keeps them going); workers are
  persistent by default, and the leader decides when to start a fresh one; brainstormers are new
  every round.
- `initialize()`: the manager and the human define the task in a multi-turn conversation
  (correctness, hard constraints, loopholes, assumption ledger); round 1 waits for the human's OK.
  No separate bound analysis.
- Human checkpoints: the only blocking one by default is the task definition in `initialize()`.
  Beyond that, `CONFIG.human_checkpoints` is a free-text instruction placed into the manager's
  prompt (default: a short summary after every round). The human can step in at any time.
- Failure paths: no special mechanisms. Agents handle failures with judgment plus the existing
  rules: failed attempts are filed like any other record (an observation); a fixed version is a new
  record with a new official eval (budget limits retries); only official, valid results with GSM8K
  holding can become champion; all state is on disk, so a restarted agent resumes from it.
- Stragglers: no time limit, no rule against slow leaders. While waiting at the barrier, the
  manager checks that the leaders still running are making progress: they or their workers are
  actually doing something (new jobs, files, records, messages), not necessarily improving. Slow
  but active: wait. Hanging: ask, then kill or restart it. (A per-round time limit was considered
  and deferred: too much complexity for v1.)
- Running out of tokens or plan usage: the manager stops the run and tells the human.
- What to work on next is decided by `search_policy.md` (an isolated, editable variable).
- One direction per leader per round (k = number of leaders).
- Ideas not selected go to `ideas.md` (with the round): an archive, not a queue.
  Only this round's ideas are selected; old ones come back only if a brainstormer re-proposes them,
  so stale ideas never run.
- Ideas come only from brainstormers (leaders don't propose next steps). Brainstormers are listed
  in CONFIG, one persona prompt each; all can access everything, and variety comes from the
  personas. Default: a systematic engineer and a smart colleague from an adjacent field.
- The task provides the baseline profile; round 1's brainstormers start from it.
- Leaders decompose their direction into experiments; max workers per leader is a limit, not a quota.
  How to explore a direction (what to read, experiments, budget split, starting points, when to
  request an official eval) is the leader's judgment, guided by its role prompt. Cheap tests
  before expensive ones is guidance, not a gate. Leaders may leave their direction when their own
  evidence points elsewhere, and say so; new ideas without evidence go to the next brainstorm.
- Two idea scores only, novelty and upside, each 1-5, assigned centrally by the manager.
- The manager is the only writer of the shared KB.
- One interface for persistent agents: `run(task: str) -> report: str`.
- Global state is not passed in assignments, except the champion: it is part of each round's
  task to every leader (synchronous; no mid-round notifications). The champion is a performance
  reference, not a required starting point.
- The manager promotes a new champion between rounds: the round's best official result, if it
  beats the champion by more than the noise and GSM8K holds. No special integration step:
  combinations come from brainstorming.
- Bank records: id (by team), team, performance (string), short summary, validity review, and a
  diff against the task baseline (no branches).
- Leaders set up worker workspaces, choose starting points, and file experiment records into the
  bank; workers are confined to their workspace.
- The leader reviews validity itself (no separate judge) and is responsible for what it
  submits. Only leaders request official evals; any agent may run unofficial GPU jobs within its
  GPU budget (manager → leader → worker). Official eval = clean build, c=16 bench + GSM8K-250.
- Budget = GPU-minutes per team per round, enforced (team total only) by the GPU queue; agents are
  trusted to use the queue. Leaders stop at any useful checkpoint; the run stops at max rounds, when
  the manager finds nothing worth running, or when the human stops it.
