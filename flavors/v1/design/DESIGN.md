# Flavor v1

The baseline AutoInfer algorithm: one run manager, round-based brainstorming, leaders that each own
one direction per round, workers that implement, and a trusted evaluator.
Bodies marked `TBD` are not designed yet. See [TODO.md](TODO.md). Detailed notes per topic are in
this folder (e.g. [evaluation.md](evaluation.md)). For a visual walkthrough of one round with a
concrete example, see [example_round.md](example_round.md).

## Agents
- **Manager**: **persistent** (one session for the whole run, with compaction). Owns the search
  tree, the shared KB (the only writer), the journal, the champion pointer and assignments; scores
  and selects ideas; talks to the human.
- **Leaders** (the brain): **persistent**: one long-running session each, carrying its context
  across rounds (with compaction). Each drives one direction per round: plans experiments, picks
  starting points, sets up worker workspaces, files experiment records into the bank, keeps a local
  KB, and reports findings back. Also brainstorms (exploitation end). **Responsible for the validity
  of everything it submits**: it reviews the raw diff against the task's constraints itself.
- **Workers** (the arms): implement and test what their leader asks, **confined to the workspace**
  the leader set up. **Persistent by default** (with compaction): the leader keeps giving follow-ups
  to the same workers, and decides when to start a fresh one instead (e.g. a new, unrelated
  experiment, or a worker stuck in a rut).
- **Brainstormers**: **temporary**: new sessions every round (fresh context is the point), each
  with a persona (a reasoning method). They see observations, not insights, and may chat with any
  leader (exploration end).

## What is recorded
| Store | Contents | Written by |
|---|---|---|
| Experiment records (bank) | One per experiment: who ran it (leader/team), starting commit, hypothesis, change (git commit), results, the leader's validity review, and two separate sections: **Observed** (what was measured/seen) and **Interpretation** (what we think it means) | the leader, from its workers' results; the evaluator appends official numbers |
| Search tree | Index of ideas; each node = an idea with its experiments, a summary/conclusion, novelty + upside scores, who proposed it, status (incl. `proposed` = not yet selected) | manager |
| Shared KB | Understanding of the system not tied to a single idea; each entry cites the experiments behind it | **manager only** (leaders propose insights in their reports) |
| Local KB | A leader's working notes, hypotheses, plans | that leader |
| Journal | The order of decisions and why | manager |

**Observations vs. insights.** Insights close questions; observations leave them open, so someone
new can draw a different conclusion from the same data. Brainstormers get observations; insights
are not their starting frame (they may still hear them when chatting with a leader).

## Global state
These are global variables in the pseudocode: available to every agent that needs them and never
passed in assignments. (Workers are confined to their workspace, so their leader copies what they
need into it.)
```python
TASK       # task folder: TASK.md (objective, metric, hard constraints, correctness criteria), harness
FOCUS      # the human's natural-language scope for this run
CONFIG     # number of leaders, max workers per leader (a limit, not a quota), max rounds, personas,
           # gpu_minutes_per_team_per_round (set by the human per run)
SEARCH_POLICY  # design/search_policy.md: plain text; how the manager picks what to work on next
               # (portfolio, continuing a winning leader's line, fallback). Isolated variable:
               # a run can override it, and the human can edit it mid-run.
LEDGER     # assumptions: frozen vs open
CHAMPION   # current best version: commit, official numbers, profile.
           # A PERFORMANCE REFERENCE, NOT A REQUIRED STARTING POINT: agents may start from any bank
           # entry. v1: synchronous, snapshotted at the start of each round and fixed until the next.
BANK       # experiment bank: every attempt (git branch/commit + record), pass or fail
SHARED_KB  # confirmed understanding of the system (written by the manager only)
TREE       # search tree (manager)
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
    return finalize()                                   # final report: champion, tree, journal
```

```python
initialize():
    CHAMPION  = evaluate(TASK.baseline)                 # official eval; also checks the harness works
    CHAMPION.profile = profile(CHAMPION)                # where the time goes
    facts     = bound_analysis(TASK, CHAMPION.profile)  # speed-of-light floor + gap breakdown
    LEDGER    = assumption_ledger(TASK, FOCUS)          # frozen vs open assumptions; human reviews
    TREE      = new_tree(root = FOCUS)
    SHARED_KB = [facts]; BANK = [CHAMPION]; JOURNAL = []
```

```python
brainstorm() -> directions:
    # out: k directions, each = (tree node, rationale, bottleneck addressed)

    # 1. Leaders: exploitation end, full context. Their last reports' "what I'd do next"
    #    (round 1: ask them with a task).
    ideas = [leader.next_steps for leader in leaders]

    # 2. Manager writes the round briefing: what you'd tell a colleague in an adjacent field.
    briefing = manager.write_briefing()     # problem, facts (profile/bound numbers, not just
                                            # interpretation), high-level summary of what was
                                            # tried and observed. Also a status report for the human.

    # 3. Brainstormers: exploration end. Fresh sessions, run in parallel, generators are pluggable
    #    (more of them, other models).
    for persona in CONFIG.personas:
        sample = random_sample(BANK, n)     # different per brainstormer (AlphaEvolve-style);
                                            # Observed sections only, no insights
        ideas += brainstormer(persona, briefing, sample,
                              tools = [query BANK (observations),
                                       multi-turn chat with any leader]).propose()

    # 4. Central scoring: one scorer with full context, on one scale.
    ideas = manager.dedupe(ideas)           # many proposals of the same idea = weak signal only
    for idea in ideas:
        idea.novelty, idea.upside = manager.score(idea)   # each 1-5
        idea.proposed_by = generator tag    # persona/model/leader, to learn which generators win

    # 5. Selection: a portfolio, not a ranking.
    directions = manager.select(ideas, k, SEARCH_POLICY)   # e.g. "if a leader improved the champion,
                                                            # continue its next step; 1 slot most novel"
    TREE.add(directions, status = assigned)
    TREE.add(ideas - directions, status = proposed)                  # backlog, visible to the human
    JOURNAL.append(why these directions)
```

Brainstormer personas: each persona is a **reasoning method**, not a character trait ("be
creative" does little). Initial set:
- **First principles / assumption challenger** (always included): list the assumptions the
  current engine makes (from the ledger and beyond) and ask what happens if each is broken; build up
  from what is truly fixed (hardware limits, the math of the model).
- **Physicist**: speed-of-light bounds: what must be moved or computed, and what doesn't have to be.
- **Systems / scheduling**: overlap, batching, pipelining, idle time.
- **Cross-domain transfer**: how HPC, databases or compilers solve the same shape of problem.
- **Literature scout**: web search for recent papers and repos.
- **Combiner**: take two random past experiments and propose their combination.

```python
manager.assign(directions, leaders) -> tasks:
    # One task string per leader:
    #   what to work on: the direction + rationale
    #   GPU budget for this round (CONFIG default; the manager may adjust it per direction)
    #   optional: scope constraints, pointers to related experiments in BANK
    # NOT included (global): correctness criteria, metric, CHAMPION, BANK, SHARED_KB.
    # Which leader continues which line is decided by SEARCH_POLICY.
```

```python
execute(tasks) -> reports:                      # all leaders in parallel; barrier at the end
    return parallel([leader.run(task) for leader, task in tasks])

leader.run(task) -> report:
    # The direction is high level: the leader first thinks about how to break it into experiments,
    # in what order, and what can run in parallel (up to CONFIG.max_workers_per_leader; fewer is fine).
    # TBD: what exactly a leader does (e.g. reviewing past results in BANK/SHARED_KB, planning,
    # when to profile); see TODO.md
    loop:
        plan = leader.plan(task, local_kb)              # next experiments; may revise the plan
        for exp in plan (in parallel, one worker each):
            start = leader.pick_start(exp)              # CHAMPION or any BANK entry
            ws = leader.setup_workspace(worker, start)  # clone/worktree + branch; copy in what the
                                                        # worker needs (task brief, metric, scripts,
                                                        # relevant excerpts) as files in the workspace
            wreport = worker.run(exp.task)              # first task says what to read first and
                                                        # gives a GPU budget; follow-ups: another task
                                                        # TBD: validation funnel (cheap tests before
                                                        # GPU-heavy ones); see TODO.md
            record = leader.file_record(wreport, ws)    # leader moves branch + record into BANK
            record.validity = leader.review(record.diff, TASK)   # the leader signs off: raw diff vs
                                                                 # hard constraints / correctness
            if record.validity.ok and leader.wants_official_eval(record):
                evaluate(record.candidate)              # appends official numbers
            local_kb.update(observations)
        if leader reached a useful checkpoint or manager says stop: break
            # out of budget = no new GPU jobs, but GPU-free work may continue.
            # Need not finish the direction: propose continuing in "what I'd do next".
            # See budget_and_stopping.md
    return report:
        # summary of findings
        # + experiments run, each linked to its BANK record (candidate commit + performance)
        # + proposed insights (the manager decides what enters SHARED_KB)
        # + what I'd do next (feeds the next brainstorm)

worker.run(task) -> report:                     # confined to its workspace
    # task:   the experiment: what to try, what to measure, when it's done, GPU budget
    # may submit unofficial GPU jobs (smoke tests, microbenchmarks, short benches, profiles)
    # through the GPU queue, within its budget
    # report: summary + experiment record (Observed / Interpretation) + candidate commit + numbers
```

```python
evaluate(candidate) -> record:          # OFFICIAL; requested by a leader only, after its review.
                                        # Trusted: frozen harness agents can't modify.
    job = gpu_queue.submit(official, candidate)   # exclusive GPUs; charged to the team's budget
    build(candidate.commit)                       # clean build from the commit
    numbers = bench(c=16, 400 prompts) + gsm8k(n=250)   # one server start; bench also gives
                                                        # MTP acceptance
    return append_only_record(candidate, numbers) # only official numbers count
```
Unofficial GPU jobs (any agent, within budget) go through the same queue and never count. Details,
costs and an end-to-end example: [evaluation.md](evaluation.md).

```python
collect_and_update(reports):
    TREE.update(reports)                    # manager turns reports into idea nodes + experiments;
                                            # may reorganize the tree
    winners = BANK records with official numbers that beat CHAMPION     # TBD: champion lifecycle
    if winners:
        CHAMPION = integrate(CHAMPION, winners)   # TBD who; re-evaluate combined
        CHAMPION.profile = profile(CHAMPION)      # the bottlenecks have changed
        SHARED_KB.update(bound_analysis(...))
    SHARED_KB.update(accepted insights from reports)   # manager is the only writer
    JOURNAL.append(round summary)
    report_to_human()
```

These run throughout, outside the loop:
- **Human ↔ manager at any time**: status, redirect, change the focus or ledger, pause or kill a
  leader. The manager can do the same to leaders in the middle of `execute`.
- **Asynchronous messages**: manager ↔ leader, leader ↔ worker.

## Decisions specific to v1
- Barrier rounds (tentative; rolling is a candidate for a future flavor).
- The search tree is one free-form markdown file; a node = an idea with experiments attached.
- Agent lifetimes: manager and leaders are persistent (compaction keeps them going); workers are
  persistent by default, and the leader decides when to start a fresh one; brainstormers are new
  every round.
- What to work on next is decided by `search_policy.md` (an isolated, editable variable).
- Leaders decompose their direction into experiments; max workers per leader is a limit, not a quota.
- Two idea scores only, novelty and upside, each 1-5, assigned centrally by the manager.
- The manager is the only writer of the shared KB.
- One interface for persistent agents: `run(task: str) -> report: str`.
- Global state is not passed in assignments. The champion is a performance reference, not a
  required starting point; v1 snapshots it at the start of each round.
- Leaders set up worker workspaces, choose starting points, and file experiment records into the
  bank; workers are confined to their workspace.
- The leader reviews validity itself (no separate judge) and is responsible for what it
  submits. Only leaders request official evals; any agent may run unofficial GPU jobs within its
  GPU budget (manager → leader → worker). Official eval = clean build, c=16 bench + GSM8K-250.
- Budget = GPU-minutes per team per round, enforced (team total only) by the GPU queue; agents are
  trusted to use the queue. Leaders stop at any useful checkpoint; the run stops at max rounds, when
  the manager finds nothing worth running, or when the human stops it.
