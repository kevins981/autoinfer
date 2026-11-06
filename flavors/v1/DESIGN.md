# Flavor v1

The baseline AutoInfer algorithm: one run manager, round-based brainstorming, leaders that each own
one direction per round, workers that implement, and a trusted evaluator with an independent judge.
Bodies marked `TBD` are not designed yet. See `../../design/open_questions.md`.

## Agents
- **Manager**: **persistent** (one session for the whole run, with compaction). Owns the search tree, the shared KB (the only writer), the journal, the champion
  pointer and assignments; scores and selects ideas; talks to the human.
- **Leaders**: **persistent**: one long-running session each, carrying its context across rounds
  (with compaction). Each owns one direction (a subtree) per round; plans experiments, keeps a
  local KB, reports its subtree and proposed insights back. Also brainstorms (exploitation end).
- **Workers**: implement and test experiments in their own worktree. **Persistent by default** (with
  compaction): the leader keeps giving follow-ups to the same workers, and decides when to start a
  fresh one instead (e.g. a new, unrelated experiment, or a worker stuck in a rut).
- **Brainstormers**: **temporary**: new sessions every round (fresh context is the point), each with a persona (a reasoning
  method). They see observations, not insights, and may chat with any leader (exploration end).
- **Judge**: independent; sees only the diff, the rubric and the numbers.

## What is recorded
| Store | Contents | Written by |
|---|---|---|
| Experiment records (bank) | One per experiment: who ran it (leader/team), baseline, hypothesis, change (git commit), results, and two separate sections: **Observed** (what was measured/seen) and **Interpretation** (what we think it means) | the worker that ran it, finalized by its leader; the evaluator appends official numbers + judge verdict |
| Search tree | Index of ideas; each node = an idea with its experiments, a summary/conclusion, novelty + upside scores, who proposed it, status (incl. `proposed` = not yet selected) | manager |
| Shared KB | Understanding of the system not tied to a single idea; each entry cites the experiments behind it | **manager only** (leaders propose insights in their reports) |
| Local KB | A leader's working notes, hypotheses, plans | that leader |
| Journal | The order of decisions and why | manager |

**Observations vs. insights.** Insights close questions; observations leave them open, so someone
new can draw a different conclusion from the same data. Brainstormers get observations; insights
are not their starting frame (they may still hear them when chatting with a leader).

## Algorithm
```python
AutoInfer(task, focus, config):
    # task:   task folder (TASK.md, harness, baseline engine, correctness rubric)
    # focus:  natural-language scope from the human ("speed up the FFN layer; keep KV cache, ...")
    # config: number of leaders, workers per leader, rounds, budgets, ...
    state = initialize(task, focus)
    for round in 1..config.max_rounds:
        directions = brainstorm(state)                  # what to work on next
        if directions is empty: break                   # manager thinks the scoped space is exhausted
        assignments = manager.assign(directions, leaders)
        results = execute(assignments, state)           # leaders work in parallel; barrier
        state = collect_and_update(state, results)
    return finalize(state)                              # final report: champion, tree, journal
```

```python
initialize(task, focus) -> state:
    champion = evaluate(task.baseline)                  # official eval; also checks the harness works
    profile  = profile(champion)                        # where the time goes
    facts    = bound_analysis(task, profile)            # speed-of-light floor + gap breakdown
    ledger   = assumption_ledger(task, focus)           # frozen vs open assumptions; human reviews
    tree     = new_tree(root = focus)
    shared_kb = [facts]; bank = [champion]; journal = []
    return state(tree, shared_kb, bank, champion, profile, ledger, journal)
```

```python
brainstorm(state) -> directions:
    # in:  tree, shared KB, experiment bank, champion + its profile, ledger, focus, selection policy
    # out: k directions, each = (tree node, rationale, bottleneck addressed,
    #      starting baseline(s) from the bank, budget)

    # 1. Leaders: exploitation end. One extra conversation turn to each persistent leader.
    ideas = [leader.propose_next() for leader in leaders]          # full context

    # 2. Manager writes the round briefing: what you'd tell a colleague in an adjacent field.
    briefing = manager.write_briefing(state)    # problem, facts (profile/bound numbers, not just
                                                # interpretation), high-level summary of what was
                                                # tried and observed. Also a status report for the human.

    # 3. Brainstormers: exploration end. Fresh sessions, run in parallel, generators are pluggable
    #    (more of them, other models).
    for persona in config.personas:
        sample = random_sample(state.bank, n)   # different per brainstormer (AlphaEvolve-style);
                                                # Observed sections only, no insights
        ideas += brainstormer(persona, briefing, sample,
                              tools = [query experiment bank (observations),
                                       multi-turn chat with any leader]).propose()

    # 4. Central scoring: one scorer with full context, on one scale.
    ideas = manager.dedupe(ideas)               # many proposals of the same idea = weak signal only
    for idea in ideas:
        idea.novelty, idea.upside = manager.score(idea, state)   # each 1-5
        idea.proposed_by = generator tag        # persona/model/leader, to learn which generators win

    # 5. Selection: a portfolio, not a ranking.
    directions = manager.select(ideas, k, config.selection_policy)   # plain text, human-editable,
                                                                      # e.g. "2 best upside, 1 most novel"
    state.tree.add(directions, status = assigned)
    state.tree.add(ideas - directions, status = proposed)            # backlog, visible to the human
    state.journal.append(why these directions)
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
execute(assignments, state) -> results:                 # one per leader, in parallel
    leader.run(direction, baselines, budget):
        local_kb = []
        loop:
            experiments = leader.plan(direction, local_kb)  # may add child ideas to its subtree
            for exp in experiments (in parallel, by workers):
                candidate = worker.implement(exp, baseline)    # own worktree; cheap proxy tests freely
                if candidate.promising:
                    record = evaluate(candidate)               # queued official eval + judge
                    bank.add(record)                           # every attempt kept, pass or fail
                local_kb.update(observations)
            propose_insights(confirmed insights)            # to the manager; cite records + baseline
            if budget exhausted or leader thinks direction exhausted or manager says stop: break
        return subtree      # idea nodes, each with experiments, summaries, links to records
```

```python
evaluate(candidate) -> record:                          # trusted; agents can't modify it
    wait_for_gpu_queue()
    numbers = bench(candidate) + gsm8k(candidate)
    verdict = judge(candidate.diff, task.rubric, numbers)   # independent agent, sees only the diff
    return append_only_record(candidate, numbers, verdict)
```

```python
collect_and_update(state, results) -> state:
    state.tree.merge(results.subtrees); manager may reorganize the tree
    winners = records that pass the judge and beat the champion
    if winners:
        state.champion = integrate(state.champion, winners)   # TBD who; re-evaluate combined
        notify all teams
        state.profile = profile(state.champion)               # the bottlenecks have changed
        state.shared_kb.update(bound_analysis(...))
    state.shared_kb.update(accepted insights from leaders' reports)   # manager is the only writer
    state.journal.append(round summary)
    report_to_human(state)
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
- Two idea scores only, novelty and upside, each 1-5, assigned centrally by the manager.
- The manager is the only writer of the shared KB.
