# Flavor v1

The baseline AutoInfer algorithm: one run manager, round-based brainstorming, leaders that each own
one direction per round, workers that implement, and a trusted evaluator with an independent judge.
Bodies marked `TBD` are not designed yet. See `../../design/open_questions.md`.

## Agents
- **Manager**: owns the search tree, the journal, the champion pointer and assignments; talks to
  the human.
- **Leaders**: each owns one direction (a subtree) per round; plans experiments, keeps a local KB,
  reports its subtree back.
- **Workers**: implement and test one experiment in their own worktree.
- **Judge**: independent; sees only the diff, the rubric and the numbers.
- **Ideators**: TBD (separate panel or a leader mode).

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
    # in:  tree, shared KB, champion + its profile, ledger, focus, explore/exploit mix
    # out: k directions, each = (tree node, rationale, bottleneck addressed, est. ceiling,
    #      starting baseline(s) from the bank, budget)
    candidates = generate_ideas(state)                  # TBD: panel, lenses, context exposure
    candidates = filter_against_kb(candidates, state)   # drop already-tried unless conditions changed
    directions = manager.select(candidates, k, mix)
    state.tree.add(directions, status = assigned)
    state.journal.append(why these directions)
```

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
            shared_kb.push(confirmed insights)              # cite eval records + baseline
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
    curate(state.shared_kb)
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
