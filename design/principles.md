# Design principles

These are agreed in discussion. Flavors may differ in *how* they follow them.

## Structure and control
1. **Two loops.** Outer: flavors (compare AutoInfer algorithms). Inner: runs (optimize a task).
   Two kinds of agents match them: **designer agents** (launched in `autoinfer/`) iterate on
   flavors with the human; **run agents** use one flavor and never see the design discussion.
   A session the human launches in `flavors/<flavor>/` **is the run manager** (that folder's
   CLAUDE.md is the manager's instructions). The manager launches every other run agent with an
   explicit role prompt (`agents/<role>.md`) and with both CLAUDE.md files excluded.
2. **One run manager per run.** A single top-level agent with full visibility and control: start,
   pause, stop or kill, reallocate. The human talks to the manager, not to the logs. The same system
   serves human-in-the-loop and fully autonomous use; there are no separate modes.
3. **Scale by parameters, not redesign.** More teams, workers or ideators should mean better
   results without restructuring. Baseline shape: a two-level tree of leaders (persistent,
   strategy, compaction) and workers (implementation; may be short-lived).
4. **Agents are launched as Claude Code CLI sessions**, not via the built-in subagent tool, for
   flexibility and peer-to-peer messaging.
5. **Asynchronous communication at every level**: human↔manager, manager↔leader, leader↔worker,
   leader↔leader.
6. **Keep leaders' contexts clean.** Implementation tokens (code, builds, benchmarks, web research)
   are spent by workers.
7. **Bet on model capability.** Keep rigid structure and schemas to a minimum (free-form markdown
   tree and KBs); let agents decide (which baselines, tools, plans).

## Task and evaluation
8. **The task is separate from AutoInfer.** A task folder defines the objective, hardware, hard
   constraints (e.g. FP8 weights + FP8 KV cache), allowed and forbidden changes, and the
   correctness rubric. Its harness is read-only to agents. Serve flags and spec-decode settings are
   allowed if they keep the math the same.
9. **Trusted evaluation.** Only the evaluator's records count: frozen benchmark + GSM8K + an
   **independent LLM judge** that reads the raw diff (not the author's description) against the
   rubric. Records are append-only.
10. **The shared GPU goes through a job queue**, both for fair sharing and because a benchmark
    needs exclusive GPUs to be valid.

## Search
11. **Navigating the search space with diverse ideas is the core problem.** A single agent is too
    slow and too in-distribution.
12. **The human scopes each run** to a subset of the search space (focus + frozen assumptions).
    Starting from zero assumptions every time is wasteful.
13. **Ground ideas in specifics**: the speed-of-light bound analysis, the profile, and explicit
    assumptions about this model, hardware and workload. Generic questions get generic ideas.
14. **Facts vs. attempt history.** Ideators should get facts (measurements, bounds, conclusions with
    their conditions); a list of past attempts anchors them. Ideas are filtered against the full KB
    *after* generation (drop what was already tried unless its conditions changed).
15. **Exploration vs. exploitation is a scale, not two modes**: how much context an ideator sees,
    what it starts from (champion, a random bank entry, a combination, an empty region of the tree),
    and how ideas are selected (expected gain vs. ceiling × what we'd learn). The manager tunes the
    mix. Bold ideas need protected budget.
16. **Population, not just the champion** (AlphaEvolve). Every attempt is kept and can serve as a
    baseline. A failure records *why* it failed, so the idea can come back when conditions change.

## Knowledge and understanding
17. **Human understanding is a first-class output**, not just the best number: what was tried, what
    was observed, and why. The human must be able to step in at any time.
18. **The search tree is the index.** One markdown file owned by the manager. A node is an
    **idea**, with its experiments attached (the bottleneck it targets is an attribute, not the
    structure). Every experiment record maps to a node; full records are stored separately and
    linked. Cross-references are allowed, and the manager may reorganize the tree (git keeps
    history). Inner nodes summarize their children.
19. **Knowledge stores**: each team has a local KB (working notes, hypotheses); the shared KB holds
    confirmed insights, readable by every agent. Insights cite eval records and the baseline they
    were measured on.
20. **The journal** records the order of decisions and why each was made (the tree shows the
    current state).
