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
7. **Keep it simple; bet on model capability.** Don't over-constrain agents: interfaces are
   string in, string out (`run(task) -> report`); tree and KBs are free-form markdown; agents
   decide baselines, tools and plans. Add structure only where it gives something agents
   wouldn't get on their own, e.g. the mechanisms for diverse, out-of-distribution ideas, or
   trusted evaluation.

## Task and evaluation
8. **The task is separate from AutoInfer.** A task folder defines the objective, hardware, hard
   constraints (e.g. FP8 weights + FP8 KV cache), allowed and forbidden changes, and the
   correctness criteria. Its harness is read-only to agents. Serve flags and spec-decode settings are
   allowed if they keep the math the same.
9. **Trusted evaluation.** Only official records count: frozen harness (benchmark + GSM8K) that
   agents can't modify, append-only records. Validity (does the diff respect the task's
   constraints?) is checked by an agent reading the raw diff, not the author's description;
   flavors decide who (v1: the leader, which is responsible for what it submits).
10. **The shared GPU goes through a job queue**, both for fair sharing and because a benchmark
    needs exclusive GPUs to be valid.

## Search
11. **Navigating the search space with diverse ideas is the core problem.** A single agent is too
    slow and too in-distribution.
12. **The human scopes each run** to a subset of the search space (focus + frozen assumptions).
    Starting from zero assumptions every time is wasteful.
13. **Ground ideas in specifics**: the speed-of-light bound analysis, the profile, and explicit
    assumptions about this model, hardware and workload. Generic questions get generic ideas.
14. **Separate generation from selection.** Many pluggable idea generators (leaders, fresh
    brainstormers with different personas, other models), one central scorer with full context,
    and one selection policy. The selection policy is the main control knob; adding generators is
    how idea generation scales. Record which generator proposed each idea.
15. **Observations, not insights, for fresh thinking.** Insights close questions; observations
    leave them open. Brainstormers start from a briefing plus a random sample of raw observations
    (AlphaEvolve-style), not from our conclusions, and can question any leader the way you'd talk
    to a colleague in an adjacent field. Already-tried ideas are caught by central scoring.
16. **Exploration vs. exploitation is a scale, not two modes**: leaders (full context) at one
    end, fresh brainstormers at the other; selection is a portfolio (e.g. best upside plus most
    novel), not a ranking. Bold ideas need protected slots. Ideas that aren't selected stay in the
    tree as a visible backlog.
17. **Population, not just the champion** (AlphaEvolve). Every attempt is kept and can serve as a
    baseline. A failure records *why* it failed, so the idea can come back when conditions change.

## Knowledge and understanding
18. **Human understanding is a first-class output**, not just the best number: what was tried, what
    was observed, and why. The human must be able to step in at any time.
19. **The search tree is the index.** One markdown file owned by the manager. A node is an
    **idea**, with its experiments attached (the bottleneck it targets is an attribute, not the
    structure). Every experiment record maps to a node; full records are stored separately and
    linked. Cross-references are allowed, and the manager may reorganize the tree (git keeps
    history). Inner nodes summarize their children.
20. **Knowledge stores**: experiment records keep **Observed** and **Interpretation** separate;
    each leader has a local KB (working notes); the shared KB holds confirmed understanding of the
    system, cites the experiments and baseline behind it, and has **one writer, the manager**
    (leaders propose insights), so concurrent writes never need reconciling.
21. **The journal** records the order of decisions and why each was made (the tree shows the
    current state).
