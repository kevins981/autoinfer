# v2: open items

These are not agreed yet. Where the designer has a suggestion it is labelled; none of it is
decided.

## Incentives and funding
- **Manager reserve.** Suggestion: about 10% of the pool for the manager's own work (profiling a
  new champion, spot-checking cited jobs).
- **Pool size for the first v2 run.** Suggestion: about 1,500 GPU-min. Test-5 used 562 GPU-min.
- **Reports must cite evidence** (GPU job ids, records). Funding now depends on reports, so the
  pull to oversell is stronger. Should the manager spot-check by re-running cited jobs?
- **Where the track record lives.** Suggestion: a per-team file the manager reads, because
  leaders get restarted and replaced.
- **Should forecast accuracy affect trust in a team's later forecasts?**
- **Thinking time costs nothing in GPU.** Should there also be a wall-clock reporting interval?
- **Credit for enabling another team's win.**

## Asynchronous mechanics
- **Promoting champions.** Suggestion: one candidate at a time through the official eval. A
  candidate built on an older champion is rebuilt on the current one and re-measured.
- **Who pays for porting onto the current champion?** Suggestion: the team.
- **What replaces v1's per-round reports to the human** (e.g. time-based, or one per promotion)?
- **The run's stopping rule** without `max_rounds`. Suggestion: when the pool is spent, or earlier if
  the manager sees nothing worth funding.
- **GPU contention.** There is one GPU pair, and benchmarks need both. Could microbenchmarks run on
  one GPU each? The `gpu` tool currently allocates both.

## GPU queue changes
- Charge every job to a team. Enforce the grant. Print the remaining balance on every job, which
  also serves as a built-in reminder.

## Idea generation (next discussion topic)
How explore teams come up with genuinely new ideas. Candidates from the human's notes:
- **First principles:** what are the real givens (hardware features, programming primitives),
  versus conventional wisdom?
- **Random subsets of features or principles per brainstormer** (AlphaEvolve-style), for
  diversity.
- **Old ideas revived** (the systolic-array example).
- **No list of "common optimizations" for explore teams.** It's fine for exploit, but limiting for
  novelty (the chess-heuristics analogy).
- **Proposers can say honestly "this is my best idea, but it's probably weak".**
- **The brainstormers' new role.**

## Other
- **Reminder prompts via hooks**, so long-running agents keep their instructions.
- **Judging v2:**
  - GPU-minutes per win by team type;
  - whether explore teams found anything outside the answer key;
  - how forecasts compared with outcomes.
- **A sceptic reviewer** for forecasts, instead of the manager.
