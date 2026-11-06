# CONFIG (v1 defaults; copied into each run as config.md, where the human may edit it)
Mostly free text: the manager reads it every round and passes the relevant parts into agents'
tasks. The `models` lines are also read by `agent start` and `new_run` (keep their format).

- leaders: 1 (one direction each per round; team ids T1, T2, ...)
- max workers per leader: 2 (a limit, not a quota)
- max rounds: 5
- GPU budget per team per round: 80 GPU-minutes. How it was sized (qwen task):
  - one unofficial GPU job ≈ 10 min: server startup 1.5-4 min + a batch of tests or a short
    bench 3-6 min;
  - each worker ≈ 3 jobs per round (e.g. measure, iterate, confirm) → ~30 min, x 2 workers = 60;
  - plus ~2 official evals x ~7 min = 14 → ~75, rounded up to 80.
  One GPU job runs at a time for the whole run, so a round needs at least
  leaders x 80 GPU-minutes of wall time (plus brainstormer profiling, unbudgeted).
- brainstormers (persona files in flavors/v1/agents/personas/, one brainstormer each):
  - systematic_engineer
  - adjacent_field
- human checkpoints: no blocking checkpoint after the task definition; a short summary to the
  human after every round.
- models (role: model / effort; Opus 5.5 to think, Sonnet 5.5 to implement):
  - manager: claude-opus-5-5 / medium
  - leader: claude-opus-5-5 / medium
  - worker: claude-sonnet-5-5 / medium
  - brainstormer: claude-opus-5-5 / medium
