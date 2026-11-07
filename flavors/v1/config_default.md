# CONFIG (v1 defaults; copied into each run as config.md, where the human may edit it)
Mostly free text: the manager reads it every round and passes the relevant parts into agents'
tasks. The `models` lines are also read by `agent start` and `new_run` (keep their format).

- leaders: 2 (one direction each per round; team ids T1, T2, ...)
- max workers per leader: 1 at a time (the leader may stop one and start a fresh one)
- max rounds: 5
- GPU budget per team per round: 55 GPU-minutes. How it was sized (qwen task):
  - one unofficial GPU job ≈ 10 min: server startup 1-4 min + a batch of tests or a short
    bench 3-6 min;
  - one worker ≈ 3 jobs per round (e.g. measure, iterate, confirm) → ~30 min;
  - plus ~2 official evals x ~11 min = 22 → ~52, rounded up to 55.
  One GPU job runs at a time for the whole run, so a round needs at least
  leaders x 55 GPU-minutes of wall time, plus the brainstormers'.
- GPU budget per brainstormer per round: 15 GPU-minutes (measurements, profiles, prototypes:
  their choice).
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
