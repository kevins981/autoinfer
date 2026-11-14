# CONFIG (run framework-eval; from the v3_framework defaults)
Mostly free text: the manager reads it. The `models` and `slot` lines are also read by `agent start` and
`new_run`, the `GPU pool` and `seed grant` lines by `new_run`, and the `sparring` line by `spar`. Keep
their format.

- teams: 2 sweep teams open at all times (team ids T1, T2, ...; a closed team is replaced by a new
  sweep team with a new id while the pool can fund one; agents/leader_sweep.md)
- sweep framework: framework/ (the framework documents, copied into the run by new_run)
- max workers per leader: 1 at a time (the leader may stop one and start a fresh one)
- GPU pool: 1000 GPU-minutes for the whole run (one GPU job runs at a time; queue time is free)
- seed grant: 20 GPU-minutes per new team (other grant sizes: funding_policy.md)
- human checkpoints: no blocking checkpoint after the task definition; a short report to the
  human on every new champion and every resync, and whenever asked.
- models (role: model / effort):
  - manager: claude-opus-5-5 / medium
  - leader_sweep: claude-opus-5-5 / medium
  - worker: claude-sonnet-5-5 / medium
- model slots:
  - slot A: leader claude-opus-5-5 / medium, worker claude-sonnet-5-5 / medium
- sparring: codex gpt-6.1-sol / high (the `spar` tool's partner: backend codex|claude, model / effort;
  for Claude, `claude leader / medium` uses the calling leader's own model)
- speed of light: on (the first task points to task/SPEED_OF_LIGHT.md, if the task has one)
