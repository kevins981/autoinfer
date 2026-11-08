# CONFIG (v2 defaults; copied into each run as config.md, where the human may edit it)
Mostly free text: the manager reads it. The `models` lines are also read by `agent start` and
`new_run`, and the `GPU pool` and `seed grant` lines by `new_run`. Keep their format.

- teams: 3 (team ids T1, T2, ...; a replacement team gets a new id)
  - explore: 1
  - exploit: 2
- max workers per leader: 1 at a time (the leader may stop one and start a fresh one)
- GPU pool: 1500 GPU-minutes for the whole run (one GPU job runs at a time; queue time is free)
- seed grant: 20 GPU-minutes per new team (other grant sizes: funding_policy.md)
- human checkpoints: no blocking checkpoint after the task definition; a short report to the
  human on every new champion and every resync, and whenever asked.
- models (role: model / effort):
  - manager: claude-opus-5-5 / medium
  - leader_explore: claude-opus-5-5 / medium
  - leader_exploit: claude-opus-5-5 / medium
  - worker: claude-opus-5-5 / medium
