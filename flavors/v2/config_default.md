# CONFIG (v2 defaults; copied into each run as config.md, where the human may edit it)
Mostly free text: the manager reads it. The `models` and `slot` lines are also read by `agent start` and
`new_run`, the `GPU pool` and `seed grant` lines by `new_run`, the `inspiration` lines by
`inspire`, the `scout` line by `scout`, and the `sparring` line by `spar`. Keep their format.

- teams: 3 (team ids T1, T2, ...; a replacement team gets a new id)
  - explore: 1
  - exploit: 2
  - external: 0 (direction from a scout's deck of published techniques; design/external_teams.md)
- max workers per leader: 1 at a time (the leader may stop one and start a fresh one)
- GPU pool: 1500 GPU-minutes for the whole run (one GPU job runs at a time; queue time is free)
- seed grant: 20 GPU-minutes per new team (other grant sizes: funding_policy.md)
- human checkpoints: no blocking checkpoint after the task definition; a short report to the
  human on every new champion and every resync, and whenever asked.
- models (role: model / effort):
  - manager: claude-opus-5-5 / medium
  - leader_explore: claude-opus-5-5 / medium
  - leader_exploit: claude-opus-5-5 / medium
  - leader_external: claude-opus-5-5 / medium
  - worker: claude-opus-5-5 / medium
- model slots: none (optional: per-team models for comparing models. One line per slot,
  "  - slot <S>: leader <model> / <effort>, worker <model> / <effort>"; the manager opens each
  team with `agent start ... --slot <S>`, and a replacement team keeps its slot)
- sparring: codex gpt-6.1-sol / high (the `spar` tool's partner: backend codex|claude, model / effort;
  for Claude, `claude leader / medium` uses the calling leader's own model)
- scout: codex gpt-6.1-sol / medium, cards 15 (the `scout` tool, for external teams: agent, model /
  effort, cards per deck; one fresh deck per external team)
- inspiration cards per draw: principles 5, hardware 5, software 5 (the `inspire` tool; 0 turns a
  kind off)
  - inspiration hardware cards: platform-supermicro-741ge-2x-rtx4090
  - inspiration software cards: software-cuda13.0-torch2.14-linux5.15
