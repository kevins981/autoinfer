# CONFIG (v1 defaults; copied into each run as config.md, where the human may edit it)
Free text: the manager reads it and passes the relevant parts into agents' prompts.

- leaders: 3 (one direction each per round)
- max workers per leader: 3 (a limit, not a quota)
- max rounds: 5
- GPU budget per team per round: 40 GPU-minutes
- brainstormers (one persona prompt each):
  - systematic engineer: careful and methodical; works from the profile and the bank: where the
    time goes, what has been tried, what the numbers imply, what hasn't been covered; also spots
    combinations of past wins.
  - smart colleague from an adjacent field: very capable, not an inference specialist; explores
    widely, questions assumptions, aims for surprising ideas the specialists would not think of.
- human checkpoints: no blocking checkpoint after the task definition; a short summary to the
  human after every round.
- models (role: model / effort; read by `agent start` and `new_run`, so keep this format):
  - manager: opus / medium
  - leader: opus / medium
  - worker: sonnet / medium
  - brainstormer: opus / medium
