# v3_framework (2026-11-14)

A flavor with one team type, **sweep**, to measure what a systematic framework adds. Forked from
v2 at the state after test-6 (T56-T58 were v2's sweep teams).

What changed from v2:
- Only sweep teams (`agents/leader_sweep.md`); CONFIG keeps N of them open at all times and
  replaces closed ones. No inspiration cards, scouts, personas or other team types.
- **No shared map; each team keeps its own** (`reports/<team>-sweep/`, plus `update-<n>.md` at each
  resync and at closing). Leaders read other teams' maps only after their own phase 1. graph.md
  is gone too (record lineage lives in `builds_on` and champion.md).
- Phase 1 is blind to the whole history (map, KB, journal, reports, other teams), for every team,
  not only the first ones.
- `new_run` copies the framework (`--framework <folder>`, docs 01-07 only) into the run's
  `framework/` and denies the framework's source folder to agents.
- Exploit-style funding: seed + start grant, judged on wins.

Kept from v2: tools (gpu queue, ws, agent, spar, official_eval), sparring, speed of light, the
bank, shared_kb.md, track records, reminders. Tool specs: `infra/`, sparring: `sparring.md`.

## Decision log
- 2026-11-14, human: **dropped the shared map.md** ~15 min into framework-eval (switched in place, no
  restart: both teams had done phase 1 blind, no contested entries yet, 3.6 GPU-min spent). The first
  design had one map.md, owned by the manager, built from teams' map proposals (PR-like, manager as a
  light quality gate; disagreements kept as attributed "contested" entries). Reasons: two teams'
  maps already diverged on verdicts within minutes; a single shared verdict is a point of failure
  (a wrong "not worth doing" makes every later team skip that part of the space), and it made the
  manager the referee of technical truth. Each team's own map keeps disagreements as separate
  documents; later teams weigh them. Human progress tracking moves outside the run (the designer
  builds a coverage view from the teams' maps and records; agents never see it).
