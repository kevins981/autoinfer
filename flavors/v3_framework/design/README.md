# v3_framework (2026-11-14)

A flavor with one team type, **sweep**, to measure what a systematic framework adds. Forked from
v2 at the state after test-6 (T56-T58 were v2's sweep teams).

What changed from v2:
- Only sweep teams (`agents/leader_sweep.md`); CONFIG keeps N of them open at all times and
  replaces closed ones. No inspiration cards, scouts, personas or other team types.
- **map.md replaces graph.md.** The shared map of the search space: what the space is, and what
  is known about each part (built, tested and why it won or lost, not worth doing and why,
  untried). Free-form: its purpose is set, its format comes from the teams. Leaders read it only
  after their own phase-1 sweep (files 1-4), may disagree with it, and propose changes as
  `proposals/<team>-map-<n>.md` (a change + why) at the end of each round. The manager is the
  only writer: it reviews each proposal as a light quality gate (reasoned, evidence for verdicts)
  and applies it, so parallel edits never collide.
- Phase 1 is blind to the whole history (map, KB, journal, reports, other teams), for every team,
  not only the first ones.
- `new_run` copies the framework (`--framework <folder>`, docs 01-07 only) into the run's
  `framework/` and denies the framework's source folder to agents.
- Exploit-style funding: seed + start grant, judged on wins.

Kept from v2: tools (gpu queue, ws, agent, spar, official_eval), sparring, speed of light, the
bank, shared_kb.md, track records, reminders. Tool specs: `infra/`, sparring: `sparring.md`.
