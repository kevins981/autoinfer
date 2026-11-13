# CONFIG (test-6: v2 with 2 explore teams, seeded from test-5)
Mostly free text: the manager reads it. The `models` and `slot` lines are also read by `agent start` and
`new_run`, and the `GPU pool` and `seed grant` lines by `new_run`. Keep their format.

- teams (from the 2026-11-10 resume, human): 1 external team only (design/external_teams.md), in
  slot C; no explore or exploit teams. A replacement for it is another external team (fresh scout
  deck) in slot C. (Before: 2 explore teams.) Team ids continue: T32, T33, ...
  - exploit-long: 1   (renamed from "explore", human 2026-11-11: role leader_exploit_long, type exploit-long; T34 closed, not replaced. Human, 2026-11-11: one new exploit-long team in slot B, alongside the exploit team; same settings, no persona)
  - exploit-long, megakernel pair (human, 2026-11-11): 2 more exploit-long teams, opened at the same
    time, slot F each (Opus leader and Opus worker), no persona; speed of light, sparring, cards as
    usual. Both get this direction, word for word in the first task:
    > **Your direction (set by the human): megakernels.** Merge the forward pass's many small
    > kernels into larger ones, from a whole layer up to the whole forward pass. The goal is to
    > remove the time the GPU sits idle or underused between and around kernels, and to schedule and
    > overlap compute, memory traffic and communication in ways separate kernels can't. How you do
    > it is your call: the granularity, the design, and whether you build on existing research or
    > libraries or write your own. Your proposal and any pivots stay within this direction.
  - exploit: 1   (human, 2026-11-11: one exploit team in slot A, after the research sets; standard exploit grants, no persona)
  - external: 0   (T32/T33 closed, not replaced)
  - moonshot: 4   (human, 2026-11-10: T35 slot A, T36/T37 slot D, all closed; a fourth in slot E; agents/leader_moonshot.md)
  - research: 4   (human, 2026-11-11: a framing test, agents/leader_research.md; slot F each.
    T39-T41 were closed before doing any work (the prompt changed); open 4 new teams.
    Personas: the first none, the second research_ownership, the third research_reviewer, the
    fourth research_radical.
    No stop after the first proposal any more (human, 2026-11-11): research teams continue as
    usual. T42 and T43 are reopened with their leaders resumed (same session); their first
    proposals (T42-1, T43-1) are handled as normal research proposals.)
  - research, second set (human, 2026-11-11): 4 more research teams in slot F, personas: the
    first none, the second research_ownership, the third research_reviewer, the fourth
    research_radical (re-added by the human: the two sets are compared). Open them now,
    alongside T42/T45 while those finish.
  - research-handoff: 2   (human, 2026-11-13: agents/leader_research_handoff.md; slot A each (Opus
    leader, Sonnet worker), opened at the same time, no persona; memo from the `research handoff
    memo` line in place of cards; full history; speed of light; sparring; funded like moonshots
    (FUNDING_POLICY). Not replaced when they close.)
- max workers per leader: 1 at a time (human, 2026-11-10; was 2) (the leader may stop one and start a fresh one)
- GPU pool: 2000 GPU-minutes for the whole run (one GPU job runs at a time; queue time is free)
  (raised from 1500 by the human, 2026-11-11)
- seed grant: 20 GPU-minutes per new team (other grant sizes: funding_policy.md)
- human checkpoints: no blocking checkpoint; a short report to the human on every new champion and
  every resync, and whenever asked.
- models (role: model / effort):
  - manager: claude-sonnet-5-5 / medium
  (manager on Sonnet: human, 2026-11-10)
  - leader_explore: claude-opus-5-5 / medium
  - leader_exploit: claude-opus-5-5 / medium
  - leader_external: claude-opus-5-5 / medium
  - leader_moonshot: claude-opus-5-5 / medium
  - worker: claude-sonnet-5-5 / medium
- model slots (from the resume on 2026-11-09; each explore team is in one slot, a replacement team
  keeps its slot; the other `models` lines then apply only to the manager):
  - slot A: leader claude-opus-5-5 / medium, worker claude-sonnet-5-5 / medium   (leader changed from Fable 5.1 by the human, 2026-11-10; leader-31 keeps Fable)
  - slot B: leader claude-opus-5-5 / medium, worker claude-sonnet-5-5 / medium   (workers changed from Opus by the human, 2026-11-10; workers started before keep Opus)
  - slot C: leader claude-opus-5-5 / medium, worker claude-sonnet-5-5 / medium   (external teams; added by the human, 2026-11-10)
  - slot D: leader claude-fable-5-2 / medium, worker claude-sonnet-5-5 / medium   (second moonshot team; added by the human, 2026-11-10)
  - slot E: leader claude-opus-5-5 / high, worker claude-opus-5-5 / medium   (fourth moonshot team; added by the human, 2026-11-10)
  - slot F: leader claude-opus-5-5 / medium, worker claude-opus-5-5 / medium   (research teams; added by the human, 2026-11-11)
- sparring: codex gpt-6.1-sol / high (the `spar` tool; added by the human 2026-11-10)
- scout: codex gpt-6.1-sol / medium, cards 15 (the `scout` tool, for external teams; one fresh deck per external team)
- research handoff memo: research_handoff_1113.md (human, 2026-11-13: written outside the run by a separate
  analysis system, without running this machine or reading this run's history)
- speed of light: moonshot, research, exploit, exploit-long, research-handoff (task/SPEED_OF_LIGHT.md; human, 2026-11-10; research added 2026-11-11; exploit and exploit-long added 2026-11-11)
- inspiration cards per draw: principles 5, hardware 5, software 5 (the `inspire` tool; 0 turns a
  kind off)
  - inspiration hardware cards: platform-supermicro-741ge-2x-rtx4090
  - inspiration software cards: software-cuda13.0-torch2.14-linux5.15
