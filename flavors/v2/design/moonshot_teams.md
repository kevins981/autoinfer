# Moonshot teams

Status: built 2026-11-10, not yet run.

**Why.** In test-5/6, new leaders read the shared KB, the graph and earlier write-ups first, and
their ideas mostly built on that history, including its "walls" (limits later teams overturned).
Sparring, inspiration cards and external decks help, but every team still inherits what the run
already believes. A moonshot team starts from the system itself.

**What it is.**
- **Objective:** the same measured speedup as every team; novelty is not a goal. The proposal
  names the assumption of the current design it rejects, an upside if it fully works of at least
  5% (argued from hardware limits and the team's own measurements), and its riskiest assumption
  with the cheapest test that could show it wrong (stage 1).
- **Two roles, two attitudes.** The manager invests like a venture investor: high risk for high
  reward, most moonshots not paying off is expected, the upside is weighed, not the odds; weird,
  long or many-step is fine. The leader is a founder: it plays to win (a failed attempt is a
  problem to solve) and is honest with its investor (measured vs. assumed).
- **History-free (prompt-only).** The leader and workers read task/, the constraints,
  champion.md and the champion's code, and their own team's files. Not shared_kb.md, graph.md,
  journal.md, announcements.md, teams/, reports/, other teams' files or the champion's record.md.
  `agent announce` skips moonshot teams; the manager tells their leaders new champions directly
  (record id + numbers) and never mentions the run's history otherwise.
- **One history check:** after a proposal, the manager replies only "not tried" or "this exact
  mechanism was measured in record X: result Y". The leader decides what it means.
- **Funding:** seed 30, staged grants up to ~200 GPU-min. Close only when the core idea is shown
  wrong or nothing new was measured for two stages in a row. Sparring and inspiration cards as for
  explore.

**Files:** agents/leader_moonshot.md, agents/reminders/leader_moonshot.md, manager.md (moonshot
sections), common.md, worker.md, design/funding_policy.md, config_default.md, tools/gpu
(`--type moonshot`), tools/agent (announce skips moonshot teams).

**To check after a run:** did the leader and workers stay out of the history (their tool calls)?
Did their ideas differ from the explore teams'? Did the manager stay history-free and invest by
upside?
