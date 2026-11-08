# Flavor v2 (draft)

**Status:** design draft, written with the human on 2026-11-08. Nothing is implemented yet.

v2 is written as a **delta from v1** ([../../v1/design/DESIGN.md](../../v1/design/DESIGN.md)).
Anything not mentioned here stays as in v1: task definition, trusted official eval, bank and
records, workspaces, leaders and workers, the shared KB with the manager as its only writer, the
journal, and "only measurements count".

This file holds only what the human and the designer have agreed. Everything still open is in
[TODO.md](TODO.md).

## Why v2: what v1 showed

v1 runs: test-3 → test-5 (qwen38-27b-fp8-2x4090, 14 rounds, 631 → 942 tok/s, +49%).
Judge report: `evals/v1/test-5/judge-a.md`.

- **Progress came from many scattered near-term wins.** Teams grabbed whatever was cheapest to win
  next: comms, overlap, kernel tweaks, spec-decode tweaks. Every answer-key win was found, and
  there were 10 novel wins. But each one refined the existing design; none restructured it.
- **Returns fell off.**
  - Rounds 1–5 gained +35% (~4–5%/h).
  - Rounds 6–14 gained +11% (~1%/h).
  - By the end the run was digging in the same place.
- **The three wins the run didn't originate came from outside.** They were ported from
  `prior/`, another run on the same task. The run had not proposed two of them, and had ruled out
  the third on a wrong estimate.
- **The structure favours near-term gains.**
  - Brainstormers and the manager decide each team's direction every round.
  - A team can leave its direction only with evidence from that round, which in practice means a
    measured near-term gain.
  - A genuinely new direction almost always starts out worse than the champion, so under this
    rule it gets dropped before it can mature.
  - This is a greedy algorithm.
- **Synchronous rounds block fast teams.** At each round's barrier, the fastest team waits for
  the slowest.

## Hypothesis v2 tests

> The main reason agents stay in-distribution is the incentive and selection structure, not only
> how ideas are generated. If the reward is still speedup but long-horizon directions are judged
> on their **expected future speedup**, and resources (GPU time) follow results and promise,
> then long-horizon, assumption-challenging work will emerge without telling agents what to do.

**Inspirations.**
- **Successful research organisations** (Bell Labs): autonomy, openness, stability, a stimulating
  environment.
- **Google's TPU team:** an autonomous team that revived an old idea (systolic arrays) and took a
  long time to beat GPUs.
- **Startups and investors:** fund a pitch, renew on traction, advise rather than control.

Human values don't carry over to agents automatically. For an agent, **the incentive is whatever
decides whether its work continues and gets resources.**

## Key ideas

### 1. GPU time is the currency
- **The only resource the manager hands out is GPU time** (GPU-minutes). Priority is not used for
  now.
- **Every team, explore or exploit, wants more GPU time.** To get it, a team shows results and
  promise.
- **This is the incentive.** We don't design a separate reward; competition for GPU time plus each
  team's track record does the work.
- **Each team has a track record:**
  - grants received;
  - GPU used;
  - results;
  - realized gain per GPU-minute, which is the efficiency measure.
- **The manager funds partly on history.** Example: "historically strong, slowing lately, still
  worth funding".
- **The GPU queue needs changes** to account for and enforce grants per team.

### 2. The manager is an investor
- **It funds teams and advises them.** It does not direct their work, just as good investors let
  their companies decide what's best.
- **It shapes behaviour only through resources:** how much GPU time a team gets, whether it gets
  more, or whether it is shut down.
- **The manager's funding policy is the central control.** It is plain text. The human can steer
  it, e.g. "favour long-horizon explore teams more".
- **For now the manager also checks explore teams' forecasts.** A separate sceptic reviewer may do
  this later.
- **The manager's objective:** the highest total speedup from a fixed pool of GPU-minutes, across
  all its teams. The human sets the size of the pool.
- **The manager doesn't do the teams' work.** It may investigate a proposal to understand it
  better, but it doesn't run the direction itself.

### 3. Two team types: explore and exploit
- **Each team is an explore team or an exploit team.** Teams don't declare individual bets.
- **The ratio of explore to exploit teams controls the explore/exploit balance.** The human sets
  it.
- **Exploit teams** work as in v1: beat the current champion now. Reward: near-term measured gain.
- **Explore teams** work on longer-horizon directions that may challenge prior wisdom. They are
  judged on **expected future value**, described next.

### 4. Explore teams: expected future value
- **The reward is still speedup, with a longer time horizon.** Novelty itself earns nothing.
  "Novel, but about equal to or slightly worse than the champion" is not worth pursuing. Every
  direction must aim to beat the champion eventually.
- **Proposal:** the team first works alone (microbenchmarks, napkin math) and then sends the
  manager:
  - a roadmap;
  - a forecast: "when the roadmap is done, I expect X% speedup", built from its measurements and
    calculations;
  - the GPU time it needs.
- **Bar:** the forecast must beat the current champion by **at least 10%**. The champion keeps
  moving, so the bar is checked against the current champion at every review.
- **Review:** the manager checks three things:
  1. the forecast is broken into parts, each marked as measured (citing the job) or assumed;
  2. the total clears the current champion by at least 10%;
  3. the GPU time asked for is within limits.

  If they pass, the team gets a grant. If not, it revises.
- **The manager is not a gate.** It checks that the forecast is plausible and honestly built. It
  does not have to fully understand the idea or agree with the estimate. If the manager funded
  only what it agreed with, it would become the bottleneck on diversity again. Its prompt must
  leave this room: when in doubt, fund it and let the measurements decide. The 10% bar also limits
  the manager: it can't fund small, safe directions as "exploration".
- **Revising the plan is allowed.** Learning and pivoting along the way is how good ideas develop,
  so the team doesn't have to stick to its first estimate.
- **A pivot doesn't reset the grant.** The team keeps spending from the same grant. The risk is
  still that a team rewrites its forecast endlessly to keep its funding. How else to handle that
  is open ([TODO.md](TODO.md)).

### 5. Updates and the end of a grant
- **While working, a team sends the manager progress reports**, like a startup's investor
  updates. The manager decides how often, e.g. after every X GPU-minutes spent.
- **A team can request a resync at any time.** It doesn't have to wait for the grant to run out.
  Usually it asks when the grant is nearly used up, keeping a little in reserve. At the resync the
  team reports what it planned, what it achieved, and any pivots.
- **Decisions at a resync.** Exploit teams read "progress" as "a win".

  | What the team shows | Decision |
  |---|---|
  | Beat the champion, or a milestone measurably confirmed | Extend, possibly with a bigger grant |
  | The measured part of the forecast grew, and the forecast still clears the bar | Extend |
  | Nothing new measured, or the forecast fell below the bar | **Last chance:** a smaller grant. The team says what it will measure with it |
  | A second last chance in a row | **Shut the team down** and launch a new team of the same type, so the ratio holds. The closing team first writes up what it learned |

- **Re-forecasting isn't progress.** Rewriting the forecast is fine, but a forecast that stays as
  high with nothing new measured counts as no progress. That way the manager never has to judge
  whether the new forecast is believable. Each re-forecast is logged in the team's track record.
- **The human's explore/exploit setting enters the policy in two places:** the team ratio, and
  patience, e.g. "explore teams get 2 last-chance grants instead of 1".

### 5b. Pool and grants
- **There is a fixed total pool of GPU-minutes for the run,** set by the human. Every grant comes
  out of it. Time a closed team didn't use goes back into the pool.
- **Every new team gets a seed grant of 20 GPU-min,** e.g. for the microbenchmarks behind a
  proposal.
- **Every team starts with an 80 GPU-min grant,** explore and exploit alike.
- **Extensions are up to the manager,** including the size of last-chance grants.

### 6. Asynchronous, no rounds
- **Teams run independently.** v1's synchronous rounds and barrier are gone.
- **The manager reacts to reports and to grants running out**, not to a round loop.
- **Every team gets two kinds of notification:**
  - **A new champion**, from any team. This is a reminder that the bar has moved.
  - **Other teams' proposals and resync reports**, e.g. an explore team's roadmap and progress.
    They are for inspiration: every team can see what the others are proposing and learning.

### 7. No brainstormers (for now)
- **v2 has no brainstormers.** Teams own their direction and come up with their own ideas.

### 8. Separate prompts for explore and exploit leaders
- **Exploit leader** ([../agents/leader_exploit.md](../agents/leader_exploit.md)): make the champion
  faster now, choose its own direction, and know how funding works.
- **Explore leader** ([../agents/leader_explore.md](../agents/leader_explore.md)). The prompt's job
  is to give the incentive plus a small seed of inspiration, without narrowing where the agent
  looks:
  - **The incentive, explicitly.** Long-horizon plans are welcome. The team is judged not on
    immediate wins but on how much it ultimately gains. The prompt explains how grants,
    extensions and resyncs work.
  - **Autonomy within vLLM.** Any part of vLLM may change, from the scheduler to the kernels, as
    long as the change respects the task's constraints. There is no list of "areas". (Teams don't
    write a new engine outside vLLM.)
  - **First principles, in one line:** start from what the hardware and model fundamentally
    require, not from how vLLM does it today.
  - **A few examples of *kinds* of moves, as illustrations only:**
    - an old idea in a new setting (how systolic arrays came about);
    - an overlooked hardware capability;
    - questioning what every inference engine does by default.

    Nothing specific to this task (e.g. no "is the KV cache needed?"), because LLMs copy their
    examples.
  - **No list of common optimizations.** That is fine for exploitation but limits novelty.

### 9. What can and can't be challenged
- **Constraints from the human can't be overridden:** the task, the task definition, and the
  assumption ledger agreed with the human.
- **Decisions and conclusions by agents can be challenged,** with measurements: records, SHARED_KB
  entries, other teams' conclusions.

### 10. Web research
- **Every agent may search the web:** the manager, leaders and workers. Nothing routes research
  through workers.

### 11. Duplicate proposals (manager's search policy)
- **By default, the manager rejects a proposal only if it is exactly the same as an already-funded
  direction.** It then picks one team by track record, or at random.
- **Similar but different proposals are the manager's call.** Funding several teams on the same
  problem can be deliberate, as investors sometimes do, because they may reach different
  solutions.

### 12. Manager mechanics ([../agents/manager.md](../agents/manager.md))
- **`initialize()` is as in v1:** task definition, loopholes and assumption ledger, agreed with the
  human.
- **The funding rules live in [funding_policy.md](funding_policy.md).** It is copied into each run,
  the human can edit it mid-run, and the manager re-reads it before each decision. It replaces
  v1's `search_policy.md`.
- **Reports to the human:** on every new champion and every resync, plus whenever asked.
- **Reminders go only to explore leaders**, once after every resync with that team, and after every
  context compaction through a hook. Nothing is time-based, and exploit leaders, workers and the
  manager get none.
- **Champion:** the best valid official record that beats the current one consistently and passes
  the gates, as in v1, whatever it started from. Combining wins is the teams' business.
- **The run ends** when the pool is spent, when nothing is worth funding, or when the human stops
  it.

## Considered and not adopted (for now)
- **Rewarding novelty directly.** It invites reward hacking: new designs that don't work. That's
  the "novel but useless paper" failure.
- **Comparing rates.** The idea: continue an explore direction only if its expected gain per hour
  beats the exploit teams' recent gain per hour. Too much for a first version, so v2 ignores the
  time dimension for now.
- **Scoring forecast accuracy as a reward.** In RL, the gap between predicted and actual outcome
  trains the predictor; it doesn't reward the actor. Forecast accuracy may still inform how much
  the manager trusts a team's next forecast. That is undecided.
- **Special prompting for honesty.** The incentives already cover it: a team that oversells and
  then doesn't deliver loses funding, and the manager can tell the difference.
- **A separate "bet" mode inside a team.** Rejected in favour of the simpler model: a team is
  either an explore team or an exploit team.
