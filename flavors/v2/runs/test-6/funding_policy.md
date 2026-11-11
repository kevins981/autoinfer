# Funding policy (v2 default)

How the manager decides who gets GPU time. `new_run` copies this file into each run as
`funding_policy.md`. The human may edit the run's copy at any time, and the manager re-reads it
before each funding decision. This is the human's main control over the run, including how much
it explores versus exploits.

## The objective
The highest total speedup the run achieves from its fixed GPU pool, across all teams.

## Grant sizes (GPU-minutes)
- **Seed:** 20, granted automatically when a team opens (`gpu team open`; set up with the pool).
- **Start:** 80 for every team. Exploit teams get it when they open. Explore teams get it once
  their proposal is funded.
- **Extensions and last-chance grants:** the manager decides the size. A last chance is smaller
  than a normal extension.
- **Second seed:** an explore team whose seed went into a test that failed for a fixable reason
  (not a hard limit) may get one more seed-sized grant (`--kind extension`, note "second seed")
  to test a variant aimed at that cause, before it has a funded proposal.

## Explore proposals
A proposal has a roadmap, a forecast and the GPU time requested. The forecast gives the
**payoff if the roadmap works** and the team's honest **central estimate and odds**. Fund it if:
1. the forecast is broken into parts, each marked measured (citing a job or record) or assumed;
2. the **payoff if it works** clears the **current** champion by at least **5%**;
3. the GPU time requested is within reason for the roadmap.

**The central estimate and odds are not gated.** A hard bar on the central estimate would reward
optimistic forecasts and penalise honest ones. Weigh the odds when sizing grants instead: a
low-odds, high-payoff direction is worth funding in small steps.

**The manager is not a gate.** It checks that the forecast is plausible and honestly built, not
that it agrees with the idea. When in doubt, fund it and let the measurements decide.

**Duplicates:** reject a proposal only if it is exactly the same as an already-funded direction.
Then pick one team, by track record or at random. Similar but different proposals are the
manager's call.

## Moonshot teams
Moonshot teams (`leader_moonshot.md`) chase high-risk, high-reward directions that reject an
assumption of the current design, without the run's history. The manager funds them as a venture
investor: most moonshots don't pay off, and that is expected; the point is the size of the ones
that do.
- **Moonshot seed:** 30 (the automatic seed plus a top-up when the team opens).
- **Proposal:** fund it if (1) it names the assumption of the current design it rejects, (2) the
  **upside if it fully works** is at least **5%** over the current champion, argued from hardware
  limits and the team's measurements, (3) it names its riskiest assumption and the cheapest test
  that could show it wrong (stage 1), and (4) its parts are marked measured / assumed. Odds are
  reported, not gated. A larger upside justifies more risk, a longer roadmap and more GPU time.
- **History check:** after a proposal, the manager gives the team pointers to related past
  records ("related work: records X, Y; the decision is yours"), or "nothing related found". No
  results or verdicts. Past work is never a reason to reject a moonshot or shrink its grant.
- **Stages:** grants come one stage at a time, up to about **200** GPU-min per team in total. Fund
  the next stage when the last one measured what it set out to and the idea is still alive. A
  small early gain or a fixable failure is not a reason to stop. Close only when the core idea is
  shown wrong (the riskiest assumption is false, or a working prototype shows the upside can't
  reach 5%), or nothing new was measured for two stages in a row.

## Resyncs
A team can request a resync at any time.

| What the team shows | Decision |
|---|---|
| Beat the champion, or a milestone measurably confirmed | Extend, possibly with a bigger grant |
| The measured part of the forecast grew, and the payoff if it works still clears the bar | Extend |
| Nothing new measured, or the payoff if it works fell below the bar | Last chance: a smaller grant. The team says what it will measure with it |
| A second last chance in a row | Close the team and open a new one of the same type. The closing team writes up what it learned first |
| The team recommends closing itself, and its failure hit a hard limit or it already tried a variant aimed at the cause | Close it and open a new one of the same type, as above |
| The team recommends closing itself after one failed attempt with a fixable cause | Encourage it to try a variant first (a second seed if its seed is spent). If it still wants to close, close it |

- **For exploit teams, progress means a win.**
- **Re-forecasting isn't progress.** A forecast that stays as high with nothing new measured
  counts as no progress.
- **A pivot doesn't reset the grant.** After a team's first funded proposal, a pivot stays related
  to its direction (the same bottleneck or mechanism, or the cause its last test found). A
  completely different idea is a new team's job.
- **Failed tests: a failed attempt isn't a dead direction.** Ask the team why it failed: a hard
  limit (hardware, math, a measured ceiling) closes the direction; a failed implementation or
  design is a cause to work on. When it's the second kind and the team has budget, say so
  plainly: you'd support it continuing. Be honest: don't push a team past a hard limit.
- **No penalty for closing.** Encouragement only.

## Explore vs. exploit
- **The team ratio** is set in `config.md`.
- **Patience with explore teams: 1 last-chance grant** before a team is closed. Raise it to lean
  more towards exploration.
