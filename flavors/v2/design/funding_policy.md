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

## Resyncs
A team can request a resync at any time.

| What the team shows | Decision |
|---|---|
| Beat the champion, or a milestone measurably confirmed | Extend, possibly with a bigger grant |
| The measured part of the forecast grew, and the payoff if it works still clears the bar | Extend |
| Nothing new measured, or the payoff if it works fell below the bar | Last chance: a smaller grant. The team says what it will measure with it |
| A second last chance in a row | Close the team and open a new one of the same type. The closing team writes up what it learned first |

- **For exploit teams, progress means a win.**
- **Re-forecasting isn't progress.** A forecast that stays as high with nothing new measured
  counts as no progress.
- **A pivot doesn't reset the grant.**

## Explore vs. exploit
- **The team ratio** is set in `config.md`.
- **Patience with explore teams: 1 last-chance grant** before a team is closed. Raise it to lean
  more towards exploration.
