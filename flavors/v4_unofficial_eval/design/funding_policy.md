# Funding policy (v4_unofficial_eval default)

How the manager decides who gets GPU time. `new_run` copies this file into each run as
`funding_policy.md`. The human may edit the run's copy at any time, and the manager re-reads it
before each funding decision.

## The objective
The highest total speedup the run achieves from its fixed GPU pool.

## Teams
Every team is a sweep team. CONFIG sets how many are open at all times; a closed team is
replaced by a new one while the pool can fund its opening grant.

## Grant sizes (GPU-minutes)
- **Opening grant:** 45, granted automatically when a team opens (`gpu team open`). No other
  grant at opening.
- **Extensions (last chances included):** 45 each.
- **At most 3 grants per team** (the opening grant + 2 extensions, last chances included). At
  the resync after its third grant, close the team and open a new one, even after a win: fresh
  teams keep the map fresh. The closing team writes up what it learned and adds a last map update
  first.

## What counts
- **A result is a win:** a record promoted to champion (its measurement, the task's harness, beats
  the champion's and passes the checks). Small gains count. A promotion later reverted (found
  invalid) counts against the team that made it.
- **The team's sweep report** (reports/<team>-sweep.md and its six files) is its first
  deliverable, not progress by itself.

## Resyncs
A team can request a resync at any time.

| What the team shows | Decision |
|---|---|
| A win since the last grant (net of reverts) | Extend (45), unless the team has had 3 grants: then close and replace |
| No win, but a measured lead it can name (what it will test next and why the evidence points there) | Last chance: one extension. The team says what it will measure with it |
| A second last chance in a row | Close the team and open a new one. The closing team writes up what it learned and adds a last map update first |
| The team recommends closing itself, and its failure hit a hard limit or it already tried a variant aimed at the cause | Close it and open a new one, as above |
| The team recommends closing itself after one failed attempt with a fixable cause | Encourage it to try a variant first. If it still wants to close, close it |

- **Re-forecasting isn't progress.** A forecast that stays as high with nothing new measured
  counts as no progress.
- **Failed tests: a failed attempt isn't a dead direction.** Ask why it failed: a hard limit
  (hardware, math, a measured ceiling) closes the direction; a failed implementation or design is a
  cause to work on. Be honest: don't push a team past a hard limit.
- **No penalty for closing.** Encouragement only.
