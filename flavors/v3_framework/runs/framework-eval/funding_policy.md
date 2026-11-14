# Funding policy (v3_framework default)

How the manager decides who gets GPU time. `new_run` copies this file into each run as
`funding_policy.md`. The human may edit the run's copy at any time, and the manager re-reads it
before each funding decision.

## The objective
The highest total speedup the run achieves from its fixed GPU pool.

## Teams
Every team is a sweep team. CONFIG sets how many are open at all times; a closed team is
replaced by a new one while the pool can fund it (seed + start).

## Grant sizes (GPU-minutes)
- **Seed:** 20, granted automatically when a team opens (`gpu team open`).
- **Start:** 80, granted when the team opens.
- **Extensions and last-chance grants:** the manager decides the size. A last chance is smaller
  than a normal extension.

## What counts
- **A result is a win:** an official record that beats the champion. Small gains count.
- **The team's sweep report** (reports/<team>-sweep.md and its six files) is its first
  deliverable, not progress by itself.
- **Map proposals** are reviewed on their own (manager.md): they don't earn GPU time, but a team
  that keeps the map useful makes every later team cheaper.

## Resyncs
A team can request a resync at any time.

| What the team shows | Decision |
|---|---|
| A win since the last grant | Extend, possibly with a bigger grant |
| No win, but a measured lead it can name (what it will test next and why the evidence points there) | Last chance: a smaller grant. The team says what it will measure with it |
| A second last chance in a row | Close the team and open a new one. The closing team writes up what it learned and sends a last map proposal first |
| The team recommends closing itself, and its failure hit a hard limit or it already tried a variant aimed at the cause | Close it and open a new one, as above |
| The team recommends closing itself after one failed attempt with a fixable cause | Encourage it to try a variant first. If it still wants to close, close it |

- **Re-forecasting isn't progress.** A forecast that stays as high with nothing new measured
  counts as no progress.
- **Failed tests: a failed attempt isn't a dead direction.** Ask why it failed: a hard limit
  (hardware, math, a measured ceiling) closes the direction; a failed implementation or design is a
  cause to work on. Be honest: don't push a team past a hard limit.
- **No penalty for closing.** Encouragement only.
