# v2 inspiration cards (spec)

**Status:** behavior agreed with the human on 2026-11-09, after test-6. The principle cards are
written ([../inspiration/principles/](../inspiration/principles/)). **Not implemented yet:** the
`inspire` tool, and the changes to the manager and leader prompts. Platform cards are still under
discussion (see "Open").

This file is the source of truth for how the mechanism should behave.

## Why
- **Test-6's ideas were interesting but not far out, and teams converged.** Three teams tried
  the same idea independently.
- **The amount of exploration should be adjustable.** This mechanism is one way to raise it.
- **A team deep in one direction loses the wider view.** In research, an outside prompt often
  unlocks something new. The cards are that outside prompt.

## What it is
- **A set of cards in the flavor, and a tool that prints a few of them.** The idea comes from
  AlphaEvolve: keep a pool of inspirations and sample a subset each time.
- **No agent is involved.** The tool is a deterministic script apart from the random draw. There
  are no brainstormer agents and no personas. The leader applies the cards itself.
- **The cards hold first principles and givens, not techniques.** They describe ways of thinking
  and facts that can't be changed. They don't list known optimizations.

## The cards
All cards live in the flavor, under `flavors/v2/inspiration/`. None goes in a task folder.

### Principle cards (`principles/`, 21 cards, written)
General principles of where systems lose time. Each card is one kind of waste, except
"Trade-offs", which is a way out when two goals conflict.

- **Nothing specific to inference, to a task, or to an earlier run.** No field terms such as
  "kernel" or "GPU".
- **The weight is on the principle, not on instances.** Format:
  - **The principle:** the way of thinking, stated generally;
  - **The waste:** what is lost, in one sentence;
  - **General directions:** what follows from the principle (directions, not techniques);
  - **Illustrations:** a few examples from other fields, explicitly marked as examples only.
- **Left out on purpose:** specific questions to ask, how the waste shows in measurements, and
  what a fix costs. The human judged these too close to instances.
- **Sources:** the kinds of waste in lean manufacturing; Park et al., "Principles and
  Methodologies for Serial Performance Optimization" (OSDI 2025); TRIZ for the trade-offs card.

### Platform cards (not written yet)
Features of a specific hardware and software platform: things the teams can't change. They live
in the flavor in a folder named for the exact platform (server model, GPUs), so it is clear which
machine they describe.

### Not included
- **ML techniques** such as speculative decoding. They are choices, not givens, and they follow
  from the principles.

## The tool: `inspire`
- **It draws a few cards at random and prints them** in full.
- **It logs every draw:** the team, the time and the cards drawn. The log shows later whether the
  cards had any effect.
- **The framing text comes from the tool,** so every draw reads the same. It says:
  - these are general principles, offered for inspiration only; nobody chose them for this team,
    and they are not a direction or an assignment;
  - look at the system yourself first (read the task, profile, find where the time goes), then
    come back to the cards; using none of them is fine;
  - you can draw again at any time;
  - you can always do your own web research to develop an idea or find more;
  - after you have considered a draw, write 2-3 lines in your notes on what it made you think
    about, even if the answer is "nothing".
- **It doesn't tell the agent that the draw is random.**

## When it runs
| When | Who runs it | Required? |
|---|---|---|
| A team opens | The manager; it attaches the output to the leader's first task | Yes |
| Any time | The leader | Up to the leader |
| A resync that ends in a last chance | The manager offers a draw | Up to the manager |

- **The manager passes the output on unchanged.** It doesn't choose cards, comment on them or
  filter them, so it stays an investor.
- **The first draw arrives before the leader has profiled anything.** That is why the framing
  text tells the leader to look at the system first.

## What the leader does with a draw
- **It considers the cards against what it has measured,** and may use none of them.
- **It writes 2-3 lines in its notes** (LOCAL_KB) on what the draw made it consider.
- **Ideas from cards get no special treatment.** They go through the normal proposal and funding
  process, and every claim still cites evidence.

## The exploration dial
- the number of cards per draw;
- the mix of card types in a draw;
- how often draws happen.

## Open
- The platform cards: which platform facts and features, and their format.
- Whether there are other kinds of givens worth cards (the software platform, the model).
- The number of cards per draw and the mix.
- Where the draw log lives in the run folder.
