# v2 inspiration cards (spec)

**Status:** behavior agreed with the human on 2026-11-09, after test-6. The principle cards are
written ([../inspiration/principles/](../inspiration/principles/)), and so are the hardware and
software cards for the 2x RTX 4090 machine. The `inspire` tool (`tools/inspire`, tests in
`tools/tests/test_inspire.sh`) and the manager and leader prompt changes are implemented; not yet
used in a run.

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

### Platform cards (written for one machine)
Facts and capabilities of a specific machine and software stack: things the teams can't change,
or tools that are already there. Each folder is named for exactly what it describes, so it is
clear which machine the cards belong to. A run on another machine needs its own folders.

- **`platform-supermicro-741ge-2x-rtx4090/` (28 cards):** hardware. 19 on the GPU, 6 on
  communication (PCIe, no direct GPU-to-GPU path), 3 on the host. The GPU gets the most weight,
  the host the least.
- **`software-cuda13.0-torch2.14-linux5.15/` (13 cards):** software. Ways to produce GPU code,
  ready-made libraries, communication software, the Python and framework runtime, and the
  operating system as an ordinary user sees it.
- **Format:**
  - **What it is:** the fact or capability, with numbers;
  - **What it makes possible:** stated generally. A card never proposes a use in the engine;
  - **How this was checked:** queried from the device, assembled for the chip, read from
    documentation, or not checked. Agents should know how far to trust a card;
  - **Where to read more:** the name of the documentation.
- **Coverage is wide rather than complete,** and leans towards capabilities that are easy to
  overlook. What the platform lacks is a card too.
- **Field terms are fine here** (unlike in the principle cards), but nothing from a task or an
  earlier run: no measurements of the engine, no findings.
- **Software is only partly a given.** The version limits, the operating system and the lack of
  administrator rights are fixed. The compilers and libraries are a toolbox: teams may ignore
  them or install others in user scope.

### Not included
- **ML techniques** such as speculative decoding. They are choices, not givens, and they follow
  from the principles.

## The tool: `inspire`
- **It draws a few cards of each kind at random and prints them** in full: by default 3
  principle, 3 hardware and 3 software cards. The counts and the platform folders are set in the
  run's `config.md` (0 turns a kind off).
- **A team sees every card of a kind before it sees one twice.**
- **It saves each draw** to `inspiration/<team>-<n>.md` in the run folder, so the manager can
  attach it unchanged.
- **It logs every draw** in `inspiration/draws.jsonl`: the team, the time, who ran it and the
  cards drawn. The log shows later whether the cards had any effect.
- **`inspire --list` and `inspire --show <file>`** list all titles and reprint one card.
- **The framing text comes from the tool,** so every draw reads the same. It says:
  - these are general principles and facts about the platform, offered for inspiration only; nobody chose them for this team,
    and they are not a direction or an assignment;
  - look at the system yourself first (read the task, profile, find where the time goes), then
    come back to the cards; using none of them is fine;
  - the hardware and software cards are high-level pointers, not a reference: they may be
    incomplete, out of date or wrong for the case at hand, so do your own research and verify any
    detail before relying on it;
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
- Whether the model is worth cards of its own.
- Whether 3 + 3 + 3 is the right size and mix (set after seeing a run use it).
