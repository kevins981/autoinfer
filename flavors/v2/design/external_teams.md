# v2 external teams (spec)

**Status:** agreed with the human on 2026-11-10, during test-6; first trial in test-6 (one team).

## Why
- Teams so far are "internal": their ideas come from optimization principles, earlier experiments
  and the code. Later teams mostly build on earlier work, and none considered some widely
  discussed techniques (e.g. megakernels).
- Agents hardly use the web on their own: in test-6, 1 web call across 61 agents, although the
  prompts say web research is welcome. "Go search the internet" isn't enough.

## What it is
- **A separate team type, `external`.** Same funding rules as explore. The difference: the
  direction comes from a deck of external technique cards. (Exploit teams may be dropped later;
  the types can be merged later if they turn out the same.)
- **A scout writes the deck when the team opens** (`tools/scout --team <team>`, manager only).
  - One fresh deck per external team, ~15 cards. No seed or topic is given to the scout; the
    randomness of the scout and of the leader is the variety (overlap between decks is accepted
    for now).
  - The scout's whole context is `agents/scout.md`: no run, task, project or machine. It runs in
    an empty folder outside the run (`.cache/scout_work/`). Cards stay history-free.
  - Content: techniques published in the last two years that make LLM inference faster without
    changing its output (exact; lossy techniques left out, or their exact form). Variety over
    coverage; structural over tuning; unfamiliar over famous. Primary sources only, opened;
    numbers as reported.
  - Card format: what it is / what it exploits (the condition it needs) / what it requires /
    exactness / reported results / sources. Plus `index.md` with what was left out and why.
  - Agent: Codex (`codex --search exec`), model and effort from config.md's `scout:` line
    (default gpt-6.1-sol / medium). Codex home: `.cache/codex` (auth copied there by the human).
- **The leader** (`agents/leader_external.md`) starts orienting while the scout runs (10-30 min)
  and is messaged when the deck lands in `scout/<team>-<n>/`.
  - Its direction comes from one or more cards, adapted to this machine. The proposal names the
    card(s) and makes a **transfer argument**: a seed measurement of how much of the card's
    condition exists in the champion.
  - "None is good enough" is allowed, with evidence (measured or costed for the most promising
    cards). The manager then closes the team; a replacement gets a fresh deck.
  - It reads the run's history like an explore team (one factor changed: the deck). No regular
    inspiration cards.

## Later
- A blind variant (no run history until after the first proposal).
- Measures without a judge: share of the proposal's citations that are external vs. KB; dead
  ends rediscovered; speedup.
