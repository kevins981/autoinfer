# Insights log

What worked (or clearly didn't) in AutoInfer's design. Only insights worth keeping, not a record
of everything. Newest last.

## 2026-11-10 · test-6 · T37: moonshot + Fable + speed of light
- Setup: moonshot team (history-free, venture funding), Fable 5.2 leader, Sonnet 5.5 worker,
  speed-of-light page, ownership line, sparring, cards.
- Result: most ambitious plan of the run (+18% upside, 3 stages); stage 1 alone +1.34% (same-job
  ABBA), the first win since T31; also found a bug in earlier teams' code.
- Caveat: first try of this config, one team. It may not generalize, and credit can't be split
  between the modules, the model and chance.

## 2026-11-11 · test-6 · T38: moonshot + Opus high + speed of light
- Setup: as T37, but Opus 5.5 leader (high effort) and Opus 5.5 worker (medium).
- Result: most ideas of any team, each argued from the floors and tested cheaply (five ideas, about
  3 GPU-min of tests in all, much of it CPU analysis); no win. Its best idea was a pipeline with
  dynamic draft trees in place of TP=2: prefill 0.51x TP per token, but decode short (~+3.7%
  modeled before pipeline bubbles).
- Didn't work: every idea already had a name (overlap, self-distilled drafter, pipeline parallel,
  draft trees). Two of its ideas were closed on the history pointers, and the leader then asked
  the manager whether other ideas were covered before proposing them.
- Caveat: first try of this config, one team; effort and model are confounded with the page.

## 2026-11-11 · test-6 · T42-T49: research teams, four framings x 2
- Setup: research team (goal: novel insights that help future teams; speedup a bonus; full history,
  speed of light, cards, sparring); Opus 5.5 medium. Framings: base, ownership, reviewer bar,
  radical; one team each, run twice (the second set read the first's results).
- Result: no idea without an existing name. Cheap, honest insight instead: ~12 KB entries for
  ~50 GPU-min, including a corrected long-held belief (PCIe duplex) and several firm negatives. The
  only win (new champion, +2.7%) came from a radical team's routine kernel check while it killed
  its own idea.
- Framing made no visible difference to novelty. Every team read the history first and concluded
  "components are at their floors" before forming ideas. Radical teams turned into auditors and
  proposed least.
- Caveat: first try, n=2 per framing, one task near its floors; history and framing are confounded.

## 2026-11-11 · test-6 · T50/T51: exploit vs exploit-long after the research teams
- Setup: one exploit and one exploit-long team (renamed from explore); Opus 5.5 leader, Sonnet 5.5
  worker; full history including the research teams' findings, speed of light, sparring, cards.
- Result: the exploit team's first move merged two exact gains that research teams had measured but
  never merged: new champion, +0.7% official, in ~1 h and 28 GPU-min. Then it closed (all else
  <=0.4%). The exploit-long team re-priced an idea earlier teams had dropped (piggyback prefill)
  and closed it with measured causes in 14 GPU-min.
- Insight: measured-but-unmerged gains are the cheapest wins; they sit idle until a team's job is
  to merge them. Research findings pay off through exploit teams.
- Caveat: one team each.
