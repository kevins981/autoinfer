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

## 2026-11-12 · test-6 · T52/T53: human-set direction (megakernels) + sparring
- Setup: two exploit-long teams, Opus 5.5 leader and worker, same direction from the human
  ("megakernels", no examples), speed of light, sparring, cards, full history.
- Result: the teams split the work themselves (producer/consumer side of the all-reduce). Decode
  megakernels: measured ceiling <1.5 ms/step, closed (KB132). T52 pivoted within the direction to
  overlap comm in mixed steps, porting two unmerged earlier wins (T30 waves, T32 captured core):
  new champion T52-005, +2.6% official. No web search by any agent.
- Sparring worked, as a check on the leader's own claims:
  - T52-1: caught an arithmetic slip in the payoff napkin (60 us x 48 = 2.9 ms, not 3.7) and set the
    falsifier test before any build.
  - T53-1: agreed the closure was defensible but named one untested alternative (half-batch
    interleave); the leader tested it (1 job), then closed on measured evidence.
  - T52-2: leader recommended closing; the partner showed the "nothing left" ceiling was
    unmeasured (link not saturated, one schedule tried). The leader withdrew the closure, ran a
    0.5 GPU-min screen with a stop rule, it came back negative, and closed.
  - Pattern: sparring turns "I think it's done" into one cheap test with a kill rule. It didn't
    find extra speed here; it made the closures trustworthy at ~1 GPU-min each.
- Caveat: one direction, two teams.

## 2026-11-13 · test-6 · T56-T58: sweep teams (a systematic framework, then wins)
- Setup: three sweep teams, Opus 5.5 leader, Sonnet 5.5 worker. Each had a framework snapshot
  (graph of data/compute nodes + 6 knobs + napkin cost), full history, speed of light, sparring.
  T56/T57 read the history first; T58 had to sweep blind first (no history), writing one file per
  step (inputs, graph, calibration, knobs), then compare with history and rank.
- Result: two champions, +1.4% for 123 GPU-min (1212.4 -> 1229.6). Smaller and costlier than
  the last non-sweep win (T52-005: +2.6% for ~51 GPU-min).
  - T57-004 (+1.08%): mostly a drafter table that T56 derived from Rewrite + Keep knobs, the only
    row in its sweep marked new. T57 had pruned the same rewrite (counted weight bytes only), then
    built it after measuring the drafter.
  - T58-001 (+0.33%): found in the blind phase. Calibrating each step type showed gate_up at
    152 vs 80 us in small mixed steps; the cause was a 1 MiB size gate on the all-reduce's L2
    prefetch. The history check found no one had raised it since T1-019.
  - Both history-first teams ranked the same idea first (SM-split prefill lane); it was dead
    within ~7 GPU-min (DRAM/L2 arbitration, not SM count).
- Insights:
  - Blind first, history second, with a file per step, makes the framework's work checkable. T58's
    files show where every candidate came from; T57's single report couldn't.
  - The win came from calibrating per node and per step type against measurement, not from the
    knob enumeration itself. Most enumerated candidates were already in the history.
  - Every team listed where the framework didn't fit (per-transfer latency, link/DRAM coupling,
    SMs and CPU as resources, contract-dependent compute). These lists agreed across teams,
    including the blind one: the most reliable output for the framework's next version.
- Caveat: one task near saturation; n=1 for the blind variant.
