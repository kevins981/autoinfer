# v4_unofficial_eval (2026-11-15)

v3_framework (sweep teams, each with its own map) + **cheaper measurement** (expected behavior: measurement_spec.md): the goal is a faster
rate of improvement per GPU-minute. Forked from v3_framework after framework-eval.

## Why
framework-eval's GPU time (494 GPU-min at the time of the analysis): ABBA A/Bs 54%, official evals
24%, other server runs 7%, nsys 6%, microbenchmarks 5%, ncu 3%. In an ABBA, half the starts re-measure
the champion, whose numbers don't change. Every candidate also paid a full official eval (~8-11 GPU-min).

Measured on framework-eval's data: identical code measured in different jobs drifts by a median of
~0.55% (max 1.6%), about the same as the start-to-start noise within one job (median pair
difference 0.74%). So a cached reference is about as good as re-running it.

## What changed from v3_framework
- **`unofficial_eval <record>`** (tools/unofficial_eval; runs the task's harness/eval.sh with PROMPTS=200, its own
  gates and a stop-loss; eval.sh's defaults are unchanged): replaces the
  candidate-vs-champion ABBA and the per-candidate official eval. Two fresh starts of the record
  (200 prompts each) against HEAD's **cached** starts: the reference is never re-run. The cache key
  is the exact code (bank diff hash) + serve args + bench settings + harness version, so it never
  expires. HEAD's reference starts are its own unofficial eval's starts; missing ones (e.g. BASE at run
  start) run once, in the first unofficial eval that needs them. Early stop after one start if it is more
  than 1.5% below the reference mean. PASS = both starts above the reference mean (+ the gates run).
  `--gsm8k` (numerics) and `--long` (memory) run those gates in the unofficial eval.
- **Two-tier champion** (tools/champion, champion.json -> champion.md): HEAD = unofficial (promoted
  on an unofficial PASS); CONFIRMED = official. Confirmation is due at fixed thresholds: summed unofficial
  gains since CONFIRMED >= 5%, or 4 unconfirmed records (not a manager judgment). The manager runs
  one official eval of HEAD (charged to MGR) and `champion confirm`: MATCH if the official number
  beats CONFIRMED's, passes the gates and is within max(1.5%, 2 sigma) of the expected number
  (CONFIRMED x the unofficial gains); otherwise bisect the chain with official evals, then revert the
  culprit and everything built on it. Each confirmation bounds how far unofficial-eval errors can carry.
- `official_eval` is manager-only. No `abba.sh` in v4 runs (removed from the run's task copy:
  `unofficial_eval` is the only comparison path). No `--ignore-eos` (tested: fixed output lengths don't
  reduce start-to-start noise).
- **Patch after sweep-unofficial-eval** (see Decision log): `unofficial_eval` only measures any code
  (records or workspaces; no cache, no verdict, no notion of the champion: agents compare numbers); no champion tool
  (the manager keeps champion.md); HEAD/CONFIRMED renamed to unofficial/official champion; one opening grant of 30 per team (no separate seed/start), extensions 30 each; TASK.md's same-job A/B
  protocol replaced in the run's copy.
- Funding: a promotion to HEAD is a win; a later revert counts against the team.
- Grants and pool as in v3 (pool 1000, seed 20), except the start grant: 60 (v3: 80).

## Known biases
- Winner's curse: HEAD's reference is the starts that passed its own unofficial eval, which skew high, so
  later unofficial evals are slightly conservative.
- Unofficial-eval noise per step (sigma ~0.75%) is larger than an ABBA's; confirmations catch chains whose
  summed gains don't hold up.

## Considered, not done
- Overlapping the next server's CPU-only startup (~24 s of ~50 s) with the current bench: worked in
  a test with no measurable effect; shelved by the human (extra machinery for now).
- A steady-state throughput metric (trim ramp-up and drain): noise 0.79% -> 0.64%, not enough.

## Decision log
- 2026-11-15, human: B-only unofficial evals against cached references; batched promotion with fixed
  thresholds (5% / 4 records); C confirmed with 2 trials + GSM8K + 128K (no same-job C rerun);
  teams rebase themselves; the manager announces promotions, confirmations and reverts; pool 1000,
  seed 20, start 60. No abba.sh, no --ignore-eos; one GPU harness (eval.sh) for both evals;
  "unofficial" (not "provisional"); the manager measures BASE's reference starts at run start.
- 2026-11-15, human, after sweep-unofficial-eval. Result: v4 ~3.5% behind framework-eval at matched
  GPU-min. Measurement fell 450 -> ~274 GPU-min, but workers' own inference-server jobs rose (checks,
  debugging, ~66 GPU-min of home-made A/Bs written because the run's TASK.md still taught ABBA).
  **Principle: simplicity. The agents are smart: give them simple tools and goals, not procedures,
  formulas or step lists; don't script what they can judge.** Changes:
  - `unofficial_eval` only measures: a record or a workspace (`--ws --label`, `--env`, `--args`),
    2 starts x 200 prompts, optional `--gsm8k`/`--long`, optional `--stop-below <tok/s>` chosen by the
    agent. No cache/fingerprints, no verdicts, no notion of the champion or a reference: agents
    compare numbers themselves (the champion's are in its bank record; promotions are announced).
    Anyone may run it, workers included. Records don't change once filed.
  - No "built on the champion" rule: any code competes on its number (`builds_on:` says what a
    record contains; a revert also removes records built on the reverted one).
  - No `champion` tool: the manager keeps champion.md itself (unofficial champion, official
    champion, history) and judges
    promotions and confirmations; official evals at ~5% / 4 promotions / the end.
  - Terms: "unofficial champion" and "official champion" (not HEAD / CONFIRMED).
  - Budget: one opening grant of 30 per team (no separate seed/start), each extension 30 (last
    chances included). Pool 1000.
  - TASK.md's FP8-activation rule (GSM8K in the same job as a bf16 reference) removed from the
    task: the normal GSM8K gate applies. The run's copy of TASK.md/lib.sh gets the noise facts
    instead of the same-job A/B protocol (new_run).
  - Stale prompts fixed (ABBA, chain, same-job A/B, "must build on HEAD"); a leader that sends a
    candidate keeps working, it doesn't wait for the manager.
  - Not done (overkill): a shared check-helper script; `gpu breakdown`; rules on what agents may
    run locally. Considered and left as is for now: shortening manager.md/common.md, the funding
    table, the manager's bookkeeping files.
