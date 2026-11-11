# LOCAL_KB leader-37 (T37, moonshot)

## Inspiration cards (first set)
- Imbalance + Generality shaped the direction: decode is bound by weight streaming with a large fixed cost per pass,
  prefill by compute/PCIe; the engine keeps separate specialised paths and ~75 captured graphs instead of filling
  one pass. Predictability -> speculation depth is priced wrongly (memory cap), not at its limit.
- Representation: checked lossless recoding of FP8 weights (6.5 bits/byte entropy): <=19% of weight bytes, dropped.
  Distance/four lanes: the all-reduce is already compressed, pipelined and at ~43 us; nothing structural found.

## Measured (see bank/T37-001, jobs 366-371)
- Decode step 23.3 ms = 16.9 + 0.10*rows + 0.83*draft steps. Acceptance geometric 0.774 then ~0.72/position.
- Prefill = 24% of run: 15 ms/event fixed (host, eager launches) + 166 us/token.
- State pages per request grow with k; pool 162 pages (5 GiB) -> k=3 seats 14 seqs, c=32 seats 19. Memory is full:
  14.46 model + 5.0 pool + 2.48 graphs GiB; +0.6 GiB pool OOMs.
- nsys node trace inflates piecewise-graph steps; use the step logger (T37_STEPLOG) for wall times.

## Sparring T37-1
- Supported by measurement: fixed-cost share, acceptance geometry, the seat cap.
- Assumed: 0.10 ms/row holds to 64-96 rows and for prompt rows (weakest link; milestone 1 tests it).
- Cheap test: equal-work elapsed time of a mixed static pass vs today's separate paths.

## Status
- Proposal proposals/T37-1.md sent to manager; seed left 13.8 GPU-min.

## After the manager's pointers (2026-11-10)
- k=3 with all 16 running was measured before via "two-slot GDN state" (T28-003 +2.9%, T28-005 null, T36-001 +2.1% at c=14).
  My +4.5% model estimate was too high; expect +2-3%. Not yet run on T31-003 at c=16 -> stage 1 redefined (worker w1).
- The ~15 ms/prefill-event cost is GPU-side (large-M GEMM at ~104 rows 19.3 vs 12.5 ms, ARs 12 vs 5.9 ms), not host
  launches: T32-003 removed 33 ms of host enqueue per mixed step and was neutral. My proposal's host-side claim was wrong.
- Confidence-budgeted depth: T3/T5/T27 sim +4-5% decode, measured losses. Dropped.
- Revised forecast sent to manager: upside ~+6%, central +2-3%, P(>=5%) ~10%.
- Lesson: per-step NVTX/CPU windows are not GPU time; attribute kernels by launch correlation, and check e2e.

## T37-002 (2026-11-10): champion + two-slot GDN state (+ boundary fix) + k=3
- ABBA vs T31-003 (job 375): +1.34%, all B > all A. Official #2: 1142.2 tok/s (1138.4/1146.1), gates pass, GSM8K 94.0%.
  Cross-job vs champion's 1088.9 looks like +4.9%; trust the ABBA (~+1.3-2%).
- Bug found: ported two-slot code corrupted output after a 1600-token state-page boundary (job 377); fixed in mamba_utils.py
  (zero temporal token bias), verified jobs 378/379. Short-request benchmarks never hit it: always test long requests.
- official_eval default timeout is 20 min and must fit the balance; a refused attempt leaves eval-<id>-1 behind and the retry
  fails at apply_diff once, then works.
- Open: FP8 A/B reuse at M=64 argued from the kernel (K-block-boundary tiling), not measured. Big T37-1 idea not supported.

## Closing (2026-11-10)
- T37-002 promoted champion (1142.2 official; same-job gain ~+1.3-2%). FP8 A/B at k=3: job 381, 94.4 vs 93.6.
- Spar turn 3: my "all-reduce bytes are a hard floor" is ASSUMED, not established. Argument only: in any split where a
  sublayer's weights sit on both GPUs each side needs the other's 5120-wide partial per row (10 KB bf16); a split where a
  sublayer sits on one GPU still ships the same vector one way and gives up parallel weight streaming (memory-bound decode).
  Not measured; no alternative partition was prototyped. Scope of my search: TP=2 with the champion's partition.
- Untested levers left: k=4 (kMaxLogTokens kernel change, model ~+1%), fully static mixed step (est. 2-4%), a different
  tensor partition / collective schedule that lowers PCIe bytes per row (unknown).
