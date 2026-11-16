# T8 sweep 5: my map vs other teams' maps and the history (read after files 1-4)

## My candidates: status
| my candidate | status | by whom / on what code |
|---|---|---|
| K3 k=3 drafts (#1) | tested twice, neutral/negative: T1-009 (764.2, 1 start, older code), T7-001 (888.9 vs 895.4, 1 start, on T5-001 + KV 5.5 GiB, job 123) | T7's lesson: per-row cost +4 ms/step (my napkin +3.4) cancels +15% acceptance; plus k=3's GDN spec slots eat KV capacity (my map missed that) |
| K6 larger L2 prefetch (#2) | tested: T5-002, 24 MiB on T5-001: 889.9 (896.6/883.2), weak negative | — |
| K3 reduced-vocab draft head (#3) | tested T1-002 before the FP8 head: acceptance -4.6%, loses; "not refuted" on the FP8 head per T4/T6 | low |
| K6 sampling SendRecv (#4) | T3 built the logits all-to-all (T3-002, in the champion lineage?); still 0.45-0.52 ms/step in the champion; other maps call it rank-skew wait, not transport | not worth it (wait, not work) |
| K1 multi-step state log | in T5's write-up as "DeltaLog follow-ups" (freeing spec-state slots, DL on mixed steps); multi-step not built | untried, large |
| K2 AR+norm fusion | T4 took it (see T4 writeup); no win recorded | tested by T4 |
| K4 PP / DBO | dropped by all maps (same dominance argument) | agree |

## What my sweep lacks that others have
- Capacity as a Speculate dimension (T7, T1): GDN spec states count against the KV budget. I priced k only in time.
- Mixed steps: T4 measured small-prefill mixed steps at 72-122 ms vs ~33 ms decode, and treated the excess as fixed overhead.
  I priced mixed steps only as a remainder (29 s). My reading differs (below).

## Where I disagree / what my map adds
- **Mixed steps are mostly compute, not overhead (napkin):** a mixed step with one ~260-token prompt + ~45 decode rows is ~305 rows x
  25.6 GFLOP = 7.8 TFLOP/GPU -> 47 ms at 165 TFLOP/s (fp32-acc FP8 peak), ~70 ms at ~65% efficiency, which matches T4's 72-122 ms.
  So the lever for the ~29 s of mixed steps is **prefill GEMM throughput**, which no map priced beyond the fp32-accumulate peak.
- **New dimension: accumulator precision inside a K-block (Rewrite knob, implementation-level).** The contract (task_definition.md)
  explicitly allows the FP8 tensor cores' own accumulator inside each 128-element K-block, with fp32 scaling/accumulation across blocks.
  On sm_89 FP8 mma with fp16 accumulate runs at ~330 vs ~165 TFLOP/s (SPEED_OF_LIGHT). Other maps note the rate (T1, T2, T6) but call
  it forbidden or ignore it (T6-sweep/1-inputs: "which the accumulation rule forbids for W8A8"); I read the rule as allowing it within a
  block. Napkin: prefill GEMMs ~65% of mixed-step time; at 1.5-1.8x faster GEMMs, mixed steps -20..-28%, i.e. -6..-8 s of 101 s:
  **+6-8% overall**. Risk: fp16 overflow of a 128-term partial sum (|product| can reach 448^2 = 2e5 > 65504 at the scaled-e4m3 extremes),
  so it needs either a per-32 (single mma k) fp16 accumulation with a proof or a range guard, plus a ruling from the manager.

## Addendum (07:11): ruling and measurement
- fp16 accumulation inside a K-block: **ruled not allowed** by the manager (task_definition.md, 2026-11-16). Microbench job 124
  (bank/T8-001): 1.09-1.23x best-vs-best, smaller than the 2x spec ratio. The rest of the map's disagreement still holds in a revised
  form: fp32-acc W8A8 already runs at 240-275 TFLOP/s at M>=512 (not 165), so a ~305-row mixed step needs ~29 ms of GEMM compute.
  T4's 72-122 ms per small mixed step therefore does contain a large non-GEMM excess (T4's K6-c overhead thesis), and my
  "mixed steps are mostly compute" reading was wrong by the peak-rate assumption.
