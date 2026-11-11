# leader-43 (T43, research) notes
## Cards (first set)
- Overqualified/unused results -> ~1/3 of verification rows are discarded; on this stack a row's cost is mostly PCIe AR bytes (0.06 of 0.10 ms/row) -> row-selective AR (T43-1).
- Coordination/four lanes: decode AR write and read phases each use 2 of the 4 PCIe lanes; but SM read+write duplex already measured (KB73), CE latency kills it (KB74). No new angle.
- Host processors / GDRcopy-style CPU push into GPU BAR: impossible without gdrdrv (/dev/gdrdrv absent; BAR1 256 MiB).
## Findings
- GPU0 clock lock (KB8) gone since the reboot after job 345: T31-003 official GPU0 2235 MHz, T37-002 official #2 GPU0 2655. Manager recorded KB113.
- Prior art for mid-verification pruning: HiSpec 2510.01336, FASER 2604.20503 (compute-side). T21 M2 = prompt-as-draft rows (not built).
## Spar T43-1 (2 turns)
- Supported: 2 x 5120 per row per layer is the minimal cut for the current sharding; recurrent state never crosses GPUs.
- Open (assumed): that this cut is minimal across all placements/schedules; PP e2e never measured end to end (T38 only estimated, r=0.91-0.92).
## T43-1 stage 1 result (T43-001, jobs 394-396, 4.3 GPU-min)
- Decision depth: logit-lens AUC accepted-vs-rejected ~0.6 through L48, 0.66 at L56, 0.99 final; linear probe 0.68/0.70/0.75 at L16/32/48
  (<=7% of rejections at 1% false drops). Drafter confidence AUC ~0.87 (pre-forward).
- Value of confidence masking with row-selective AR: best chained-prob tau 0.15-0.2 -> +0.6% decode AR-only, +1.4-1.7% with GEMM
  compaction (needs ragged plumbing). Break-even P(row useful) ~10% (token worth ~0.61 ms, row AR 0.06 ms).
- Gate failed -> direction closed. Untested: nonlinear / draft-aware (tuned-lens) probes; data on d4+ (deeper k) confidence.
- Lesson for me: on this stack a token is worth ~10x a row's AR bytes, so pruning rows only pays for nearly hopeless rows.
## Close (spar T43-2, on the reasons for closing)
- Withdrawn: "the target decides in its last ~8 layers" -> only "logit lens and linear probes find little before the final layers".
- Supported: oracle ceiling of mid-forward byte pruning at k=3, L32: ~+2.7% decode / ~+1.8% e2e; deeper k is drafting-bound
  (0.83 ms per extra position for every request; positions 4-6 add ~0.47 tokens if conditional acceptance stays ~0.68 (extrapolated)).
- Assumed: ragged GEMM compaction must cost what budgeted-K's plumbing did (-8.5%); acceptance beyond position 3.
- Cheap tests left: a short k=6 run measuring per-position acceptance + step time; the compaction path benchmarked in isolation.
- Second set of cards (imbalance, repetition, predictability...): nothing new; the spare resource (SM compute) can't buy back PCIe bytes without
  extra HBM bytes (the binding resource).
