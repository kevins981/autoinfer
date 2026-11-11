# leader-13 (T13, explore) notes
## Reading summary (2026-11-09)
- Floors per T2/T10: decode ~22 ms floor vs 23.8-25.5; prefill ~20 s tensor floor; small mixed steps (<150 pt, 215/trial) ~6.5 s excess (T7), unclaimed (T9 works R>=256).
- Dead: FP8/INT8 exact MMA, PP, replication, spatial/temporal sharing, AR-under-consumer (T4), CE decode AR (T9-006: 23-43 us slower than hostar+pf), drafter training/vocab, n-gram, deeper spec, 2-warp GEMM.
## Direction candidate: producer-fused AR (FLUX-style, through host memory)
- Producer Marlin (out_proj/o_proj/down) writes partial tiles to host as they complete (posted writes, overlapped with GEMM); consumer reads peer + add + norm. Removes AR kernel boundary + write half from critical path.
- Napkin: save ~10-13 us/AR net of lost L2-prefetch window -> ~1.3-1.7 ms/decode step; + small mixed steps ~3.8 ms/step. ~+4.5-5.5% if it works. Borderline bar.
- Risk: T8-001 SM round trip x1.45 beside HBM; T4 AR beside consumer ~100 us. Unknown: writes-only interference. Marlin tile completion spread.
- T13-001 premise (w1, <=6 GPU-min): A interference split, B tile timeline.
- Manager notes on T13 premise: (1) gate A next to real M=48 producers, NUMA verified; (2) count bytes: epilogue must encode (0.781) or pay 491 vs 383 KB; (3) small-step credit only R<256 (T9-005 covers >=256). Bar 996.3 (5% over 948.9).
- T13-001 (jobs 111-113, 0.8 min): KILL. Writes cost GEMM ~+9us/383KB (out_proj +26%); Marlin tiles land at ~87-94% of kernel. Recommended voluntary close; write-up agents/leader-13/T13-writeup.md.
