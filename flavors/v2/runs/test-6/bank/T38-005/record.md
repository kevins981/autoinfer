team:         T38
builds_on:    T38-003, T38-004 (measurement only; no engine change)
performance:  n/a. Job 393 (0.9 GPU-min, leader-38.w1); vocab study on CPU.
summary:
  hypothesis: (T38-3 stage 2) the decode side of PP+trees can be recovered to r >= 0.93 of the champion's decode
              throughput: GEMMs at <= 1.10x floor at M=96-160 per microbatch, and drafting cheap enough per level.
  change:     none. agents/leader-38.w1/s2/ (NOTES.md, splitk.json, stage_total.py, draft_level.json, summary.txt),
              agents/leader-38.w1/tree/vocab_analysis.txt.
  results:    Observed:
              - Best existing kernel per shape (champion's bs_gemm_marlin, all 43 tile configs x split-K 1-8; exact:
                fp32 partials over whole 128-K blocks, fixed split order), per 32-layer PP stage, floor 13.22 ms:
                M=96 14.49 ms (1.10x), M=128 15.24 (1.15x; N=5120 shapes 1.21-1.33x even with split-K 3),
                M=160 18.88 (1.43x), M=192 19.27. At M>=160 compute-limited (190-260 of 330 TFLOPS mma peak).
              - One draft level on one GPU: MTP-layer GEMMs at the weight floor (~470-480 us at 8/32 rows); draft head
                98K FP8 full 642/568 us (M=8/32), champion's t27 screen 161/323 us (scores kernel reads U twice at M=32;
                BM=32 would likely give ~130-150, untested). Level ~0.65 ms (8 rows), ~0.9 ms (32 rows), ~0.75 with the fix.
              - Draft vocab = first V ids (as the champion's 98K subset). Level-wise K=4 D=6 N=15 tokens/step: 98K 4.09,
                64K 3.89 (-0.20 [-0.26,-0.15]), 48K 3.73, 32K 3.60, 16K 3.26. Cutting the vocab saves less than it costs.
  conclusion: Interpretation: decode per round with measured parts and zero pipeline bubble: N=11/D=6 (M=96) 39.4 ms for
              16 x 3.94 -> ~1600 tok/s; N=15/D=6 (M=128) 41.2 ms, 16 x 4.09 -> ~1588; N=15/D=8 -> ~1604; champion ~1747
              (16 x 2.73 / 25 ms). r = 0.91-0.92 < 0.93 (my stage-2 pass rule). Wall model t = 0.78/r + 0.22 x 0.51:
              +3.7% before any bubble (stock PP: 12%), tree-kernel or scheduling loss; with a hypothetical better
              narrow-K kernel, r ~0.935 -> ~+5.8% ceiling. Stage 2 FAILS; T38-3 closed (see agents/leader-38/T38-writeup.md).
validity:     n/a (no engine change)
