team:         T38
builds_on:    T38-002 (offline measurement; no engine change)
performance:  n/a (measurements for the PP+tree cost model). GPU: jobs 389 (failed at start, Marlin workspace), 390 (0.3 min)
summary:
  hypothesis: (A) realistic level-wise tree drafting keeps most of best-first acceptance at few drafter levels; (B) at
              pipeline-parallel shapes (full width, half the layers per GPU) the champion's decode GEMMs stay at the weight-read
              floor up to ~96-128 rows, so tree rows are cheap.
  change:     none. leader-38.w1: (A) agents/leader-38.w1/tree/tree_level.py, level_analysis.txt/.json (CPU, exact, as T38-002);
              (B) agents/leader-38.w1/pp_gemm/pp_gemm.py, summary.txt, pp_gemm.json (workspace t38-champ unmodified, GPU0,
              CUDA graphs, cold weights; Marlin vs dequantized reference relF 0.0028).
  results:    Observed (A), tokens/step, K=4 nodes expanded per level, D levels (K=8 adds <= +0.04):
                N=11: D=4 3.58, D=6 3.94, D=8 4.09, D=10 4.18
                N=15: D=4 3.67, D=5 3.92, D=6 4.09, D=8 4.28, D=10 4.38
                N=19: D=5 4.00, D=6 4.18, D=8 4.39, D=10 4.51   (best-first, N MTP steps: 4.21 / 4.53 / 4.74)
              Observed (B), GEMM time per 32-layer PP stage (24 GDN + 8 attention layers), best champion kernel per M:
                M=16 13.4 ms, 32 13.7, 48 14.1, 64 15.4, 96 16.9 (Marlin; W8A8 bs untested <=96), 128 16.3 (bs), 160 18.9,
                192 19.3, 256 25.9. Floor (920 GB/s) is reached by the best kernels (917-928 GB/s) at M<=48. At M=128,
                bs is 1.07-1.10x floor on wide-N shapes but 1.47-1.54x on K=6144/17408 shapes. t31 has no good config for
                N>=16K at M=64-96 (2.0x). TP cross-check at M=64: 64 layers 15.4 ms of GEMM (same bytes per GPU as a PP stage).
  conclusion: Interpretation: with 2 GPUs, PP keeps both busy only with 2 microbatches in flight, so every weight byte (layers,
              lm_head, MTP layer, draft head per level) is read twice per round. Using (A)+(B), N=15/D=6/K=4 (8 seqs x 16
              rows = M=128 per microbatch): period ~ 32.6 GEMM + 3 non-GEMM (assumed) + 1.6 head + 6.3 drafting (6 levels x
              ~1.05) + 0.3 transfers = 43.8 ms for 16 x 4.09 tokens -> ~1490 tok/s decode vs the champion's 16 x 2.73 / 25 ms
              = 1747: about -15%. Break-even would need T = 4.78 at this cost. The tree half of the PP+trees idea does not pay
              for itself with today's kernels; with all GEMMs at 1.07x floor at M=128 and a cheaper draft head (~0.6 ms/level),
              decode is ~break-even (+1%). What is left of the direction is PP's prefill saving (no all-reduce): see T38-3.
validity:     n/a (no engine change)
