team:         T42
builds_on:    BASE (CPU analysis only; no engine change; no GPU)
performance:  "n/a (CPU study): certified argmax of the target lm_head from a 6-7-bit integer plane, mean 4-24 candidates/row"
summary:
  hypothesis: Greedy verification consumes only the argmax of each row's logits. A cheap representation of the
              lm_head plus a RIGOROUS bound on its error can certify a small candidate set that always contains the
              baseline argmax; only candidates need exact logits.
  change:     none (analysis). Data: T3's dumped post-final-norm hidden states (agents/leader-3.w1/data/seq, exactly
              the lm_head input), response positions only; PCA basis/statistics from 360 train prompts (14,400 rows),
              tests on held-out prompts (2,000 rows for A, 400 rows for B). Real bf16 lm_head (248,320 x 5120, full vocab).
              Reference = argmax of bf16-rounded fp32 logits.
  results:    Observed:
              A. Low rank (screen.py): eigenbasis of E[h h^T] captures 35% of energy at r=32, 57% at r=256, 67% at r=512,
                 79% at r=1024 (near-isotropic). Bound = per-block Cauchy-Schwarz on the complement (B=8..128 blocks).
                 r=64..512 x B=8..128: candidate set = ALL 248,320 rows for every test row (12 configs).
              B. Integer plane (sweep.py): q = round(w/s) with b-bit signed ints, scale s per group of G along K;
                 rigorous Hoelder bound |h.(w - s q)| <= sum_g ||h_g||_1 s_g/2; center = exact h.(s q);
                 candidates = {v: approx+slack >= max(approx-slack) - (1e-3 + 2^-8|LB|)}. Ref argmax in set: 100% in all.
                 bytes/weight = b/8 + 2/G (bf16 scale)  [the 'bytes_per_w' field in sweep_res.json is wrong: it used 16/G]
                   b=8 per-row (1.000 B/w): mean 2.7, p50 2, p99 14, max 20
                   b=7 G=512   (0.879):     mean 5.2, p50 2, p99 41, max 76
                   b=7 G=128   (0.891):     mean 4.0, p50 2, p99 26, max 51
                   b=6 G=128   (0.766):     mean 24,  p50 6, p99 320, max 769
                   b=6 G=512   (0.754):     mean 57,  p50 10, p99 940, max 1771
                   b=5 G=128   (0.641):     mean 7,923, p50 320 (fails)
                   b<=4: essentially all rows (fails)
                 Top1-top2 logit gap: median 3.2, p10 0.42 (top logit ~25).
  conclusion: Interpretation: the lm_head has no usable low-rank structure for exact screening, but plenty of
              PRECISION redundancy: ~6-7 bits per weight (0.38-0.44 of bf16 bytes; today's lossless code is 0.78)
              certify the greedy argmax with a few to a few dozen candidates. At the champion's lm_head (1.07 ms at
              M=64, ~95% of read roof for 0.993 GB/rank, T1-037) this projects to ~0.55-0.6 ms + rescoring, i.e.
              ~0.45 ms/step (~1.7% decode, ~1.2% e2e). NOT yet established: GPU kernel cost; a complete error
              enclosure for the GPU screen (screen-center fp32 error ~ gamma_K * slack * 2*qmax ~ +4% slack, plus the
              baseline's own accumulation error and bf16 rounding) -- see proposals/T42-1.md. Corrects my own napkin
              belief (sparring T42-1) that worst-case bounds in 5120 dims must be too loose: truncation fails,
              quantization (all coordinates kept, small bounded perturbation) works.
validity:     no engine change. Any implementation must keep the decision bitwise-equivalent to argmax over the
              baseline's bf16 logits (rigorous enclosure + exact rescoring), per TASK.definition.
