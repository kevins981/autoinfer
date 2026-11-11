team:         T1
builds_on:    T1-013 + T1-014 (+ Marlin tiling, run-1 patch 0005)
performance:  "official #1: 805.7 tok/s (+7.5% vs T1-012 749.7, +26.7% vs BASE), GSM8K 92.8%, TTFT p99 1283, TPOT p99 41.0, accept 2.335, all gates + 128K pass"
summary:
  hypothesis: the three exact run-1 wins stack independently on T1-012 (attention planning idle, draft head bytes, gate_up tiling).
  change:     T1-013 code (draft_lm_head_vocab_size=98304) + run-1 patch 0005 (csrc marlin.cu: {64,128,128} instead of {64,256,256}
              for 2-3 m-block batches when valid) + --attention-backend TRITON_ATTN. ws leader-2.stack (leader-2 assembled).
  results:    Observed: Marlin (w7, agents/leader-2.w7/job2 + mb): C (TRITON+Marlin) 719.4/715.6 vs B (TRITON) 707.5/721.1, +0.45%
              (B spread 1.9% > gap); microbench gate_up M=32 108.2->105.0 us, M=48 114.2->107.8 us (est ~0.3-0.4 ms/step, ~+0.8%);
              16/18 real-shape outputs bitwise equal, gate_up M=32/48 differ by ~1 ulp (fp32 reduction order). 128K request with
              TRITON + Marlin: pass, KV 267,721 tokens unchanged. Stack itself not measured unofficially; official eval directly.
  conclusion: Interpretation: expected ~+6-7% over T1-012 if additive.
validity:     leader-2 review: TRITON = different attention kernel, same math (fp reordering); vocab subset = drafter only, target
              verifies full vocab exactly, N from repo text (ruling 2); Marlin = tiling change, same math, fp32 reduction order only
              (shown ~1 ulp). No benchmark-specific logic. Memory unchanged; 128K passed unofficially. OK.

official #1 (2026-11-08T02:57): 805.7 tok/s at c=16 (mean of 2 trials: 806.1 / 805.2, CV 0.1%) (+26.7% vs BASE), GSM8K 92.8%, MTP acceptance 2.335, TTFT p50 191 ms, TPOT p50 18.0 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)

profiles/ (leader-2.w7, agents/leader-2.w7/prof_stack, sqlite there): 16-req decode step 30.2 ms, GPU idle 0.12 ms (0.4%, TRACE=graph);
kernels/step (node): Marlin 17.96 ms (60%), host AR ~6.8 ms (~128 calls x 53.6 us incl. peer wait, 23%), GDN 2.37, lm_heads 1.77,
attention 0.78, norms etc 0.29. vs KB9 (T1-005 38.5 ms): lm_heads -2.5, logits AG -1.65, GDN -1.3, idle gap gone; AR grew 5.7->6.8.
Mixed prefill steps ~98 ms period (vs 30 decode).
