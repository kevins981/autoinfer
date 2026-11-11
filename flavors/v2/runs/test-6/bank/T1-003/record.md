team:         T1
performance:  "n/a (diagnosis of T1-002: in-server nsys + M=48 overlap microbench)"
summary:
  hypothesis: (a) find where T1-002's gain comes from in the server and what is left in AR / all-gather time;
              (b) check whether the M=48 write->flag->read chain can be pipelined (manager's question).
  change:     none beyond T1-002 (same diff). Worker leader-1.w2, ws leader-1.hostar2 (out/nsys, out/ana*.py,
              out/bench2.py, out/job8.log, out/notes.md).
  results:    Observed:
              (a) nsys, 64 prompts c=16, both GPUs paired in order, 135 ARs/step in both runs, vs BASE nsys_c16:
                - decode ARs: mean 49.6 us vs 60.8 (BASE); post-arrival (min of 2 ranks) 34.0 vs 48.1; start skew
                  mean/median 30/4.4 vs 28.6/9.6 us. AR per decode step GPU0 4.71 vs 6.24 ms, GPU1 5.29 vs 7.36 ms.
                - prefill ARs: mean 622 vs 626 us; post-arrival 512 vs 522; start skew mean/median 316/247 vs 242/186 us.
                  Prefill-step AR total GPU0 78.6 ms / 212 ms step vs BASE 66.8 / 196 (different run, nsys-inflated).
                  -> no gain in prefill steps.
                - logits all-gathers per decode step: 21, 330, 946, 41, 330 us = 1.14 ms (BASE 1.11).
              (b) M=48 graph us/call, G writer=reader blocks x K chunks: K=1 51-53 for G 8..64; K=2 54-59; K=4 58-65;
                  K=8 69-85. Rank-1 timeline: writer fence done at 21-23 us (K=1), reader sees flag immediately, ends ~22 us
                  later. Each chunk fence costs a fixed ~5 us. Concurrent write-only + read-only kernels (no flags, G=16):
                  26-27 / 25 alone, 41 us together.
  conclusion: Interpretation: T1-002's gain is all in decode (~1.5 ms/step). Prefill-step AR time is dominated by
              RANK SKEW (~250-316 us mean start skew on ~620 us calls), not transport: the slower rank arrives late
              (likely eager-mode CPU launch differences or load imbalance before the AR). The M=48 chain is near its floor:
              the link only partly overlaps up/down (41 us for both vs 27 if fully parallel), and per-chunk fences cost
              ~5 us each, so pipelining loses. At most ~5 us/call (<2%) left via a warp-specialized async-fence design; not
              recommended.
validity:     diagnosis only; no new code.
