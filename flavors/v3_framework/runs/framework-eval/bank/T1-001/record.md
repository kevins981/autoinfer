team:         T1
builds_on:    BASE
performance:  "unofficial: +3.08% at c=16 (ABBA, 200 prompts, all B > all A), T1 GPU job 9"
summary:
  hypothesis: The drafter reads the full bf16 target lm_head (1.27 GB/GPU) twice per step and all-gathers
              full draft logits (2 x 3.5 MB) — ~3.3 ms of a 38.5 ms decode step (T1 sweep 3-calibration).
              FP8 draft head (halves bytes) + vocab-parallel local argmax (8 B/row instead of logits) should
              save ~1.9 ms/step (napkin ~+3.6% overall) with unchanged acceptance.
  change:     serve flags only (existing engine options): speculative-config draft_lm_head_quantization=fp8,
              use_local_argmax_reduction=true. No code change (empty change.diff).
  results:    Observed (job 9, agents/leader-1/jobs/flags1/abba.log): A 626.2 / 635.9, B 646.3 / 654.6 tok/s;
              acceptance A 2.336/2.351, B 2.339/2.346; p99 TTFT/TPOT B 1108-1220 / 44.7-49.4 ms (A 1117-1256 / 42.1-49.5).
              Server logs of B confirm both options active ("Using local argmax reduction ...").
  conclusion: Interpretation: +3.1% measured vs ~+3.6% napkin; acceptance unaffected (fp8 draft head flips almost
              no draft argmaxes). Exact: target verification unchanged (bf16 target lm_head, full-vocab).
validity:     Ruling 1 allows the FP8 draft lm_head (draft side only; verification exact at baseline precision).
              Local argmax is an exact rewrite of greedy draft selection (draft only). No benchmark-specific
              behaviour. Memory: the fp8 drafter copy of the lm_head adds ~0.64 GB/GPU outside the fixed 5 GiB
              KV budget; benchmark load ran fine; the 128K request is checked by the official eval's gate
              (and by T1's next job).
