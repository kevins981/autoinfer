team:         T1
performance:  "unofficial: 670.1 vs same-job T1-007 719.7 tok/s (-6.9%); TTFT p99 2133 ms (fails the 1784 gate)"
summary:
  hypothesis: with cheap drafts (T1-007), a 3rd MTP token raises acceptance enough to pay for a longer step.
  change:     T1-007 flags with num_speculative_tokens=3. Worker leader-1.w3, ws leader-1.logitsflags (out/C3, out/job2.sh).
  results:    Observed (job 24, same job as T1-007 rerun "C2b"): k=3 670.1 tok/s vs 719.7; TTFT p50/p99 792/2133 vs 210/1422 ms;
              TPOT p50/p99 19.3/41.6 vs 20.5/48.9; acceptance 2.715 vs 2.356 (napkin est ~2.65); per-position (steady)
              .952/.881/.786. gsm8k50 88% both. 128K PASS; memory same as k=2 (model 14.99 GiB, graphs 1.34 GiB, KV 259,441 tok).
  conclusion: Interpretation: decode improves (TPOT -6%) as predicted, but TTFT blows up ~4x, so prefill/mixed steps got
              much slower or prefills are delayed; cause not diagnosed (no profile). Not viable as is.
validity:     exact spec decode (method unchanged, verification exact); memory change, 128K tested. Valid but fails TTFT gate.
