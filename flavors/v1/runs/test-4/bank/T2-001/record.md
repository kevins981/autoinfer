team:         T2
performance:  "unofficial: +2.4% vs BASE same job (608.0 vs 593.6 tok/s, 150 prompts, 1 run each; within noise)"
summary:
  hypothesis: FP8 (W8A8) draft lm_head + vocab-parallel local argmax in the drafter cut ~2 ms/decode step (~+4%).
  change:     flags only (serve_args.txt): use_local_argmax_reduction=true, draft_lm_head_quantization=fp8. Empty diff.
  results:    Observed (gpu job by leader-2.w1, 150 prompts c=16, same job as BASE arm): 608.0 tok/s (BASE 593.6),
              TPOT med/p99 23.1/54.8 ms (BASE 23.6/48.8), TTFT p99 1405 (BASE 1357), acceptance 2.366 (.790/.576) vs BASE 2.365.
              Log: "Quantized a fp8 copy of the lm_head (124160 x 5120 per rank)", CutlassFP8ScaledMMLinearKernel (no slow fallback),
              "Using local argmax reduction". Caveat: worker edited workspace code during this arm; the untested target fast path
              may have been imported (no errors, acceptance unchanged). Re-measured in job 2.
  conclusion: Interpretation: acceptance unchanged by the FP8 head; gain smaller than the +4% estimate, not yet distinguishable
              from noise. p99 TPOT 54.8 close to the 55.3 gate in a single short run (p99 noise ~±15%) - needs watching.
validity:     leader-2: flags only. FP8 W8A8 draft head ruled allowed by manager (drafter only); target verifies bf16 full vocab.
              Local argmax is exact (drafter-only anyway). OK.
