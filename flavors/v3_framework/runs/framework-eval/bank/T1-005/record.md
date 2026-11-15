team:         T1
builds_on:    T1-004
performance:  "unofficial: +1.78% at c=16 vs T1-004 (ABBA, 200 prompts, all B > all A), T1 GPU job (t1-bss)"
summary:
  hypothesis: The target's vocab-parallel logits are all-gathered to both ranks (42 x 248K bf16, 0.84 ms/decode step) and
              both ranks sample all requests. With batch-sharded sampling each rank receives full-vocab logits only for
              its half of the requests (all-to-all, half the bytes) and samples only those -> ~-0.5 ms/step, ~+1%.
  change:     existing engine flag --enable-batch-sharded-sampling (Qwen3.5 implements compute_logits_local). No code.
  results:    Observed (agents/leader-1/jobs/bss/abba.log): A 680.8/670.0, B 685.2/689.7 tok/s; acceptance A 2.351/2.344,
              B 2.349/2.360; p99 TTFT/TPOT B 1262/1214 / 47.8/47.6 ms vs A 1200/1233 / 48.9/49.1. B logs: "Batch-sharded
              sampling enabled."
  conclusion: Interpretation: small consistent gain, a bit above napkin (sampling work on full logits also halves).
validity:     Exact: each request is still sampled from its full-vocab target logits (same values, same argmax/rejection
              math), only which rank does it changes. Engine option, generic; parsers unaffected. No memory change of note.
