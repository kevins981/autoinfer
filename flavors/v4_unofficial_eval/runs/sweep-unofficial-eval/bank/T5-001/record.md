team:         T5
builds_on:    T3-004
performance:  "offline only: no engine change evaluated (C1 step A, compressibility gate)"
summary:
  hypothesis: The all-reduce payload (bf16 partial sums) can be encoded losslessly with ~23% fewer bytes: within a
              block of 32 neighbouring values the exponent is predictable, while sign and mantissa are not. The host AR
              takes ~10 us + 78 us/MB (T3-001), so fewer bytes means less time with no overlap needed.
  change:     change.diff = HEAD + an env-gated dump of AR inputs (VLLM_AR_DUMP_DIR, in cuda_communicator.all_reduce)
              used only to collect real payloads (eager server). Offline codec + analysis: agents/t5-w1/codec.py, analyze.py.
              Codec: per 32 values a block max exponent E; each value = sign + 4-bit offset (E-e, with a code for +/-0)
              + 7-bit mantissa; a block with any subnormal/inf/NaN or offset > 14 is sent raw (1-bit flag).
  results:    Observed (jobs 206 and a re-dump with rows >= 40, eager HEAD, ShareGPT prompts; worker t5-w1):
              B=32, 4-bit offsets: ratio 0.77 of raw bytes incl. headers, 0.2-0.6% raw blocks;
              decode(encode(x)) bitwise identical on all 160 files checked. 3-bit offsets: 0.83 (41% raw blocks).
              The first dump was dominated by 2-/16-row warm-up tensors; final per-kind numbers from the re-dump,
              if any, are in agents/t5-w1/summary.md.
  conclusion: Interpretation: passes the 0.82 gate provisionally. Napkin: decode AR bandwidth part ~34 of 42 us
              -> ~-8 us x 135 AR/step ~ -1 ms/step (~2% of wall); prefill-size ARs at ~60 instead of ~80 us/MB
              ~ -2.4%; stacks with T4's P4 (two-half overlap, where the link is the bottleneck). NOT built: the
              compressed kernel (step B) and its microbench were stopped by the run's end.
validity:     Manager ruling 2026-11-15: lossless AR encoding ALLOWED if the peer decodes the identical bf16 tensor for
              every input (incl. -0, subnormals, inf/NaN, raw fallback), with a recorded bitwise stress test vs the
              current AR and a generic codec. This record changes no served computation (dump is off by default);
              the stress test is still owed by whoever builds step B.
