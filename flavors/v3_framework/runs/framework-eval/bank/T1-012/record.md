team:         T1
builds_on:    T2-006
performance:  "unofficial: mixed, +1.18% (ABBA 200 prompts: B 790.3/795.1 vs A 776.2/790.6), worker job 51"
summary:
  hypothesis: With batch-sharded sampling each decode step still all-to-alls ~5 MB of half-vocab target logits (NCCL SendRecv
              ~0.4 ms/step, T2-006 profile). For all-greedy batches without logits needs, the verified tokens only need the
              target argmax: local (max, idx) per vocab shard + an 8 B/row all-gather -> ~-0.4 ms/step, ~+1.3%.
  change:     new v1/worker/gpu/sample/greedy_local.py + ~100 lines in model_runner.sample: fast path (env
              VLLM_GREEDY_LOCAL_ARGMAX, default on) used only if every request is temperature 0, no logprobs, no grammar /
              structured output, no logits processing (penalties, bias, bad words, min_tokens, processors), plain sampler
              (no watermark / adaptive verification); otherwise the existing path runs unchanged. greedy_verify Triton
              kernel reproduces the greedy branch of the rejection sampler. prompt_logprobs unaffected (computed separately).
  results:    Observed: ABBA mixed +1.18%; acceptance B 2.345/2.361 vs A 2.347/2.360; TTFT/TPOT p99 B 996/961 / 34.3/35.6 vs
              A 844/1101 / 36.9/34.1. B log: "Using target-side greedy local-argmax sampling fast path." gsm8k 50 on B 90.0%.
              CPU unit check of the shard combine vs torch.argmax incl. ties: pass (agents/leader-1.w1/test_greedy_local.py).
  conclusion: Interpretation: likely a real ~1% (napkin 1.3%), not resolved by one ABBA. Stackable.
validity:     Exact: argmax over the full vocab, ties to the lowest global id (= torch.argmax / rejection kernel rule); accept/
              reject logic identical to the greedy rejection path. Keyed to request properties (greedy, no logits needs),
              not to the benchmark; generic fallback for everything else.
