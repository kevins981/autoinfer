team:         T2
builds_on:    T2-006
performance:  unofficial: stopped after 1 start, 704.6 tok/s (champion T2-006 738.8 / 747.8)
summary:
  hypothesis: with cheap draft steps (64K draft head, local argmax, W8A8) a 3rd MTP draft pays: T1-001 on BASE measured acceptance 2.70 but +22% step.
  change:     serve args: T2-006's with num_speculative_tokens 3 (no code).
  results:    Observed: unofficial_eval --ws t2-k3 --label t2-k3 --long --stop-below 740: 704.6 tok/s after 1 start; acceptance 2.654; p99 TPOT 32.3 ms (T2-006: 37.1), p99 TTFT 1828 ms (over the 1609 gate); server log: only 13-14 requests running (KV 259,441 tokens in 1,600-token blocks: k=3 needs more GDN state slots per request).
  conclusion: Interpretation: screen negative caused by lost concurrency (14 of 16 slots) and queueing (TTFT), not by step cost: per-token decode got faster (TPOT p99 -13%). Retry with a larger KV budget so 16 requests run.
validity:     serve flag only; exact MTP verification; memory change -> 128K needed if pursued.
retry 1 (label t2-k3-kv6, --kv-cache-memory 6 GiB): OOM during CUDA graph capture at server start (unofficial/ws/t2-k3-kv6/1).
retry 2 (label t2-k3-kv575-mbt512): KV 5.75 GiB (16 requests need ~185 blocks of 1,600 tokens; 5 GiB = 162 blocks held 14) + max-num-batched-tokens 512 and prefill graphs to 512 to free activation memory.
retry 2 failed at server start (allocation failure in aten::empty: memory), unofficial/ws/t2-k3-kv575-mbt512/1.
retry 3 (label t2-k3-nopc, + --no-enable-prefix-caching, allowed by manager ruling): 687.4 tok/s after 1 start, acceptance 2.632, p99 TPOT 31.6 ms / TTFT 1630 ms; still "Running: 14 reqs, Waiting: 2, GPU KV cache usage 95.3%".
  Interpretation: the cap is not prefix caching but the GDN state slots: every speculative position keeps its own fp32 recurrent state (24 heads x 128 x 128 x 4 B x 48 layers = ~75 MB per request per slot per GPU), drawn from the same KV pool; k=3 needs 4 slots per request (~4.8 GB for 16 requests). This is the framework's "in-place overwrite keep" rollback trap (02 §4.4) showing up as capacity C. K1 (replay commit: only the committed state is kept) removes the per-position states, which would make k=3 fit. k=3 parked until K1 lands.
