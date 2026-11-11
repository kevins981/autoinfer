# Round 4 report from leader-2 (T1)
Best: bank/T1-016 = T1-012 + draft vocab subset (T1-013) + Marlin {64,128,128} tiling (run-1 0005) + --attention-backend TRITON_ATTN.
Official 805.7 tok/s (806.1 / 805.2, CV 0.1%): +7.5% vs T1-012 749.7, +26.7% vs BASE. GSM8K 92.8%, TTFT p99 1283, TPOT p99 41.0,
acceptance 2.335, all gates + 128K pass.

Experiments (all on T1-012 unless noted; unofficial = 200 prompts c=16, fresh servers, interleaved A B B A):
- T1-013 (builds_on T1-012): draft_lm_head_vocab_size=98304 (port of run-1 0008). +1.0% (715.3/715.0 vs 726.0/718.8), accept same.
  N from coverage on vllm repo docs+code (98304 = 99.93%, plateau), ruling 2. KV unchanged (fixed 5 GiB), 0.38 GiB/rank headroom.
- T1-014 (T1-012): --attention-backend TRITON_ATTN (default was FLASHINFER). +5.5% (666.7/678.4 vs 708.2/711.6), accept same,
  gsm8k50 88%, KV same, Running 16. Largest single win; ~1.2 ms/step.
- Marlin {64,128,128} (in T1-016): +0.45% end-to-end (noise; 719.4/715.6 vs 707.5/721.1); microbench gate_up 114->108 us at M=48;
  ~1 ulp differences (fp32 reduction order) only on shapes it applies to. 128K pass with TRITON+Marlin.
- T1-015 (T1-013): R3-B v2 gate: k=3 vs k=2 at c=14 (both fit, Running 14/Waiting 0): +0.1% (680.2 vs 679.6). Acceptance 2.33->2.67
  but step time +14%. NO-GO for v2 (fitting 16 at k=3 would only remove the stall; k=3 has no intrinsic gain here). Caveat: c=14,
  FlashInfer backend, 2 trials/arm.
- --enable-batch-sharded-sampling: not tested. Code read: enabling it disables our greedy local-argmax verify (_greedy_ids_eligible
  requires batch_sharder None); our greedy path has no full logits to save.
- T1-016 official (above). Profile (TRACE=graph + node) in bank/T1-016/profiles.

Proposed SHARED_KB:
- KB-a: FlashInfer's host-side planning cost ~1.2 ms/decode step on our stack; TRITON_ATTN removes it (+5.5% same-job), acceptance and
  KV unchanged. After it, GPU idle is 0.4% of a decode step (TRACE=graph): launch/host gaps are no longer a target. (T1-014, T1-016)
- KB-b: T1-016 16-req decode step 30.2 ms: Marlin 18.0 ms (60%), host-staged AR ~6.8 ms (~128 calls x 53.6 us, incl. peer wait; 23%),
  GDN 2.4, lm_heads 1.8, attention 0.8. Mixed prefill steps ~98 ms. (T1-016 profiles)
- KB-c: k=3 with all requests admitted (c=14) = k=2 throughput on our stack (acceptance +14.5%, step +14%). Shrinking GDN spec state
  for k=3 (R3-B v2) has no payoff. (T1-015)
- KB-d: Draft head vocab subset 98304 (non-benchmark coverage 99.93%) costs no acceptance; +1.0% same-job. (T1-013)
- KB-e: Marlin {64,128,128} for 2-3 m-blocks: gate_up -6 us at M=48, end-to-end below noise (~+0.5%). (T1-016)

Left open: Marlin alone not officially separated; stack not A/B'd unofficially (official only); AR wait vs copy split not measured;
mixed prefill steps not investigated.

Next (one direction): decode all-reduce, now 23% of the step (~6.8 ms) and the largest non-weight-bandwidth cost. Per-call time grew
(49.6 -> 53.6 us) as other work shrank, so part of it may be peer wait/skew (KB8: GPU0 clock). First split copy vs wait per call
(cheap: timestamps in the kernel), then pick: cut calls (e.g. fuse the AR of attention/GDN out-proj with the next norm + skip one sync),
or overlap AR with independent work (two-microbatch decode, run-1's top idea). Evidence: T1-016 profile; KB2/KB5 (copy near floor at
~34 us post-arrival -> if the 53.6 us is mostly wait, the lever is skew, not transport).

Notes: GPU used 58.1/80 (w8 23.6, w7 25.0, eval 9.5). No GPU contention. Workers w7, w8 stopped. Workspaces kept: leader-2.triton,
leader-2.triton.marlin, leader-2.vocab, leader-2.stack (= T1-016, built by w7).
