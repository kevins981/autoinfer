team:         T2
builds_on:    T2-001
performance:  unofficial: stopped after 1 start, 645.3 tok/s (champion T2-001: 648.6 / 658.6); first try OOM'd
summary:
  hypothesis: FP8 copy of the shared lm_head for the MTP drafter (speculative-config draft_lm_head_quantization="fp8", ruling 1) + use_local_argmax_reduction=true (draft-side vocab-parallel argmax instead of logits all-gather) cut ~1.3 + ~0.4 ms per decode step.
  change:     serve flags: T2-001's + --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_lm_head_quantization":"fp8","use_local_argmax_reduction":true}'
  results:    Observed: unofficial_eval --ws t2-cg --label t2-cg-dlm8 failed at server_start: torch.OutOfMemoryError in speculator.capture() (GPU 1: 15.75 MiB free of 23.52 GiB) (unofficial/ws/t2-cg-dlm8/1/b/trial1/server.log).
  conclusion: Interpretation: the extra 0.64 GB/GPU FP8 head copy does not fit beside 2048-token prefill graphs with a 5 GiB KV budget. Retried with --kv-cache-memory 4 GiB (KV pool holds 2.04x a 128K request at 5 GiB) as label t2-cg-dlm8-kv4.
validity:     flags only; draft side only, verification exact (ruling 1). Attribution: use_local_argmax_reduction suggested for this test by T1's lane split (leader-1).
retry (label t2-cg-dlm8-kv4, + --kv-cache-memory 4294967296; serve_args.txt now holds this version):
  Observed: 645.3 tok/s, stopped after 1 start (below 648); MTP acceptance 2.346 (unchanged vs 2.351: the FP8 draft head does not hurt acceptance); p99 TPOT 65.2 ms (gate 55.3) and TTFT 1515 ms (unofficial/ws/t2-cg-dlm8-kv4/1).
  Interpretation: screen negative, not a measured loss (1 start). The TPOT p99 jump is unexplained: suspect the FP8 lm_head kernel path on sm89 (per-token-scaled FP8 GEMM, 248K x 5120 at M=16) is slower than the bf16 CUTLASS head, or the local-argmax path; needs a profile before more starts.
