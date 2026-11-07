# Champion (run test-4)
record: T1-003 (round 1) = T1-001 diff (host-staged 2-rank all-reduce through pinned shared host memory, bitwise = NCCL)
  + T2-003 serve_args:
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true,"draft_lm_head_quantization":"fp8"}'
  --enable-batch-sharded-sampling
official #1: 691.3 tok/s at c=16 (+8.7% vs BASE; trials 696.1 / 686.4); TTFT p50/p99 226/1303 ms; TPOT p50/p99 21.3/46.1 ms;
GSM8K 92.8%; MTP acceptance 2.346; 128K request pass. bank/T1-003/official/1/results.json
Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms, GSM8K >= 89.6%. All pass.
History: BASE 635.9 -> T1-003 691.3 (r1).
A performance reference, not a required starting point.
