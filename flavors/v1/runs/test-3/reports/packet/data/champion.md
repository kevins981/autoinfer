# CHAMPION (run test-3)
record: T1-009 (round 2) = T1-005 (host-staged AR + AG; prefill-graph flag) + greedy target-verification
local-argmax fast path (code) + spec config flags use_local_argmax_reduction and draft_lm_head_quantization=fp8.
serve_args.txt: --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true,"draft_lm_head_quantization":"fp8"}'
official #1: 730.9 tok/s at c=16 (+15.8% vs BASE; 2 trials 729.7/732.1); TTFT p50 204 / p99 1438 ms;
TPOT p50 20.0 / p99 41.3 ms; GSM8K 93.6%; MTP acceptance 2.344; 128K request pass. bank/T1-009/official/1/results.json
Gates (vs BASE): TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%. All pass.
History: BASE 631.4 -> T1-005 699.2 (r1) -> T1-009 730.9 (r2).
A performance reference, not a required starting point.
