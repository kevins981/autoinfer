team:         T1
builds_on:    T1-017
performance:  "unofficial negative (1 start): -6.10% vs T1-017; p99 TTFT 2559 ms gate failed"
summary:
  hypothesis: T1-017's W8A8 route used vLLM's default Triton config (no 4090 config file for these shapes). Tuned configs
              are 13-26% faster at every tuned (shape, M) (job 156). With them, W8A8 also beats Marlin and W8A16 from
              M=80 (job 158), so the thresholds move from 128/160/160 to 80.
  change:     3 tuned config JSONs (vllm/model_executor/layers/quantization/utils/configs/N=..,K=..,device_name=NVIDIA_GeForce_RTX_4090,
              dtype=fp8_w8a8,block_shape=[128,128].json; keys M=80..2048) for gate_up 17408x5120, down 5120x8704 and out 5120x3072.
              They are read by vLLM's get_w8a8_block_fp8_configs. W8A8_MIN_M is 80 for all 3 shapes; warm-up also covers M=80/96.
              Worker t1-w1, workspace t1-w13. No other code change vs T1-017.
  results:    Observed:
              - tuning job 156: tuned vs default, e.g. gate_up M=512 365.8 vs 430.0 us; down M=128 81.9 vs 111.4;
                out M=1024 124.3 vs 149.4 (agents/t1-w1/w8a16/tune_table.txt).
              - threshold job 158: at M=80-128, tuned W8A8 is 6-35% faster than the better of Marlin/W8A16
                (agents/t1-w1/w8a16/out6.txt).
              unofficial #1: B 782.3 vs HEAD mean 833.1 (-6.10%), p99 TTFT 2559 ms (gate fail), TPOT 43.2; server log
              shows "Triton kernel JIT compilation during inference: _w8a8_triton_block_scaled_mm".
  conclusion: Interpretation: runtime JIT stalls. The tuned JSON has ~8 distinct configs per shape, and warm-up compiled them
              only at M multiples of 16, while Triton specializes integer args on divisibility by 16. Real M values triggered
              new compiles during serving (T1-017 had a single default config, so only one stall). Not a kernel-speed result.
              Superseded by a record with complete warm-up.
validity:     Same FP8 scheme as T1-017 (per-token-group-128 fp8 activations x the checkpoint's 128x128 block weights, fp32
              accumulation across K blocks; BK=128 in all configs). Only tiling changes, plus a wider M range on the W8A8
              route (allowed per batch size). Needs its own same-job FP8 GSM8K A/B (below). Configs are keyed on
              shape/device, not the workload. Reviewed by leader-1.

unofficial #1 (2026-11-15T11:50): negative (1 start): -6.10% vs HEAD T1-017 (B 782.3 vs reference mean 833.1 over 2 starts, 0 run in this job; 200 prompts); GSM8K/128K not run (stopped after 1 start), p99 TTFT 2559 ms / TPOT 43.2 ms, MTP acceptance 2.345; gates failed: latency p99_ttft_ms 2559 > 120% of BASE 1341 (unofficial/1/unofficial.json)
