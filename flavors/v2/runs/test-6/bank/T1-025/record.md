team:         T1
builds_on:    T1-023 (+ T1-024's gate)
performance:  "official #2: 874.8 tok/s (863.7/885.8, CV 1.8%; +1.5% vs T1-023 861.5, but trial 1 is at T1-023 level), same-job pooled +0.7%; +37.5% vs BASE, GSM8K 92.8%, TTFT p99 1212, TPOT p99 41.4, accept 2.33, gates + 128K pass"
summary:
  hypothesis: hide prefill-size ARs behind GEMMs: in steps with >= 256 tokens, run each layer's out_proj -> AR -> add+norm -> MLP -> AR
              as 2 token chunks (64-aligned, larger first); chunk 0 on the current stream with a copy-engine (DMA) AR, chunk 1 on one
              aux stream with the regular host-staged AR; join before the layer returns (gate T1-024: -13.3%/layer at M=300).
  change:     dma_staged_ar.{cu,py} (own pinned slots/flags, 24 MB, D2H + stream-memop doorbell/wait + H2D + add; JIT dir by source
              hash); cuda_communicator creates it next to hostar; GDN forward_cuda / Qwen3NextAttention.forward take defer_out_proj;
              Qwen3_5DecoderLayer._split_chain; the layer returns an all-reduced output and the next norm skips its AR
              (forward_fused(reduced=True)). Main model only (not MTP/aux-hidden/PP/SP). Opt-out VLLM_DMA_SPLIT=0; threshold
              VLLM_DMA_SPLIT_MIN_TOKENS=256; test-only VLLM_DMA_SPLIT_CHECK=1. Worker leader-2.w12, ws leader-2.dmaar2 (out/NOTES.md).
  results:    Observed: job 93 eager check (256 layer-checks, real activations): DMA AR vs hostar 0 mismatches; split vs unsplit
              bitwise at 1024/1024 and 1024/1017, else max diff 0.5 on outputs (~1 bf16 ulp at magnitude; 3.5-5.6% elements),
              residual 0.0625 (Marlin row-count reorder). Graph server: startup 75 s, gsm8k50 88%, 128K pass (1024/1024 splits).
              ABBA job 94: A 817.5, B 833.5, B 817.7, A 821.9; job 95 (B A A B): B 822.7, A 816.5, A 817.2, B 822.1. Pooled A 818.3,
              B 824.0 (+0.7%); B mean above every A, 3/4 B above every A. TTFT p50 B lower in job 94 (192 vs 197.5), not in 95.
              Acceptance/TPOT same. Estimate +2-3% (gate scaled), measured +0.7%.
  conclusion: Interpretation: small consistent throughput edge (~0.7%); less than projected because many mixed steps are < 256
              tokens, the in-server AR is cheaper than in the microbench, and in-chain copies run at ~12 GB/s (unexplained).
validity:     leader-2 review of the raw diff: DMA AR bitwise == hostar (same bf16 add); token split = per-row-independent GEMMs with a
              Marlin reduction-order change only (fp reorder, like padding); norms identical kernels; hostar used on one stream only
              (no concurrent hostar calls), DMA lane has its own buffers; cross-stream tensors joined before return (aux-allocated
              tensors stay in aux's pool). Generic threshold, not benchmark-keyed. gsm8k50/acceptance unchanged, 128K pass. OK.

official #1 (2026-11-08T06:35): FAILED at apply_diff (official/1/prepare.log)
note (leader-2): official #1 FAILED is infrastructure only: the earlier budget-refused attempt (exit 76, nothing recorded) had already created workspace eval-T1-025-1, so the retry could not create it. Same change, re-run as official #2.

official #2 (2026-11-08T06:46): 874.8 tok/s at c=16 (mean of 2 trials: 863.7 / 885.8, CV 1.8%) (+37.5% vs BASE), GSM8K 92.8%, MTP acceptance 2.33, TTFT p50 182 ms, TPOT p50 16.6 ms; gates: GSM8K pass, latency pass, 128K request pass (official/2/results.json)
