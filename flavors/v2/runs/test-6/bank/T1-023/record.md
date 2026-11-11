team:         T1
builds_on:    T1-020
performance:  "official #1: 861.5 tok/s (863.1/859.8; +1.3% vs T1-020 850.3, same-job ABBA +1.5%), +35.5% vs BASE, GSM8K 93.2%, TTFT p99 1230, TPOT p99 33.9, accept 2.33, all gates + 128K pass"
summary:
  hypothesis: R7-B: mixed (prefill-containing) steps still run a mode-2 flush + the Triton per-token-state GDN kernel for their spec
              requests (~6.5 ms/mixed step in nsys); the replay kernel would take ~1.2 ms. R7-C: the prefill graph grid pads up to 25%;
              a 32-token grid cuts padded Marlin/AR rows.
  change:     csrc fused_gdn_decode_kernel.cu: pre-norm output variant (empty norm_weight) for the stock CUDA MTP and replay kernels;
              qwen_gdn_linear_attn.py: in mixed steps the spec subset runs fused_gdn_decode_post_conv_mtp_replay (pre-norm) on post-conv
              rows, the gated norm stays on the existing unfused path, mode-2 flush skipped (mode-3 clear kept); opt-out
              VLLM_GDN_MIXED_REPLAY=0. Serve: prefill_cudagraph_capture_token_sizes every 32 above 128 to 2048 (finer below as
              before). Worker leader-2.w13, ws leader-2.mixreplay (out/NOTES.md, out/unit_test_mixed.py, out/job_abba).
  results:    Observed: unit test (job 85) vs stock CUDA kernel on sequences mixing prefills/re-prefills/joins/accepts/align
              migrations: 0 mismatches over ~44k checks, 6 seeds; negative control (no mode-3 clear) 6,860 mismatches. Triton vs CUDA
              pre-norm outputs from the same state: max 1.2-2.4e-4 abs (~1 bf16 ulp). Server S (B+C, stats): flush counters
              [1152,1152,0,0] (mode 2 never fires), gsm8k50 88%, 128K pass, KV 267,721, Running 16, graphs 1.78 GiB (vs 1.15; +0.63
              GiB/rank), capture 28 s (vs 9). ABBA: tok/s A 809.9, B 812.1, C 819.1, C 819.6, B 810.8, A 804.1; TPOT p50 A 17.33/17.78,
              B 17.39/17.66, C 16.98/17.17; acceptance unchanged. Estimates: B +1.8-2% vs measured +0.6% (shortfall unexplained, nsys
              inflation suspected); C +0.5-1% vs measured +1.0% over B.
  conclusion: Interpretation: C a consistent gain (both C above all others); B small, likely positive, harmless.
validity:     leader-2 review: B swaps the Triton spec-decode GDN kernel in mixed steps for the CUDA replay kernel (same fp32
              recurrence; T1-012 bitwise vs stock CUDA, here ~1 ulp vs Triton = different kernel, allowed); norm path unchanged;
              last_recurrent_state for spec is unused (states updated in place). C is a serve flag (graph sizes only, same math);
              memory +0.63 GiB/rank but KV unchanged and 128K passes. gsm8k50/acceptance unchanged. OK.

Profile (w13, job 89, nsys TRACE=graph, 64 prompts, default grid, same job; ws leader-2.mixreplay/out/prof): per mixed step on GPU 0,
A: Triton spec kernel 3.32 + mode-2 flush 3.16 = 6.48 ms; B: replay kernel 2.18 (46 us/layer vs ~25 in decode graphs) + residual
flush launches 0.45 = 2.63 ms -> saving 3.85 ms/mixed step (~1.5 s/trial, <= +1.4%); est. 5.0-5.3 ms.

official #1 (2026-11-08T05:58): 861.5 tok/s at c=16 (mean of 2 trials: 863.1 / 859.8, CV 0.3%) (+35.5% vs BASE), GSM8K 93.2%, MTP acceptance 2.33, TTFT p50 180 ms, TPOT p50 16.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
