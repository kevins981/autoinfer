team:         T2
builds_on:    T2-003
performance:  "+5.2% vs the champion (T2-003, 707.0): 743.9 tok/s (starts 741.1, 746.8), eval/1 (GPU job 18)"
summary:
  hypothesis: BASE runs every quantized body linear (GDN in_proj_qkvz/out_proj, attention qkv/o_proj, MLP gate_up/down,
              and the MTP layer's) through Marlin weight-only FP8 with bf16 compute. At the decode batch (M=48)
              this is memory-bound and Marlin is ~equal to a good plain-layout kernel, but at the mixed/prefill
              M of 128-2048 it is compute-bound at bf16 speed (~150-160 TFLOP/s) while the checkpoint's own
              W8A8 scheme (dynamic per-token-group-128 e4m3 activations x 128x128 block-scaled FP8 weights)
              can use the FP8 tensor cores (vLLM's Triton block-FP8 kernel measured 180-230 TFLOP/s).
              vLLM ships no tuned Triton config for these N,K on the 4090 (default config = 64x128x128, 2 stages,
              50-120 us floor at small M), so we tuned it.
              Napkin from microbenchmarks (cold weights, CUDA-graph replay, GEMMs only): 64-layer GEMM time
              -4% at M=48, -30% at M=256, -36% at M>=1024 (Marlin -> tuned Triton incl. activation quant).
  change:     (1) serve flag: --kernel-config '{"linear_backend_per_quant": {"fp8_block_w8a8": "triton"}}'
              (vLLM's documented per-quant backend override; log: "Applied linear backend override for
              'fp8_block_w8a8': 'triton'", "Selected TritonFp8BlockScaledMMKernel for Fp8LinearMethod").
              (2) 5 new tuned-config JSON files in vllm/model_executor/layers/quantization/utils/configs/
              (N,K = 8192x5120, 7168x5120, 17408x5120, 5120x8704, 5120x3072; device_name=NVIDIA_GeForce_RTX_4090,
              dtype=fp8_w8a8, block_shape=[128,128]) read by the existing get_w8a8_block_fp8_configs lookup:
              per M in {16,48,64,128,256,512,1024,2048} the best (BLOCK_M,N,K, GROUP_M, warps, stages) found
              by a search on cold-L2 weights (180 configs; small-M and large-M families). All winners use
              BLOCK_K=128; 4 warps except gate_up at M=48 (2 warps). No Python/CUDA code change: change.diff is data files only.
              serve_args.txt = T2-003's three flags + the --kernel-config line (job 18: eval/1).
  results:    Observed (harness, 2 fresh starts x 400 prompts, GATES=long,gsm8k, GPU job 18):
              741.1 / 746.8 tok/s, mean 743.9 (+5.2% vs T2-003 707.0 whose starts were 708.9 / 705.1; both
              starts above; +17.0% vs BASE 635.9). CV 0.53%.
              TTFT p50/p99 183/918 ms (T2-003 196/1185); TPOT p50/p99 19.8/35.9 ms (T2-003 20.8/38.6; BASE
              23.0/46.1); gate_latency: both p99 below BASE. MTP acceptance 2.348 (T2-003 2.349, BASE 2.345).
              128K request: pass (48.8 s, server answers afterwards). GSM8K-250: 92.8% (7 outputs lacked an 'Answer:' marker, all had a numeric fallback; BASE
              94.4%, gate 89.6%; T2-001 93.2%).
              Server log (both starts): TritonFp8BlockScaledMMKernel selected, and "Using configuration from
              .../configs/N=...,K=5120,device_name=NVIDIA_GeForce_RTX_4090,...json" for all 5 (N,K)
              (no "Using default W8A8 Block FP8 kernel config" warning).
              Microbenchmarks (task 1/2, GPU jobs 3, 6, 12; agents/leader-2.w1/mb/): see results below.
              Cold-L2, per call at M=48: Marlin vs tuned Triton GEMM-only: gdn_in 55 vs 50 us, qkv 60 vs 44,
              gate_up 118 vs 107, down 61 vs 52, o_proj 36 vs 20. In-graph activation quant ~3 us/GEMM.
              Per layer in graph (us): M=48 GDN 249 -> 240 (-4%); M=256 698 -> 490 (-30%); M=1024 2502 -> 1613 (-36%).
  conclusion: Interpretation: the gain (+5.2%) is larger than the decode-only napkin (-0.6 ms of the 64-layer GEMM time
              at M=48), consistent with the mixed prefill+decode steps (M up to 2048 now captured as graphs by
              T2-001) are GEMM-bound and the FP8 tensor cores give 30-36% lower GEMM time (microbenchmarks; no end-to-end profile establishes the attribution), which also lowers TTFT
              and tail TPOT. Only the dense body linears changed; no profile of the end-to-end step yet.
              The tuning is per (N,K,M-bucket) from random weights; M between grid points use the nearest.
validity:     (leader to review) Checkpoint's own W8A8 scheme per TASK.md: dynamic per-token-group (128)
              e4m3 activations against the checkpoint's 128x128 block-scaled FP8 weights; no lower precision,
              weights unchanged (no re-quantization, no extra copy: Marlin repack is skipped, plain layout).
              Kernel math (read from fp8_utils._w8a8_triton_block_scaled_mm): tl.dot of fp8 A and B tiles with
              fp32 accumulation; per K-block (128) the partial is multiplied by As[m, kblk] * Bs[nblk, kblk]
              and added to an fp32 accumulator; bf16 store at the end. Activation quant: the server's existing
              QuantFP8 per-token-group-128 custom op (unchanged). Not touched: lm_head, attention/GDN
              kernels, norms, comms, KV cache, sampling. Accuracy: GSM8K-250 92.8% >= 89.6 gate. Microbench
              max-abs-diff vs fp32 reference of dequantized weights x bf16 activations: tuned/default Triton
              0.05-0.11 (output magnitude ~3) vs Marlin 0.006-0.014: activation quantization error, expected
              for W8A8; outputs of tuned vs default Triton configs identical (diff 0) on the 40 tested shape/M cases.
              Memory: model loading 14.22 GiB/GPU (T2-003: 14.40; no Marlin repack padding), KV cache size unchanged (267,721 tokens), 128K gate passes.
review:       reviews/T2-3: W8A8 path verified against the rule (dynamic (1,128) fp32 activation scales, no
              UE8M0/pow2; per-K-block scaling with fp32 accumulation; checkpoint 128x128 weight scales, no
              requant/persistent copy); override covers quantized body linears incl. the MTP decoder layer;
              lm_head, attention/GDN cores, norms, KV, comms, sampling unchanged; numbers recomputed (741.140 /
              746.754, +5.222%, both starts above; latency, 128K, GSM8K 232/250 pass); tuning provenance random
              data only; nearest-M config lookup benign. Wording corrections applied.
              worker (leader-2.w1) answer: agrees with all review points as applied; leader-2 judgment: adopted.
