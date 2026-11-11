team: T9
builds_on: T1-042 (all its changes) + M2/S3c (ws diff t9-tbo)
performance: OFFICIAL #1 948.9 tok/s (947.9/949.8; +0.7% vs T1-042 942.2), GSM8K 92.8%, acc 2.327, 128K pass. Earlier unofficial: pp1 (S3c, prefill-graph cap 1536) 875.7 tok/s vs pp0 champion 865.2 at c=16, 100 prompts, ABBA x3: +1.2%, near noise
summary:
  hypothesis: per-chunk next-layer add+norm+projection inside the 2-chunk chain, chunk 1 on DMA lane 1, hides more AR time in mixed steps.
  change: VLLM_T9_PINGPONG=1 (S3c), see change.diff (= ws diff t9-tbo). NL=2 DMA lanes, persistent per-parity byte region for next-layer qkvz|ba / qkv, pp1 arm served with max_prefill_cudagraph_capture_tokens 1536 (grid truncated) because 2048 OOMs at capture (pp1 needs ~130 MB more than pp0; free after start 360 vs 490 MiB per GPU).
  results: Observed (start_server_bench 100 prompts c=16, order A B B A B A; A = pp0 champion config, B = pp1+cap1536; tok/s / p99 TTFT ms / p99 TPOT ms / MTP accept):
    1A 867.55 / 1224.9 / 38.44 / 2.36
    2B 875.04 / 1152.2 / 39.14 / 2.37
    3B 871.18 / 1169.3 / 39.26 / 2.34
    4A 864.24 / 1344.9 / 44.68 / 2.37
    5B 880.97 / 1212.3 / 41.11 / 2.35
    6A 863.89 / 1223.3 / 41.31 / 2.35
    means A 865.2 (spread 3.7), B 875.7 (spread 9.8): +1.2%. p99 TTFT A 1264 vs B 1178 (noisy). median TTFT A 215 vs B 209.
    gsm8k 50 on pp1 start: 0.88 (44/50; BASE 88.0%). long_request 128000 on pp1 server: pass. KV tokens 267,721 both arms.
    Correctness: eager PP_CHECK n=2048: residual/qkvz/qkv bitwise, ba ULP-level (30-50% elements, maxabs <=0.031). c=1 greedy: pp1 round0==round1; 3/6 prompts identical to pp0, others diverge later (ba ULP).
  conclusion: Interpretation: S3c is correct and gives a small gain, +1.2%, versus microbench expectation; per-start ranges (A 863.9-867.6, B 871.2-881.0) do not overlap, but 3 starts each, unofficial. Steps >1536 tokens not counted (no per-step token log exists in server logs).
validity: see leader-9 review at the end of this file

validity (leader-9 review, 2026-11-09): PASS for official eval.
  - Diff reviewed (m2_s3c.diff + fixes: persistent per-parity buffers outside the graph pool, NL=2 lanes, pre reset per layer). change.diff = ws diff t9-tbo (T1-042 + S3c); test-only switches default off.
  - Math: scheduling only. Residual stream and qkvz/qkv bitwise vs the full-row path in PP_CHECK (eager, n=2048, ~30 layers); ba differs at ULP level because its GEMM runs on chunk rows — the same fp-reordering class as the champion's own token-split GEMMs (task_definition: reordering allowed). AR sums unchanged (DMA lane = hostar bitwise, KB35). GDN/attention cores still on full rows. gsm8k 50 = 0.88 (BASE 88.0), MTP acceptance unchanged (2.34-2.37).
  - Memory: +~130 MB/rank (2nd DMA lane tmp + 35 MB/parity buffers + shm); prefill-graph capture cap 2048 -> 1536 (serve_args.txt; T5-004 measured that cap neutral, KB63) because 2048 OOMs at capture. KV unchanged 267,721 tokens; 128K request passes on this config (job 99).
  - Nothing benchmark-specific; switch is generic (VLLM_T9_PINGPONG=0 restores T1-042 behaviour, serve_args aside).

official #1 (2026-11-09T05:23): 948.9 tok/s at c=16 (mean of 2 trials: 947.9 / 949.8, CV 0.1%) (+49.2% vs BASE), GSM8K 92.8%, MTP acceptance 2.327, TTFT p50 177 ms, TPOT p50 15.3 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)

## ABBA #2 (job 104, 12.5 GPU-min)
400 prompts, c=16, fresh server per start, order A B B A (3rd pair skipped: time guard). A = T1-042 serve_args, VLLM_T9_PINGPONG=0; B = bank/T9-005/serve_args.txt (cap 1536 grid), PINGPONG=1; workspace t9-tbo, default VLLM_T9_PP_ROWS.
Observed (tok/s / p99 TTFT ms / p99 TPOT ms / MTP accept):
  1A 940.2 / 983 / 40.2 / 2.332
  2B 945.1 / 951 / 33.6 / 2.322
  3B 950.5 / 873 / 33.7 / 2.328
  4A 942.9 / 1042 / 32.2 / 2.323
Means: A 941.5, B 947.8 (+0.7%). A range 940.2-942.9, B 945.1-950.5 (no overlap, 2 starts each).
Interpretation: consistent in sign with ABBA #1 (+1.2%) and official (948.9 vs 942.2, +0.7%); small gain, ~noise-scale per start.

## How to reproduce / switches
- Workspace t9-tbo (T1-042 + S3c); change.diff = `ws diff t9-tbo`. Arm A (champion): bank/T1-042/serve_args.txt with VLLM_T9_PINGPONG=0. Arm B: serve_args.txt in this folder (= T1-042 with the prefill CUDA-graph grid capped at 1536: max_prefill_cudagraph_capture_tokens 1536, sizes <= 1536) with VLLM_T9_PINGPONG=1 (default).
- VLLM_T9_PINGPONG=1|0: S3c (next-layer add+norm+projection per chunk inside the split chain, chunk 1 ARs on DMA lane 1) vs the champion chain. Default 1.
- VLLM_T9_PP_CHECK=1: test-only, eager steps only (use --enforce-eager or ungraphed steps); logs "PPCHECK" lines comparing chunked vs full-row next-layer projection and residual (residual/qkvz/qkv bitwise, ba ULP-level). Leave 0 for serving.
- VLLM_T9_PP_ROWS (default 2048): rows of the persistent ping-pong buffer regions (2 sets, one byte region per layer parity).
- Memory: pp1 needs ~130 MB more than pp0 at the same grid; with the 2048 grid the prefill-graph capture OOMs (hence the 1536 cap). KV size unchanged (267,721 tokens). Free GPU memory after start: pp1 360 MiB vs pp0 490 MiB.
- JIT extensions (built at first use under TORCH_EXTENSIONS_DIR; pre-build in a plain shell before GPU jobs): host_staged_ar (hostar), dma_staged_ar (copy-engine AR with NL=2 lanes; source dma_staged_ar.cu). Microbench-only exts: t9ce / t9ce7 (T9-001..004, T9-007 ce.cu), t9pfk (T9-006 pfk.cu).
- Test scripts: agents/leader-9.w1/smoke2_job.sh (greedy c=1 compare), abba_job.sh, abba2_job.sh (A B B A, 400 prompts).
