team:         T3
builds_on:    T3-003
performance:  +3.3% vs the champion (T3-003, 826.1): 853.6 tok/s, starts 854.1 / 853.2 (eval/1, GPU job 75); both above 826.1.
summary:
  hypothesis: with FlashInfer on sm89 the MTP drafter cannot update its decode attention metadata in place (speculator.py:427-441), so
              every draft decode step rebuilds it eagerly (CPU work, H2D copies, a blocking seq_lens.cpu(), flashinfer.py:1689) instead of
              a FULL draft graph replay; TRITON_ATTN supports the in-place update (triton_attn.py:99/212). Profile: ~2 ms GPU idle per
              decode step (period minus kernel union, agents/leader-3/prof1).
  change:     T3-003 + --attention-backend TRITON_ATTN (serve_args.txt) + one code change (by leader-3): vllm/v1/attention/backends/
              triton_attn.py: supports_quant_query_input defaults to False on CUDA (opt-in VLLM_TRITON_ATTN_FP8_QUERY=1, unset here), so the
              query stays bf16, K/V are dequantized to bf16 in-kernel and P stays bf16 (same precision class as FlashInfer's bf16-Q/fp8-KV
              path). change.diff = ws diff t3-tri (checked with cmp) = the measured workspace.
  results:    Observed: eval/1 (job 75): 853.6 (854.1, 853.2), acceptance 2.337, TPOT p50 17.28/17.19 ms (T3-003: 18.05/17.78), TTFT p99
              893/1002 ms, TPOT p99 33.6/32.4 ms, gate_latency pass, 128K pass, GSM8K-250 93.6% (gate pass), preemptions 0/0. Server log:
              "Using AttentionBackendEnum.TRITON_ATTN backend" on both ranks; the "cannot update draft decode metadata" line is gone.
              Earlier flag-only screen (T3-006, FP8 query, invalid): 855.9 vs 822.2 quick (job 71) -> the bf16 query costs ~nothing.
  conclusion: Interpretation: removing the drafter's eager metadata rebuild (and the step's CPU-bound gap) is the likely mechanism, consistent
              with TPOT p50 -0.6..0.8 ms; not isolated by a profile (backend swap also changes target-side attention kernels and planning).
validity:     Review reviews/T3-5 turn 2: Q stays bf16 (query_quant not built), _cast_kv_tile dequantizes FP8 per-tensor K/V to bf16
              (triton_unified_attention.py:155), USE_FP8_Q_DESCALE false, P x V with bf16 operands and fp32 accumulation, split-K partials
              merged in fp32, output bf16; same for prefill, split-K decode and the MTP layer; reshape_and_cache unchanged (K/V only).
              Floating-point reordering vs FlashInfer, no precision change; no persistent higher-precision KV copy. GSM8K 93.6% (ran, gate
              applies). Turn 1 of the same review found the flag-only variant invalid (FP8 Q/P) -> T3-006, superseded by this record.
