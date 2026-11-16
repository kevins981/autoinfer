team:         T3
builds_on:    T3-003
performance:  "INVALID (contract): quick bench only, job 71, 1 start each: 855.9 vs 822.2 tok/s (+4.1%); harness cancelled after start-up (job 73)"
summary:
  hypothesis: FlashInfer cannot update the MTP drafter's decode attention metadata in place (speculator.py:427-441), so draft steps
              rebuild it eagerly (incl. a blocking seq_lens.cpu(), flashinfer.py:1689); TRITON_ATTN supports the in-place update.
  change:     flag only: T3-003 + --attention-backend TRITON_ATTN.
  results:    Observed (worker, job 71, agents/leader-3.w1/run13.out): 855.9 vs 822.2 tok/s, TPOT p50 17.04 vs 17.87 ms, acceptance 2.333 vs
              2.331, GSM8K-50 92%; the "cannot update draft decode metadata" log line disappears.
  conclusion: Interpretation: the gain is plausible from removing the eager rebuild, but this configuration changes attention precision.
validity:     NOT VALID. Review reviews/T3-5: with an FP8 KV cache, TRITON_ATTN enables static FP8 query quantization
              (attention.py:493/537; triton_attn.py:556 supports_quant_query_input on CUDA) and computes P x V with FP8 operands
              (triton_unified_attention.py:705); FlashInfer on sm89 keeps Q and P in bf16. Attention math must stay at baseline precision
              (task_definition.md). Superseded by T3-007 (bf16 query).
