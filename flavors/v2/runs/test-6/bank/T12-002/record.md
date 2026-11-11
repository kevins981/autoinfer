team:         T12
builds_on:    T1-042
performance:  "unofficial: -7.3% (single smoke, job 106); G1 gate failed. Detail: A 801.7 tok/s acc 2.430 vs B' (freq-ranked 32K + FP8 fc) 743.1 acc 2.229; C (bf16 fc) 751.8 acc 2.258; draft step 0.835 -> 0.574 ms (step gate met, throughput gate failed)."
summary:
  hypothesis: cheaper draft step (32K draft vocab, no fc all-gather via replicated FP8 fc) pays at k=2 despite small acceptance loss.
  change:     speculative-config options draft_lm_head_token_file (npy of kept ids; generic remap through LocalArgmaxMixin draft_id_to_target_id),
              draft_fc_fp8 (replicated per-output-channel FP8 fc, bf16 shard freed), env-gated step timers (ai_timers, AUTOINFER_TIMERS). Worker leader-12.w1, ws leader-12.g1. serve_args.txt.
              Subset: build_draft_vocab.py (log build_draft_vocab_32768.log): ranking = token counts in T3 TRAIN split (self-generated prompts+target outputs, 169K tokens) ->
              22,551 ids; then vllm repo docs/*.md + vllm/*.py (14.5M tokens) -> 10,217 ids; no ShareGPT/benchmark data. Held-out (T3 40 heldout responses, 14.2K tokens) coverage 91.64% vs 87.04% for first-32768 ids.
  results:    Observed (jobs 105, 106; timers on rank-0, pure decode steps, 16 req):
              draft step: A 0.835 ms (propose 1.670/2), B' 0.574, C 0.583; first-32K+fc(job 105) 0.573.
              acceptance length: A 2.430, B' 2.229, C 2.258, first-32K+fc 2.248. tok/s: A 801.7, B' 743.1, C 751.8. TPOT p50 A 16.07, B' 16.60, C 15.75 ms.
              A2 diagnosis: draft step has 4 host-staged collectives (embedding AR, fc all-gather (gather_output=True), attn AR, MLP AR+norm), none NCCL; T5 trace: 171 us (draft1) / 112 us (draft2); attn+MLP AR (40-55 us each) match the host-AR model, embedding AR carries skew.
              Replicated FP8 fc removes the fc all-gather. Unit effect of fc FP8 on acceptance: B' vs C -1.3%.
  conclusion: Interpretation: 32K draft vocab costs ~7-8% acceptance even frequency-ranked (91.6% held-out coverage is not enough; the benchmark target text is more diverse than T3's replies), outweighing the ~0.26 ms/step (x2) saving => net -7%. Gate: draft step <= 0.65 met; ABBA not run (B' below A in the smoke).
validity:     drafter only; target verification untouched; vocab list from non-benchmark sources (ruling 2).
