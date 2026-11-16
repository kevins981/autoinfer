team:         T3
builds_on:    T3-001
performance:  "within noise, not a candidate: 823.7 tok/s (starts 811.0 / 836.3), +0.9% vs the champion T3-001 (816.6); start 1 below 816.6 (eval/1, GPU job 54)"
summary:
  hypothesis: batch-sharded sampling exchanges full-vocab target logits each step (vocab-shard all-to-all, ~6 MB/rank, ncclSendRecv
              ~454 us per decode step in agents/leader-3/prof1); when every request is greedy, a vocab-parallel local first-max + an
              all-gather of [rows,2] pairs + direct greedy verification gives the same tokens. Napkin -0.45 ms/decode step (~+1-1.5%).
  change:     by worker leader-3.w1, in workspace t3-greedy (= T3-001 + this). NEW vllm/v1/worker/gpu/sample/greedy_fast.py
              (GreedyFastPath.try_sample: qualification checks, Triton _greedy_verify_kernel); LogitsProcessor.get_first_max_tokens;
              compute_greedy_tokens on the Qwen3_5 model classes; hook in model_runner.sample(). Falls back unless: all temperature 0,
              plain Sampler, default rejection method, no logprobs/grammar/logits processing/masks/watermark/trace/adaptive, drafts
              present. Env VLLM_T3_GREEDY_FAST=0 disables. serve_args = T3-001's.
  results:    Observed: harness (job 54): 823.7 (811.0, 836.3), acceptance 2.333, TPOT p50 18.02/17.72 ms (T3-001: 18.06/18.01),
              TTFT p99 1019/1052, TPOT p99 40.8/37.7, gate_latency pass, 128K pass; fast path logged active in both starts.
              Quick bench (job 53, 1 start each): on 826.7 vs off 809.5. Unit test (greedy_unit.py): 0 mismatches vs rejection_sample
              on finite bf16 logits incl. 104 planted ties; c=1 4x200 tokens identical on/off.
  conclusion: Interpretation: a small gain (~+1%, TPOT p50 -0.2/-0.3 ms) that this measurement does not resolve (start 1 below the
              champion's mean; CV large). Not pursued further with this team's budget.
validity:     Exact for finite logits (tie-breaking lowest id; same head arithmetic). Review reviews/T3-3 found two gaps to fix before
              any promotion: (1) NaN handling differs from the old verifier (old: NaN-containing 8192-token block max -> -inf; new:
              per-element NaN -> -inf), so a NaN row can pick a different token; (2) lm_head LoRA not excluded (crash instead of
              fallback when batch sharding is off). Worker found: no guard for num_new_sampled_tokens_per_step != 1 (not Qwen).
              Not affecting the benchmark config, but they block calling it unconditionally exact.
