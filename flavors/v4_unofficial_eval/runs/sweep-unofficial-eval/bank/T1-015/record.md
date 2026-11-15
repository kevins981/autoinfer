team:         T1
builds_on:    T1-012
performance:  "unofficial: PASS +1.29% vs T1-012 (GSM8K 93.2%, 128K pass)"; quick ABBA job 113: +2.2% (B 749.8/747.7 vs A 737.7/727.8, 100 prompts)
summary:
  hypothesis: T1-010 planned decode attention from host upper-bound lengths and had to disable split-KV. The kernel
              derives chunk counts from the exact device lengths, while the merge used the plan's upper-bound counts,
              so the two mismatched. Without split-KV, decode attention costs 76.6 us/call vs 16.4 us
              (leader-2/t2-w1 job 101, 18 calls per step), i.e. +~1 ms per decode step (~3%). Fix the layout, keep the
              split, keep no sync.
  change:     vllm/v1/attention/backends/flashinfer.py (on T1-012): with cuda-graph decode wrappers, plan from
              the upper bound with split-KV on. Three small Triton kernels then run (eager, in the metadata builder,
              stream-ordered after the plan's H2D). From the exact device seq_lens they rewrite o_indptr and
              merge_indptr with the exact per-request chunk counts, and clear block_valid_mask for CTAs whose chunk
              lies beyond the exact length (those CTAs exit early). Non-graph wrappers keep the no-split path.
              VLLM_FI_EXACT_SPLIT=0 restores T1-012's behaviour. Worker t1-w1, workspace t1-w9.
  results:    Observed: offline (job 112, agents/t1-w1/split/test.py, 120 cases, q_len 1 and 3, 232 requests with
              UB chunk count != exact chunk count): bitwise equal to the exact-length split plan; max abs diff 0.031
              vs the exact no-split plan (fp reordering). Garbage beyond the length (fp8 448, NaN) leaves the
              output unchanged. Quick ABBA (job 113, same job): A 737.7/727.8, B 749.8/747.7 tok/s (+2.2%);
              median TPOT 19.7 -> 19.5 ms.
              unofficial #1: B 766.0 / 766.6 vs HEAD mean 756.5 (+1.29%); p99 TTFT 1138 / TPOT 39.5 ms; acceptance 2.327.
  conclusion: Interpretation: restoring split-KV recovers most of the decode attention cost that T1-010 traded for removing
              the host sync. The gain is smaller than the quick ABBA (+2.2%, 100 prompts), but both starts are above HEAD.
validity:     Exact: identical to the exact-length split-KV plan (the planner's chunk size from upper-bound lengths
              only changes the chunk partition, i.e. fp reordering of the attention merge). No precision change,
              no benchmark fitting, and an off switch. Reviewed by leader-1: the tmp buffers are sized by the
              UB plan, and the exact counts are <= UB, so writes stay in bounds; padded CTAs keep their old mask.

unofficial #1 (2026-11-15T10:02): PASS: +1.29% vs HEAD T1-012 (B 766.0 / 766.6 vs reference mean 756.5 over 2 starts, 0 run in this job; 200 prompts); GSM8K 93.2%, 128K pass, p99 TTFT 1138 ms / TPOT 39.5 ms, MTP acceptance 2.327 (unofficial/1/unofficial.json)
