team:         T1
builds_on:    T2-006
performance:  unofficial: 758.8 / 762.4 vs T2-006 738.8 / 747.8 (+2.3% mean)
summary:
  hypothesis: In decode steps the worker thread blocks ~30 ms in a pageable D2H `seq_lens.cpu()` inside the
              drafter's FlashInfer metadata build (flashinfer.py:1699), then launches ~1.8 ms of eager
              next-step prep while the GPU idles (T2-006 TRACE=graph profile: ~1.9 ms idle per 34.1 ms
              decode step, agents/leader-1/prof_t2006_graph). Napkin: up to ~4% overall; likely less,
              since the target build's own seq_lens.cpu() may become the next blocking point.
  change:     (leader-1.w1) Drafter-only: the speculator wraps its `build_attn_metadata` in
              `draft_upper_bound_planning()`. In that context, for pure-decode cuda-graph FlashInfer
              wrappers (no DCP, no prefills, an upper bound exists), the host plans with
              `seq_lens_cpu_upper_bound` (no sync), then the exact paged_kv_indptr / last_page_len /
              indices are written on the GPU from the device seq_lens (same formulas, stream-ordered
              after plan()'s H2D copies). Everything else keeps the exact path. VLLM_DRAFT_UB_PLAN=0
              disables. Delta over T2-006: delta_over_T2-006.diff.
  results:    Observed:
              - job 58 (standalone): an upper-bound plan (+0..8 tokens) with exact GPU buffers gives
                bitwise-identical decode attention outputs to an exact plan in 32/32 cases (page 32/256,
                lens 100-40000, 16 reqs, bf16 KV).
              - job 59 (in-server self-check): 150 drafter steps / 584 rows, 0 mismatches of indptr,
                last_page_len, indices vs the host-exact values; UB >= exact always.
              - same server, greedy 150-token request: identical text new vs old path.
              - 6-request burst: 2.63 s new vs 2.83 s old (1 sample each, noisy).
              - unofficial #1: 758.8 / 762.4 tok/s, p99 TTFT 936 / TPOT 39.2 ms, acceptance 2.318
                (T2-006: 738.8 / 747.8, acceptance 2.302): both starts above T2-006's best.
              - mechanism, nsys TRACE=graph, same recipe (job t1-prof-t1008-graph, agents/leader-1/prof_t1008_graph
                vs agents/leader-1/prof_t2006_graph), 334 decode steps each: decode period 34.11 -> 32.78 ms
                (-3.9%), GPU idle in step window 1.92 -> 0.61 ms, "other" eager kernels 1.41 -> 0.09 ms
                (next-step prep now overlaps the graphs). Mixed steps: GPU span 120.0 -> 115.7 ms, period
                120 -> 134 ms (38 steps only; period includes scheduling of arrivals, noisy).
  conclusion: Interpretation: the drafter's seq_lens.cpu() was the decode-step host stall; removing it lets
              the host prepare the next step while the GPU runs, cutting ~1.3 ms of idle per decode step,
              which matches the +2.3% end to end (decode ~70% of time x 3.9%).
validity:     (leader-1) Diff reviewed. Only the draft model's attention metadata changes; target verify
              is untouched, so output tokens are exactly the target's greedy tokens whatever the drafter
              proposes (exact spec decode). Draft attention reads the same exact KV ranges (checked
              bitwise on real steps); the plan may differ in split-KV scheduling only, and that was
              bitwise identical in the tested shapes; at worst it's an fp reordering of draft-side
              attention, allowed. No precision, memory, parser or batch-size change; nothing workload
              specific. Debug toggles are off by default.

unofficial #1 (2026-11-15T17:37): 758.8 / 762.4 tok/s (200 prompts); p99 TTFT 936 ms / TPOT 39.2 ms (BASE official 1341 / 46.1), MTP acceptance 2.318 (bank/T1-008/unofficial/1/unofficial.json)
