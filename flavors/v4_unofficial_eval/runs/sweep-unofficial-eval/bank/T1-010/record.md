team:         T1
builds_on:    T2-012
performance:  "unofficial: PASS +1.89% vs T2-012 (GSM8K 92.8%, 128K pass)"; quick ABBA job 84 +1.1%
summary:
  hypothesis: With spec decode (k=2) on, FlashInferMetadataBuilder.build() calls seq_lens.cpu() each
              step (no upper-bound fast path when num_speculative_tokens>0). The CPU blocks on the GPU,
              so the next step's launches start late: ~2.1-2.5 ms of GPU idle per decode step. If the
              plan uses the host-side upper bound and the exact lengths/bounds are read from device
              buffers, the sync goes away without changing the math.
  change:     vllm/v1/attention/backends/flashinfer.py: when VLLM_FI_EXACT_DEV_LENS=1 (the default), the
              decode/spec plan is built from seq_lens_cpu_upper_bound (no sync), and the kernel's exact
              per-request length and chunk bounds are overwritten on device from the true seq_lens.
              Split-KV is disabled in this mode (disable_split_kv=True), because the merge step uses the
              planned chunk counts. Setting VLLM_FI_EXACT_DEV_LENS=0 restores the old path.
              Worker t1-w1, workspace t1-h12, on top of T2-012.
  results:    Observed:
              - Offline exactness: outputs are bitwise equal to the exact-length no-split plan. Against the
                split-KV plan the difference is fp reordering only (max abs 3.1e-2, bf16). A test that
                writes garbage beyond each true length passed (the kernel never reads past the exact lengths).
              - Job 76: with VLLM_FI_CHECK_UB=1 the run raised 0 asserts (the upper bound always holds).
                GSM8K-50 = 88%.
              - Job 82 (TRACE=graph): idle per decode step 2.56 -> 0.14 ms. Profiled decode period
                33.93 -> 29.55 ms (job 65). Idle in mixed steps 10.7 -> 5.4 ms; what remains comes from
                prefill target builds, which this change does not touch.
              - Job 84 (unprofiled ABBA, T2-012 vs T1-010): A 721.2/718.3, B 734.1/721.5 tok/s, +1.1% mean.
  conclusion: Interpretation: removing a host sync in the attention metadata gives a small, exact win.
              No-split decode attention costs a little at long context; at this workload's lengths, the
              saved idle outweighs it. The --long gate checks the 128K request.
validity:     Same math: attention over exactly the true lengths (shown bitwise equal to the exact no-split
              plan; split vs no-split differs only by fp reordering). No precision change, no benchmark
              fitting, and the original path stays behind an env flag. Uses the device seq_lens that the
              runner already keeps. Reviewed by leader-1 against task_definition.md: allowed.

unofficial #1 (2026-11-15T08:34): PASS: +1.89% vs HEAD T2-012 (B 753.9 / 751.3 vs reference mean 738.7 over 2 starts, 0 run in this job; 200 prompts); GSM8K 92.8%, 128K pass, p99 TTFT 1146 ms / TPOT 36.4 ms, MTP acceptance 2.340 (unofficial/1/unofficial.json)
