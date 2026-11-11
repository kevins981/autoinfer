team:         T50
builds_on:    T45-005 + T42-003 (leader-42/leader-42.w1's certified target lm_head, delta applied unchanged) + T50-001 (gate_up cfg)
performance:  "official #1: 1181.3 tok/s (1179.1/1183.5; +0.67% vs T45-005 1173.4 cross-job, both trials above both champion trials); unofficial: ABBA vs T45-005: job 431 (300 prompts, ABBA) mixed +1.04%; job 432 (400 prompts, BAAB) all B > all A +1.25%"
summary:
  hypothesis: two exact, measured-but-unmerged small gains stack on the champion: T42's certified target lm_head
              (-0.2 ms/decode step, KB121; +0.53% mixed vs T37-002) and cfg 103 for the gate_up W8A8 GEMM at M=192..1024+
              (T50-001: 3-7% faster at M=192-320, 1-2% at M>=768). Est. +0.6-1.0%.
  change:     workspace leader-50.stack = T45-005 + bank/T42-003/delta_vs_T37-002.diff (git apply -p2, clean; copy in
              t42_delta.diff) + t29_bs_gemm/configs.json "17408x5120": buckets 192/256/320/768/1024 -> [103,1] (cfg_delta.diff).
              serve_args = T45-005's. VLLM_LMH_CERT=0 disables the lm_head part.
  results:    Observed (outputs agents/leader-50/out/abba1, abba2):
    - job 431, 300 prompts, A=T45-005 (ws leader-50.base) B=stack: A 1153.8/1151.1, B 1175.4/1153.6 -> mixed +1.04%.
      acceptance A 2.684/2.684, B 2.699/2.666; p99 TPOT A 30.3/30.2, B 25.6/26.7 ms; p99 TTFT A 836/834, B 828/818.
    - job 432, 400 prompts, BAAB: B 1183.6/1180.2, A 1166.3/1168.2 -> all B > all A, +1.25%. acceptance B 2.680/2.664,
      A 2.670/2.658; p99 TPOT B 24.0/27.9, A 31.0/26.7 ms.
    - Server logs, every B start, both ranks: "target lm_head shard certified ... self-test 201 rows 0 mismatches"; plane
      0.566 GB replaces the 1.271 GB shard (allocated 18.38 -> 17.74 GB); 10 full-logit host calls at startup only.
    Pooled over 8 starts: B 1173.2 vs A 1159.8 mean (+1.15%); 7 of 8 cross-pairs favour B.
  conclusion: Interpretation: +~1.1% from the stack, in line with the estimate (T42 ~0.6-0.8% + gate_up cfg mixed steps
              ~0.3-0.4%); the parts were not separated (GPU saved). Small but consistent at the official prompt count.
validity:     Reviewed by leader-50 against TASK.definition:
    - lm_head: the task_definition ruling (manager 2026-11-11) allows certified screening if the greedy decision is bitwise
      the baseline's: candidates are rescored with lmh_codec's exact chain (the champion's own lm_head arithmetic), the
      cross-rank combine and tie-break are T37-002's unchanged code, rows failing the certificate fall back to the exact
      full shard, and the startup self-test must pass on all ranks or the server reverts to lmh_codec. Non-greedy /
      logprob paths compute exact full logits. HBM goes DOWN 0.64 GB/GPU. ITL/TPOT improved, so the host-backed rescoring
      adds no decode latency. FLAG for the manager: T42 states one assumption in its proof (tensor-core error bound for
      the plane chain, 2^-11 relative assumed, 2^-19.4 measured, PROOF.md in bank/T42-002/code). The ruling asks for a
      bound, not an empirical margin; please confirm the ruling covers it (it was written for T42's request).
    - gate_up cfg: same kernel, same W8A8 scheme and M range as the champion (M>96 already W8A8); only the tile shape
      changes, so per-output K accumulation (128-K blocks, fp32 promotion) is the same arithmetic up to reordering.
    - Nothing keyed to the benchmark; serve args unchanged. GSM8K/128K: covered by the official eval (T42 measured 128K pass,
      c=1 greedy outputs identical to T37-002).

official #1 (2026-11-11T14:36): 1181.3 tok/s at c=16 (mean of 2 trials: 1179.1 / 1183.5, CV 0.3%) (+85.8% vs BASE), GSM8K 93.6%, MTP acceptance 2.663, TTFT p50 142 ms, TPOT p50 12.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
