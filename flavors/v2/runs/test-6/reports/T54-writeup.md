# T54 write-up (leader-54, research-handoff, memo research_handoff_1113.md). Recommendation: CLOSE, 0 GPU-min used.
Triage: reports/T54-triage.md (one row per idea, plus a measured-cost addendum for trees). Sparring: T54-1. No records, no proposal.
Split with T55: T55 took ideas 1/5 (GDN state), T54 took 2/6/7. Both triages agree.

## The memo, idea by idea
- **1 DeltaLog Keep.** The memo predicted -43% GDN HBM traffic.
  - Already the champion's mechanism: replay (T1-012), lazy checkpoint every 8 tokens (T1-035), two-slot rollback-free state (T37).
  - The assumption that broke: the 16-bit state. The checkpoint's mamba_ssm_dtype is float32.
  - What's left is the read floor (25 MB/layer/GPU) and ~13 us/layer of latency. That headroom is ~1.5% e2e; T12/T22 costed it at <=1%.
- **2 Dual-form tree speculation.** The memo predicted ~free verification of 32 tree tokens.
  - The assumption that broke: the ridge. It is ~80 rows on this exact path, not 330, and c=16 already uses 64 rows.
  - Rows above 64 cost ~0.14 ms each (GEMM + AR, measured), and below N=7 the best static trees are chains (T38 data).
  - Every tree size nets -6..-8% except N=3 (+1.6% tokens at the same rows). The ancestor-masked WY algebra holds (T38 noted it) but has no payoff here.
- **3/4 PCIe KV offload / Quest prefetch.** Wrong workload. The KV is 32 KB/token (the memo said 64: the config has 4 KV heads, not 8), all requests fit in the 5 GiB pool, and idea 4 is lossy.
- **5 Shared-prefix lazy state.** Wrong workload: no fan-out. Prefix-state caching paid +0.5% (KB86).
- **6 Min chunk / PP switch.** Prompts aren't chunked. PP was measured at 0.91-0.92x TP decode against a 0.93x gate (T38). End-to-end PP was never measured; it reopens only with a GEMM that holds its floor at M=128-160.
- **7A FR-Spec drafter vocab.** Done in better forms: 98K subset +1.0%, 32K -7.3%, exact PCA screen (T27). The draft head is now ~0.27 ms/step.
- **7B Gate head skipping.** Approximate, so not allowed.

## Assessment of the memo
Its reasoning is sound for its assumed regime (B=1-4, long context, 16-bit state, a 330-row ridge). This task differs on all four. Every
idea that fits this task's regime had already been found and built by earlier teams.
