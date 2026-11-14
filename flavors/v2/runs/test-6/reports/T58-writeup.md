# T58 write-up (sweep, leader-58; summary by the manager)
Full sweep: reports/T58-sweep.md and reports/T58-sweep/ (6-ranking.md has Addenda 1-2). Records bank/T58-001, T58-002. Sparring T58-1 (twice). GPU ~49 of 100.
- Win: **T58-001 CHAMPION 1229.6** (T57-004 + VLLM_HOSTAR_L2PF_MSG_MAX 1 MiB -> 2.7 MB; mixed steps of 97-255 rows prefetch like decode ARs; ABBAAB +0.72% all B > all A; KB137).
- T58-002: decode t31b tiled over M wins only for out_proj/down at <=144 rows (~+0.2-0.3% e2e), would need its own FP8 A/B: not pursued.
- Rank 3 (wave threshold via captured cores): withdrawn as "<=0.5%"; optimistic scenario +0.9-1.4% if host overhead were zero. Unmeasured: whether two waves beat joined by 8-12% GPU time at 384-767 rows (T30_WAVE_BENCH hook), and whether wave capture fits in ~0.45 GiB free.
- Rank 4 (deeper drafts in the drain): +0.4-0.8%, assumed; KB85's -8.5% plumbing loss is the risk; do a trace-based payoff count first.
- Largest open lever: a drafter fed intermediate target hidden states (Addendum 1; k=3 ceiling 4.0 vs 2.68 tokens; premise test using T43's stored residuals + T3's offline MTP pipeline): research-sized.
- Framework gaps found: no link latency/duplex model, sub-ridge GEMMs not at DRAM bandwidth, ridge ~172 rows here not 333, accumulator semantics, no host/admission resource, capacity depends on k, overlap zero-sum here.
- Closing reason: resource, not exhaustion; what remains are multi-day builds with unresolved feasibility.
