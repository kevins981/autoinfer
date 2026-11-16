# T2 map, update 2 (closing resync)

Champion moves since update 1: T2-003 707.0 → T2-004 743.9 (T2) → T1-007 771.3 (T1) → **T2-007 787.6 (T2)**.

## Tested since update 1
| candidate | result | evidence |
|---|---|---|
| #3 argmax before gather (draft: local argmax; target: batch-sharded sampling) | **win, champion** 707.0 (+2.5%), exact | bank/T2-003, job 14; review T2-2 |
| #2/R4' W8A8 block-FP8 with tuned Triton for all M (decode and mixed) | **win, champion** 743.9 (+5.2%), GSM8K 92.8% | bank/T2-004; review T2-3 |
| N3 fused activation quant (RMSNorm + SiLU producers emit dyn128 FP8) + configs for every graph M | 752.6 alone (+1.2% vs T2-004); superseded by T1-007 before its 3rd start | bank/T2-006, job 32; review T2-5 |
| N3 stacked on T1-007 (host-mapped AR) | **win, champion** 787.6 (+2.1% vs T1-007) | bank/T2-007, job 38; review T2-6 |
| torch.compile / Inductor fusion route (instead of hand fusion) | **loss**: -9.4% (compile mode forces graph-break overhead with Qwen3.5 breakable graphs) | bank/T2-005; review T2-4 |
| R2'/R2a lm_head kernel choice | **dead on napkin**: lm_head is 1.27 GB/GPU, not 0.635 (my 3-calibration error); target head already at ~0.8 TB/s | job 22 |
| N1 capture GDN chunked-prefill core | **priced, deprioritized**: TRACE=graph profile of T2-003 shows only ~7.65 ms GPU idle per mixed step → ≤ ~2%, high build cost | agents/leader-2/prof_t2003_graph |
| N2 fp8 draft head within memory | T1's lane; T1-001 kept it at prefill cap 1024 | T1 records |

## Verdicts changed
- **3-calibration lm_head bytes** were half the true size (vocab 248320/2 × 5120 × 2 B = 1.27 GB). The decode lm_heads
  are already near bandwidth; only precision (fp8 head, T1) or vocabulary changes could move them.
- **Mixed-step "eager overhead"** (my W1 dimension) was real as a win (+8.4%) but partly misattributed: the GDN core
  stays eager in breakable graphs, and after W1 the mixed-step GPU idle is small (7.65 ms). The CPU-launch dimension
  is mostly exhausted by W1.
- **W8A8 at decode**: one tuned Triton kernel beats Marlin at every M on this card. Separate activation-quant kernels
  were then a visible cost (~3 µs each, 4 per layer), which is what N3 removed.
- **Framework note**: the cost model needs (1) a CPU-launch term (W1) and (2) a per-kernel fixed-cost term. At decode,
  many of the remaining costs are launch/latency floors, not bytes or FLOPs: quant kernels, small GEMMs like GDN
  in_proj_ba, the AR floor.

## Open / handed over
- **Task 9 → bank/T2-008 (null)**: the fused quant went into the gated-norm callback, but with MTP the live path is
  the CUDA spec-decode branch (qwen_gdn_attention_core_fused_norm_packed), which bypasses it (review T2-7). The
  candidate is still open: fuse into that op's epilogue (~+0.5%, plus o_proj ~+0.15%; 3+ starts).
- To T3 (agreed with leader-3): GDN in_proj_ba split-K kernel, L2 prefetch during the AR, output-column split to overlap
  the AR, GDN decode state writes, and N1 if wanted.

## Current ranking (for whoever continues)
1. Overlap the AR with GEMM work in decode (output-column split, T3) — up to ~+5% if hmar's ~6 ms/step can be hidden.
2. Small-kernel fixed costs at decode: GDN in_proj_ba (~1.2 ms/step, T3), remaining quant launches (Task 9).
3. k=3 drafts once the draft head is cheaper (T1's verdict in its write-up).
4. N1 GDN prefill core capture (≤2%).
