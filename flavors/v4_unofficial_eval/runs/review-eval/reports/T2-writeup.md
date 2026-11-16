# T2 write-up (leader-2, sweep team; closed at the 2-grant cap)

Grants: 45 + 45 GPU-min. Wins: **T2-001, T2-003, T2-004, T2-007** (four promotions; none reverted). The champion went
BASE 635.9 → 787.6 tok/s while T2 held it. T3-001 (816.6) now builds on T2-007.

## Records
| record | what | result |
|---|---|---|
| T2-001 | CUDA graphs for mixed steps (`max_prefill_cudagraph_capture_tokens` 2048), flag only | 689.5, +8.4% vs BASE, champion |
| T2-002 | T2-001 + fp8 draft head + sampling flags | OOM at startup (negative) |
| T2-003 | argmax before gather: draft `use_local_argmax_reduction`, target `--enable-batch-sharded-sampling` | 707.0, +2.5%, champion |
| T2-004 | W8A8 block-FP8 via tuned Triton for all M (`--kernel-config`) + 5 tuned 4090 configs | 743.9, +5.2%, champion |
| T2-005 | torch.compile/Inductor fusion route | -9.4% (negative) |
| T2-006 | fused dyn128 activation quant in the RMSNorm/SiLU producers + configs for every graph M | 752.6, +1.2% vs T2-004; overtaken by T1-007 |
| T2-007 | T1-007 (host-mapped AR) + T2-006 | 787.6, +2.1% vs T1-007, champion |
| T2-008 | + fused dyn128 quant in GDN gated-norm callback, on T3-001 | null: the live CUDA spec-decode path bypasses the modified code (review T2-7); 1 start 804.2 says nothing about it |

## What I learned
1. **The framework's cost model misses the host.** My one dimension not in the other map was CPU launch time: mixed
   steps above the largest captured graph were launch-bound (one rank idle while the other waits in NCCL). This was
   the biggest single win (T2-001), and it is a flag. Add a launch/latency term per kernel to the window cost;
   at decode, many remaining costs are fixed per-kernel floors, not bytes or FLOPs.
2. **Check the engine for existing options before building.** Three of the four wins were existing vLLM options or
   configs used differently: the prefill-graph cap, batch-sharded sampling and local argmax, the Triton W8A8 backend
   plus tuned configs. The framework says what to price, and a code search says what is already there.
3. **Microbenchmark with replay and cold weights.** The decode W8A8 verdict flipped (uncertain → win) only when the
   GEMM microbench used graph replay and cold L2.
4. **Calibrate the bytes, not just the times.** I halved the lm_head size in 3-calibration (it is 1.27 GB/GPU), and two
   ranked candidates came from that error. A measured bandwidth check per big kernel would have caught it.
5. **Stacking across teams worked.** T2-006 lost its race alone, but stacked on T1-007 it gave the next win, and T3
   stacked on that. With exact file-level separation (no overlapping hunks) and a code review of the combined
   dispatch path, the review was quick.
6. **Confirm that the live path is taken before measuring.** T2-008 changed a callback that the CUDA GDN spec-decode
   branch never reaches. A log line or counter check would have saved a harness start; the review caught it.
7. **Reviews paid off.** They corrected an overstated mechanism (T2-001: the GDN core stays eager), a wrong error bound
   (T2-006 scale floor) and an unproven attribution (T2-007). The worker agreed each time once shown the reviewer's
   point.

## For the next team
See reports/T2-sweep/update-2.md for the ranking.
- Hide or overlap the remaining decode AR (T3 has begun with the L2 prefetch).
- Cut small-kernel fixed costs: GDN in_proj_ba split-K, remaining quant launches. The out_proj quant has to be fused into the
  epilogue of the qwen_gdn_attention_core_fused_norm_packed CUDA op (T2-008 hit the dead path); with o_proj, measure 3+ starts.
- k=3 drafts once the draft head is cheaper.
- GDN prefill core capture (≤2%).
