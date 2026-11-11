# T15 write-up (leader-15, explore). Seed only; no proposal clears the bar

GPU: 0.1 of 20 used (jobs 115, 117, 118). Records: bank/T15-001, bank/T15-002. Notes: local_kb/leader-15.md. No worker started.

## Measured
1. **No second exact compute pipe (T15-001).** On sm_89, bf16/f32-acc HMMA and FP32 FFMA do not co-issue: with fixed work per
   warp, mixed runs take the sum of the two alone (tensor 0.463 ms + FFMA 0.243 ms -> 0.708 ms together). Combined throughput
   never exceeds tensor alone (145 TF GPU0). The idea: GeForce Ada's half-rate f32-acc MMA makes FFMA ~50% of the tensor rate,
   and a bf16 x e4m3 product is exact in fp32, so an FFMA share would be exact-class. Dead. Exact compute roof = bf16 MMA.
2. **PCIe duplex with node-0 slots (T15-002).** KB5 is not a NUMA artifact for SM traffic: SM write + SM read of 384 KB = 25 us
   vs 17.3 / 19.0 alone. SM posted writes are the problem (SM-W + CE-H2D just as bad). Copy engines are ~full duplex
   (CE D2H + H2D 19.0 us; 4 MB at 45 GB/s combined) and **CE write + SM read is near duplex (20.8 us)**.
3. **Decode rank imbalance is small (CPU analysis of T2-006's trace, agents/leader-15/imb.py).** Decode: GPU0 Marlin +1.6%,
   rank 1 waits ~3.2 us/AR mean (KB23 holds). Mixed steps: GPU0 Marlin +15%, rank 1 hostar +47% (waiting).
   Also: decode "attention" in T2-006's table is mostly the flashinfer fused add+RMSNorm kernel (name matched "flash");
   kernel_unified_attention itself is ~0.16 ms/step. The FP8 draft lm_head is 309 us/call, ~86% of its byte roof.

## Costed, below the bar (vs T9-005, decode ~62-70% of wall)
| item | payoff if it works | central | basis |
|---|---|---|---|
| Hybrid decode AR: CE D2H in pieces + cuStreamWriteValue32 flags, SM reader polls/decodes/adds | ~+2% (~-10 us raw, ~-6 us net/AR) | ~+0.8% | transport floor measured (T15-002); CE start/tail latency assumed (T9-006 saw ~35 us per-op overhead all-CE) |
| Static uneven MLP split (block-aligned, e.g. 67/69 of 136 blocks) | ~+0.8% | ~+0.4% | imbalance measured (T2-006 trace); memory +125 MB on GPU1 (360 MiB free per T9) |
| Exact top-1 target lm_head screen (T10) | ~+0.8% | ~+0.3% | assumed |
| Drafter collective anomaly (T12: 110-183 us/draft step vs ~50) | ~+0.7% | ~+0.3% | measured size, cause unknown |
| **sum** | **~+4.3-4.7%** | **~+1.8%** | below +5% even if all work |

## Opinion
- I agree with T2/T4/T8/T10/T12: the champion's decode step is pinned by coupled HBM, bf16-tensor and PCIe floors, and the
  prefill by the bf16 tensor roof, which no exact trick on sm_89 lifts (FP8 MMA: KB69; INT8 slicing: T12; FFMA: T15-001).
- The hybrid CE-write/SM-read AR is the one new transport fact; it is an exploit-sized item (~1-2%), best stacked into a
  package that also owns the mixed-step items (T14).
