# T2 sweep, step 5: my map against T1's map and the history

State when written: HEAD = CONFIRMED = BASE. Bank: BASE, T1-001 (draft lm_head FP8 + draft local argmax, flags,
pending unofficial eval). shared_kb.md empty, no resync reports, no sparring notes. T1's map: reports/T1-sweep/1-6.
T1 read my partial map (files 1-2) when writing its 5-compare.

## My candidates: status
| id | candidate | status |
|---|---|---|
| W1 | CUDA-graph mixed steps (> 96 tokens run eagerly) | **untried, and absent from T1's map** (T1 R5 "torch.compile / prefill graphs ~1-2%" is the nearest row; T1 attributes the prefill idle to nsys overhead) |
| PL1 | L2 prefetch of next GEMM's weights during AR | untried; T1 ranks it 3rd (adopted from my file 2); my worker t2-w1 is microbenchmarking it |
| KE1 | faster decode FP8 GEMM | untried; = T1 R3 |
| KR1 | GDN single-write state (replay accepted rows) | untried; = T1 K2, priced differently (below) |
| W2 | AR ‖ GEMM column split | untried; = T1 W1/W3 (T1 prices it for prefill steps) |
| L1/L2 | NCCL tuning / custom host AR | **being tested by T1** (X1, t1-w1 microbenchmark) |
| AP1 + RW2 | FP8 draft head + draft local argmax | **built by T1: T1-001** (unofficial eval pending) |
| SM4 | reduced-vocab draft head | untried; = T1 R6b |
| SP1 | 3 drafts | untried; = T1 S1 |
| RW1 | target local argmax | untried; = T1 R1 (target part) |
| W4 | draft-step CPU gaps (FlashInfer metadata rebuilt eagerly) | untried; not in T1's map |
| SM3 | batch-sharded sampling | untried; T1 notes the code path |

## What my sweep found that T1's map lacks
1. **Mixed steps are not CUDA-graphed and are CPU-launch-bound.** Server config: `max_prefill_cudagraph_capture_tokens: 96`,
   `max_cudagraph_capture_size: 96`, compilation mode NONE. With a graph-level trace (GPU job 2, M1, whose 100-prompt
   throughput under profiling, 599 tok/s, equals the unprofiled reference ~596), small mixed steps (M≈140) take 133 ms
   with ~53 ms of measured GPU idle plus rank-skew waiting inside NCCL (napkin GPU work ~55-65 ms). T1 attributed the
   idle to nsys overhead on its node-level trace; the graph-level trace says it is real. This is a missing *resource*
   in the framework's cost model (host dispatch rate), worth ~8-10% here.
2. **The GDN MTP kernel writes 3 state copies per layer** (one per verified row, `spec_state_indices_tensor` [reqs, 3],
   csrc/gdn/fused_gdn_decode_kernel.cu): 1.57 MB x 16 x (1 read + 3 writes) = 100 MB/layer, not T1's 50 MB. So the
   ReplaySSM row (my KR1 = T1 K2) saves 2 writes per layer, ~2.4 GB/step: ~4%, not ~1.8%. Also explains part of
   Marlin's in-situ slowness (deferred write drain after the GDN kernel).
3. **Draft-step CPU gaps** (~1 ms/step, "FLASHINFER cannot update draft decode metadata in place" in the server log).

## What T1's map has that mine missed
1. **Official-run time model** (143.2 s split, 42% prefill-containing): consistent with my M1 (38%).
2. **fp16 in-block accumulation question for W8A8 on GeForce, with the e4m3 product overflow risk** (T1 R2b). I parked
   the same row (KE3) for a ruling; T1's overflow point is a concrete reason it may fail numerically.
3. **Custom one-shot host-memory AR as a concrete design (X1).** I had it (L2), T1 is building it.

## Where I disagree with T1's verdicts
- **Prefill GEMM "re-read bound" (T1 R2a, ~5%)**: M1 shows Marlin in mixed steps at 6.07 s for ~33.1k rows, i.e. ~80%
  of the bf16 peak (4.9 s at 100%). T1's own 437-row number (91.9 ms vs 66.7 ms at peak = 73%) is also near compute.
  A read-once kernel is bounded by compute at fp32 accumulation: realistic gain ≤ 1-3%, not 5%. (My own file 2/1 said
  "Marlin re-reads per 64 rows"; M1 shows L2 absorbs much of it. Correction to my file 1.)
- **Prefill AR overlap (T1 W1/W3, ~13%)**: part of the measured prefill NCCL time is waiting for the slower rank's
  eager launches (rank skew), not transfer. After W1 graphs the mixed steps, the transfer part is smaller than T1's
  66.6 ms/step implies; re-price after W1.
- **T1 R5 (torch.compile/graphs ~1-2%)**: the graph part of it is my W1, ~8-10% by M1.
