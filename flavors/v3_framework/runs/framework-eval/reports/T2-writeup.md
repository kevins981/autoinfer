# Team T2 write-up (sweep team, leader-2 + worker leader-2.w1)

Map: reports/T2-sweep/ (1-inputs ... 6-ranking, update-1/2/3). Worker index: agents/leader-2.w1/SUMMARY.md.
GPU: 220 granted, ~203 used (17 left at closing).

## Official wins (BASE 635.9 tok/s at c=16)
| record | what | official | vs previous champion |
|---|---|---|---|
| T2-002 | CUDA-graph capture sizes up to 512 (mixed steps were running eager and CPU-bound) on T1-004 + T1-005 | 757.6 | +7.1% |
| T2-006 | W8A8 Triton block-FP8 GEMMs with 4090-tuned configs (+ T1-009's TRITON_ATTN) | 819.9 | +4.3% |
| T2-008 | DeltaLog: GDN MTP decode writes row 0's state + a 784 B/row/head log; accepted rows replayed lazily; bitwise exact | 880.1 | +4.4% |
| T2-009 | Fused add-RMSNorm / SiLU-mul + group-128 FP8 quant feeding the GEMM | 906.9 | +1.06% |
| T2-011 | Reduced-vocab MTP draft head (ids [0,151643) + specials, data-free; target verifies full vocab) | 931.8 | +1.39% |
Also T2-004 (788.8, superseded the same day). Not promoted: T2-012 (hostmem AR release/acquire flags, official 953.1
vs 951.5, within noise; a same-job A/B vs T3-009 was mixed, +0.54%).

## What we learned
1. **The biggest wins were not where the FLOPs/bytes napkin pointed.**
   - T2-001/002 was CPU launch time in eager mixed steps.
   - DeltaLog was L2 write-back that one kernel leaves for the next to pay.
   - Both were found by measuring per-step time with GPU events, not by pricing nodes.
2. **Interactions dominate.**
   - W8A8 gained nothing until the mixed steps were graphed (T2-003 null → T2-004 +4.7%).
   - My fused norm+quant silently disabled T1-013's row-split overlap (found by T3).
   - A shorter AR shrinks T3's prefetch window.
   - Before every A/B, check the earlier features' log lines in every server.
3. **Use measured peaks, not spec sheets.**
   - DRAM ~850-890 GB/s, not 1008.
   - SM-issued PCIe duplex ~31-34 GB/s total for both GPUs. Copy-engine numbers (22.6 GB/s) are a different floor.
   - FP8 block-scaled MMA: ~275-285 TFLOPS realistic vs the 330 spec.
   - Against these, the decode GEMMs, GDN decode and target lm_head are at their floors.
4. **The decode AR is latency-bound.** 44 us = 19 write + 21 read + handshake. Each extra flag handshake costs ~5 us,
   so pipelining loses. Cheaper ordering (release/acquire instead of a system fence) saves ~3 us.
5. **Exactness pitfalls worth knowing:**
   - Triton's f32→fp8e4nv cast double-rounds through fp16. Use PTX cvt.rn.satfinite.e4m3x2.f32.
   - vLLM's silu_and_mul_per_block_quant skips the bf16 rounding of silu.
   - FlashInfer's norm reduction order differs from Triton's by 1 ulp in ~1e-5 of elements.
   - DeltaLog's first version corrupted requests at prefix-cache block boundaries: an in-place replay destroyed another slot's log.
6. **Spec depth k=2 is measured optimal on current code** (k=3: acceptance +16% but step +32%, TTFT p99 gate broken; T2-013).
7. **Sparring was useful at the end.** It made me re-test k=3 instead of inheriting T1's verdict, and mark
   "mixed GEMMs at ceiling" as assumed until we wrote down an independent bound.

## What's left (see update-3.md)
- Hostmem AR release/acquire (built, ws t2-rel9): ≤0.5% on T3-009 (mixed A/B); stack it with a prefetch retune.
- Prefetch-aware GEMM configs with a bigger prefetch window (T3 took it).
- A better block-scaled FP8 kernel for mixed steps: ≤0.9% of wall.
- DeltaLog materialize folded into the Triton spec path: ~0.5% of wall.

## Framework feedback
The graph + knobs sweep found the right *sites*: DeltaLog = Keep/Recompute on the state node, reduced vocab = Window on
the vocab index, fusions = Merge. It mis-ranked them because the cost model lacks:
- (1) a CPU resource;
- (2) L2 carry-over edges between adjacent nodes;
- (3) measured per-resource peaks;
- (4) a check for interactions between combined changes (shared gates, shared idle windows).
