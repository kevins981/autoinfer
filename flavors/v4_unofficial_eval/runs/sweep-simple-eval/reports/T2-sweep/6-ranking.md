# T2 sweep, step 6: ranked candidates

Napkin gain against BASE (143.4 s benchmark; 3-calibration.md). Throughput gain ~ 1/(1-saving) - 1.

| rank | id | candidate | napkin saving | wins if | riskiest assumption | cheapest test that could show it wrong | owner |
|---|---|---|---|---|---|---|---|
| 1 | W1 | CUDA graphs for mixed prefill steps up to 2048 tokens (serve flag) | ~17 s (12%, +13% tok/s) | mixed-step fixed cost is host/skew, not NCCL bandwidth | that breakable graphs capture GDN chunked prefill correctly at all sizes and memory still fits 128K | **running**: unofficial eval t2-cg2048 with --long (job 4) | T2 |
| 2 | W4 | faster all-reduce (NCCL tuning or custom pinned-host AR) | ~4 ms/decode step -> ~9.7 s (7%) + bandwidth part of mixed steps | intrinsic 0.5 MB AR can go from ~60 to ~25 us | that the 60 us is transport cost, not cross-rank wait (p5 is 14-19 us!) | AR microbenchmark in a tight 2-rank loop (running in both teams) | T1 (I-AR), T2 shares its worker data |
| 3 | R8 | Marlin decode efficiency 68% -> ~85-90% BW | ~4.5 ms/step -> ~10 s (7%) | a tile config / split strategy for M=48 streams at ~900 GB/s | that in-situ loss is per-kernel, not inter-kernel (tails, L2) | microbenchmark Marlin at M=48 for the 4 shapes vs a copy kernel's BW | open |
| 4 | R5+R4 | draft lm_head: reduced vocab (first 32K-64K token ids) + FP8 | 2.2-2.5 ms/step -> ~5.5 s (4%) minus acceptance loss | acceptance drops < ~3% | how often the target's argmax lies outside the first N ids on chat text | offline: tokenize generic (non-benchmark) text, fraction of tokens with id >= N; then 1 quick bench for acceptance | T2 |
| 5 | R6 | W8A8 FP8 GEMMs for mixed steps (M > ~128) | ~8.6 s (6%) only if FP8/fp32-acc = 330 TF | FP8 MMA with fp32 accumulate is 2x bf16 on 4090 | the 330 vs 165 TF question (5-compare, disagreement 2) | 10-second torch._scaled_mm / triton microbenchmark | T2 (after W1) |
| 6 | S1 | MTP k=3 | +2% now; +8-10% after R4/R5 | p3 (unconditional) >= ~0.35 and a draft step costs ~1 ms | p3 value | quick bench with num_speculative_tokens=3 (+128K test) | T1 listed it as cheapest; split |
| 7 | R1 | distributed argmax (no logits all-gather for greedy batches) | 1.6 ms/step -> 3.9 s (2.7%) | all-greedy batches (benchmark: yes; fallback otherwise) | that the rejection sampler path only needs argmax at temp 0 | read the sampler code; then microbench | T1 (R-ARGMAX) |
| 8 | K1 | GDN rollback-free state commit (write only accepted state) | 1-2 ms/step -> 2.4-4.9 s (2-3%) | the 2 extra state writes cost what the napkin says | measured GDN time is below the napkin already | ncu dram bytes of gdn_decode_post_conv_mtp_kernel | T2 |
| 9 | W3 / W-DBO | overlap AR with GEMM (chunked GEMM+AR in decode; DBO in mixed steps) | decode ~5 ms/step if hidden; mixed ~4 s | NCCL can run concurrently with Marlin without slowing it | SM/L2 contention | after W4 | later |
| 10 | S5 | async host prep / host gaps | 1.3 ms/step -> 3.2 s (2%) | async scheduling not already on | - | grep config / server log | T2 quick check |

Plan: W1 now (job 4). Then R5+R4 (offline vocab coverage first), R6 microbenchmark, K1 ncu, S5 check.
Coordinate with T1 so we don't duplicate: T1 owns all-reduce (W4) and distributed argmax (R1);
T2 owns W1, draft-head (R4/R5), W8A8 prefill (R6), GDN commit (K1); k=3 (S1) after draft-head work.
