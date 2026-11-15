# T1 map update 1 (after the first wins; HEAD = T1-005, confirmed 760.8 tok/s official, +19.6% over BASE)

## Tested since 6-ranking.md
| id | record | result | verdict change |
|---|---|---|---|
| R6a + R1-draft | T1-001 | PASS +3.26% vs BASE, confirmed in the chain | napkin (~2 ms/step, ~3%) matched |
| S1 (3 MTP drafts) | T1-003 | negative (1 start) -4.89% vs T1-001; acceptance 2.708 (p3 0.38 as napkin assumed), step +~21%, TTFT p99 gate failed | napkin's "~0% now" was optimistic: step cost grew more than priced (AR bytes +33% on the slow link, heavier mixed steps). Revisit only if steps get much cheaper |
| R1-target (logits) | T1-004 (batch-sharded sampling) | PASS +1.61% vs T1-001, confirmed | napkin ~1-1.5% matched |
| X1 (fast AR) | microbench (t1-w1 job 3) -> T1-002 / T1-005 | NCCL tuning <= ~10% (noise); host path ceiling ~13-14 GB/s, not 25; custom one-shot host-pinned AR: 48 vs 60 us at M=48, 20 vs 36 at M=16, ties at 256, loses at >=512. T1-002 PASS +3.40% vs T1-004; rebased T1-005 PASS +5.07% vs T2-004, confirmed +4.4% official | **X1 napkin (15%) was wrong by ~3x**: I priced the link at 25 GB/s spec and assumed NCCL was the loss; measured, the host path itself caps near 13-14 GB/s and NCCL is near it at large sizes. Real gain is only at decode sizes (latency-bound). Lesson: measure the link before ranking by it |
| W1/W3 (prefill AR overlap) | feasibility (t1-w1 job 20, skew analysis) + T1-006 | side-stream NCCL hides 74% behind GEMMs in isolation; but half of in-step prefill AR time is **rank skew** (GPU0 runs compute-bound Marlin ~12% slower: same kernels, not launch-bound); full DBO blocked by GDN align-mode state (1.5-2 weeks); chunked RowParallel GEMM+AR (T1-006): bitwise exact, no gain on T1-005 (1 start per arm) | dropped. Napkin overestimated: "AR time" in prefill steps was ~43% wire + ~57% waiting for the slower rank |

## New dimensions / findings
- **Rank asymmetry**: GPU0 is ~12% slower on compute-bound kernels (prefill Marlin), ~3% on memory-bound ones (decode). The framework's asymmetric split (alpha != 0.5, file 05 §4.1) is real here, but napkin only ~1% (prefill steps are GPU0-critical by ~8.5 ms of ~145). Untried.
- **The model defers the AR into the next norm** (qwen3_5 forward_fused, reduce_results=False): AR+add+RMSnorm fusion inside the host-AR kernel is a natural next step (~0.5 ms/decode step, ~1%).
- From T2 (5-compare credit): prefill CUDA graphs up to 512 tokens (T2-004, +4.5%): my map listed R5 (graphs/compile for prefill) at 1-2% because I trusted the nsys-inflated idle less than I should have. T2 measured a graph-level trace and was right.

## Current ranking (remaining, against HEAD T1-005)
1. Host AR closer to link floor (overlap write upstream / read downstream; copy-engine pull) — microbench in progress (t1-w1 task 4). Up to ~3 ms/step if 48 -> ~25 us. ~5%?
2. AR + residual + RMSnorm fusion in the host AR kernel: ~1%.
3. Decode GEMM efficiency (Marlin 0.67 of BW at M=48): ~5%, high effort.
4. GDN state writes (T2's KR1) — T2's.
5. Reduced-vocab draft head (T2's SM4) — T2's.
6. Asymmetric TP split for the slower GPU0: ~1%, large plumbing.
7. 3 drafts: only after steps get cheaper.
