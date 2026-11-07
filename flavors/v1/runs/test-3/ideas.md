# IDEAS (run test-3): scored but untried. An archive, not a queue.
These are never selected directly: an idea comes back only if a brainstormer proposes it again.
Scores: novelty (N) and upside (U), 1-5. Sources: SE = systematic_engineer, AF = adjacent_field.
Brainstorm files: brainstorm/r<round>/*.md.

## Round 1 (base: BASE)
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R1-C | NCCL env tuning (protocol / channels / buffsize) | SE | 1 | 1 | AF measured it: at most -6% time at prefill size, worse at decode. Effectively ruled out. |
| R1-D | L2 prefetch of the next GEMM's weights during each AR | SE, AF | 5 | 3 | Re-proposed in r2 as R2-C. |
| R1-E | Marlin per-shape retune for M=48 | SE | 2 | 3 | Re-proposed in r2 as R2-D. |
| R1-F | Local argmax for draft logits | SE, AF | 1 | 1 | Later done as T1-006 (r2, under R2-A). |
| R1-G | Greedy target-verification local argmax | AF | 2 | 2 | Later done as T1-009 (r2, under R2-A). |
| R1-H | In-place FlashInfer draft-metadata update (~1 ms CPU gaps) | SE | 2 | 2 | Gaps measured under nsys. |
| R1-I | FP8 draft lm_head | AF | 1 | 2 | Later done as T1-007 (r2, under R2-A). |
| R1-J | Reduced-vocab draft lm_head (non-benchmark frequencies) | SE, AF | 3 | 3 | Allowed (ruling 2); untried. |
| R1-K | GDN: write one spec state per step instead of three | SE | 4 | 2 | Re-proposed in r2 as R2-B. |
| R1-L | Prefill AR/compute overlap with micro-batches | AF | 4 | 2 | Mostly ruled down in r2: the skew is from GPU 0's clock (KB8). |

## Round 2 (base: T1-005)
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R2-B | GDN spec-state checkpoint + log replay | AF1, SE4 | 4 | 3 | 100 MB/layer of state traffic, spill measured (~5 ms/step); est. +3.5-5%; medium-high effort. |
| R2-C | L2 prefetch during the host-AR windows | SE2 | 5 | 3 | Est. +3-7%, low-medium confidence; 1-GPU-min kill test. |
| R2-D | Marlin decode M=48 retune (670 vs 800 GB/s at M=3) | SE3 | 2 | 3 | Est. +5%. |
| R2-E | Copy-engine full-duplex AR for large M | AF3 | 2 | 2 | ~+1.5-3%; half the prefill AR is the GPU 0 skew. |

Ruled down with numbers (from the brainstorms):
- PP=2.
- Chasing the prefill skew, or uneven TP (KB8).
- The decode step-start gap (an nsys artifact).
- Dequant + cuBLAS for prefill GEMMs.

## Round 3 (base: T1-009)
Selected: R3-A + R3-B (see tree.md). Scored for the record: R3-A (k=3 with a KV budget that holds 16 requests; both brainstormers; N2 U3, est. +2-5% alone) and R3-B (GDN spec-state checkpoint + replay, = R2-B; both; N4 U4, est. +4.5-6%, frees spec pages so k=3 fits without extra KV).
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R3-C | Bitwise-lossless compression of the AR payload (bf16 exponent packing, raw overflow fallback) | AF2 | 5 | 2 | Est. +2-3%, depends on measured compressibility; ruled baseline-precision by the manager (lossless for all inputs, generic parameters). Near-free kill test (1 dump + CPU analysis). |
| R3-D | L2 prefetch of gate_up during the host-AR window (= R2-C, re-scoped) | SE3 | 5 | 2 | Do only after R3-B: the dirty-L2 drain from GDN state writes would fight the prefetch. Est. +3-5%, low confidence. |
Ruled down (r3, with numbers):
- Marlin M=48 retune (R2-D): the in-kernel penalty is only 1.6 ms/step; the rest is GDN spill. SE microbench agents/bs-r3-systematic_engineer/mb/.
- k=4: Marlin cliff at M=80 (17.2 -> 24.7 ms/step).
- Reduced-vocab draft head (R1-J): net ≈ 0 now that the drafts are FP8.
- Rebalancing weights for the GPU 0 skew: ≤3%, and it costs memory.
Left open after round 3 (from leader-1's report, not yet scored by a brainstorm):
- R3-B v2: shrink the GDN spec blocks per request to 1 (the log slot), so k=3 fits 16 requests in 5 GiB. Needs the stock per-token layout removed everywhere; risk: silent state corruption from an unknown reader.
- Redundant conv-state work in the replay kernel.
- T1-012 preemption/resume test.
