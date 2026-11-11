# IDEAS (run test-5, resumed from test-3): scored but untried. An archive, not a queue.
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
Selected: R3-A + R3-B (see graph.md). Scored for the record: R3-A (k=3 with a KV budget that holds 16 requests; both brainstormers; N2 U3, est. +2-5% alone) and R3-B (GDN spec-state checkpoint + replay, = R2-B; both; N4 U4, est. +4.5-6%, frees spec pages so k=3 fits without extra KV).
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

## Round 7 (base: T1-020; brainstorm/r7/*.md)
Selected (one direction, "mixed prefill steps"): R7-A prefill AR overlap with GEMMs via copy-engine (DMA) AR + 2-way token split (SE2 + AF1, also leader-2's proposal; N4 U4, est. +4-6%, premise measured: DMA under Marlin costs it 0.2%); R7-B GDN replay on the mixed-step path (SE1; N3 U3, est. +1.8-2%, high confidence, exactness proof from T1-012); R7-C finer prefill CUDA-graph grid, serve flag (AF2a; N4 U2, est. +0.5-1.1%).
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R7-D | Bitwise-lossless AR payload compression (= R3-C) | SE3 | 5 | 2 | Compressibility now measured: 0.768 of bytes, 0.3% raw blocks (bank/BASE/profiles/ar_payload_compressibility.md). Est. +2.5% from prefill ARs, ~0 at decode; overlaps with R7-A (a hidden AR gains nothing from fewer bytes). |
| R7-E | Marlin partial last chunk in-kernel (no remainder launch re-streaming weights) | AF2b | 4 | 2 | Remainder launch costs 7.6-14.5 ms per non-64-multiple M (measured). With R7-C, est. up to +1.2-2% combined. |
| R7-F | Host AR sends only real rows (device-side token count) | AF3 | 4 | 1 | ~+0.5-1% alone; overlaps R7-C and R7-A. |
Ruled down (r7): adaptive k for the drain (drain is 4.3 s, <= 0.7%); AR fused into Marlin epilogue (prefill GEMMs are ~1 wave); PCIe duplex (+10% aggregate only); disabling async scheduling.

## Round 10 (base: T1-026; brainstorm/r10/*.md)
Selected (one leader, two parts): R10-A lossless AR payload compression at decode AND prefill sizes (SE1 = R7-D re-scoped, also leader-2's proposal; N5 U3; est. +2.5-3.5%, kill if codec adds > ~3 us per decode AR); R10-B Marlin single launch for any M with a masked last slice (AF1 = R7-E; N4 U3; est. +1.0-1.3%, +0.7% more with an every-16 grid; 0.2 GPU-min gate).
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R10-C | Copy-engine pre-pull of the peer partial by the slow rank in non-split mixed steps (uses rank 1's 26-47 us lead) | SE2 | 4 | 2 | Est. +0.5-0.8%; overlaps R10-A. |
| R10-D | Pad Marlin M to 64 when M%64 in {16,48} | SE3 | 2 | 1 | +0.2-0.3%; superseded by R10-B. |
Ruled down (r10, measured): staggered two-microbatch AR overlap without per-layer join (AF job 114: no better than T1-025's join); new large-M GEMM kernel (Marlin ≈ cuBLAS bf16 at M=112-448, SE job 116).

## Round 12 (base: T1-030; brainstorm/r12/*.md)
Selected (one leader, two parts): R12-A lazy GDN checkpoint (commit the fp32 state every ~3 steps; SE1; N4 U3; est. +1.3-2.0%, exact by construction; 2-GPU-min timing gate); R12-B lossless compression of the HBM weight streams (AF; N5 U4): first the Marlin ALU-slack probe (gate for PFOR-coded FP8 weights in Marlin, est. +2-5%), with the lm_head codec (AF1, est. +1.0-1.4%, ratio 0.766 measured) as the fallback/first build. Manager ruling on lossless weight re-encoding: journal.
| id | idea | from | N | U | note |
|---|---|---|---|---|---|
| R12-C | Certified FP8 prefilter for the greedy target argmax (exact final result) | SE2 | 4 | 3 | Est. +1.5-1.8%; needs a human ruling (an extra FP8 copy of a target weight used for screening) and +636 MB/rank (KV peak 83.8% of 5 GiB). Candidate count unmeasured (CPU check first). Competes with R12-B's lm_head codec. |
| R12-D | Residual-add + RMSNorm fused into the hostar epilogue | SE3 | 2 | 2 | Est. +0.7-1.2% (run-1 measured only 1.5-3 us/call for a similar fusion). |
| R12-E | Frequency-ranked draft vocab | SE4 | 2 | 1 | +0.3-0.7%. |
| R12-F | Compressed fp32 GDN replay state | AF3 | 4 | 2 | ~+1%, compressibility unmeasured; overlaps R12-A. |
Ruled down (r12, with numbers): k=3 / dynamic draft length (Marlin knee: ~0.29 ms per extra token); chunked-prefill piggybacking; tree drafts within M=64; exact FP8/INT8 prefill decompositions; uneven TP; host gaps (GPU idle 0.6 s of gaps >= 0.2 ms in a 21 s trace).
