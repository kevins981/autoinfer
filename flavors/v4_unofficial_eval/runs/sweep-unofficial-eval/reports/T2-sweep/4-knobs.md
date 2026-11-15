# T2 sweep, step 4: the sweep, knob by knob

Reference: champion = BASE, M1 profile (3-calibration.md): total 37.9 s = 644 decode steps x 38.6 ms
(23.6 s) + 85 mixed steps (14.3 s). Conversion: **1 ms saved per decode step = 0.64 s = 1.7% of
total**; 1 ms per mixed step = 0.085 s = 0.22%. Gains below are napkin, against that measured time.

## Knob 1: Keep / Recompute (data nodes across steps)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| GDN state S[r,l,h] (fp32, 1.57 MB/req/layer/GPU) | **KR1: keep S[t] + the 3 rows' (k, v, g, β) instead of 3 per-row state copies** (today: read 1 + write 3 states per layer so the next step can start from the accepted row). Next step replays the ≤2 accepted rows from S[t] (rank-1 updates in SRAM) before its own rows, writes 1 state. = framework 02 §4.3 Row B / 04 §5 row 4 ("rollback-free SSM verify"). | bytes/step 4.8 GB -> 2.4 GB: **-2.4 ms/step (≈4%)**, plus the same on decode rows of mixed steps | **candidate** (medium-high effort: slot bookkeeping, prefix-cache align mode, prefill/preemption paths). Exact: same fp32 updates in the same order |
| same | KR1b: store per-row states in bf16 | -1.2 GB | **forbidden** (state precision is part of GDN math; approx row kept only for completeness) |
| conv state, draft KV | small (<1 MB/step) | ~0 | dropped: < 0.1% |
| kv cache | already kept; recompute loses (02 §3) | — | baseline |
| prefix cache across requests | ShareGPT prompts share only the chat-template head; hit rate 0% in BASE logs | ~0 | dropped (and tuning it to the benchmark is forbidden) |
| target / draft logits | kept nowhere (Rule 2: depend on t) | — | dead |
| dequantized bf16 weights | 2x bytes and forbidden ("higher-precision copies") | negative | dead |

## Knob 2: Rewrite
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| argmax_v over vocab-split logits (target) | **RW1: local (max, idx) per rank + merge of 2x48 pairs** instead of all-gathering 48 x 248k bf16 (11.9 MB). Exact for greedy rows (ties -> lowest index, as torch.argmax); non-greedy batches fall back | -0.9 ms/step (≈1.5%) | **candidate** (code: sampler/rejection path) |
| argmax_v (draft) | RW2: same for the draft: existing flag `use_local_argmax_reduction` | -2 x 0.3 ms = -0.6 ms (≈1%) | **candidate, flag only** |
| AR + residual add + RMSNorm | fuse (saves 2 launches/layer; bytes unchanged) | ~0.3 ms/step | weak (graph-replayed launches are ~2 µs) |
| RMSNorm over split d (sequence-parallel: reduce-scatter + norm + all-gather) | same link bytes as AR, 2 exchanges instead of 1 | ≥ 0 | dropped: no byte saving, more latency |
| GDN chunk form (prefill) | re-bracket (QKᵀ)V vs Q(KᵀV) at M vs d=128 | prefill GDN is 0.6 s total | dropped: ≤1.5% upper bound |

## Knob 3: Speculate
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| token edge, draft count k | **SP1: k=3** (M=64 rows: still one Marlin pass, below ridge 82). Acceptance 1+0.78+0.56+~0.38 = ~2.72 vs 2.345 (+16%). Cost: +1 draft step (~2.5 ms today, ~1 ms after RW2/SM4/AP1), AR payload +33% (+~2 ms), GDN 4 rows (+1.2 GB writes today) -> step +4..5.5 ms (+11-14%) | **+2..5% decode throughput**, more after KR1 and cheap drafts | **candidate** (flag); riskiest: 3rd-position acceptance of a 1-layer MTP head |
| same | k=1 | 1.78 vs 2.345 tokens for -~3.5 ms | dropped: -24% tokens for -9% time |
| tree drafts | top-2 branching | GDN verification of a tree needs per-branch states (in-place S) -> large state traffic | dropped (cost > gain at 16 x 3 rows already) |
| other drafters (retrained MTP / EAGLE) | ML-layer work, acceptance unknown | ? | parked (no non-benchmark training pipeline here) |
| n-gram / suffix | ShareGPT replies rarely copy the prompt | small | dropped |
| sel/expert edges | dense model: none | — | n/a |

## Knob 4: Split / Merge
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| layer index l across GPUs | **PP=2** | per step each GPU reads its 12.2 GB once per micro-batch: with 2 micro-batches 2x weight reads (+18 ms), with 1 micro-batch 50% idle; saves only AR (8.2 ms) | **dominated** (proof: weight bytes per GPU per step double while AR saving < weight cost) |
| r for the GDN mixers (DP attention-like), TP for MLP | replicate GDN weights (+57.6 MB x 48 = 2.8 GB/GPU/step reads, +2.8 GB memory) to remove 48 of 128 ARs (-2.9 ms) | ~0, memory worse | dominated |
| asymmetric α ≠ 0.5 | both ranks do identical work; nothing to rebalance | 0 | dead (05 §4.1 condition T_other,0 = T_other,1 holds) |
| r for sampling | **SM3: batch-sharded sampling** (existing flag; all-to-all half the logits) | -0.5 ms/step (≈0.8%) | candidate, superseded by RW1 if both work |
| v for the draft head | **SM4 (approx, allowed draft-side): reduced draft vocab** (top-N token ids from non-benchmark data, e.g. 32k) | lm_head draft 2 x 1.37 -> 2 x 0.18 ms: -2.4 ms (≈4%) minus acceptance loss | candidate (needs a clean vocab source; ruling 2 allows it) |
| i of prompts (chunk size) | max_num_batched_tokens vs CUDA graph capture size | interacts with W1 | folded into W1 |
| merge the 2 draft steps | serial token edge | — | dead |

## Knob 5: Place
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| next GEMM's weights -> **L2 tier** during each AR (HBM idle ~61 µs, L2 72 MB) | **PL1: prefetch kernel on a side stream during the AR** (novel; Place on memory tier + Window HBM ‖ Link) | up to 128 x ~50 µs = **-5..6 ms/step (≈9%)** if L2 keeps ~45 MB and the GEMM then runs at L2 speed | **candidate, high uncertainty**: microbenchmark running (worker t2-w1) |
| lm_head on one GPU | 2x bytes on one GPU | negative | dead |
| KV / GDN state to host | no capacity pressure at c=16 | 0 | dead |
| draft head on one GPU, target on other | serial dependency; no overlap without 2 cohorts (doubles weight reads) | negative | dead |

## Knob 6: Window (and the missing CPU resource)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| **mixed steps run eagerly** (> 96 tokens) | **W1: capture piecewise/breakable CUDA graphs for mixed steps** up to max_num_batched_tokens (`max_prefill_cudagraph_capture_tokens`, `max_cudagraph_capture_size`, capture sizes), or cap chunk size to the captured size | small/medium mixed steps 133 -> ~60-110 ms: **-3..4 s of 37.9 (≈8-10%)** | **top candidate, flags only**; riskiest: memory (graph pool) and the 128K request |
| AR ‖ GEMM | W2: split the out/down GEMM's output columns into 2-4 chunks, AR chunk c while computing c+1 (non-reduced index -> weights still read once) | hides ~(1-1/k) of AR behind ~45-60 µs GEMMs: **-2..3 ms/step (≈4%)** | candidate, medium effort; competes with PL1 for the same AR shadow |
| draft steps | W4: CPU gaps 0.99 ms/step (FlashInfer cannot update draft metadata in place -> eager rebuild); try a backend that can, or the fused multi-step draft path | -0.5..1 ms/step (≈1-1.5%) | candidate (flag test) |
| decode rows of two cohorts (DBO) | split r into 2 cohorts, overlap AR of one with GEMM of other | weights read twice at M=24 (memory-bound) -> +18 ms | dominated (06 §3: wins only near ridge) |

## Link efficiency (Place/Link: implementation-level, but the 2nd biggest term)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| NCCL config | **L1: env tuning** (NCCL_PROTO LL/LL128/Simple per size, NCCL_MIN_NCHANNELS / NCCL_NCHANNELS, NCCL_BUFFSIZE, SHM copy mode) | decode 61 -> ~45 µs: -2 ms (3.4%); mixed AR 12 -> 18 GB/s: -1 s (2.6%) | candidate, flags only, high uncertainty |
| all-reduce kernel | L2: custom one-shot AR through pinned host memory (each GPU writes its 0.49 MB partial once, reads the peer's; full duplex) | 61 -> ~35 µs: -3.3 ms/step (≈5.6%) + mixed steps | candidate, medium-high effort |

## Kernel efficiency (implementation layer; listed because the napkin shows the gap)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Marlin decode GEMMs (0.65 of BW in situ) | KE1: better M=48 FP8 weight-only kernel / Marlin config tuning | 18.6 -> ~14 ms: -4.6 ms/step (≈8%) | candidate, high effort; part of the gap is GDN write drain (KR1 attacks it) |
| target lm_head (bf16 cuBLAS 0.8 TB/s) | KE2: better GEMM choice | -0.25 ms (0.4%) | weak |
| mixed-step GEMMs at ~80% of bf16 peak | KE3: W8A8 FP8 with fp16 accumulation inside a 128-K block (330 TFLOP/s) | -2.5 s (≈6.6%) | **needs a ruling** (is fp16 accumulation "the tensor cores' own accumulator"?) + new kernel + GSM8K-250 A/B; parked |
| draft lm_head | **AP1 (approx, allowed draft-side): `draft_lm_head_quantization=fp8`** | 2 x 1.37 -> 2 x ~0.7 ms: -1.4 ms (≈2.3%) | candidate, flag only |

## Pruned groups (summary of dominance arguments)
- Any split that duplicates weight reads at M=48 (PP micro-batching, DBO cohorts, DP mixers) loses:
  decode is weight-bandwidth-bound (N=48 < N_ridge≈82), so each extra weight read costs ~18 ms per step
  against AR savings ≤ 8.2 ms.
- Keep-set changes other than the GDN state are below 0.1% (bytes too small).
- Approximate rows on the target side (FP8 communication, bf16 state, truncated target vocab,
  early exit) are forbidden by the contract; approximate rows on the draft side (FP8 / reduced-vocab
  draft head) are allowed and kept.

## Candidate list (napkin gain of total time)
| id | candidate | gain | cost to test |
|---|---|---|---|
| W1 | CUDA-graph mixed steps | 8-10% | flags |
| PL1 | L2 prefetch during AR | 0-9% | microbench, then kernel + graph plumbing |
| KE1 | faster decode FP8 GEMM | ~8% | high |
| KR1 | GDN 1-write state (replay accepted rows) | ~4% | medium-high |
| W2 | AR ‖ GEMM column split | ~4% | medium |
| L2 | custom host AR | ~5% | medium-high |
| L1 | NCCL env tuning | 0-6% | flags |
| SM4 | reduced-vocab draft head | ~4% (net of acceptance unknown) | medium |
| SP1 | 3 drafts | 2-5% | flag |
| AP1 | FP8 draft head | ~2.3% | flag |
| RW1 | target local argmax | ~1.5% | low-medium |
| RW2 | draft local argmax | ~1% | flag |
| W4 | draft-step CPU gaps | ~1-1.5% | flag/low |
| SM3 | batch-sharded sampling | ~0.8% | flag |
