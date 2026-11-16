# T4 sweep, step 4: knob by knob

Reference times (3-calibration.md): decode step 32.7 ms (62% of benchmark time), mixed step ~120-140 ms
(38%). Gain of a site = saved ms/decode step x 0.62/32.7, or saved ms/mixed step x 0.38/140.
Note (after writing): the champion moved to T3-003 (T3-001 + FP8 draft lm_head, KV 4.4 GiB) at 826.1
during this step; K2-a below is therefore built by T3.

## Knob 1: Keep / Recompute (data nodes across steps)
| site | candidate | napkin | verdict |
|---|---|---|---|
| K1-a GDN state S[r,l] under spec verify | today: Keep with in-place overwrite + 3 branch copies (state written after each of the 3 rows: 1 read + 3 writes of 1.57 MB per req-layer). **DeltaLog variant (02 §4.3 Row B)**: write only the state after row 0 (row 0 is always accepted), store rows 1-2's inputs (k, v, a, b: ~1 KB per head) in the now-unused slot 1; next step reads slot 0 and replays the accepted rows 1..a-1 in registers before the new rows. Bit-exact if the replay uses the same fp32 arithmetic. | traffic 4 -> 2 state units/req-layer: measured 72.7 us/layer, read-only floor ~25 us; expected ~40-45 us -> saves ~1.4 ms/decode step | **+2.6%**; risk: every path that reads spec slots (mixed-step Triton GDN path, prefix-cache state copies) must follow the new convention |
| K1-b same, write nothing (keep S[t-L] for L=2 steps, replay up to 2+3 rows) | reads 1, writes 0.5 per step | ~+0.4% more than K1-a | after K1-a |
| K1-c KV for the draft (MTP layer) / target KV | append-only, fine | 0 | no site |
| K1-d keep x (hidden) instead of KV for attention | recompute k,v projections: compute >> 2 KB read | loses | pruned (02 §3 table) |
| K1-e logits | not kept (Rule 2: carry t) | — | illegal |
| K1-f prefix caching | on by rule; no shared prefixes in the benchmark | 0 | no site |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| K2-a draft lm_head bf16 -> FP8 (ruling 1) | 2.54 GB -> 1.27 GB per step | -1.3 ms/decode -> +2.5% | **built by T3 (T3-003, +1.16%)** |
| K2-b reduced-vocab draft lm_head (ruling 2: ids from non-benchmark data, e.g. BPE merge rank or a generic corpus) | draft head 248K -> 32K rows: 2 x 0.16 GB (bf16) | -2.3 ms (vs bf16) / -1.1 ms (vs FP8) per decode step -> +2.1% on top of T3-003 | acceptance risk (tokens outside the subset are never proposed); try 64K |
| K2-c target lm_head | must stay bf16 (contract) | — | pruned (contract) |
| K2-d fold norm weights / scales into GEMMs | norms are ~0.8 ms launch-bound, not math | ~0 | pruned |
| K2-e merge qkvz + ba GEMMs (dedup the activation read; one launch) | ba is bf16 (unquantized) while qkvz is FP8: different kernels; ba already on an aux stream | ~0.3 ms | low |
| K2-f GDN dual form (03 §4.2) for 3-row verify | already recurrent form per row; M=3 < d=128 -> attention form would be cheaper in FLOPs but the kernel is state-bandwidth-bound | 0 | pruned (bandwidth, not FLOPs) |
| K2-g full-attention output gate / RoPE | tiny | 0 | pruned |

## Knob 3: Speculate
| site | candidate | napkin | verdict |
|---|---|---|---|
| K3-a draft count 2 -> 3 | tokens/step 2.33 -> ~2.73 (p3 ~0.40 est. from 0.775/0.558); cost: 3rd draft step (~1.5 ms bf16 head, ~0.9 ms FP8), M 48->64 (+~1 ms GEMM, hmar +~3 us x 132), GDN +1 state write (+~1.2 ms; ~0 with K1-a) | step 32.7 -> ~36.5: x 1.17 / 1.12 = **+4.6%** (more with K1-a) | memory: KV/graph sizes change (128K test needed); acceptance p3 is the riskiest number |
| K3-b draft count 2 -> 1 | tokens 2.33 -> 1.78; saves ~2.5 ms | loses | pruned |
| K3-c tree speculation | GDN state per branch -> state writes x branches; vLLM GDN spec kernel is chain-only | loses on state traffic | pruned |
| K3-d speculate the accept count to avoid the 3 state writes | that is K1-a by other means (guess accepted = all, recover by replay) | = K1-a | merged into K1-a |
| K3-e speculate on AR (begin next layer on a partial sum) | high-cardinality edge (p ~ 0) | — | pruned (Gate 2) |
| K3-f n-gram / prompt-lookup drafter in addition | needs exactness (fine) but tuning on benchmark forbidden; ShareGPT replies rarely copy the prompt | small, uncertain | low |

## Knobs 4 & 5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| K45-a TP=2 (today) vs PP=2 | PP: no per-layer AR (saves 6.2 ms/decode, ~40 ms/mixed) but each GPU idles half the time unless 2 micro-batches, which read every weight twice (M=24 x 2): +12 ms/decode | decode loses (+12 vs -6); mixed steps (compute-bound) would win | pruned for decode; asymmetric hybrid (TP decode / PP prefill) needs 2 weight layouts: memory pruned |
| K45-b DP=2 | 28.75 GB of weights > 24 GB | — | pruned (capacity) |
| K45-c **AR transport for M > 96 rows (mixed steps)** | today NCCL ring through host, 293 us per AR at 3.6 MB (2x the 145 us link napkin); a host-mapped AR like hmar but pipelined (write chunk i / peer reads chunk i, using both PCIe directions at once) for up to ~2048 rows | 132 x (293 -> ~170 us) = -16 ms/mixed step | **+4.3%** |
| K45-d AR for decode (hmar, 46 us at 0.49 MB) | pipeline write/read per sub-chunk so upstream (write) and downstream (read) overlap; today all blocks write, flag, then read | 46 -> ~32 us x 135 = -1.9 ms/decode | **+3.6%**; riskiest: flag/poll latency dominates at this size |
| K45-e place GDN states / KV on host | capacity is not binding at c=16 | — | pruned |
| K45-f vocab split of lm_head (today) vs row split | today fine | 0 | no site |
| K45-g asymmetric split (05 §4.1): MTP draft + lm_head imbalance | both GPUs do the same work (vocab split) | 0 | no site |

## Knob 6: Window (merge and overlap)
| site | candidate | napkin | verdict |
|---|---|---|---|
| K6-a **chunked prefill into decode steps (Sarathi, 06 §2.1)**: cap the tokens per step (e.g. 48 decode + 32-80 prefill rows) so prompts ride the decode steps' unused compute instead of separate ~120 ms mixed steps | total prompt rows 104.5K over ~2,300 decode steps = ~45 rows/step; a decode step at M~96 costs ~+3-5 ms (GEMMs stay memory-bound below ridge ~80-160 rows; AR 0.49 -> 1 MB) | removes ~300 mixed steps x ~(120 - 37) ms = ~25 s, adds ~2,300 x 4 ms = ~9 s: **~-16 s of 112 (+14%)** if the small mixed step costs what its napkin says. BUT measured small-P mixed steps cost 72-122 ms (not ~37): with today's mixed-step overheads it LOSES. | the winning condition is "a mixed step with ~50 prefill rows costs < ~40 ms", i.e. fix the mixed-step fixed costs first (K45-c, K6-c) |
| K6-b dual-batch overlap in mixed steps (06 §3): split rows into 2 micro-batches, AR of one overlaps GEMM of the other | mixed GEMMs are compute-bound (M~350): re-reading weights for 2 halves costs ~+12 ms of memory time but it hides under compute; hides up to min(AR, GEMM) ~ 38 ms | **up to +10%**, realistic +5% | vLLM ubatching path is built for DP/EP; hmar disabled under ubatching; complex |
| K6-c **mixed-step fixed overhead** (GPU idle 5-68 ms, piecewise graphs with eager attention/GDN, ~45-85 ms excess for small P) | cause being analysed (agents/leader-4.w1/mixed.md) | if half of a ~50 ms excess goes: -25 ms x 300 = 7.5 s -> **+7%** | top priority to understand |
| K6-d decode idle 1 ms/step | CPU bookkeeping between graphs | -0.5 ms -> +1% | low |
| K6-e overlap draft step with target (draft of half-batch during the other's verify) | same weights? no (different layers): HBM-bound on both -> no gain at M=48 (Gate 1) | 0 | pruned |
| K6-f L2 prefetch extension (T3's site) to qkv/out_proj/down and lm_head | AR windows: ~46 us each, HBM idle; 16 MiB measured optimum per GEMM | ~0.3 ms | owned by T3's line; low |
| K6-g merge lm_head target + draft step 1 window | serial (draft needs sampled token) | — | illegal (dependency) |

## Groups dropped and why (summary)
- Anything lowering precision of weights/KV/activations beyond the W8A8 rule, or communication: contract.
- PP / DP / disaggregation: capacity (DP) or doubled weight reads at M=48 (PP, DBO in decode).
- Tree speculation: GDN state per branch.
- Approximate variants (early exit, sparse attention, quantized AR): forbidden by the contract (marked
  approx, gate = "not allowed", not just accuracy).

## Napkin ranking (before comparing with others)
1. K6-c/K45-c/K6-a mixed-step cluster: +4% (AR transport alone) to +14% (all of it).
2. K3-a 3 drafts: +4.6% (and K1-a makes it cheaper).
3. K45-d pipelined hmar for decode: +3.6%.
4. K1-a GDN DeltaLog state writes: +2.6%.
5. K2-b reduced-vocab draft head: +2.1% on top of T3-003.
