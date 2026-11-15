# T3 sweep, step 4: the sweep, knob by knob

Reference time (T2-008, the champion while this was written): official 880.1 tok/s = 103.3 s for
90.9K tokens. Split (from 3-calibration, T2-008 = T1-014 minus ~1.2 ms per decode step):
**decode-only steps ~2490 x ~30.3 ms = ~75.5 s (73%)**, **mixed steps ~234 x ~119 ms = ~27.8 s (27%)**.
Decode step (GPU span ~32.3 ms): body GEMMs 17.05, AR 6.56, lm_heads 3.1, GDN ~2.3, small
kernels 2.1, gaps 0.7, MTP layer 0.25.
"Gain" = napkin saving as a share of the whole benchmark time unless marked "decode".

Per-row cost (a framework number this machine changes): one more verify row in a decode step costs
~135 ARs x 20 KB / 22.6 GB/s = **~0.12 ms of link time** + GDN/attention ~0.01 ms + GEMM ~0 (weight-bound).
So on this machine the "ridge" for adding rows to a decode window is set by the link, not by
compute: 16 extra rows (one per request) cost ~2 ms.

## Knob 1: Keep / Recompute
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| GDN state `S[r,l]` (fp32, in-place) | Keep S + 3 row writes (T1-014) vs **Row B / DeltaLog: keep S after row 0 + a log of draft-row inputs, replay accepted rows next step** | 4 -> 2 state passes per request-layer: 16x48x3x1.57 MB = 3.6 GB/step less HBM traffic (~1.2-2.4 ms decode) | **Built by T2 (T2-008, the champion)**; I priced it independently (job 73: 120 -> 59 us per call isolated). |
| same | DeltaLog with checkpoint every L=2 steps (write S every other step, replay up to 2 steps of logged rows) | writes halve again: 16x48x1.57 MB/2 = 0.6 GB/step -> ~0.6 ms decode (~1.4%) | untried; medium-hard (T2's kernel + a 2-step log) |
| same | State in bf16 / fp16 | halves state bytes | **Illegal** (GDN math/state precision fixed by the contract) |
| KV cache (fp8) | keep k,v | ~0.1 GB live | fixed by contract; already minimal |
| Conv state | keep last 3 rows | tiny | nothing to gain |
| Prefix-cache mamba "align" copies | keep extra state copies at block boundaries | 15 us/decode step + copies in mixed steps | <0.2%: dropped |
| Activations / q | keep nothing else | Rule 1: no future readers | dead |

## Knob 2: Rewrite
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| norm(AR(P0+P1)+res) -> column GEMM | Distributivity: W(P0+P1) = W P0 + W P1 so each GPU multiplies its own partial | needs W's columns of the peer (2x weights or 2x reads) and RMSNorm is a nonlinear barrier (Rule 3) | dead |
| RMSNorm scale | pull rstd out of the GEMM (x*rstd)W = rstd (xW) to start the GEMM before the norm's reduction | the GEMM still needs the AR'd x; and the FP8 group quant of the normalized row needs the full row | dead (gains nothing: the AR, not the norm, is the blocker) |
| Draft lm_head `W_draft[d,v]` | **reduced-vocab draft head**: drop rarely-used vocab rows (only the draft; verify stays full-vocab, exact). Allowed by ruling 2 if the vocab subset comes from non-benchmark data (e.g. BPE merge rank = id order of the tokenizer, or a generic corpus) | draft lm_head reads 2 x 0.64 GB per step (1.53 ms measured). 32K vocab -> 2 x 0.08 GB: **saves ~1.3 ms decode (~4.3% decode, ~3.1% total)** minus acceptance loss (tokens outside the top-32K are rare in chat text: guess -0.5 to -1% acceptance) | **candidate** |
| Target lm_head | local argmax per shard (T1-012) | done | built (champion) |
| Elementwise chain per layer (act_and_mul -> quant; GDN out -> quant; norm -> quant) | fuse into GEMM epilogues / single kernels (Dedup of passes over the activation) | ~264 quant + 66 act + 135 norm kernels ~1.25 ms; fusing half of it ~0.5 ms decode (~1.2%) | candidate, low priority |
| GEMM order (associativity) | none at these shapes (no low-rank factors) | - | nothing |

## Knob 3: Speculate
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Next token (MTP chain) | k=2 (champion) -> **k=3** | +16 rows: AR +2.2 ms, draft step +0.9 ms, GEMM/GDN +~0.6 ms -> +3.7 ms on 30.3 (+12%); tokens 2.34 -> 2.34 + 0.56 p3 (p3 ~0.65: +0.36, +15%) | **marginal (+0-3%)**; cheap flag test, but memory change needs the 128K test |
| same | k=1 | -3.4 ms but tokens 2.34 -> 1.78 (-24%) | dead |
| same | tree (top-2 at depth 1, 5 rows/request) | +2 rows per request x 0.12 ms x 16 = +4 ms per step for +~0.15 tokens (top-2 hit gain) | dead here: rows cost link time (see per-row cost) |
| Acceptance outcome (3 values) | speculate the accepted length to start draft step 1 before verify ends (3 candidate positions in one draft window) | draft and target lm_head both HBM-bound: no orthogonal resource; saves only sync latency | dead |
| Better drafter (retrained MTP head on generic data) | raises p1,p2 | each +1% acceptance ~ +0.7% throughput; training cost and data provenance burden | out of our budget |
| AR result / hidden state | high cardinality | Gate 2 fails | dead |

## Knobs 4/5: Split / Merge and Place
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Layer index across GPUs | PP=2 (+2 micro-batches) | each GPU reads its 12.2 GB once per micro-batch: 24.4 ms/step vs TP 12.2 + 6.6 AR | dead in decode (weight-bound, shared-read rule); prefill would win but both layouts don't fit in memory |
| lm_head / drafter placement | target lm_head whole on GPU0, drafter whole on GPU1, concurrently | ~4.4 ms vs 3.8 ms now (the full fp8 draft head on one GPU is slower than half on each) | dead |
| Request split r inside a GPU | two request groups so one group's AR overlaps the other's GEMM | GEMM weights read twice at M=24 (weight-bound) except L2-sized ones; AR halves cost 33 us each vs 48.6 for the whole (latency floor) | dead in decode; it is the champion's row-split in mixed steps (compute-bound, so it wins there) |
| **Output index d of the row-parallel GEMMs (down, out_proj) + AR, at tile granularity** | split d into tiles; push each finished tile to the host buffer and reduce it while the GEMM computes the next tiles (fused GEMM -> AR pipeline) | per layer: down GEMM ~50 us + AR 48.6 -> ~max + tail ~55 us (save ~40 us); out_proj 25 + 48.6 -> ~54 (save ~20 us). x64 layers: **~3.8 ms decode (~12% decode, ~9% total)**; mixed steps gain too | **candidate (largest decode napkin), hard**: Triton GEMM epilogue writing to mapped host memory + per-tile flags + a consumer kernel; exact (same rank0+rank1 bf16 sum) |
| same, coarse | 2 column chunks as separate kernels + side-stream ARs | 2 ARs of half size cost 2 x ~29 us (latency floor): saves ~15 us per down AR -> ~1 ms decode (3%) | candidate, weaker version of the above |
| Head split for attention / GDN | TP over heads (current) | balanced | nothing |
| Context split j | context-parallel decode | contexts are short (~500) | dead |

## Knob 6: Window
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| **AR in mixed steps (messages > 1.31 MB go to NCCL)** | route all AR sizes through the hostmem kernel (env `VLLM_HOSTMEM_AR_MAX_BYTES`, no code) | in the server NCCL ran 2.6-2.7 MB in ~650 us vs 242 us isolated (hostmem 218-254 us isolated). If hostmem holds its isolated rate in the server: big mixed steps lose ~50-100 ms of exposed AR. **~+5-12% total** | **candidate; ABBA running (job 76)**. Risk: hostmem may be no faster in the server if the cause is link contention, not NCCL's CPU proxy |
| same | more row chunks (3-4) in mixed steps / lower ROWSPLIT_MIN_TOKENS | hides more AR once AR ~ GEMM time | follow-up after the above |
| AR (link) || next GEMM's weights -> L2 (HBM idle during the AR) | prefetch up to ~45 MB into the 72 MB L2 during each 48 us AR | qkvz (42 MB) and half of gate_up could be served from L2: up to ~30-50 us per layer -> 2-3 ms decode | uncertain: my isolated test (job 75) was confounded; the GEMM in L2 is still compute/latency-limited at M=48. Untried in the server |
| Chunk size (max-num-batched-tokens 2048, graphs up to 512 tokens) | cap mixed steps at 512 tokens so they stay in CUDA graphs | eager mixed steps show ~18 ms idle under nsys (inflated); unknown without a plain bench | cheap flag test |
| CPU/GPU overlap between decode steps | gaps 0.7 ms per step | <2% decode | low |
| GDN state traffic || GEMM (different requests) | GDN kernel per request group overlapping the next GEMM | both HBM-bound | dead |

## Implementation layer (not a knob, but the largest measured gap)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Decode GEMM efficiency (W8A8 Triton at M=48: 0.68-0.88 of 1 TB/s; body GEMMs 17.05 ms vs 12.2 napkin) | better tile configs for M=48 (e.g. one M-tile BLOCK_M=48/64, split-K, more stages), or a streaming FP8 kernel | gate_up to 0.88 TB/s: -2.2 ms; all shapes to 0.9: **-3.5 ms decode (~11% decode, ~8% total)** | **candidate; cheapest test: an isolated config sweep at M=48 (~2 GPU-min)** |
| lm_heads (0.8 of peak) | better GEMV | ~0.5 ms decode | low |

## Dropped groups and why
- Everything that adds verify rows (trees, k>=4) loses on this machine because each row costs
  ~0.12 ms of link time per step (the AR scales with rows).
- Everything that runs the same weights in two windows at M<=48 (PP micro-batches, request-split
  overlap in decode) loses by the shared-read rule; the L2 (72 MB) only rescues weights <= ~40 MB
  read back-to-back.
- Precision changes beyond the contract (state, comm, lm_head) are illegal.
