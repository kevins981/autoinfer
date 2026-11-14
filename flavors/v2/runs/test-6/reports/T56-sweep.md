# T56 sweep: the framework (framework/01-07) applied to the champion T52-005

leader-56, 2026-11-13. Written before any build. Evidence: job 483 (fresh nsys of T52-005, 160 prompts,
agents/leader-56/prof_champ/), the T37-era trace agents/leader-38/prof_champ_node, shared_kb.md, bank/.
Scripts: agents/leader-56/step_dump.py (one step's kernel sequence).

## 1. Inputs

### Model (config.json, Qwen3.8-27B-FP8), per GPU at TP=2
| node | shape (full) | bytes/GPU | per decode step (M=64) |
|---|---|---|---|
| GDN layer x48: qkvz | 5120 -> 16384 (q,k 16x128; v,z 48x128) fp8 | 41.9 MB | |
| ba | 5120 -> 96, bf16 | 0.5 MB | |
| out_proj | 6144 -> 5120 fp8 | 15.7 MB | AR #1 (M x 5120 bf16) |
| gate_up / down | 5120 -> 2x17408 / 17408 -> 5120 fp8 | 89.1 / 44.6 MB | AR #2 |
| GDN state S | 24 v-heads/GPU x 128 x 128 fp32 per request | 1.57 MB x 16 req = 25 MB/layer | read every step |
| Attention layer x16 (24 q / 4 kv heads, dim 256, output gate) | qkv 5120 -> 14336, o 6144 -> 5120, same MLP | 186 MB | 2 ARs |
| target lm_head | 248320 x 5120 bf16 (certified 7-bit plane in champion) | 1.27 GB (plane 0.57) | |
| embedding | 248320 x 5120 bf16, vocab-split | 1.27 GB | 1 AR |
| MTP drafter (1 layer + fc 10240->5120 bf16 + PCA-screened FP8 draft lm_head 98K) | | ~238 MB weights | 3 levels x 4 ARs |
Body weights 12.18 GB/GPU; 24.4 GFLOP per token per GPU.

Indices: r (16 requests), l (64; GDN/attention alternate 3:1), i/t (positions; k=3 drafts -> 4 rows per request
per step), h (q/v heads split by TP), d=5120 (the narrowest cut in a layer: the residual stream), m (17408 MLP,
split by TP), vocab V (split by TP). Lookup indices: token ids only (no MoE).

### Hardware (per GPU; measured values from the KB where they exist)
DRAM 957 GB/s (KB129; P2 memory clock); L2 72 MB, ~3.4 TB/s (KB97); 128 SMs; bf16 MMA 146-165 TF; e4m3 MMA
291-329 TF peak, ~280-296 TF achieved with the mandatory per-128-K fp32 promotion at M>=1024, 136-271 at M=128-512
(KB130). Link: no P2P; PCIe 4 x16 per GPU via host: copy engines ~21-24 GB/s per direction and full duplex
(45 GB/s combined), 20-35 us start latency inside graphs (KB73/74); SM-driven host traffic ~22 GB/s total
whatever the direction mix unless paced (KB109/115), and it slows beside HBM-heavy kernels (KB62/66).
Ridge (rows per weight load): bf16 path 146/(2*0.957) ~ 76 rows; e4m3 path ~146 rows.

### Workload and objective
400 ShareGPT requests, closed loop c=16, prompt ~260 tokens incl. template, output ~228; temperature 0; MTP k=3,
acceptance 2.66 (official). Objective: output tok/s; contract in task_definition.md (exact math up to fp
reordering; checkpoint W8A8 allowed; comm at baseline precision; drafter-only changes may be approximate since
verification is exact).

## 2. Calibration: napkin vs the measured champion (job 483, GPU0, under nsys: 1068 tok/s vs 1212 official)
Wall split: decode-only steps 66% (863 steps, median 24.7-24.9 ms at 64 rows), mixed steps 34% (138 steps).

Decode step 24.7 ms (one step dumped kernel by kernel):
| part | measured | napkin floor | how the floor is priced |
|---|---|---|---|
| body GEMMs (incl. qkvz on the side stream) | ~12.3 ms | 12.7 ms | 12.18 GB / 957 GB/s (AR-window L2 prefetch moves ~0.4 ms into AR windows) |
| 128 ARs + embed AR | 6.66 ms | 6.0 ms | 128 x 2 x 511 KB (0.78 codec) / 22 GB/s, SM path, write then read |
| GDN conv + replay | 2.0 ms | 1.26 ms | 1.21 GB state read |
| small ops (455 kernels) + attention | 1.3 ms | ~0.5 | |
| target lm_head (certified screen + exact rows) | 0.75 ms | 0.6 | 0.57 GB plane |
| drafter, 3 levels + bookkeeping | 2.06 ms | 0.71 + ARs | 238 MB x 3; 4 ARs/level ~100 us |
| total | 24.7 | ~22.8 | napkin within 8% |

Mixed steps: GPU busy 45 ms at <128 prompt tokens (64 of 138 steps), 64-72 ms at 256-511, 120-138 ms at 768-1151;
period fit 48.7 ms + 0.111 ms/prompt token (nsys-inflated periods). Mixed excess over decode = 8.2 s of 33.9 s
(24% of wall) for 43.7K prompt tokens: **~0.19 ms per prompt token all-in** (~0.165 scaled to the official speed).
Per-token floors: compute 0.087 ms (24.4 GFLOP at 280 TF), link 0.05 (copy-engine duplex) to 0.09 ms (SM path).
Small mixed step (60 prompt + 60 decode rows, M=120), kernel by kernel: 256 bs_gemm 20.4 ms (vs 12.3 in decode; at
M=128 the DRAM floor 12.7 ms and the e4m3 compute floor ~11 ms coincide), 137 hostar 11.6 ms (vs 6.7), GDN chunked
prefill ~2 ms, rest glue.

The framework's Window cost (max of F, BW, Link inside a window) does NOT reproduce this machine without one
correction: Link and DRAM are coupled when the link is driven by SMs (host-mapped loads/stores queue behind HBM
traffic: KB62, KB66, KB97 "1.2 us of AR per MB prefetched beyond 32 MB"). Only copy-engine traffic is nearly
independent of DRAM (x1.0-1.12, KB66), and copy engines pay 20-35 us start latency in graphs. So for decode I price
AR and DRAM additively (that is what reproduces 24.7 ms); the max() form would predict ~17 ms.

## 3. Sweep, knob by knob (status vs run history; "built" = in the champion)

### Knob 1 Keep / Recompute
| candidate | verdict | status |
|---|---|---|
| keep k,v (FP8) | fixed by contract | built |
| keep S (fp32) vs DeltaLog (S[t-L] + inputs) vs rollback-free verify | DeltaLog + replay + lazy checkpoint + two-slot | built (T1-012, T1-035, T37) |
| Row C: keep raw k,v instead of S for short contexts | crossover ~196 tokens/request/layer (8 KB/token vs 1.57 MB state per GPU); requests are 300-500 tokens; gated delta rule's attention form needs the WY/UT triangular solve; <=~0.5 ms/step for the early part of requests | untested; pruned (small, complex) |
| embedding on GPU vs host tier | host zero-copy gather 64 rows 31 us vs ~50 us embed AR; frees 1.27 GB/GPU | microbench only (T46); not built |
| **drafter: keep a per-token table T[v] = W_fc,e . norm(E[v]) (Rewrite by distributivity over the fc concat + Keep on host)** | removes the drafter's embed lookup + embed AR and half of its fc GEMM: ~55-90 us per level, ~0.2 ms/step (~0.5-0.6% e2e); drafter-only, so rounding of the table only touches proposals | **untested (new)** |
| prefix state cache (shared system prompt) | +0.5% n=1 | tested (T21-002, KB86), not built |
| keep activations / logits across steps | Rule 1: no future readers | pruned |

### Knob 2 Rewrite
| candidate | verdict | status |
|---|---|---|
| 1/rms past the next linear (bf16 and W8A8) | rulings 11-09/11-11 | used in T52-002/T53 (parity) |
| distribute the AR sum into the consumer GEMM (consume partials as they arrive) | parity in situ | tested (T52-002, T53) |
| GDN dual form for the 4-row verify (attention form from S[t] in SRAM) | this is the replay kernel | built |
| associativity in lm_heads (screen then exact rescoring) | | built (T27 draft, T42 target) |
| replicate out_proj/o_proj and all-gather their FP8 W8A8 inputs (3.2 KB/row) instead of AR'ing bf16 outputs (8 KB/row) | 64 of 128 ARs carry 2.5x fewer bytes; +6% prefill FLOPs, +1 GB/GPU; decode ~wash (T47 napkin), prefill ~+1-2.5% | untested; NEEDS A RULING (link carries FP8 GEMM inputs, though they are bitwise what the GEMM consumes) |
| subgraph dedup across requests | only the 45-token system prompt is shared | tested (T21) |

### Knob 3 Speculate
| candidate | verdict | status |
|---|---|---|
| tokens, chain k=3 | | built |
| k=4 | +10.2% oracle tokens (KB127) vs +16 rows: gate_up m-tile step +37 us/layer, t31b +9, AR ~1.2 ms, 4th level 0.54, replay +0.3: ~+5 ms/step (+20%) | priced, loses; needs a 16-row-granular gate_up and still loses (+14% for +10%) |
| trees | -6..-8% at measured row prices (KB135) | priced, loses |
| deeper k only in the drain (fewer running requests -> cheap rows) | drain ~2-3% of slot time; <=0.5% | untested, small |
| layer outputs / AR outputs (high cardinality) | Gate 2 fails exactly; early-rejection signal weak (KB114) | pruned/tested |
| verify outcome -> start the next draft early | draft level 1 needs the target's final hidden: dependency is real | pruned |
| stage-speculative PP | PP decode r=0.91 (KB112) | tested |

### Knobs 4/5 Split, Merge, Place
| candidate | verdict | status |
|---|---|---|
| TP=2 symmetric | narrowest cut d=5120; 2 ARs/layer; AR = RS+AG bytes, no 2-rank scheme moves fewer bytes | built |
| PP=2 / PP for prefill only | decode r 0.91 (KB112); PP-prefill + TP-decode needs 1.5x weights | tested / pruned (memory) |
| prefill/decode or attention/FFN disaggregation across the 2 GPUs | full model 24.4 GB FP8 > 24 GB: cannot place on one GPU | pruned (capacity) |
| asymmetric alpha | GPU clocks now symmetric (KB113); uneven split negative (KB82) | tested |
| replicate the drafter (no drafter ARs) | saves ~0.3 ms/step of ARs, costs +0.71 ms of weight reads | pruned |
| split r into 2 cohorts on one GPU (AR of one under GEMM of other) | weights read twice at M=32; 1.31-1.47x (KB132) | tested, loses |
| weights partly on host (link as extra bandwidth) | ~2.4% of weight bandwidth, ~1% ceiling (KB125) | tested (microbench) |
| CPU as the reducer / RS-AG through host | same per-GPU link bytes (S up + S down) | pruned |
| **SM-level Place: decode on 128-f SMs, prefill on f SMs (spatial lane)** | see section 4: the only candidate with a >5% napkin; T2's no-go was measured with a bf16-Marlin lane and 128-SM decode grids, both since changed | **re-test warranted** |

### Knob 6 Window
| candidate | verdict | status |
|---|---|---|
| merge decode rows; chunked prefill joined with decode | | built |
| two-wave layer pipelining of big mixed steps + link priority | | built (>=768 rows) |
| lower the wave threshold (capture GDN/attention cores per wave size) | ~+1% est., memory-limited (0.45 GiB free) | open (T52) |
| AR window || next GEMM's weight prefetch | partial: Link-DRAM coupling caps it ~32 MB/window | built |
| prefill GEMM slabs in decode AR windows | chain +20-33% | tested (T11) |
| decode wave beside prompt wave inside a mixed step | overlap 0.43-0.54 vs 0.5 needed | tested (T47) |
| **GEMM-internal Window: weight stream || e4m3 MMA at M=96-192 (mixed steps sit at the ridge, where both floors bind)** | bs_gemm M=120 costs 20.4 ms/step vs ~13 ms max(DRAM, MMA); ~5-8 ms per mixed step -> ~2-3.5% e2e if half recovered | tested only as configs/split-K (T50-001, T51): "needs a kernel change"; untested as a kernel |
| coalesce prompts into fewer, larger mixed steps | slot dead time ~= savings | tested (KB77) |

## 4. Ranked candidates (gain vs the champion's measured time; official trial ~75.2 s)

1. **Spatial prefill lane on today's FP8 prefill path (Place on SMs + Window Mechanism 2).**
   - Mechanism: new prompts prefill on f SMs (green context) while decode steps keep running on 128-f SMs; mixed steps
     disappear. Win condition: cost per prompt token in lane mode, (1 - D) / r_lane (D = decode speed fraction while
     the lane runs, r_lane = lane prompt tokens/s), below the joined mixed step's ~0.165 ms/token all-in.
   - Napkin: T2 measured D = 0.61 at f=48 with a bf16-Marlin lane at 36 TF/rank (1475 tok/s): 0.26 ms/token, a loss.
     The lane now runs W8A8 bs_gemm (1.5-2.1 TF/SM at M=256 vs ~0.75): r_lane ~2.9-4K tok/s -> 0.10-0.13 ms/token
     if D holds -> saves 20-40% of the mixed excess (24% of wall): **+3% to +8%**. With decode kernels sized for
     128-f SMs (KB129: DRAM saturates from 16-32 SMs; today's grids step up when they spill a wave) D could rise.
   - Riskiest assumption: that a 2-3x faster lane does not lower D (it streams prefill weights at ~150-200 GB/s,
     15-20% of DRAM, vs ~70 GB/s for T2's lane; T2 found in-server interference 3x the microbench).
   - Cheapest test: port T2's in-server synthetic lane (bank/T2-002: real per-rank linear modules of all 64 layers at
     M=256 on a green-context stream + copy-engine AR volume) onto T52-005; the real modules now dispatch to W8A8.
     Measure decode ITL and lane TF at f=32/48/64 vs off on one server with phases (~10 GPU-min).
     Go threshold (after sparring T56-1): the synthetic lane's r is GEMM-only (no dependent ARs, GDN core, attention),
     so a real lane runs ~25% slower unless its ARs are covered by in-prompt waves (cross-prompt pipelining is not
     available: 126/138 mixed steps carry one prompt). Go if GEMM-only (1-D)/r <= ~0.11 ms/token; a failed probe
     kills only this arrangement (decode on full-size grids), not resized decode kernels.
2. **Mid-M GEMM (M=96-192) at max(DRAM, e4m3 MMA) for mixed steps.** ~5-8 ms per mixed step of excess; +2-3.5%
   if half recovered. Wins if a warp-specialised stream+MMA kernel (T33's design, KB101) holds >=85% of the e4m3
   roof at M=128 with per-128-K promotion. Riskiest: T51/KB131 found math-pipe throttle at 74% of roof on padded
   rows. Cheapest test: standalone T33-style kernel at M=128 on gate_up (one shape) vs bs_gemm, cold, read flush.
   Heavy kernel work; better suited to a GEMM-focused team.
3. **Drafter fc/embedding table (new).** ~0.2 ms/step, ~+0.5% e2e. Exact for outputs (drafter only); costs 2.5 GB
   pinned host memory. Riskiest: zero-copy gather latency of 16-64 rows on the critical path (T46: 11-31 us).
   Cheapest test: in-server timing of one draft level with the table (A/B within one job).
4. **Lower the wave threshold with freed memory** (host embedding 1.27 GB/GPU + KV pool ~2x oversized per T47) and
   cores captured per wave size: ~+1% (T52's estimate). Riskiest: 128K gate with a smaller KV pool.
5. Out_proj/o_proj replication + FP8 all-gather: needs a ruling; ~+1-2.5% prefill-only napkin.
Not worth testing now: k=4, trees, Row C, drain-only depth, replicated drafter, cohorts.

## 5. Where the framework didn't fit this machine
- Link and DRAM are not independent resources when the link is SM-driven (host-mapped loads/stores): the Window
  cost needs a coupling term (AR time grows ~1.2 us per MB of concurrent DRAM traffic beyond ~32 MB, KB97), or
  separate resources for copy-engine vs SM-driven link traffic, with a start-latency term for copy engines.
- The ridge "333 rows" is wrong here for exact math: the bf16 path ridges at ~76 rows, the checkpoint W8A8 path at
  ~146; and at the W8A8 ridge (M~128, i.e. every small mixed step) both floors bind at once, so the max() form
  under-prices real kernels, which need stream/MMA overlap inside the GEMM to approach it.
- SMs are a placeable resource (green contexts) that the framework has no index for; spatial sharing (candidate 1)
  is a Place on SMs + Window combination it can express only loosely.

## 6. Results after the analysis
- Candidate 1 (spatial lane), tested in-server (T56-001, job 489, 7 GPU-min):
  - Decode ITL is x1.28 / x1.39 / x1.52 at f=32/48/64, with the lane at 34 / 44.5 / 55 TF/rank under load.
  - The GEMM-only cost per lane token is 0.151-0.156 ms (0.137 at lane M=512), vs ~0.167 for the joined mixed step and a go threshold of 0.11.
  - With the real lane's dependent ARs, GDN work and slot wait, it is break-even to a loss: NO-GO.
  - T57-002 (microbench, the resized-grid branch) found the contention is DRAM/L2 arbitration, not SM count, so resized grids don't reopen it.
  - The framework's Window/Place view over-predicted here for the same reason as in decode: co-running DRAM-bound and compute-bound work is not max(F, BW) on this GPU (decode loses ~1/3 of its rate at f=48 while the lane reads <=20% of DRAM).
- Candidate 2 (mid-M GEMM), cheapest no-new-kernel variant tested (T56-002, job 495, 0.3 GPU-min): t31b in row slices of <=96 rows loses 66-84% per layer vs bs_gemm (no L2 reuse between slices; t31b at 96 rows is MMA-bound).
  - bs_gemm at M=112-128 is 291 us per GDN layer vs max(DRAM 200, MMA ~170) us, so a new kernel's ceiling is ~25% of mid-M GEMM time, ~+1-2% e2e: a kernel project, not a dispatch change.
  - My earlier ~13 ms 'floor' per forward for this band was optimistic (it ignored that DRAM and MMA floors nearly coincide here, so reaching either needs near-perfect overlap).
- Candidate 5 (replicated out_proj/o_proj + FP8 all-gather): sent by the manager to the human for a ruling; not built.
- Candidate 3 (drafter fc/embedding table): handed to T57, together with the drafter overhead.
- Sparring T56-1 (turns 6-7): two ARs per layer are required under TP (norm + full-d contraction after each); FP8 weight bytes are a contract floor and lossless coding was tried; decode stays closed. Remaining space: mixed steps.
- Candidate 5 (replicated out_proj/o_proj + FP8 all-gather; human ruling 2026-11-13 ALLOWED): S1 (T56-003, job 499, 0.2 GPU-min).
  - Bitwise premise holds.
  - Net per layer, AR minus AG minus the extra full-K GEMM: -10 / +10 / +55 / +109 us at M = 64 / 128 / 256 / 512.
  - The framework priced link bytes but not the duplicated compute the replication adds to the critical path at the row counts where steps actually sit.
  - ~+0.3-0.8% e2e for ~50 GPU-min plus memory work: not built.
