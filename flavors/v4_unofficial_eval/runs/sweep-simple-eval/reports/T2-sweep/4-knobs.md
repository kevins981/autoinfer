# T2 sweep, step 4: knob by knob

Reference: BASE official 143.4 s for the benchmark; calibrated model (3-calibration.md): decode steps
97 s (39.9 ms x 2,430), prefill fixed excess 19.3 s (48 ms x ~400), prompt rows 25.4 s (0.244 ms/row).
"Gain" = napkin seconds saved on the benchmark (and % of 143.4 s). Resources: F, BW, Link, C, plus
**Host** (CPU launch/step time and the cross-rank skew it causes; added in 3-calibration).

## Knob 1: Keep / Recompute (data nodes across steps)
| id | site | choice vs BASE | napkin gain | verdict |
|---|---|---|---|---|
| K1 | GDN state S under MTP verify | BASE writes S for all 3 speculative positions (in-place keep x3 slots). Row B of `02` §4.3 ("rollback-free verify"): write only the accepted position's S. Variant (a): keep S[t] + the 3 positions' (q,k,v,decay,beta) (~4 KB/head), and at the next step replay the `accepted` tokens from S[t] in SRAM before the new tokens (one read + one write of S per step). Variant (b): verify kernel keeps all 3 states in registers/smem and writes one after acceptance is known: impossible in the same kernel (acceptance needs the lm_head) | napkin: -2 x 25 MB writes x 48 layers = -2.4 ms/step -> 5.8 s (4%). But measured GDN time (3.66 ms) is already below the BASE napkin (4.8 ms), so the real write cost is unclear: **realistic 1-2 ms/step, 2.4-4.9 s (2-3%)** | candidate; exact (same fp32 ops in the same order). Cheapest test: ncu dram__bytes_write of gdn_decode_post_conv_mtp_kernel |
| K2 | GDN state dtype | fp32 is the model's `mamba_ssm_dtype`; bf16 would halve bytes but is lossy | (would be ~2 ms) | **dropped: contract** (not FP8-scheme, lossy) |
| K3 | KV cache, conv state | FP8 KV fixed; conv state ~25 KB | ~0 | dropped: nothing left |
| K4 | prefix cache | on (generic); ShareGPT prompts share only the chat template | ~0 | dropped |
| K5 | keep target logits / hidden for the drafter | MTP already reuses the target hidden states | 0 | dropped (already kept) |

## Knob 2: Rewrite
| id | site | rewrite | napkin gain | verdict |
|---|---|---|---|---|
| R1 | argmax over the vocab-split logits (target verify + 2 draft steps) | push the max/argmax reduction below the all-gather: each rank computes (max, first argmax) of its shard; exchange 48 (value,index) pairs, combine with (max, lowest index). Only for all-greedy batches (fallback: gather when any request needs the distribution/logprobs) | removes AG 0.95 + 2 x 0.33 ms + ~40 us elementwise = 1.6 ms/step -> 3.9 s (2.7%) | candidate; exact (same argmax incl. tie rule) |
| R2 | rmsnorm after each all-reduce | factor the per-row 1/rms out of the next GEMM (`(x*g/rms)W = (1/rms)((x*g)W)`) so the next GEMM can consume all-reduced chunks of d before the whole row is reduced | enabler for W3 (overlap with the next GEMM) | kept as enabler only (fp reorder; complex) |
| R3 | fuse all-reduce + residual add + rmsnorm | BASE disables fuse_allreduce_rms (FlashInfer AR unsupported at world 2) | -0.48 ms norms + launches = ~1.2 s (0.8%) | low; folds into W4 if we write our own AR |
| R4 | draft lm_head precision | FP8 draft lm_head (ruling 1): 1.27 -> 0.64 GB per draft step | -1.27 ms/step -> 3.1 s (2.2%), acceptance may drop slightly | candidate |
| R5 | draft lm_head vocab | reduced-vocab draft head (ruling 2), vocab = first N ids of the tokenizer (BPE merge order ~ generic frequency: data-free, not fitted to the benchmark). N=32K: 0.16 GB per step bf16 | -2.2 ms/step -> 5.3 s (3.7%) minus acceptance loss (target argmax outside the subset -> draft wrong) | candidate; combine with R4 |
| R6 | prefill/mixed GEMMs (M > ~100) | W8A8 block-FP8 (the checkpoint's own scheme, allowed per batch size): FP8 MMA 330 vs bf16 165 TFLOP/s | GEMM per-row 0.178 -> ~0.095 ms -> 104K rows x 0.083 = 8.6 s (6%) | candidate; needs a good sm89 FP8 block-scaled GEMM; GSM8K gate |
| R7 | decode GEMMs W8A8 | memory-bound: no byte saving | only if a W8A8 kernel streams weights better than Marlin (implementation) | merged into R8 |
| R8 | (implementation under Rewrite) Marlin decode efficiency | in-situ 68% of BW (ncu isolated 77-80%); at 90%: 18.1 -> 13.6 ms | -4.5 ms/step -> 10.9 s (7.6%) | candidate, high effort (kernel tuning: tile config for M=48, stream-K, fewer partial-tile waves) |
| R9 | merge in_proj_qkvz + in_proj_ba (same input x) | one GEMM, also merges the small bf16 GEMM launch | ~0.3 ms/step -> 0.7 s (0.5%) | low |

## Knob 3: Speculate (two gates of `04`)
| id | edge | gates | napkin | verdict |
|---|---|---|---|---|
| S1 | t -> t+1 chain length k=2 -> 3 | both pass (shared W, token cardinality) | tokens/step 2.345 -> ~2.75 (p3 ~ 0.57 x 0.7 = 0.40, to measure); step cost +~6 ms (3rd draft step 2.3 ms incl. a full lm_head read, 64 verify rows: GDN writes +1 slot, AR +33% bytes, Marlin m-tile 4) -> net ~+2% now; **+8-10% if R4/R5 make a draft step ~1 ms** | candidate, ordered after R4/R5; memory change -> 128K test |
| S2 | tree speculation (2 branches at depth 1) | Gate 1 passes on weights, but each GDN branch needs its own S copy (the `02` §4.4 rollback trap): +1.57 MB x 48 layers per branch per request | +25 MB/layer writes per extra branch ~ +1.2 ms; rows 48 -> 64+ | dropped for now: needs a branch-aware GDN verify kernel (K1 first) |
| S3 | all-reduce result, layer outputs | high cardinality, p=0 | - | dropped (Gate 2) |
| S4 | prompt-lookup / n-gram drafts | exact, but ShareGPT replies copy little from prompts | low p | dropped |
| S5 | host step t+1 prep (scheduler output) | Speculate/overlap the host's input prep for step t+1 with GPU step t (async scheduling) | GPU idle 1.3 ms/step -> 3.2 s (2.2%) | candidate if async scheduling is not already on with spec decode (check) |

## Knobs 4/5: Split / Merge and Place
| id | split | rules | napkin | verdict |
|---|---|---|---|---|
| P1 | DP=2 (split r) | no comm | needs 24.4 GB fp8 + 2.5 GB lm_head per GPU | **dead: capacity C** |
| P2 | PP=2 (place layers 1-32 / 33-64) | 1 transfer per microbatch | to keep both GPUs busy at decode, 2 microbatches -> each GPU reads its 12.2 GB twice per step: 2 x 18.1 = 36 ms GEMM vs TP 18.1 + 8.2 AR = 26.3 ms | dead while AR < weight time (dominance: Shared-Read rule across microbatch windows); breaks only if AR cost > ~18 ms/step |
| P3 | asymmetric TP (alpha != 0.5) | | ranks do identical work (lm_head also split) | dead (no asymmetry to balance) |
| P4 | context parallel over j | | KV is 0.1 GB/step | dead |
| P5 | TP split of lm_head by v (BASE) | combine = gather | keep; the gather is removed by R1 | - |
| PL1 | KV / GDN state / weights to CPU | Link-bound | no capacity pressure (KV fixed 5 GiB) | dead |

## Knob 6: Window (incl. the Host resource)
| id | site | move | napkin | verdict |
|---|---|---|---|---|
| W1 | mixed prefill steps (> 96 rows run eager) | capture prefill/mixed CUDA graphs up to 2048 tokens (`max_prefill_cudagraph_capture_tokens`): one graph launch instead of ~1,500 eager launches; ranks stay in lockstep so the 135 all-reduces stop waiting | fixed excess 48 ms -> ~5 ms per prefill step: **-17 s (12%)** | **top candidate; testing now (t2-cg2048)**; memory change -> 128K test |
| W2 | prefill chunk size <= graph size | alternative to W1: chunks of <= 48 prompt rows so every step is graph-captured | marginal cost per prompt row ~0.1 ms + 5 more steps per prompt; TTFT +~200 ms | dominated by W1 if W1 works |
| W3 | all-reduce || producing GEMM | split the GEMM output (d) into chunks; all-reduce chunk c while computing c+1 (distinct weight columns: no re-read) | hides most of 8.2 ms -> ~3 ms: -5 ms/step -> 12 s (8.5%) | candidate, high effort (two streams inside graphs, SM sharing with NCCL) |
| W4 | (implementation) all-reduce cost itself | NCCL protocol/channels, or a custom all-reduce through pinned host memory (one-shot: each rank writes 0.5 MB, reads the peer's) | 60 -> ~25-30 us x 135 = -4 ms/step -> 9.7 s (6.8%) | **candidate; worker microbenchmark running** |
| W5 | dual-batch overlap (split r, AR(R1) || GEMM(R2)) | Shared-Read rule: weights read twice | +18 ms GEMM vs -8 ms AR | dead at N=48 |
| W6 | host gaps at step start | see S5 | 1.3 ms/step | candidate (S5) |

## Pruned groups, summary
Capacity kills DP; weight re-reads kill PP and DBO at this batch; the contract kills a bf16 GDN state
and lower-precision anything; Gate 2 kills speculation on continuous tensors. What remains is
(1) Host-resource waste in mixed steps (W1), (2) communication (W4, W3, R1), (3) bytes per step on the
draft side (R4, R5, then S1), (4) compute rate of prefill GEMMs (R6), (5) kernel efficiency (R8, K1).
