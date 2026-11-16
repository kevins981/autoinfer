# T5 sweep, step 4: knob by knob

Gains are napkin estimates against the champion's measured 106.6 s per 400-prompt trial (3-calibration.md):
decode 2,368 steps x 28.2 ms = 66.8 s, mixed 371 steps x ~107 ms = 39.8 s. "Host" = the CPU resource added in
3-calibration C (the framework's window cost has F, BW, Link only; on this machine a 4th resource, the host's
op-issue rate, bounds the mixed steps).

## Knob 1: Keep / Recompute
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| GDN state S[r,hv] across verify rows | **K1 DeltaLog (02 §4.3 Row B)**: keep a checkpoint S at the last accepted position + the 3 rows' inputs (k, v, decay, beta: ~1 KB/row/head vs 64 KB/state); next step replays the accepted rows in-kernel, then writes one checkpoint | writes 3 -> 1 state per req/head: measured GDN 3.69 ms/step -> ~1.8 ms (read+write floor, 2.4 GB at ~1.3 TB/s apparent): -1.9 ms x 2,368 = 4.5 s (**+4.2%**) | candidate; risk: vLLM's mamba 'align' prefix-cache mode and the state-slot/num_accepted plumbing (prefill writes slot 0, block-boundary copies) must see exact states; heavy |
| GDN state, Row C (keep raw k,v, no S for drafts) | same saving as K1 for spec rows | dominated by K1 (K1 keeps one S anyway) | merged into K1 |
| KV cache (append-only) | budget/precision fixed; no preemptions in the champion | 0 | drop |
| Draft hidden state | already kept (MTP reads target hidden) | 0 | drop |
| logits / attention scores across steps | Rule 2 (contain t) | illegal | drop |
| Host-side metadata across steps (persistent buffers instead of per-step tensor construction) | part of the host lever H1 | see H1 | folded into H1 |

## Knob 2: Rewrite
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| GDN prefill form (03 §4.2 dual form) | **R1 state (recurrent) form for short prefill segments**: one fused recurrent launch per layer instead of ~10 FLA chunk kernels when the prefill segment is short (p <~ 128); exact up to fp reordering | host: ~9 launches x 47 layers x ~37 us = ~16 ms per mixed step; GPU: recurrent over p rows ~1-3 ms/step extra vs chunk. On host-bound steps (~2/3 of mixed): ~250 x 14 ms = 3.5 s (**+3%**) | candidate (part of the host family) |
| GDN mixed spec/non-spec glue (index_select of qkv/a/b for spec and non-spec rows, scatter of outputs, transposes, cat) | **R2 fuse the glue**: kernels read rows through the index arrays / one varlen kernel handles spec rows and prefill rows; drop ~13 gather/index/elementwise launches per layer | ~13 x 47 x ~37 us = ~23 ms per host-bound mixed step: ~250 x 20 ms = 5 s (**+4.7%**) | candidate (host family) |
| Draft lm_head vocab (ruling 2) | R3 reduced-vocab draft head (e.g. 32K most frequent ids from non-benchmark text) | 875 us -> ~120 us per draft step: -1.5 ms/step = 3.6 s (+3.4%) minus acceptance loss (each -1% acceptance ~ -0.4% throughput) | candidate, acceptance risk; needs a generic corpus |
| Target lm_head, norms, attention math | fixed by the contract | — | drop |
| GEMM tile configs (decode M=48, 0.66-0.81 of BW) | not an algebraic rewrite; implementation | up to ~3 ms/step (+6%) in theory | low confidence, already tuned by the champion's authors |

## Knob 3: Speculate
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Next token x[t+1] (chain length) | **S1 3 drafts** | tokens/step 2.34 -> ~2.7 (pos-3 acceptance ~0.65 x 0.56 = ~0.36); step cost +GDN writes 1.2 ms + 1 draft step 1.4 ms + AR/attention +0.5 ms = +3.1 ms (+11%) -> net ~+4% on decode, mixed steps unchanged: **+2-3%**; with K1 the GDN term vanishes (+1%) | candidate, measure; acceptance unknown |
| Tree drafts | GDN in-place state (02 §4.4): per-branch state copies | prune: branch cost ~1.6 MB/req/layer per branch | drop |
| Expert / layer outputs / AR payload | no low-cardinality edge (dense model, high-cardinality activations) | — | drop (Gate 2) |

## Knob 4/5: Split / Merge, Place
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| Layer index l across GPUs (PP=2) | removes 128 ARs (6.2 ms) | at c=16 each stage must run 2 micro-batches -> weights read twice per token: +12 ms per 16-request step | drop (dominated) |
| Asymmetric TP / PP split (05 §4.1) | no asymmetry here (both ranks hold the same share incl. vocab-split heads) | 0 | drop |
| Drafter on one GPU (TP=1) | saves ~4 drafter ARs (~0.3 ms) | doubles drafter weight reads on that GPU (+0.85 ms) | drop |
| Split r inside a decode step (overlap AR of half 1 with GEMM of half 2) | Window mech. 2 | shared-read rule: weights read twice at M=48 (+15 ms) | drop |
| **P1 AR rows > 96 (mixed steps) via the host-mapped path** | split the payload into <=96-row chunks or raise MAX_ROWS | 130 ARs x (105 -> ~60 us) = -6 ms GPU per mixed step; pays only on GPU-bound mixed steps (p > 350: ~1/3 of mixed): ~120 x 6 ms = 0.7 s (+0.7%); more after the host fix | candidate, secondary |
| Host work split across processes (each TP rank already has its own process) | — | — | drop |

## Knob 6: Window
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| **W1 L2 prefetch volume during each decode AR** (champion: 16 MB of the next GEMM) | 24-32 MB: the 47 us AR could stream ~40 MB | gate_up 89 MB at 0.75 TB/s; prefetching 8-16 MB more could save ~10-15 us x 64 = 0.6-1 ms/step (+1-2%); L2 is 72 MB | cheap screen (env var VLLM_T3_L2_PREFETCH_MB, no code) |
| W2 admission window for prefills (merge 2 arriving prompts into 1 mixed step) | halves host floors for merged pairs | -~70 ms per merged pair, but waiting slots idle (~3%) and TTFT grows; Policy layer, validity question (behavior must not be keyed to the benchmark) | drop for now (ask manager only if the host fix fails) |
| W3 FULL CUDA graph for mixed steps | removes the host floor entirely | GDN prefill kernels have data-dependent grids (chunk indices); needs bucketed padding per prefill length: large engineering | drop (same target as R1/R2, cheaper there) |
| Decode launch gaps | 0.15 ms/step idle | ~0 | drop |
| Drafter overlap with the next target step | serial edge x̂ -> next step | illegal | drop |

## Summary: the families
1. **Host family (R2 + R1, plus trimming other per-layer host ops)**: the mixed-step host floor (~110 ms vs GPU 50-90 ms). Combined napkin +5-10%; the largest untouched resource in my calibration.
2. **K1 DeltaLog GDN state**: +4%, heavy.
3. **S1 3 drafts**: +2-3%, flag-level, measure.
4. **R3 reduced-vocab draft head**: +3% minus acceptance loss.
5. **W1 L2 prefetch volume**: +1-2%, env var.
6. **P1 hmar for >96 rows**: +0.7% now, more after (1).
