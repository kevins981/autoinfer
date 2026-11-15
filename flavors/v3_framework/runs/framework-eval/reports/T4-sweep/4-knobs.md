# T4 sweep, step 4: the sweep, knob by knob

Gains are napkin estimates against the champion's calibrated wall time (95.9 s for the 400-prompt benchmark; 3-calibration.md):
decode-only steps 54.9 s (2,070 x 26.5 ms), mixed steps 42.0 s (small P<211: 17.4 s, mid: 8.3 s, P>=467 eager: 16.3 s).
"+x%" = throughput gain if the saving is fully realized.

## Knob 1: Keep / Recompute (data nodes across steps)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| GDN state S[r,l,hv] (fp32, 1.57 MB/layer/request/GPU) | champion: row 0 writes full S every step, rows 1-2 log (DeltaLog). **ReplaySSM depth L**: keep checkpoint S[t-L] + a ring of L step logs; write full S only every L steps | GDN decode kernel 1.78 ms/step, ~half of its bytes are the write -> saves <= ~0.8 ms/step x (1-1/L) -> ~1.2-1.7 s (+1.3-1.8%). Measured kernel is already faster than its bytes (L2), so likely less | **keep, low rank**: real but small, and the vLLM state-slot plumbing is the hard part |
| GDN state dtype | bf16 state halves bytes | -0.9 ms/step | **dropped: changes precision** (config mamba_ssm_dtype fp32; contract) |
| KV cache (fp8) | already kept; 16 KB/token/GPU | ~0.3 ms/step total attention | dropped: too small to matter |
| Prefix (chat template) | prefix cache | ShareGPT shares only the template (~10 tokens) | dropped: nothing to share; benchmark-tuned caching forbidden anyway |
| Logits (target) | keep nothing (only argmax needed) | logits write 12 MB/step | see Rewrite R6 |
| q, attention scores | Rule 1/2 kill them | — | dead (as the framework derives) |

## Knob 2: Rewrite
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| FP8 GEMM accumulation | fp16 accumulator inside a 128-K block (2x tensor rate, 330 TF/s) for compute-bound mixed steps | prefill GEMM ~17 s -> ~9 s: +8% | **dropped**: e4m3 x e4m3 products reach 448^2 = 2e5 > fp16 max 65,504: overflow, not a bounded reordering; would also need a ruling |
| RMSNorm row scalar (2-graph §4): `norm(x)W = rstd(x) * ((x⊙g)W)`, x = p0 + p1 + res | start the next GEMM on the local partial before the AR completes | needs W read twice (once per partial) or FLOPs doubled | **dropped by the Shared-Read Rule**: decode is HBM-bound (2x weight bytes), mixed is compute-bound (2x FLOPs) |
| in_proj_ba (bf16, 48 x 5120) | fold into the qkvz GEMM | 9 us + 3 us per layer | **dropped**: it already runs on a side stream overlapped with in_proj_qkvz (node trace, stream 17 vs 4524); merging into FP8 would quantize a bf16 weight (lossy) |
| Target lm_head bf16 GEMM (0.80 TB/s) | better GEMV-shaped bf16 kernel at ~0.93 TB/s | 1.58 -> 1.37 ms: 0.2 ms/step -> ~0.5 s (+0.5%) | keep, low |
| Draft lm_head FP8 cutlass (0.76 TB/s) x2 | same | 1.03 -> 0.84 ms: ~0.45 s (+0.5%) | keep, low |
| R6: argmax fused into lm_head epilogue | skip writing/reading 12 MB of logits | ~25 us/step | dropped: < 0.1% |
| Decode W8A8 GEMMs at 0.75-0.87 TB/s | kernel/config with more CTAs (grid is 80 CTAs on 128 SMs for every shape), split-K or stream-K for M=48 | if they reached 0.95 TB/s: 14.4 -> ~13.2 ms; if the L2 prefetch then also paid off (5.1 GB/step prefetched during ARs, see W-knob): up to ~4 ms/step -> up to 8 s (+8%) | **keep, needs a diagnostic first** (ncu on gate_up: DRAM bytes vs 89 MB tells whether the prefetch hits; achieved occupancy/BW tells whether 80 CTAs cap it) |

## Knob 3: Speculate
| Edge | Candidate | Napkin | Verdict |
|---|---|---|---|
| x[t+1] (serial sampling edge) | champion: chain MTP, 2 drafts (acceptance 0.775 / 0.556, 2.33 tokens/step) | — | baseline |
| | **3 drafts**: if pos-3 cumulative acceptance ~0.40 -> 2.73 tokens/step (+17%) for +16 verify rows (AR payload +33% ~ +1.3 ms, GEMM ~free at 64 rows, GDN rows +) and one more draft step (~1.1 ms): step ~26.5 -> ~29.5 ms (+11%) | decode net ~+5%, mixed ~0: **~+3%** | keep (cheap: serve flag; needs 128K/memory check). Riskiest: pos-3 acceptance |
| | tree speculation (top-2 at depth 1) | gains acceptance; but GDN state must fork per branch (framework 02 §4.4: in-place S breaks branching; Row C keep raw k,v for the branch suffix) | dropped for now: high build cost; the recurrent kernel would need branch-aware logs |
| | retrained draft head (non-benchmark data) | acceptance up | dropped: training budget |
| sel / continuous edges (AR payloads, layer outputs) | p = 0 (high cardinality) | — | dead (Gate 2), as the framework predicts |

## Knobs 4 & 5: Split / Merge and Place
| Index / node | Candidate | Napkin | Verdict |
|---|---|---|---|
| Layer l across GPUs | PP=2 (2 microbatches of 8) | each GPU reads its 12.2 GB twice per token step (Shared-Read Rule) -> ~24 ms weights vs 12 + ~7 AR | **dropped**: decode N=48 << ridge |
| Request r across GPUs (DP) | each GPU needs 24.4 GB of weights | > 24 GB | dropped: capacity |
| Prefill/decode disaggregation (GPU0 prefill, GPU1 decode) | same capacity problem | | dropped: capacity |
| Asymmetric TP split (alpha != 0.5) | GPUs symmetric (both hold half lm_head) | GPU0/GPU1 within 0.2% | dropped |
| AR payload (rows x d) | reduce-scatter + all-gather instead of AR | same bytes per direction for 2 ranks | dropped |
| **Split r/i of mixed steps for the AR** | row-split halves are 1.4 MB > hostmem limit 1.31 MB -> NCCL ring at 217 us vs hostmem (est. ~70-90 us) | NCCL 36 ms/step, ~15 ms exposed, on ~168 steps -> **~1-2 s (+1-2%)** | keep: one env var (VLLM_HOSTMEM_AR_MAX_BYTES); check slot memory |
| Merge prefills across arrivals (policy: admit pending prompts together) | each mixed step carries ~45 ms of fixed CPU cost for small P; two small prompts in one step save one | up to ~5 s if half of the 232 small prompts pair up; TTFT up | keep as **fallback** for W2 (policy change, generic but a TTFT trade) |

## Knob 6: Window (merge and overlap)
| Site | Candidate | Napkin | Verdict |
|---|---|---|---|
| **Mixed steps with P+45 > 512 (76 prompts, eager: ~215 ms)** | extend CUDA-graph capture sizes to 1024 (largest real prompt 885) so these run as graphs like the 256-512 steps (~90 ms at 400 tokens; ~130-140 ms extrapolated at 700) | ~75 ms x 76 = **~5.7 s (+6%)** | **rank 1 by cost/benefit**: serve flag only; graph memory -> 128K test. Testing (E1) |
| **Mixed steps, small P (232 prompts): eager GDN core is CPU-bound** (~1.4 ms CPU vs 0.2 ms GPU per GDN layer) | put the GDN core of mixed steps in the window with the rest: graph-capture it (per-layer graph keyed on padded P bucket and number of spec decodes, static metadata buffers; padding rows with g=0, beta=0, k=0 leave S unchanged exactly), or cut its launch count (fused prep/merge kernels, C++ launches) | 75 -> ~33-40 ms per step: **~8-10 s (+9-11%)**; mid bucket gains a little too | **rank 1 by size**: biggest single gap; engineering-heavy |
| AR (link) || next weights (HBM) | champion: prefetch 40 MB into L2 during each AR | GEMM + AR = 21.3 ms vs 12.5 GB bytes floor | see Rewrite (decode GEMMs): the diagnostic decides whether the prefetch is paying off |
| AR || own GEMM (column-chunked out/down + per-chunk AR, Flux-style) | splitting N does not re-read weights (Shared-Read safe) | hides link time under the GEMM instead of under the prefetch: only wins if prefetch isn't effective | keep behind the same diagnostic |
| AR || next layer's GDN state read | state read (25 MB) does not depend on the AR | same resource budget as the weight prefetch | dropped unless prefetch slack exists |
| lm_head || draft | draft needs the verified token (serial edge) | | dead (dependency) |
| Draft step s || verify | speculative drafting of step t+1 drafts during verify | the draft needs accepted count | dead under chain MTP |
| GDN kernel || GEMMs | both HBM-bound | Mechanism 2 needs different resources | dropped |

## Summary of surviving candidates (napkin, vs 95.9 s)
1. W: mixed-step GDN core out of eager/CPU-bound: +9-11% (hard).
2. W: CUDA graphs up to 1024 tokens: +6% (trivial; E1 running).
3. Rewrite: decode W8A8 GEMM efficiency / prefetch effectiveness: 0 to +8% (diagnostic first, ~5 GPU-min ncu).
4. Speculate: 3 drafts: ~+3% (cheap flag).
5. Split: hostmem AR for row-split halves: +1-2% (env var).
6. Keep: ReplaySSM depth L: +1-2% (hard).
7. Rewrite: lm_head / draft head bandwidth: ~+1% together.
8. Merge prefills (policy): fallback for 1.

Where the framework needed help: its cost model has no CPU/launch resource. The largest gap in this system (small mixed
steps) is a host-side cost: the Window knob found it only once I added "CPU launch time" as a fourth resource in the
window max(). Its window model also assumes a kernel at full BW; here the 0.75-0.87 TB/s efficiencies are where much of
the decode gap is, which the framework leaves to the Implementation layer.
