# T1 sweep, step 6: ranked candidates

Gain = napkin share of BASE's ~143 s benchmark (decode steps 68%, mixed steps 32%), using measured
part times and the skew/transfer split of 5-compare.md. Effort: L/M/H.

| # | candidate | napkin gain (tok/s) | wins when | riskiest assumption | cheapest test that could show it wrong | effort |
|---|---|---|---|---|---|---|
| 1 | **I-AR**: faster all-reduce transport (NCCL proto/channels/SHM memcpy; or custom pinned-host AR; or flashinfer PCIe IPC AR) | decode -3.7 ms (of 38.5), mixed -25 ms transfer: **~+10%** | a variant beats NCCL RING_LL by >1.5x at 0.5 MB and 2.7 MB | that ~20 GB/s is reachable via host memory for these sizes (latency-bound at 0.5 MB?) | AR microbenchmark across variants (leader-1.w1, in flight) | L (env) .. H (custom) |
| 2 | **W1 graphs for mixed steps** (T2's, in flight as t2-cg2048) | -25..-30 ms per mixed step: **~+6%** | graph-captured mixed steps remove skew + launch idle | breakable graphs still break at attention/GDN prefill, keeping launches | T2's unofficial eval | L (flag) |
| 3 | **S-K3** MTP 3 drafts (+ ideally on top of R-DLM8 / R-ARGMAX) | 0..+5% alone, +8% with cheaper draft steps / AR | conditional acceptance of draft 3 >= ~0.65 | pos-3 acceptance of an MTP head trained for 1 step | one unofficial eval with the flag (+ --long) | L |
| 4 | **R-ARGMAX** distributed argmax (target verify + draft steps) | -1.3..-1.6 ms/decode step: **~+3%** | all requests greedy, no logprobs (fallback otherwise) | that the rejection sampler / drafter path can consume (value,index) without the full logits in this vLLM | read the sampler code; microbench gather vs (max,idx) | M |
| 5 | **R-DLM8** FP8 draft lm_head (ruling 1) | -0.65 ms per draft lm_head: **~+1.5-2%** | acceptance unchanged within noise | that vLLM supports `draft_lm_head_quantization="fp8"` for MTP here | grep the config; one unofficial eval | L |
| 6 | **W-DBO** in mixed steps (two half-batches, AR || GEMM) | hides the transfer part of mixed-step ARs (~15-40 ms): **+3..+9%** depending on #1 | mixed step GEMMs compute-bound (N >> 82) | NCCL on a side stream concurrent with Marlin without SM starvation; eager-path complexity | 2-stream microbench: Marlin(130 rows) || NCCL AR(1.3 MB) | H |
| 7 | **W3** AR || producing GEMM by output-column chunks (T2's) | -3..-5 ms/decode step: **~+6%** (less after #1) | chunked GEMM keeps Marlin efficiency | Marlin efficiency on narrower N chunks; AR latency per chunk | microbench chunked Marlin + AR overlap | H |
| 8 | **R-W8A8** prefill GEMMs FP8 (fp32 across K-blocks) | with fp16 in-block accumulate: -25..-30 ms per mixed step **~+6%**; fp32 accumulate: only kernel-efficiency gain | an sm_89 block-FP8 GEMM beats Marlin at M~260 | fp16 in-block accumulation overflow / FP8 fp32-acc rate = bf16 rate on Ada | microbench triton w8a8_block_fp8_matmul vs Marlin at M=260 | M-H |
| 9 | **I-MARLIN** decode BW 67% -> 85-90% | -4.6 ms/decode step: **~+8%** | a better M=48 tile config / split-k exists | Marlin is near its design limit in-situ (ncu isolated already 80%) | ncu of in-situ vs isolated Marlin; try other kernels at M=48 | H |
| 10 | reduced-vocab draft head (data-free first-N ids) | -1.2 ms/decode step minus acceptance loss: ~+1-3% | acceptance loss < ~3% | acceptance drop | one eval | M |
| 11 | K-B GDN rollback-free state writes | <= 1.2 ms/decode step: ~+2% | | measured GDN already near floor | ncu bytes written of the GDN MTP kernel | H |

Dropped (with reasons in 4-knobs.md / 5-compare.md): DP (capacity), PP (dominated once mixed steps
are fixed), decode DBO (Shared-Read), bf16 GDN state (contract), tree drafts (GDN branch states),
batching prompt chunks (~1%), KV/weight offload, sequence parallel.

## Plan for T1's first 30 GPU-minutes
- leader-1.w1: AR microbenchmark (#1, 6 min, in flight). If a NCCL env variant wins: one unofficial
  eval with the env set in code (ruling 3: generic default) -> candidate record.
- Me: S-K3 serve flag (#3) unofficial eval with --long (~6 min). FP8 draft lm_head (#5) if vLLM has
  the option (flag), stacked with K3.
- Next (code): R-ARGMAX (#4) by a worker, CPU-side first.
- Coordinate with T2 (after this file): they own W1 graphs; avoid running the same AR variants twice.
