# T2 sweep, step 6: promising candidates, ranked

Napkin gain = % of BASE wall time (~143 s; decode 67%, prefill-containing steps 32%), from 4-knobs.md.
Late finding while ranking (vLLM code read, phase 2): two draft-side candidates are already config
options of this vLLM: `speculative_config.use_local_argmax_reduction` (vocab-parallel argmax for greedy
drafts; Qwen3.5 MTP implements get_top_tokens) and `draft_lm_head_quantization: "fp8"` (allowed by
ruling 1). That makes them the cheapest tests, so they go first in the testing order even though their
gain is mid-sized.

| rank | candidate | napkin gain | wins when | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | R2 W8A8 body GEMMs (checkpoint FP8 scheme) for decode and prefill | 10-13% (prefill ~6%, decode ~7%) | an sm89 block-scaled FP8 GEMM beats Marlin at M~45 (>0.8 TB/s) and at M~260 (>~1.6x Marlin) | a kernel exists/attains this on sm89 (vLLM's Triton w8a8_block_fp8_matmul or CUTLASS); act-quant overhead small; GSM8K within 2 pts | microbench Marlin vs available FP8 block GEMM on the 4 real shapes at M=16/45/60/260 (~2 GPU-min) |
| 2 | W4b prefill-step lockstep/CPU-boundness | 0-10% (unmeasured) | real (non-nsys) prefill steps have >~20 ms of CPU-bound waiting | the 150 us AR skew / 24 ms idle aren't nsys artefacts | log per-step wall time + P/D counts from a plain bench (no nsys), compare with GPU busy (~3 GPU-min) |
| 3 | C1 custom one-shot AR through pinned host memory | ~8% (decode 6, prefill 2) | mapped-host loads/stores reach >~20 GB/s with both directions active; per-op latency <~10 us | PCIe zero-copy bandwidth/latency on this host | primitive microbench (running: worker task 2, ~2 GPU-min) |
| 4 | W2 L2 prefetch of the next GEMM's weights during the AR | 5-9% (less if C1 lands) | ~40 MB prefetched into L2 survives until the GEMM and Marlin then runs from L2 | L2 residency on 4090 with Marlin's cache hints; prefetch kernel co-running with NCCL | microbench: prefetch kernel (on a 2nd stream, concurrent with an NCCL AR) then Marlin on that weight; time vs cold (~2 GPU-min) |
| 5 | S2a + R1(draft) flags: FP8 draft lm_head + local argmax for drafts | ~2.3% + ~0.9% = ~3% | acceptance unchanged (FP8 draft lm_head may flip a few argmaxes) | acceptance doesn't drop >~1% | one ABBA with serve_args only (no code) — screen negative stops it after 1 pair |
| 6 | S2b reduced-vocab draft lm_head (generic-corpus top-K tokens) | ~5% (incl. S2a) | out-of-vocab target tokens rare (<2%) | a non-benchmark frequency source that covers chat/code/math; vLLM support for a draft vocab subset with MTP | offline: token coverage of a generic corpus top-32K on non-benchmark text (no GPU) |
| 7 | R1(target) argmax-combine for target logits at T=0 | ~2% | all rows greedy, no logprobs requested | rejection sampler only needs target argmax for greedy rows | code read + A/B; no flag exists for the target side |
| 8 | R2b Marlin config tuning for M 33-64 | ~3.5% (subsumed by R1 rank if W8A8 wins) | another tile config gets >0.75 TB/s | Marlin's heuristics already pick the best | microbench of Marlin thread configs on real shapes (~1 GPU-min) |
| 9 | S1 k=3 drafts (after rank 5/6) | +2-5% | p3 >~0.4 and draft pass cheap | TPOT p99 within gate | ABBA with serve_args |
| 10 | W4a decode-step CPU gaps (1 ms/step) | ~1.8% | gaps are real without nsys | — | same plain-bench step logging as rank 2 |
| 11 | K1a GDN DeltaLog (write 1 state instead of 3) | 1.2-2.6% | GDN kernel is write-bandwidth-bound | bytes: measured GDN is under the byte napkin | ncu dram__bytes_write of gdn_decode_post_conv_mtp (~3 GPU-min) |
| 12 | W1 GEMM-AR chunked overlap | shares the AR pool with C1/W2 | AR stays long | concurrency of comm and GEMM kernels | after C1/W2 |

Dropped (4-knobs.md has the proofs): PP=2 and hybrids (decode BW-bound; capacity), DP, DBO half-batch
staggering, sequence parallel, embedding replication, draft-on-one-GPU, KV/state offload, prefix
caching, tree speculation (until R2), NCCL protocol/channel tuning (measured: no lever), FlashInfer
PCIe-IPC AR (needs P2P).

Testing order (cost-weighted): 5 (flags, ABBA) -> 1 microbench -> 3 (running) -> 2 (step logging) -> 4
microbench -> 6 offline -> 8 / 11 microbench.
