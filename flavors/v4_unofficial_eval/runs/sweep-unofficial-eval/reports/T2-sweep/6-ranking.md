# T2 sweep, step 6: ranked candidates

Gains: napkin, as share of BASE's measured time (M1: 37.9 s; 1 ms per decode step ≈ 1.7%). Not additive where they share
slack (noted). Excluding what T1 is already building (T1-001 draft FP8 head + local argmax; X1 all-reduce).

| rank | id | candidate | napkin gain | wins when | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|---|
| 1 | W1 | CUDA-graph mixed steps: raise `max_prefill_cudagraph_capture_tokens` / capture sizes (and cap `max_num_batched_tokens` to the largest captured size if needed) | **8-10%** | mixed-step graphs capture and replay correctly with GDN chunked prefill + FlashInfer under breakable graphs | memory of larger graphs/buffers (128K request), and that the idle is CPU dispatch, not nsys | 100-prompt A/B in one job (BASE vs flags), then `--long` unofficial eval |
| 2 | KR1 | GDN state: write 1 state per layer per step instead of 3 (replay the accepted rows from S[t] next step) | ~4% | slot bookkeeping can be redone without breaking prefix-cache align mode / preemption | that the 3 writes really reach DRAM (L2 may absorb some) | ncu dram bytes of gdn_decode_post_conv_mtp_kernel + the following Marlin (1 profile job) |
| 3 | PL1 | L2 prefetch of the next GEMM's weights during each AR | 0-9% (shrinks if X1 shortens the AR) | L2 keeps ~40 MB of streamed weights; prefetch doesn't slow the AR | L2 residency / Marlin's cache hints | microbenchmark (t2-w1 running) |
| 4 | SP1 | 3 MTP drafts | 2-5% (more after cheap drafts / KR1) | p3 ≥ ~0.35 | 3rd-position acceptance | 100-prompt bench with k=3, read acceptance (same job as W1) |
| 5 | W4 | draft-step CPU gaps (metadata rebuild) | 1-1.5% | another attention backend or fused draft path removes the gaps without slowing attention | backend speed | flag bench |
| 6 | RW1 | target local argmax (greedy batches) | ~1.5% | all rows greedy, no logprobs | sampler/rejection plumbing | code + numeric check |
| 7 | SM4 | reduced-vocab draft head (on top of T1-001) | ~2% beyond FP8 head | acceptance loss < ~3% | vocab source (non-benchmark) | acceptance on a 100-prompt bench |
| 8 | W2 | AR ‖ GEMM column split | ~4% decode; competes with PL1 and X1 | NCCL on a side stream inside graphs | ordering/graph capture | microbench |
| 9 | KE1 | faster decode GEMM | ~8% | | high effort | after KR1 (removes write-drain noise) |

Plan (80 GPU-min, ~70 left): (1) W1 flags + SP1 probe in one screening job; (2) PL1 microbench result decides PL1;
(3) KR1 measurement (ncu) then build if confirmed; (4) RW1/W4 small code items.
