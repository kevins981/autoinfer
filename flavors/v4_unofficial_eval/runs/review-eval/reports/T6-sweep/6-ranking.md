# T6 sweep, step 6: ranking (vs T3-007, 853.6 tok/s, 106.6 s per trial)

| # | candidate | napkin gain | wins when | riskiest assumption | cheapest test |
|---|---|---|---|---|---|
| 1 | **Row-split GEMM->AR pipeline in mixed steps** (rows >= ~192): out_proj/o_proj and down_proj computed in 2 row chunks; AR (NCCL) of chunk 1 on a side stream while chunk 2's GEMM runs; norm consumes the joined output | +1-3% (mixed steps ~30-37% of wall, AR ~45% of mixed-step kernel time) | AR and Triton GEMM co-run without slowing each other; the skew/wait can be overlapped | NCCL on a side stream inside piecewise graphs keeps the same order on both ranks and doesn't deadlock | microbench of GEMM(260x8704->5120)+AR vs 2-chunk pipeline on 2 GPUs; then a quick bench |
| 2 | GDN DeltaLog | +3-4% | writes reach DRAM | slot contract | T5 owns it |
| 3 | k=3 after DeltaLog frees memory | +1-3% | p3 >= 0.35 and the draft step stays ~1.2 ms | capacity | after #2 |
| 4 | decode GEMM efficiency (76% -> 85% of DRAM peak) | up to +3% | a split-K/stream-K kernel at M=48 beats the tuned Triton tiles | the remaining gap isn't DRAM page/L2 effects | ncu of the decode GEMM |
| — | large AR transport, reduced vocab, PP, DBO via ubatching, tree drafts | closed / dropped (5-compare.md, 4-knobs.md) | | | |

## Review T6-1 (reviews/T6-1) and my answers
1. Pricing of #1 = G1 + max(AR1, G2) + AR2 vs G + AR, minus extra collective/stream overhead; the waiting does not give extra allowance: ADOPTED. Napkin for #1 lowered to **+0.5-2%**; the worker's same-job ON/OFF bench decides.
2. The 82-row ridge is contradicted: a 1,849-row mixed step ran its W8A8 GEMMs at ~239 TFLOP/s, so the effective FP8 peak is above 165 (fp32 accumulation across 128-K blocks only, not within). Effective ridge ~240+ rows -> 130-row halves are not clearly compute-bound, and re-reading weights per chunk costs something: ADOPTED (a cost in #1's pricing; it also weakens my k=3 "M=64 stays BW-bound" only in the direction that helps k=3).
3. Mixed calibration mixes medians with means; only 26/50 mixed steps have >= 192 rows (4.58 of 7.36 s of profiled mixed time): ADOPTED. #1 applies to ~60% of mixed time.
4. GDN state traffic: 1 read + 3 distinct stores; 3.93 GB at 13 requests, 4.83 GB at 16: ADOPTED (2-graph's per-GPU figure of 1.21 GB read + 1.21 GB written per step was wrong; the correct figure is 1.2 read + 3.6 written).
5. Missing edges (accept count -> state source; conv rollback; prefix checkpoint publication; logits all-to-all before batch-sharded sampling + gather; AR+norm not fused in the champion): ADOPTED as graph corrections. None change the ranking.
6. PP premise wrong (balanced PP also reads ~12.2 GB per GPU per micro-batch); column-split decode and reduced vocab are "not priced enough" rather than refuted: ADOPTED. They are now marked untried and low priority rather than dead.
7. k=3 also grows all 128 target AR payloads: ADOPTED; k=3 stays conditional.

## New evidence (worker, job 101, isolated microbench, TP=2, us/call): HMAR vs NCCL by row count
| rows | 96 | 128 | 192 | 256 | 384 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| NCCL default | 138 | 418 | 452 | 545 | 665 | 751 | 1123 | 1617 |
| HMAR (16 blocks, region raised) | 126 | 133 | 205 | 269 | 396 | 530 | 1016 | 1904 |
HMAR is bitwise-equal to NCCL at every size. **This contradicts T1's "no gain above 96 rows"** (T1 cited bare NCCL ~205 us at 2.66 MB; here NCCL is 545 us at 256 rows). New candidate **#0: HMAR up to 768 rows**: saves ~200-280 us per AR at 128-512 rows x ~130 ARs -> ~30 ms per eligible mixed step -> napkin **+3-6%**, if the isolated gap survives skew in the server.
