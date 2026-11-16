# T3 sweep, step 6: ranking (updated after review T3-1, worker microbenchmarks and T3-001)

Reference: champion T2-007 (787.6 tok/s). Original ranking (pre-review) is preserved in git history of this run only through reports/T3-sweep.md's
answers; the changes: #1 ba kernel dropped (side stream), #3 column split dropped (microbench), #4 DBO dropped (blocked), #5 skew unpriced,
#8 SendRecv promoted (real transport).

| # | candidate | napkin gain | status / wins if | riskiest assumption | cheapest test |
|---|---|---|---|---|---|
| 1 | **K5-a L2 weight prefetch during each hmar all-reduce** | ~+3% | **built: T3-001, 816.6 tok/s (+3.7% vs T2-007), candidate sent** | — | done (jobs 47-49) |
| 2 | K5-a tuning: per-target prefetch size (gate_up / qkvz / qkv / MTP), prefix vs tile-order ranges | +0.5-1% | 16 MiB covers 19% of gate_up, 40% of qkvz; job 47 shows >=32 MiB loses | gains are not flat across targets | worker microbench per target (~1 GPU-min) |
| 3 | **K2-e exact greedy target fast path** (replace the 6 MB/rank logits all-to-all with local argmax + [rows,2] gather + greedy verify, fallback otherwise) | ~-0.45 ms/decode step + mixed: **+1-1.5%** | tie-breaking and logits dtype identical to the rejection kernel's greedy path | the all-to-all is on the critical path (serial step: yes) | worker building in t3-greedy; c=1 bitwise compare + quick bench |
| 4 | K1-a GDN DeltaLog | <= +1.5-3% | all 3 slots are written (confirmed); real-server kernel 70 us vs cold 120 us: gain depends on L2 write-back | write-back is on the critical path | ncu dram bytes in the server, with a write-mask that keeps source selection |
| 5 | mixed-step launch skew / eager GDN core | <= ~2% (T2: ~7.65 ms GPU idle per mixed step) | capture or trim the eager part | size-matched waits are skew | per-rank AR offsets from existing sqlite |
| 6 | K2-d hmar epilogue fusion (add + RMSNorm + group quant) | ~+0.8% | the bf16-rounded AR result is kept before the residual; whole-row layout | hmar's block partition splits rows | after #2-#3 |
| 7 | reduced-vocab draft head on this champion | unsettled | acceptance loss < step gain, within memory | T1-002 acceptance -4.6% | low priority |
