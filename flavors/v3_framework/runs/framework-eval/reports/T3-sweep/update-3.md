# T3 map update 3 (closing)

## Tested since update-2
| candidate | record | result | evidence |
|---|---|---|---|
| row-split restore on T2-011 | T3-009 | **official 951.5 (+2.11% vs T2-011): champion** | ABBA job 123 |
| ROWSPLIT_MIN_TOKENS=128 | T3-010 | -1.15%, all B < all A | job 128 |
| copy-engine AR on the restored row-split | T3-011 | screen negative (1 pair, -2.45%); profile: CE copy time reappears as idle | jobs 130, 132 |
| prefetch 56 MB (vs 40) | T3-012 | -0.38%, within noise | job 134 |
| prefetch-aware split-K GEMM configs (T2's, retested with the in_proj prefetch delivered) | T3-013 | -0.03%, null | job 135 |
| NCCL_PROTO=Simple (microbench, alone and under GEMM load, 1.3-5.2 MB) | (microbench) | no difference (+/-2%) | job 138 |

## Verdicts changed
- Copy-engine AR: from "rank 2, ~3-4%" to dead. The microbench overlap (T3-002) never carried into the server, in two
  tests on two code bases. Framework lesson: the Window knob's "link" node may be a set of copy-engine ops whose
  ordering and launch costs the napkin doesn't see. Price ops, not just bytes.
- Prefetch tuning: 40 MB is near the window's capacity (~44 us x ~0.9 TB/s). More bytes don't help, and neither do
  GEMM configs tuned for warm weights.
- Row-split below 256 rows loses, because the 64-127-row halves go to the SM-spinning hostmem AR (T3-002: hm||gemm is
  the worst overlap).

## Final ranking of what is left (all small or uncertain)
1. Row-split below 256 rows with NCCL forced for the halves (instead of the hostmem AR): napkin ~0.2%. Not worth a test.
2. A pipelined AR from T2 (release/acquire, T2-012: +1.08% ABBA on T2-011, official within noise of T3-009) plus prefetch
   retuning: T2's line.
3. DeltaLog with L=2: ceiling ~0.33 ms/step (T3-007): +0.3-0.8% net at best, with a risky materialize rewrite.
Recommendation: close T3; the remaining items above are below what an ABBA can resolve cheaply.
