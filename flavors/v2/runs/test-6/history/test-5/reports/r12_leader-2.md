# Round 12 report: leader-2 (team T1)

## Summary
Two exact gains stacked on the champion T1-030 (897.3):
- **T1-035, lazy GDN checkpoint: official 906.3 (+1.0%).**
- **T1-037 = T1-035 + lossless lm_head codec: official 922.8 tok/s.** Trials 925.9 / 919.8. That is +2.8% vs T1-030 and +45.1% vs BASE. GSM8K 93.2%, the 128K request passes, p99 TPOT 30.9 / 28.8 ms.

GPU: 59.3 / 80 used (2 official evals, about 22 min).

## Experiments
| record | builds on | what | performance |
|---|---|---|---|
| T1-034 | T1-030 | Marlin ALU-slack probe for PFOR FP8 weights (no engine change) | killed: L2-warm GEMMs have no ALU room (+4-8% at k=8), net ~0 |
| T1-035 | T1-030 | lazy GDN checkpoint: S[0] stored only on log overflow (cap 8); every flush replays pending tokens | **official 906.3** (ABBA +1.05%, ITL -0.50 ms) |
| T1-036 | T1-035 | GDN kernel at 3 blocks/SM (1 wave) | null: 49.7 -> 50.4 us/layer; bandwidth-bound |
| T1-037 | T1-035 | bf16 target lm_head stored with the block-exponent codec (0.781), decoded in registers inside a custom GEMM; raw freed after bitwise verification | **official 922.8** (ABBA +1.17%, ITL -0.54 ms) |

## Validity notes
- **T1-035:** bitwise by construction: the same fp32 ops in the same order, only when the state is stored changes.
  - Unit tests: 36k checks vs the stock kernel, 0 mismatches; 4 negative controls all fail.
  - Greedy c=1: 10/10 identical.
- **T1-037:** satisfies the round-12 weight rule.
  - Load: every block is verified bitwise with the same decode routine the GEMM uses; raw fallback per block; the raw shard is freed. If verification fails: ERROR plus cuBLAS.
  - GEMM: bf16 MMA, fp32 accumulation over all of K, one bf16 rounding. It differs from cuBLAS only in summation order: bitwise where cuBLAS is sequential, a few ulps where cuBLAS split-K rounds through bf16.

## Proposed SHARED_KB
1. The GDN replay kernel is HBM-bound on the fp32 state read: about 50 us/layer at 16 requests in the server, linear in blocks. Occupancy/wave changes do nothing (T1-036). Skipping 2 of 3 state stores saved 0.50 ms/step, less than the microbench's 0.68 because L2 already absorbed part of the writes (T1-035).
2. Marlin FP8 at M=48 has ALU room when cold but none when L2-warm (issue/tensor bound). Any in-GEMM weight decode costs more than it saves on prefetched GEMMs (T1-034).
3. cuBLAS bf16 lm_head at M=17-64 runs well below the HBM roof (1.43-1.62 ms vs a ~1.30 ms roof). A custom stream-from-HBM GEMM on encoded bytes runs at about 95% of the read roof: -34% at M=48 (T1-037).
4. The block-exponent codec compresses bf16 weights to 0.781 with 0.27% raw blocks (lm_head), the same ratio as on AR payloads.

## Left open
- Fuse the greedy argmax into the T1-037 GEMM epilogue (~-0.04 ms).
- 3-bit exponent offsets (ratio ~0.72, ~-0.06 ms).
- The step in the GDN kernel time from 11 to 16 requests (32 -> 50 us) is unexplained; possibly L2.
- w22's survey (agents/leader-2.w22/NOTES.md): everything else is below 0.5% each, except the two items below.

## Next direction (one)
**Lossless compression of the GDN fp32 recurrent state read** (inside our own replay kernel).
- Why: the state read is now the GDN floor, about 1.6 ms/step, 25 MB per layer. If its entropy ratio is ~0.87, that is about -0.2 to -0.27 ms/step (~1%), in a kernel shown to be bandwidth-bound (ALU slack not yet measured).
- Cost: a cheap gate first. Dump real states, measure the codec ratio (~2 GPU-min), and go only if the ratio is ≤ 0.9.
- Alternative (multi-round): an entropy codec for the cold down_proj FP8 weights (2.85 GB/step/rank, -0.26 to -0.5 ms) with the prefill penalty (+2%) to manage.
