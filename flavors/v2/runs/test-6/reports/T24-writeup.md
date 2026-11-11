# T24 write-up (leader-24, explore). Voluntary close after the seed

GPU: 0.2 of 20 used (job 155, worker leader-24.w1). Record: bank/T24-001. Report: agents/leader-24.w1/W1-1/REPORT.md.
Notes: local_kb/leader-24.md.

## Direction examined: a decode GEMM built to be fast with the weights in L2, plus prefetch to keep them there
- **Premise (from KB79 and T17-001):**
  - In situ on T14-004, the decode GEMMs (M=48) run at Marlin's L2-warm floor.
  - That floor reaches only 49-67% of the HMMA peak.
  - T17's kernel ran its math alone at 130 TF, but it padded M to 64.
- **The bet:**
  - a kernel specialised for M<=48 (3 m16 tiles, no padding);
  - prefetch that leaves HBM busy during the non-GEMM windows;
  - together they would cut decode GEMM time 15-25% (~-3 ms/step).
- **Forecast (assumed):**
  - payoff if it works: ~+7-8%;
  - odds: low (T8 and T17 had failed on related kernels).
- **Kill rules on gate_up at M=48** (set with the manager's review, which pointed to KB39/49/73/78):
  - "no B loads" (A + dequant + MMA) <= ~75 us;
  - "B loads only, warm" <= ~70 us.

## Measured (job 155, GPU0, M=48, read flush per KB76, profiler median of 15, us)
Arms:
- **warm:** read flush, then evict_last prefetch of min(48 MB, the shape's weights);
- **chain:** flush + 45 us spin + concurrent prefetch (T17's harness).

| shape | Marlin cold / warm | g17 (NM=3) cold / warm | no B loads | B loads only, warm |
|---|---|---|---|---|
| gate_up 17408x5120 | 104.7 / 85.8 | 127.6 / 114.8 | 88.7 | 103.2 |
| qkvz 8192x5120 | 51.9 / 43.0 | 63.7 / 59.5 | 45.8 | 51.4 |
| down 5120x8704 | 57.4 / 47.5 | 76.5 / 69.6 | 55.3 | 59.9 |
| gdn_out 5120x3072 | 24.5 / 21.4 | 39.7 / 37.8 | 29.5 | 31.7 |
| attn_qkv 7168x5120 | 47.2 / 40.1 | 63.0 / 59.4 | 45.9 | 50.8 |

More on gate_up:
- no loads at all: 80.4 (106 TF);
- B loads + dequant, no MMA: cold 110.3, warm 106.5;
- chain: Marlin 85.2, g17 113.8.

Chain ~= warm for both kernels in every shape.

Exactness vs Marlin:
- max|d| 2.4-4.9e-4; 0.04-0.15% of elements differ;
- deterministic run to run (fp32 summation order only).

Forecast vs outcome on gate_up:
- no B loads: expected 66-72, measured 88.7;
- warm: expected 72-76, measured 114.8.

Both kill rules fail, by 14 us and 33 us.

## KB candidate
At M=48 the FP8 decode GEMM is limited by a fixed cost per weight fragment, not by the MMA:
- **The cost:** dequant ALU, the shared-memory round trip, the per-stage barrier and ldmatrix.
- **Evidence:** going from M=64 to M=48 cuts MMA work 25% but the no-loads time only ~10% (88.6 -> 80.4 us on gate_up).
- **Even with no loads at all:** a stream-K/cp.async design (T17's g17) is within 6% of Marlin's full warm kernel.
- **Its load path:** a cp.async ring can't read from L2 at Marlin's rate (qkvz loads-only warm 51.4 vs Marlin's full kernel 43.0).
- **Conclusion:** Marlin is near a realistic L2-warm floor at M=48.
  - A warm-optimised GEMM plus prefetch has at most ~5% of decode GEMM time to win, about +2% end to end.
  - This agrees with KB39, KB49 and KB68.

## Caveats
- The kernel ran unchanged and untuned: T17's g17.cu already instantiates NM=3 for M<=48, and grid, NST and warp count were not tuned.
- The 48 MB prefetch covers only about half of gate_up's 89 MB, so gate_up's "warm" arms are partly cold. The shapes that fit in L2 (qkvz, gdn_out, attn_qkv) show the same ordering.
- Not run: step 2, a decoupled L2-prefetch load path.
  - It could only lower the loads arms.
  - The no-loads floor (80.4 vs Marlin 85.8 on gate_up) already bounds the gain at ~6% of that GEMM.

## Opinion
- I agree with T8, T17 and T22: the decode GEMM at M=48 is at both its HBM and its realistic warm floors on this card.
- I found no other direction whose payoff if it works clears +5% with measured parts.
