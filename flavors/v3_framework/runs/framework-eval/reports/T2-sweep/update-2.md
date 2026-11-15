# T2 map, update 2 (resync 2, 2026-11-14)

## Tested since update-1
| candidate | result | evidence |
|---|---|---|
| K1a GDN DeltaLog (rank 1) | **WON (champion)**. The MTP decode kernel writes row 0's state plus a 784 B/row/head log of the draft inputs. Accepted rows are replayed lazily: before the Triton path in mixed steps, and before the prefix-cache align copies. It is bitwise exact. ABBA +3.91% on T1-014; official 880.1 (+4.4%). Napkin was 2-2.5 ms/step (~7%); measured ~1.2 ms. A first version corrupted requests crossing a block boundary: the replay destroyed the log of the `bias` slot. It was fixed by materializing the highest slot first. | jobs 36, 42, 60, 63, 65; T2-008 |
| Fused norm/SiLU + group-128 FP8 quant (rank 2) | **WON: official 906.9 (+1.06% vs T3-003)**, ABBA: +1.29%, all B > all A, on T3-003. Napkin ~0.5 ms/step (1.5%). The SiLU site is bitwise. The norm site is reordered by FlashInfer's fp32 variance order: 1 bf16 ulp in ~1e-5 of elements, allowed by the manager's fused-op ruling. Same-job GSM8K-250: 93.6 vs 93.2. | jobs 92, 98, 100, 102; T2-009 |
| Decode-M GEMM efficiency (rank 3) | **in progress**. nsys on T2-006 at M=48: gate_up runs at 67-78% of the byte floor, in_proj 76%, qkv 74%. Every 4090 config uses BLOCK_M=16, i.e. 3 M-tiles. Headroom is ~1.5-2 ms/step if ~88% of peak is reachable; it must now be measured with T3-003's 40 MB prefetch in place. | prof_t2006 sqlite; worker task 10 |

## Verdicts changed
- **W2 L2 prefetch (my T2-005, "LOST") was wrong as a general verdict.** T3-003 won +1.97% official with a different
  site and purpose: prefetch from extra CTAs *inside* the PCIe-bound hostmem AR, evict_last, 40 MB of the next GEMM.
  This also forces the GDN's dirty state lines to drain during the AR instead of during gate_up. My microbench
  prefetched without the state-write drain and from a separate kernel. The lesson is that a microbench without the
  real neighbours (dirty L2, AR window) can price a site at zero. T2-005 stays negative only for "separate prefetch kernel".
- **Framework gap, confirmed twice:** per-kernel cost models assume each node pays for its own bytes. On the 4090, L2
  write-back of one node (GDN state, ~25-75 MB) is paid by the *next* node (gate_up 132 vs 113 us). Both DeltaLog and
  T3-003 were found by pricing a pair of adjacent nodes, not single nodes. Suggest adding an "L2 carry-over" edge
  cost between consecutive nodes in 3-calibration-style models.
- **No torch.compile matters for pricing fusions:** under breakable CUDA graphs every elementwise op is its own kernel,
  so fusion napkins should use launch count x ~2.5 us. Real-kernel napkins (0.5 ms) matched the measured gain. An
  emulated-compiled reference (0.25 ms) would have under-ranked it.
- Exactness pitfalls worth recording: Triton's f32->fp8e4nv cast double-rounds via fp16 (0.33% mismatches); use PTX
  cvt.rn.satfinite.e4m3x2.f32. vLLM's C op silu_and_mul_per_block_quant skips the bf16 rounding of silu, so it is not
  equal to the unfused chain.

## Current ranking (on T2-009 if promoted, 906.9)
1. Decode-M W8A8 GEMM tiles / deterministic split-K, measured with prefetch: 1.5-2 ms/step napkin (4-6%), riskiest
   assumption: ~88% of DRAM peak is reachable at M=48 with block scales. Cheapest test: microbench (worker task 10).
2. Reduced-vocab draft lm_head (FP8 already): ~2-3% (T1 closed; unowned now). Exactness: the draft only affects
   acceptance, and a reduced vocab must keep the target's verification exact.
3. Remaining fusions: GDN gated norm (`_fused_sigmoid_gate_mul` / layer_norm_fwd) -> out_proj quant: ~0.15 ms (~0.4%).
4. Prefetch tuning (T3's area).
Dropped: chunking >512 rows, PP/DP/DBO, separate-kernel L2 prefetch, matching FlashInfer's norm reduction order bitwise (1 h for 0%).
