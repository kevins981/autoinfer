# T2 map, update 3 (closing, 2026-11-15)

## Tested since update-2
| candidate | result | evidence |
|---|---|---|
| Fused norm/SiLU + FP8 quant | **WON**: official 906.9 (+1.06% vs T3-003). Side effect missed by me: it made rowsplit_ok() False, which disabled T1-013's mixed-step AR overlap. T3 found this (job 117) and restored it (T3-008/T3-009, +2.1%). | T2-009; T3-009 |
| Decode-M GEMM tiles / split-K | **NOT WORTH IT cold**: current configs equal a plain streaming read (850-890 GB/s achievable; my 1008 GB/s napkin peak was wrong). With 40 MB prefetched, the micro shows -9..-24% per GEMM, but the server smoke was -0.8% (1 pair, inconclusive). Handed to T3 (pf-aware configs, gate VLLM_T2_GEMM_PF_CFG). | T2-010; jobs 103/105/107/111 |
| Reduced-vocab draft head (ids [0,151643) + specials, data-free) | **WON**: ABBA +3.14% on T2-009; official 931.8 (+1.39% vs T3-006). Acceptance unchanged. | T2-011; jobs 113, 116 |
| Hostmem AR release/acquire flags | ABBA +1.08% on T2-011 (job 129). Official on T3-009: 953.1 vs 951.5, within noise, **not promoted**. Same-job ABBA vs T3-009: mixed, +0.54% (B3 < A4), not resubmitted. Sub-chunk pipelining loses (each handshake costs ~5 us of PCIe round trip). | T2-012; jobs 122, hm-var, ra-test2, 129 |
| Mixed-step idle | **not worth it**: only ~2.5 ms of real gaps per mixed step (≈0.9% of wall). The ~9 ms that step-window attribution suggests is mostly async launch lag. | worker gaps.py on T3-006 nsys |
| Mixed-step GEMM efficiency | 65-79% of the 330 TFLOPS nominal = ~80-90% of a realistic block-scaled ceiling (cuBLAS bf16 reaches 89-93% of spec here, and block scaling costs ~7-8%). ≤0.9% of wall left, and it needs a CUTLASS-class sm89 kernel. | agents/leader-2.w1/gemm_roofline_mixed.txt, fp8_gemm_ceiling_note.md |
| Spec depth k=3 (re-test prompted by spar T2-1) | **LOST**: screen -11.7% (1 pair), acceptance 2.32→2.70, step cost +32%, TTFT p99 1989 ms (breaks the gate). DeltaLog falls back at k=3 (it caps at 2 logged rows), but even with that given back, k=3 is ~-7%. | T2-013; job 133 |
| GDN gate→out_proj quant fusion | not built: the gated norm already runs inside the GDN decode kernel; the standalone quant is 0.16 ms/step. | T3-006 profile |

## Verdicts changed
- **Achievable DRAM bandwidth on this 4090 is ~850-890 GB/s, not 1008.** Every "x% of floor" in my calibration and
  4-knobs used 1008, so they overstated headroom by ~15%. Decode weight GEMMs are at the achievable floor.
- **PCIe duplex for SM-issued traffic:** write+read concurrently is 29-32 us for 491 KB per GPU, with both GPUs at
  once (31-34 GB/s total). Copy-engine measurements (22.6 GB/s total, T3) are not the right floor for an SM-issued AR.
  The decode AR is latency-bound (flag/fence round trips), not bandwidth-bound.
- **Fusions can silently disable earlier wins.** Before any A/B, check the log lines of earlier features in every
  server log.
- **"k=2 is optimal" is now measured on current code**, not only inherited from T1.

## Framework notes (cumulative)
1. No CPU resource in the cost model (T2-001's +6.4% was CPU/launch time).
2. Per-node pricing misses L2 carry-over between adjacent nodes (DeltaLog, T3-003).
3. Fusion napkins must use the real kernel list (no torch.compile under breakable CUDA graphs).
4. The cost model needs *achievable* peaks per resource, measured on the box (DRAM ~870 GB/s, PCIe SM-issued
   duplex ~32 GB/s total, FP8 MMA ~280 TFLOPS block-scaled). Spec-sheet peaks mis-rank candidates near the floor.
5. Combinations interact through shared gates (rowsplit_ok) and shared windows (the AR window used by prefetch).
   The sweep lists sites independently; a "feature-flag regression check" belongs in the method.

## Final ranking (what's left, on T3-009 ~951 tok/s)
1. Hostmem AR release/acquire (T2-012): same-job A/B vs T3-009 was MIXED (+0.54%, B3 < A4); ≤0.5%, within noise. Built, ws t2-rel9; only worth stacking if the prefetch is retuned.
2. Prefetch-aware GEMM configs plus a larger prefetch window (T3 now owns this).
3. Mixed-step block-scaled FP8 GEMM kernel quality: ≤0.9% of wall, needs a new kernel.
4. DeltaLog materialize folded into the Triton spec path in mixed steps: ~1.4 ms per mixed step, ≈0.5% of wall.
Dropped (measured): k=3, k=1 (T1), separate-kernel prefetch, sub-chunk AR pipelining, decode-M cold tuning,
chunking >512 rows, mixed-step idle, GDN gate quant fusion, target lm_head (must stay bf16).
