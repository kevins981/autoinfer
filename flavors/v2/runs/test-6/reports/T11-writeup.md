# T11 write-up (leader-11, explore). Recommending a voluntary close after the seed

GPU: 1.6 of 20 used (jobs 94-96, worker leader-11.w1). Record: bank/T11-001. Notes: local_kb/leader-11.md.

## Direction tried: window lane (prefill compute inside decode AR windows)
- Premise: each of the ~128 decode all-reduces per step (~46 us) holds only 16 CTAs (8 AR + 8 L2-prefetch), so ~110 SMs' tensor cores idle for ~5.9 ms of a ~24 ms step (T2-006). Prefill at M>=256 is compute-bound. Fill the windows with prefill GEMM slabs; prefill small prompts (<400 tokens: 75% of requests, 41% of prompt tokens, ~12.5 s of mixed-step increment per trial, T7) in the background.
- Why it seemed distinct from KB59: there, spatial sharing ran the whole step; here, the lane would run only while decode waits on PCIe.
- Napkin payoff if it works: ~+10%. It needed the decode slowdown s <= ~3% and ~80 TF of lane rate inside the windows.

## Measured kill (T11-001, NUMA-verified 2-process chain with the champion's prefetch on)
| arm | chain ms (r0) | vs champion path |
|---|---|---|
| A_pf (champion) | 4.917 | - |
| A_nopf | 5.423 | +10.3% |
| lane + pf, joined | 5.91-6.53 | +20..33% |
| lane + pf, unjoined | 5.75-6.30 | +17..28% |
| lane, no pf | 5.49-6.10 | +12..24% |
- The lane GEMM runs 65-91 us in the window, vs 22-31 us standalone. The host AR slows 45.7 -> 59-85 us beside it. Consumer GEMMs keep their prefetch benefit.
- At the real prefill rate the lane's FLOPs are worth 0.34-0.69 ms per chain. The cheapest arm costs 0.57 ms.
- Unjoined is ~joined, so the loss comes from interference with the SM-driven AR and the GEMM tails, not from drain.

## Takeaway
KB59/KB62's zero-sum picture also holds for temporal fill of the AR windows. The window is idle in SM count, not in the shared memory/PCIe path the AR depends on. Smallest M=256 Marlin slab is ~22 us (launch/latency floor), which is half a window by itself.

## Other directions considered (napkin, no GPU), all below the bar or already killed
- Exact INT8/INT4 tensor slicing: >= 6 int8 passes for bf16 x e4m3 exactness, so it is slower than bf16 (agrees with T2/T10).
- Strassen-type GEMMs: bf16 rounding of the block sums makes them not a reordering. Ruling risk, and <= 12% of GEMM flops anyway.
- Two-microbatch decode ping-pong with L2-shared weights: the AR needs ~75 MB of weight phase per layer kept in L2 (72 MB total). Infeasible (agrees with T10).
- Uneven TP for GPU0's locked clock, or a duplicated FP8 slice on GPU1 used in compute-bound steps: ~2-2.5%, exploit-sized (T9 holds it as an add-on).
- Tree verification with the MTP top-k: +~5% E for +14% step time (M=64 cliff, KB63).
- PP=2 with deeper k: break-even with TP (29 ms per 47 tokens vs 23.8 per 37).
- Replicated out_proj (all-gather instead of all-reduce): +16 us streaming to save ~11 us per layer.

## Opinion
I found no bar-clearing direction with honest odds. I'm returning the rest of the seed (~18 GPU-min).
