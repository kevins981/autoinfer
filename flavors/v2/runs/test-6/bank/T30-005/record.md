team:         T30
builds_on:    T30-004
performance:  "unofficial (host-free GPU replay of the same eligible step): wave schedule saves 6-7 ms of 108-177 ms vs eager-joined chain (3-6%); AR stays ~33-50 ms exposed"
summary:
  hypothesis: With the host out of the way, the two-wave schedule hides most of the exposed all-reduce of big mixed steps (AR-off floor 72/90/122 ms at 527/734/1010 tokens).
  change:     Test only (T30_WAVE_BENCH=1 in workspace t30-wave, no change to the serving path): for the first eligible step in each size bin the server snapshots GDN/KV state,
              then for {joined chain (W2), two waves (W1)} x {real DMA AR, AR replaced by a copy} it (a) captures the whole eager forward of THAT step into one CUDA graph
              (works: cores are capturable when the same step is replayed) and times 4 replays with CUDA events, (b) runs it once under torch.profiler (CUDA activities),
              restoring state between runs. Replay = pure GPU time, no host.
  results:    Observed (graph replay ms; n tokens; TP2 rank 0; 4 replays each, spread <=2 ms; raw: agents/leader-30.w2/hostfree/RESULT_raw.txt):
                n=527  W2 108.2-108.7 | W1 98.6-105.3 (median ~102) | W2 AR-off 71.8-72.5 | W1 AR-off 71.7-72.7
                n=734  W2 130.2-130.9 | W1 123.5-123.8              | W2 AR-off 89.7-90.9 | W1 AR-off 88.8-90.2
                n=1010 W2 176.4-176.6 | W1 170.3-170.9              | W2 AR-off 121.4-122.1| W1 AR-off 127.6-128.5
              Exposed AR in the joined chain: 36 / 40 / 54 ms. The wave schedule recovers ~6 / 7 / 6 ms of it (~17% / 17% / 11%). W1 AR-off is not faster than W2 AR-off (equal at
              527/734, +6 ms at 1010: two M/2 GEMM sets). Profiler (eager run): busy-union 124.7/116.8 (527), 134.8/143.1 (734), 179.3/186.6 (1010) ms for W2/W1, span = busy + host stalls
              (W1 eager wall 189-212 ms, host-bound as in T30-004).
              Extra: eager-joined replay (108/131/177) vs the champion's breakable-graph step forward GPU (T30-004 W0 medians 99.8 @448, 138.8 @715, 173.6 @920): the same order, so the
              replay numbers are comparable to the graph path.
  trace (n=734, one graph replay each under torch.profiler CUDA activities, rank 0; files agents/leader-30.w2/trace/, analysis analyze_trace.py):
              Observed: W1 replay span 127.5 ms = kernel-busy-union 94.5 + 33.0 ms of idle gaps (no kernel on any stream); W1 AR-off span 86.9 = busy 86.9, no gaps (AR-on kernel busy is +7.6 ms: the
              AR add kernels). The 33 ms are 128 gaps (median 204 us, p90 377 us, max 657 us). EVERY one of the 128 gaps starts with a wave-B bs_gemm kernel ending (its out_proj/down_proj, whose output
              B must now all-reduce) and ends with wave A's AR add_kernel starting (the reduce after A's H2D); in 128/129 gaps all 4 copy engines streams (D2H+H2D of both lanes) are active, and a copy is
              active during 95% of the gap time. Copies: 2048 (1924 MB, 0.94 MB each, median 45 us); copy-union 58.9 ms, of which 27.5 ms overlaps kernels; 56% of summed copy time has no kernel running.
              Copy rate with a kernel running 20.7 GB/s vs with none 21.3 GB/s (no slowdown from compute). GEMM-like kernels (median) 143 us with copy overlap vs 161 us without (no slowdown; mixed
              GEMM shapes) and 145 us in the AR-off run. W2 (joined chain) for comparison: gaps 39.0 ms (128 x median 342 us), GEMMs with copy overlap median 217 us vs 142 us without (here contention shows).
              Interpretation (cause): NOT copy/compute contention in the wave schedule (rates and GEMM times are unchanged by concurrency) and not a mem-op/ack serialization visible in the trace; it is
              AR latency vs the other wave's available compute. Each AR is a serial D2H -> peer flag -> H2D -> add chain (~0.3-0.5 ms per ~1 MB message at 21 GB/s per copy plus signalling); the
              other wave has only one GEMM (out_proj ~ 0.1-0.2 ms or one MLP GEMM) to run before it needs its own AR, so both waves end up waiting on AR results at the same instant (B's last GEMM
              ends, A's reduce has not run yet). Two waves per layer give each AR only ~1 GEMM of cover; the schedule would need many more, smaller waves or a finer stagger (more AR concurrency),
              each with M/k GEMMs and k x the host work. Not tested.
  conclusion: Interpretation: the schedule does NOT hide the AR. With a perfect host it would give about -6 ms on a 108-177 ms step (3-6%), 34-50 ms above its own AR-off floor at 734/1010
              tokens, i.e. the criterion 'W1 <= AR-off floor + 10 ms' fails by 3-4x, and W1 ~ W2 within 6 ms. Likely cause (not tested): the AR is a PCIe-bound copy stream
              (7.5 MB x 128 ARs x 2 directions at 734 tokens) that competes with the GEMMs' HBM/SM traffic and with the other wave's AR on the same link, so overlap saves little.
              Eligible steps are ~19% of GPU time, so even the ideal -6 ms is <1% of tok/s; plus the host problem (T30-004). Direction does not pay: stop.
validity:     test-only code path (env flag, default off); serving path as in T30-004.

## Follow-up: k waves (3, 4), host-free replay, n=734 (job 294, 2.4 GPU-min; hostfreek/)
Observed (CUDA-graph replay of the same eligible step, steady-state ms, rank 0):
| variant | with AR | AR-off | exposed AR |
| W2 joined chain (champion) | 131.0 | 88.3 | 42.7 |
| wave k=2 | 123.6 | 87.4 | 36.2 |
| wave k=3 | 116.2 (still falling: 138.6/123.0/119.1/116.2) | 90.2 | 26 |
| wave k=4 | ~124 (128.7/124.1/129.4/122.6, noisy) | 92.7 | ~31 |
Interpretation: k=3 recovers a further ~7 ms over k=2 but is ~26 ms above its own AR-off floor; k=4 is worse than k=3 (smaller waves = less cover per AR, more per-AR fixed latency, higher floor). Success line (within ~10 ms of own floor) NOT met. Best wave variant is 14.8 ms (11%) under the joined chain host-free, before any host-enqueue cost (k=3 eager wall is 923 ms: host-bound far worse than k=2).
Caveats: n=1010 did not occur in the 190 s window (bin never triggered), so only n=734 was measured; k=3 replay times had not converged over 4 replays; the split label in logs is the k=2 split. Logging bug (T30WAVESTEP %d with list) is harmless.
Code: workspace t30-wave (T30_WAVE_K, EXTRA_PLANS, NL=4 in dma_staged_ar.cu, T30_WAVE_KBENCH=1). Script hostfreek.sh.
