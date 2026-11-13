team:         T53
builds_on:    T53-004
performance:  "measurement only: hybrid (chunk 0 by copy engine, chunk 1 by codec SM posters) saves 2.6/4.1 us down->AR, the same as even C2 in the same job (2.6/3.1); gate (>=18 us) FAILS. Producer side closed."
summary:
  hypothesis: SM-driven host traffic during the GEMM is throughput-limited (T53-004), and copy-engine D2H costs the
              GEMM ~0 but lands 17-20 us late (T53-002). Moving C2's chunk 0 to the copy engine hides that latency
              under the ~36 us of GEMM still to run and frees SM/PCIe capacity for chunk 1. Prediction:
              down->AR ~100-105 vs ~123 us (saving 18-23 us).
  change:     ext/t53c.cu v7: transport mode per chunk (spec[16+c]): 0 = SM posters; 1 = copy-engine raw (posters
              skip the chunk; a side stream does cuStreamWaitValue64 on the DEVICE chunk flag, then a 2D D2H copy,
              then cuStreamWriteValue64 of the host flag); 2 = posters encode into a device buffer and set a device
              flag, then the CE copies the encoded region and writes the host flag. Side ops are enqueued after
              everything that satisfies them (GEMM + release kernels), and the CE path is warmed up first.
              benchC.py: same 2-GPU setup as T53-003/004 (C2 = [2560k1, 2560k1], UN2, P80, codec x8 posters,
              8 readers with a single host poller, 8 prefetch CTAs, chain down -> AR -> qkvz). Job 459, 1.0 GPU-min.
  results:    Observed (job 459, medians r0/r1; same-job champion: down->AR 110.1/111.6, full chain 145.9/147.5. This
              job's baseline is lower than job 454's 122.9, so absolute savings are smaller here):
              - C2 even codec: down->AR 107.5/108.5 (+2.6/+3.1); full +3.6/+4.1. GEMM done 69.6/69.1; chunk 1
                posted at 85.0; readers done at 104.4/105.5.
              - Hybrid, chunk 0 CE raw: 107.5/107.5 (+2.6/+4.1); full +3.6/+4.1. GEMM done 64.5/64.0 (5 us
                earlier); chunk 0 seen by the peer at 57.9/59.4 (21-22 us after done); chunk 1 posted at 85.0/84.5;
                readers done at 104.4/105.5.
              - Hybrid, chunk 0 CE codec: 120.3/119.3 (-10.2/-7.7). Encode done at 46.1; CE landed and seen at 77.8
                (~32 us of CE latency).
              - Numerics: every AR result is bitwise equal to the sum of the two partials (kseg 1, so also equal to
                the champion's AR); ranks equal.
  conclusion: Interpretation: gate fails.
              - Taking chunk 0 off the SMs speeds the GEMM up by ~5 us.
              - The end-to-end time doesn't move. Chunk 1 still lands ~16 us after the GEMM, and the peer read +
                add of chunk 1 (~20 us) remains the critical tail.
              - CE latency for chunk 0 (~22 us) is hidden but buys nothing, because the tail is set by the last
                chunk's post + read at the effective link rate.
              - A CE path for the last chunk would be later still (17-32 us CE latency).
              - Across jobs, the best producer-side variant saves 3-12 us per down boundary against a baseline that
                itself varies 108-124 us between jobs. Producer-side chunked posting is closed.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
