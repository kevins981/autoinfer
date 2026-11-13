team:         T53
builds_on:    T53-001
performance:  "measurement only: codec posters x8 pass gate B on down (C4 +3.1 us, C2 +4.1 us GEMM slowdown); posting tail 9-14 us; copy-engine path lands too late"
summary:
  hypothesis: Gate B failed in T53-001 because the posters were slow and the bytes were raw. Posters using hostar's
              write pattern plus the hostar 0.78 lossless codec, or a copy-engine D2H path, should bring the GEMM
              slowdown to <=+5 us.
  change:     ext/t53c.cu v3/v4: codec posters (hostar's enc_block verbatim, one region per chunk); 256-thread
              posters with 8x16 B in flight per thread (= hostar's 512x4); 8 or 16 poster CTAs. Ablations:
              pmode 2 (no host stores), device-buffer target. CE path: side stream cuStreamWaitValue64 on the
              DEVICE chunk flag -> cudaMemcpy2DAsync D2H -> stamp, eager and in a CUDA graph. Bench: benchB2.py /
              jobB2.sh (down only, C2 = UN2 kseg1 P80, C4 = UN2 kseg2 P80).
              Jobs: 446 hung at the CE smoke and was killed by its 240 s timeout (4.0 GPU-min wasted, the process
              exited cleanly). Cause: the stream wait was enqueued before the GEMM that satisfies it. Fixed by
              enqueuing the GEMM + a release kernel first, on a non-blocking stream. Job 448: 0.1 min.
  results:    Observed (stock 57.1 us; no posting: C2 GEMM done 58.4, C4 63.5). Format: GEMM slowdown / tail
              (last posted - GEMM done):
              - C2: raw x8 +7.1/15.4, raw x16 +8.2/16.4, codec x8 +4.1/14.3, codec x16 +6.1/13.3.
              - C4: raw x8 +5.1/10.2, raw x16 +10.2/9.2, codec x8 +3.1/9.2, codec x16 +7.2/9.2.
              - Ablation, no host stores (raw or codec): GEMM +0 and tail 4-5. Device-buffer target: +0 and 3-5.
              - CE eager: GEMM +0, landing tail 23.6 (C2 chunk 0 done 30.7, landed 51.2; C4 chunk 0 18.4 -> 35.8).
                CE in a graph: 21.5-22.5, no better.
              - Write rate beside the GEMM: ~18 GB/s on the wire.
              - Codec round trip bitwise (self-loop readers: out == 2C). Host copies verified.
  conclusion: Interpretation: the GEMM slowdown comes entirely from the PCIe host stores, not from poster SM/L2
              work (encode included). It grows with bytes and write rate: the codec saves ~3 us, and 16 posters
              are worse than 8. Codec x8 meets gate B at C2 and C4 but leaves a 9-14 us posting tail. The CE path
              doesn't slow the GEMM, but data lands 17-20 us after the flag, so it is out.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
