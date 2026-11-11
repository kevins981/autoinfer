team: T35
builds_on: T31-003
performance: "microbench only (milestone 1 of proposal T35-1): kill criterion hit; best run-ahead streamer 340.8 us/layer vs champion-like 317.5 (AR 12) / 360.2 (AR 39/50) in the same harness"
summary:
  hypothesis: decode GEMMs that run from L2 are issue-bound and leave DRAM idle. Streaming the next GEMM's weights into L2 concurrently with the current GEMM (instead of only inside AR windows) would bring a GDN layer from ~345 us toward its DRAM floor (~240-250 us).
  change: none to the server. Standalone microbenchmark by leader-35.w1 (agents/leader-35.w1/ms1/: ext.cu, bench.py, job2.out, job3.out, res3.json; GPU jobs 358, 359): GPU0, M=48, the champion's real kernels (t31 e4m3 GEMM, Marlin fp8 gate_up), 64 distinct layers of random weights (12 GB), spin kernels for the AR and small ops, a GDN dummy (12.5 MB read + 12.5 MB write, >=37 us), CUDA graphs of 32 layers.
  results: |
    Observed:
    - per GEMM, cold / warm / after ld-prefetch (us): qkvz 69.6 / 30.7 / 24.6; gdn_out 31.7 / 16.4 / 14.3; gate_up 143 / 123 / 82 (85 MB prefetched); down 73.7 / 34.8 / 26.6.
    - chain (us/layer):
      - V0 champion-like (in-AR hint prefetch): 317.5 at AR 12 us, 360.2 at AR 39/50 us.
      - V1 serial, no prefetch: 332.0 at AR 12, 397.0 at AR 39/50.
      - V2 streamer (AR 12): best 340.8, other configs 355-840.
      - V3 streamer (AR 39/50): best 369.4.
    - with the streamer running, every GEMM got slower than cold: qkvz 63 vs 53, gate_up 136 vs 120, down 65 vs 52 us.
  conclusion: |
    Interpretation:
    - The riskiest assumption is false. In-chain cold GEMMs already stream at ~0.8-0.9 TB/s, so they are DRAM-bound and a concurrent streamer only splits the same bandwidth (and costs SM slots and L2).
    - The only DRAM-idle time is the AR, norm and small-op windows (~50 us/layer), and the champion's in-AR prefetch already uses it.
    - The remaining gap between the in-situ layer (~345 us) and the DRAM floor (~240-250 us) is mostly AR latency (in situ 39/50 us windows vs ~12 us in the harness). That is a communication problem, which weight streaming can't fix.
    - Untested: green-context SM partitioning, stream priorities, lookahead > 1 GEMM. None of them changes the bandwidth argument.
validity: microbenchmark only, nothing to review for serving.
