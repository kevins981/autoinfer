team:         T36
builds_on:    T31-003 (+ env knob T36_AR_K / T36_AR_G in host_staged_ar.py `_config`, default = champion)
performance:  "unofficial: decode-size AR with 4 cut-through chunks (T36_AR_K=4) 1029.6 vs champion K=1 1042.2 tok/s (-1.2%, one same-job pair, 100 prompts): within single-pair noise, no gain"
summary:
  hypothesis: the 43 us decode all-reduce is store-and-forward (write own payload to host, then read the peer's); the
              two directions of a PCIe link are independent, so reading chunk by chunk while the peer is still writing
              (cut-through, K=4) should cut it to ~25 us: -1.5 ms/step, ~+5% overall.
  change:     `_config()` for messages <= 600 KB returns (T36_AR_G, T36_AR_K) from the environment (default 8, 1 = the
              champion). Workspace t36-calib. Measured with the champion's own test-only trace (VLLM_HOSTAR_TRACE).
  results:    Observed:
    - job 364 (champion, 100 prompts c=16, trace on, 1033.2 tok/s): decode-size AR (30720 x 16 B, 57,948 calls/rank),
      medians from kernel start: own write published 17.4 us; peer flag seen 20.5-21.5; wait after own write 2-3 us
      (mean 3.2, p90 7.2, same on both ranks); read+add after the flag 16.4-17.4; last block end 45.1 (mean 43.0).
    - job 365 (same workspace, trace on in both arms, order K=4 then K=1): K=4: own write published 24.6 us, chunk-0
      flag seen 15.4, reader block end 41.0, last block end 42.0; K=1: 17.4 / 19.5 / 36.9 / 45.1. Throughput 1029.6
      (K=4) vs 1042.2 (K=1); ITL median 23.78 vs 23.27 ms.
    Estimate ~25 us reader end; measured 41.0 (vs 36.9 at K=1).
  conclusion: Interpretation: when the read overlaps the write, the write slows by the same amount (17.4 -> 24.6 us):
              kernel loads/stores to host-mapped memory move ~21-22 GB/s per GPU in total, whatever the direction mix.
              The decode AR is at the wire time of its two payloads (2 x 384 KB packed) for this mechanism; the peer
              skew is only ~3 us, so GPU 0's lower clock is not what the AR waits for. Chunking cannot help; only fewer
              bytes or a second transport running beside the kernel path (copy engines) could, and I did not test that.
              One pair: record as a screen, not a measured loss.
validity:     the knob only changes how the same bitwise payload is chunked (same sums, same order). Not a candidate.
