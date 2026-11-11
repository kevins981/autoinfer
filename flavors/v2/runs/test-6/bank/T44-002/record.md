team:         T44
builds_on:    T37-002 (env knobs T44_RATIO / T44_PACE_GBPS in host_staged_ar.{cu,py}, default off = champion)
performance:  "stage-1 kill: no variant cuts exposed hostar AR >= 10% at R >= 128 (best -6.5% at R=255, worse p90)" (job 401, 0.1 GPU-min)
summary:
  hypothesis: T44-001's writer-oversubscription mechanism applies to the champion's hostar AR; a 1:2 writer:reader grid or
              paced writers shorten the exposed AR in small mixed steps (<256 tokens; larger steps are DMA-only, PINGPONG=1).
  change:     writer-block count Gw and per-warp wire-byte pacing in hostar (default off). Bench agents/leader-44.w1/s1/:
              two ranks, node-0 slots, server's own AR classes (codec on, no L2-prefetch blocks), chain of [AR][~30 us
              spin consumer] x135, exposed = AR launch -> consumer start. All variants bitwise equal to the reference.
  results:    Observed (rank 0 median us; rank 1 within ~2 us):
              R=48: champ 45.1, best variant 51.2 (pace 18-20). R=64: 65.5 vs 69.7 (1:2). R=128: 105.5 vs 109.6 (pace 20-22).
              R=192: 150.5 vs 158.7. R=255: 192.5 vs 179.2 (1:2, p90 208.9 vs 201.7). Ratio variants bimodal.
              Champion hostar wire rate 9.5-10.6 GB/s per direction at R=128-255 (< half of ~22 GB/s duplex).
              DMA+DMA two lanes as the server issues them: R=300/700/1100 -> 247/438/688 us = 12.4/16.4/16.4 GB/s per
              direction vs ~21.5 GB/s CE duplex standalone (T44-001). Final-norm AR at split sizes: no hostar (reduced in-chain).
  conclusion: Interpretation: the champion's hostar is chain-latency-bound (write -> fence -> flag -> read per chunk + ramp),
              not link-bound, so there is no write backlog to relieve; pacing/fewer writers only lengthen the write side.
              T44-001's mechanism is real but has no target on T37-002: large mixed steps are copy-engine-only (already
              duplex), small ones and decode are latency-bound. Side observation (bound, not a measured lever): the two-lane
              DMA AR runs ~25% below the standalone CE duplex rate at 700-1100 rows (688 vs ~525 us at 21.5 GB/s).
              Caveats: spin consumer stand-in; no prefetch blocks; no SM contention from real GEMMs.
validity:     knobs default off; bitwise-equal outputs in all variants; no served change proposed.
