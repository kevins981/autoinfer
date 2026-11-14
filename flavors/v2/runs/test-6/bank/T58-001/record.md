team:         T58
builds_on:    T57-004 (code); first measured on T52-005 via env (jobs 498, 503)
performance:  "unofficial: same-job ABBAAB on T57-004 job 505 (400 prompts): +0.72%, all B > all A (B 1232.1/1236.6/1230.7 vs A 1225.2/1225.8/1221.7); earlier on T52-005 job 503 +0.26% mixed"
summary:
  hypothesis: (sweep candidate M1, reports/T58-sweep/6-ranking.md) mixed steps with 97-255 rows use the host-staged AR but its L2 prefetch
              of the next GEMM's weights is gated off by VLLM_HOSTAR_L2PF_MSG_MAX = 1 MiB (a 128-row AR is 1.31 MB), so their bs_gemm GEMMs
              run cold (KB131 names the loss of prefetch; the gate itself was never raised). Raising the gate lets those ARs prefetch.
  change:     default of VLLM_HOSTAR_L2PF_MSG_MAX 1 MiB -> 2.7 MB in host_staged_ar.py (one line; env 1048576 restores the old gate).
              Covers hostar ARs up to 255 rows (larger steps use the DMA split chain). Generic size threshold, nothing benchmark-specific.
  results:    Observed: screen job 498 (nsys node trace, 96 prompts, T52-005 + env) vs job 485 (same, no env), GPU1:
              hostar grid 16 -> 24 in 128-row mixed steps (prefetch blocks present); gate_up bs_gemm 151 -> 117 us/call; hostar per call
              101 -> 102 us (not slower). Median GPU span of mixed steps: 97-128 rows 43.1 -> 40.8 ms (-2.3; napkin -1.5..-2.3),
              129-160 -0.6, 161-192 -0.9, 193-255 +4.1 (n=5-6, noisy); all 97-255: 46.67 -> 46.03 ms.
              ABBA job 503 (t58-champ = T52-005, A no env, B env, 400 prompts): A 1216.9/1217.4, B 1224.7/1215.9: +0.26%, mixed.
              Acceptance 2.675-2.684 all starts.
              ABBAAB job 505 (t58-c57 = T57-004 + this default; A = env VLLM_HOSTAR_L2PF_MSG_MAX=1048576, B = default; 400 prompts,
              fresh starts): A 1225.19/1225.84/1221.65, B 1232.08/1236.57/1230.65 tok/s: +0.72%, all B > all A; acceptance A 2.669-2.674,
              B 2.668-2.685.
  conclusion: Interpretation: the mechanism works where predicted (<=128-row steps; ~150 per trial); napkin ~0.3-0.5% e2e, measured +0.72%
              same-job on T57-004 (6 starts, all B > all A; the 4-start run on T52-005 was within noise). Official eval requested.
validity:     leader-58: cache hints (prefetch.global.L2::evict_last) issued by extra blocks of the AR kernel, no data dependence: outputs
              bitwise unchanged (same mechanism already used for decode ARs since T1-019). No precision, sampling or memory change
              (prefetch blocks are part of the already-captured AR kernel; no new buffers). Exact by construction; no accuracy run needed.

official #1 (2026-11-13T23:36): 1229.6 tok/s at c=16 (mean of 2 trials: 1231.5 / 1227.7, CV 0.2%) (+93.3% vs BASE), GSM8K 93.6%, MTP acceptance 2.663, TTFT p50 136 ms, TPOT p50 11.7 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
