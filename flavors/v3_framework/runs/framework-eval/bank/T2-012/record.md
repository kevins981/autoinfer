team:         T2
builds_on:    T3-009 (champion) + hostmem AR release/acquire handshake (T2 worker leader-2.w1, task 13, ws t2-rel on T2-011)
performance:  "OFFICIAL #1: 953.1 tok/s (+0.17% vs T3-009 951.5, within trial noise: CV 0.8%), GSM8K 94.4%, all gates pass. Unofficial: +1.08% ABBA on T2-011 (job 129)"
summary:
  hypothesis: The decode hostmem AR (135 per step, ~44 us at 491 KB) costs more than its PCIe data movement. Measured
              with both GPUs at once, SM-issued, real slots (job 122): write alone 19.0 us, read alone 21.2 us, both
              concurrently 29-32 us. The remainder is handshake and fence latency, so a cheaper handshake should help.
  change:     hostmem_ar.cu: ar_kernel templated on RA. RA=true replaces each thread's __threadfence_system before the
              flag store with __syncthreads + thread-0 st.release.sys. The poll becomes thread-0 ld.acquire.sys
              (no trailing system fence). The ordering argument is in a kernel comment. RA=false is the original code.
              hostmem_all_reduce.py: env VLLM_T2_AR_RELACQ (default 1) selects it. T3-003's prefetch CTAs are untouched.
  results:    Observed:
              - Micro, job hm-var: release/acquire saves 2.5-3.1 us per AR at 491/164/82 KB vs the fence variant
                in the same job.
              - Sub-chunk pipelining (S=2/3/4, small first chunk) is always worse (+4-12 us), because each extra
                handshake costs ~5 us.
              - torch.equal vs NCCL in 66 configs. Stress test: 10k back-to-back random ARs per mode, 0 mismatches,
                with and without prefetch CTAs (worker jobs ra-test2).
              - ABBA job 129 (ws t2-rel = T2-011 + this; A RELACQ=0): +1.08%, all B > all A. Acceptance 2.33-2.35.
                TPOT p99 30.3/30.4 (A) vs 34.4/40.8 ms (B), under the 55.3 limit. HostMem AR enabled in all 4
                servers. One EngineDeadError traceback during forced shutdown of server 2_B, after the bench completed
                (shutdown race, not in serving).
              Napkin ~0.35-0.4 ms/step (~1.2%), measured +1.08%.
  conclusion: Interpretation: the decode AR is latency-bound by PCIe round trips (flag and fence), not bandwidth-bound.
              Removing the full system fence is the gain. More pipelining loses, because every handshake costs a round trip.
              Port: hostmem_ar.cu / hostmem_all_reduce.py are byte-identical in T2-011 and T3-009 (cmp), and were
              copied onto ws t2-rel9 (from T3-009).
validity:     (leader-2) Communication stays bf16 with the same reduction order on both ranks, bitwise equal to NCCL
              and to the original kernel (tests above). Only the memory-ordering primitives change: the release/acquire
              chain gives the same happens-before as the fence (the WAR case on double-buffered slots is covered by
              the kernel boundary and the release ordering prior loads). No math, precision or memory change; not
              benchmark-specific.

official #1 (2026-11-15T01:36): 953.1 tok/s at c=16 (mean of 2 trials: 958.3 / 947.9, CV 0.8%) (+49.9% vs BASE), GSM8K 94.4%, MTP acceptance 2.32, TTFT p50 143 ms, TPOT p50 15.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)

follow-up (leader-2): same-job ABBA vs T3-009 (job after 133; agents/leader-2/abba_ra9, ws t2-rel9, A RELACQ=0):
  A 936.3/943.4, B 948.1/941.8 tok/s -> MIXED, +0.54% (B3 < A4). Not consistent; not resubmitted (T2 budget ~17 min).
  Interpretation: on T3-009 the gain is at most ~0.5% and within noise; the earlier +1.08% on T2-011 doesn't carry over cleanly.
