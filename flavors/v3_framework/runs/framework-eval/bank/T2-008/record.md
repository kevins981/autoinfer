team:         T2
builds_on:    T1-014 (champion) + DeltaLog (T2 worker leader-2.w1, task 8)
performance:  "OFFICIAL #1: 880.1 tok/s (+4.4% vs T1-014 842.7, +38.4% vs BASE), GSM8K 94.4%, all gates pass. Unofficial: +3.91% ABBA job 65"
summary:
  hypothesis: The GDN MTP decode kernel writes the full fp32 128x128 state after every draft row (3 slots per request per head,
              ~75.8 MB SM->L2 writes per step on GPU0, ncu job 36). These writes drain after the kernel and slow the next
              GEMM (in-situ nsys: gate_up is 137.7 us in GDN layers vs 113.6 us in attention layers). Writing only the
              row-0 state plus a compact log of the draft rows' inputs (k fp32, v bf16, decay, beta: 784 B/row/head), and
              replaying the accepted rows lazily at the start of the next step, saves the drained bytes. It is exact:
              same recurrence and same order of operations.
  change:     csrc/gdn/gdn_decode_kernels.cuh (new) + fused_gdn_decode_kernel.cu: runtime log_mode on
              fused_gdn_decode_post_conv_mtp. Slots 1..2 hold a log (with NaN-payload markers) instead of a full state.
              New ops gdn_deltalog_materialize (hook i: before the Triton spec path in mixed steps) and
              gdn_deltalog_materialize_all (hooks ii/iii: before mamba_utils run_fused_precopy / run_fused_postprocess_align
              at prefix-cache block boundaries; materializes slots hi..max(bias,1) highest first).
              Gate VLLM_GDN_DELTALOG (default ON in this record), active only when the SSM state is fp32.
  results:    Observed:
              - Bitwise test test_deltalog.py (300 steps x 3 seeds, N=3, a in {1,2,3}, F/MF/MR steps, random bias): 0 failures
                (worker jobs 60, 63).
              - Smoke job 63 (T1-014 args, gate on): 2x8 requests crossing a 1600-token boundary all coherent;
                GSM8K-50 90.0%; acceptance 2.363 on bench 100. A first version corrupted 1-2/8 boundary requests
                (bias-slot log destroyed by in-place replay). Fixed before job 63.
              - Job 65: 128K request pass (gate on). ABBA (same workspace; A gate off = T1-014 path, B gate on):
                A 828.3 / 825.6, B 857.1 / 861.5 tok/s, +3.91%. Acceptance 2.335/2.337 (A) vs 2.33/2.33 (B).
                TPOT p99 29.5/30.7 (A) vs 34.0/34.0 ms (B); BASE limit 1.2 x 46.1 = 55.3 ms.
              Napkin was ~2-2.5 ms/step (~7%). Measured ~1.2 ms/step-equivalent.
  conclusion: Interpretation: a real win of about half the napkin. Part of the drain saving is probably eaten by the materialize
              kernels in mixed steps and boundary steps (and that may also be the TPOT p99 rise).
validity:     (leader-2) Exact rewrite of the GDN state recurrence: same fp32 math and op order, bitwise-tested against
              the original kernel. State stays fp32 and the KV path is unchanged. No change to weights, activation precision,
              communication or sampling. Not benchmark-specific. Memory unchanged (logs live inside the existing state slots).
              128K request passes. No accuracy A/B needed (exact); the official GSM8K gate still runs.
              Open: the worker is auditing the prefix-cache-hit / preemption paths by code reading (agents/leader-2.w1/deltalog_audit.md).

official #1 (2026-11-14T21:23): 880.1 tok/s at c=16 (mean of 2 trials: 880.4 / 879.7, CV 0.1%) (+38.4% vs BASE), GSM8K 94.4%, MTP acceptance 2.333, TTFT p50 153 ms, TPOT p50 16.7 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
