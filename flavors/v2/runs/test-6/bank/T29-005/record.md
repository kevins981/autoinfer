team:         T29
builds_on:    T29-003
performance:  "unofficial: within noise (abba A B B A, job 211: B 1028.4/1032.5 vs A 1034.3/1028.4, -0.08%, mixed); mixed-like ITL sum -1.8% in both B starts; decode median ITL unchanged"
summary:
  hypothesis: |
    With W8A8 GEMMs (T29-003) the two token-split chunks' GEMMs run concurrently on their two streams.
    Marlin used to fill every SM, which serialized them. Running concurrently, both chunks finish each phase together,
    and their DMA ARs bunch and interleave on the copy engines while the GPU idles (agents/leader-29.w2/crit/NOTES.md,
    from w1's job-199 nsys: ~650 us idle per GDN layer at M~734).
    Ordering chunk 1 after chunk 0 per GEMM phase should restore the ping-pong stagger, so chunk 0's AR overlaps
    chunk 1's compute.
  change: |
    Workspace t29-ord = T29-003 plus one function changed, _qwen3_5_split_chain in
    vllm/models/qwen3_5/nvidia/model.py (patch: agents/leader-29.w2/order/patch_order.py, patch_check.py;
    delta vs T29-003: notes/delta_vs_T29-003.diff).
    - VLLM_T29_ORDER (default 0 = T29-003 exactly):
      - 1: chunk 1's out_proj waits on a CUDA event recorded after chunk 0's out_proj, and chunk 1's MLP waits on an
        event after chunk 0's MLP.
      - 2: additionally, chunk 1's ARs wait for chunk 0's AR of the same phase (not tested).
    - VLLM_T29_ORDER_CHECK=1 (test-only, default off): startup bitwise self-check, described in results.
    - The A/B used T29_W8A8=2 in both arms; B added VLLM_T29_ORDER=1 VLLM_T29_ORDER_CHECK=1.
  results: |
    Observed:
    - Bitwise self-check (B servers, eager warmups before graph capture, startup only; notes/ordercheck.txt):
      - At every captured split size M=256..1536 (step 32, incl. 736), the first 2 split-chain calls ran the same
        layer input through mode 0, mode 0 again (control) and mode 1.
      - Compared bitwise: the chain output, the residual and the next layer's per-chunk projections (4 tensors).
      - 164/164 per server (2 servers, both ranks): mode0 == mode0 and mode0 == mode1 bitwise.
    - abba.sh (job 211, 300 prompts, c=16, serve_args = T29-003's, order A B B A, both arms workspace t29-ord):

      | start | tok/s | median ITL | mean ITL | TTFT p50/p99 | TPOT p99 | acceptance |
      |---|---|---|---|---|---|---|
      | A1 | 1034.3 | 24.89 | 32.94 | 149/853 | 28.6 | 2.334 |
      | B1 | 1028.4 | 24.93 | 32.86 | 152/902 | 31.5 | 2.325 |
      | B2 | 1032.5 | 24.89 | 32.84 | 161/927 | 28.6 | 2.333 |
      | A2 | 1028.4 | 24.93 | 32.99 | 143/666 | 31.9 | 2.324 |

      Verdict: mixed, -0.08%.
    - Prometheus ITL deltas (w1's hist.py, notes/per_start.txt):
      - mixed-like sum: B 295.0/297.7 s vs A 301.1/302.7 s (-1.8%, both B below both A);
      - gaps > 50 ms: B 2669/2739 vs A 2764/2781;
      - total ITL sum: B 953.1/954.2 vs A 956.2/962.0 s.
    - KV cache 267,721 tokens in all starts; model load 14.45 GiB.
    - 4 starts = ~10 GPU-min.
  conclusion: |
    Interpretation:
    - Exact: same kernels, same inputs, only stream ordering. Bitwise equality was confirmed in-server at every
      captured split size.
    - No throughput effect that 2 pairs can resolve.
    - The mixed-like ITL sum is consistently ~1.8% lower (~5.6 s of summed per-request gaps, about 0.35 s of wall per
      300-prompt run, ~0.4%). That is consistent with a small real effect below the A/B's resolution, but it is not
      shown.
    - My napkin estimate (~25 ms per M~734 step) overstated it. It came from a TRACE=node nsys of the older Triton
      mode (job 199), where launch inflation and the slower kernels exaggerate the idle. Steps at M~734 are also a
      small share of a trial.
    - Not shown in this job: whether the stagger actually happened in graph replays (no nsys taken).
    - Next if pursued: one untraced nsys of mode 1 vs mode 0 (copy/compute overlap per layer), before more A/B minutes.
    - Mode 2 not tested.
validity: |
  (leader-29.w2 draft; leader to review)
  - Scheduling-only change: cross-stream CUDA event record/wait between the two token chunks of the existing split
    chain. No change to any kernel, input, reduction order or precision.
  - Each chunk's rows are computed by the same kernel calls as in T29-003, and the ARs are the same DMA lanes. Only
    when chunk 1's work may start changes.
  - Verified bitwise in-server at all 41 captured split sizes (mode 0 vs mode 1, with a mode-0 control).
  - Default off (VLLM_T29_ORDER=0 = T29-003). The check code is env-gated (off by default) and startup-only.
  - The FP8 GEMM/quant path and configs are unchanged from T29-003, so T29-003's accuracy evidence carries over.
notes:        notes/abba.log, notes/job211.log, notes/per_start.txt, notes/ordercheck.txt, notes/delta_vs_T29-003.diff; run dir agents/leader-29.w2/order/abba/; analysis agents/leader-29.w2/crit/
