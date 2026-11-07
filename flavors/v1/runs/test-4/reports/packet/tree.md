# TREE (run test-4): what was tried
Root: whole-engine speedup of c=16 output tok/s (focus.md). ★ = champion path.
- BASE: 635.9 tok/s official ★
- [r1, T1] R1-A host-staged all-reduce over pinned shared host memory (no P2P)
  - T1-001 ★: host AR backend for bf16 <= 2 MiB (NCCL above); bitwise = NCCL. OFFICIAL 672.5 (+5.7%), gates pass.
    - T1-002: lower-overhead AR kernel variants (unified blocks, per-warp flags). Microbench null (-2 us / worse). ✗
    - T1-003 ★ CHAMPION: T1-001 + T2-003 flags. OFFICIAL 691.3 (+8.7% vs BASE, +5.7% vs T2-003), gates pass.
      - [r2, T1] overlap prefill-step AR with compute
        - T1-004: chunked row-parallel GEMM + side-stream NCCL AR, Marlin reserves 2 SMs. Big prefill steps -10% (nsys),
          but same-job 400-prompt A/B 0.0%. ✗ null (unmeasurable at c=16)
      - [r2, T2] cost of the spec-decode step
        - T2-005: depth 3 on the champion: step +12.6% for +16.7% tokens; flat end-to-end; TTFT p99 2185 (gate fail). ✗
        - T2-006: + T2-004 greedy target fast path, without batch-sharded sampling. OFFICIAL 699.5 (+1.2%) but ABAB same-job
          +0.2% (noise). ≈ tie, not promoted.
        - T2-007: analysis: real decode idle ~1.6 ms/step (CPU Python, attention-metadata builds); node-level nsys inflates it. no change
  - (check) NCCL_PROTO=Simple at prefill sizes: no gain. ✗
- [r1, T2] R1-B vocab-parallel head diet -> draft depth
  - T2-001: local-argmax drafting + FP8 (W8A8) draft lm_head (flags). Unofficial +2.4% same-job; acceptance unchanged.
    - T2-002: + num_speculative_tokens 3. Acceptance 2.757 but -3.9% tok/s; TTFT p99 2225 (gate fail). ✗
    - T2-003 ★ (merged into T1-003): + --enable-batch-sharded-sampling. OFFICIAL 653.9 (+2.8%), gates pass.
    - T2-004: custom greedy vocab-parallel target argmax (exact). No gain over T2-003. ✗ null
