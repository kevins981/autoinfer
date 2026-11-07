# TREE (run test-4): what was tried
Root: whole-engine speedup of c=16 output tok/s (focus.md). ★ = champion path.
- BASE: 635.9 tok/s official ★
- [r1, T1] R1-A host-staged all-reduce over pinned shared host memory (no P2P)
  - T1-001 ★: host AR backend for bf16 <= 2 MiB (NCCL above); bitwise = NCCL. OFFICIAL 672.5 (+5.7%), gates pass.
    - T1-002: lower-overhead AR kernel variants (unified blocks, per-warp flags). Microbench null (-2 us / worse). ✗
    - T1-003 ★ CHAMPION: T1-001 + T2-003 flags. OFFICIAL 691.3 (+8.7% vs BASE, +5.7% vs T2-003), gates pass.
  - (check) NCCL_PROTO=Simple at prefill sizes: no gain. ✗
- [r1, T2] R1-B vocab-parallel head diet -> draft depth
  - T2-001: local-argmax drafting + FP8 (W8A8) draft lm_head (flags). Unofficial +2.4% same-job; acceptance unchanged.
    - T2-002: + num_speculative_tokens 3. Acceptance 2.757 but -3.9% tok/s; TTFT p99 2225 (gate fail). ✗
    - T2-003 ★ (merged into T1-003): + --enable-batch-sharded-sampling. OFFICIAL 653.9 (+2.8%), gates pass.
    - T2-004: custom greedy vocab-parallel target argmax (exact). No gain over T2-003. ✗ null
