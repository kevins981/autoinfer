# TREE (run test-3): what was tried, and how the champion came about
Root: whole-engine speedup of output tok/s at c=16 (focus.md). BASE = 631.4 tok/s official.
Champion: **T1-009, 730.9 tok/s official (+15.8% vs BASE)**. Best measured: T1-012, 749.7 (+2.6% over T1-009, below the 3% bar; pending the human's decision).
Legend:
- ★ = on the champion's path.
- ✗ = failed or worse.
- ○ = null result (within noise) or diagnosis only.
- A child builds on its parent.
- Scored but untried ideas are in ideas.md.

## Direction R1-A: CUDA-graph mixed prefill steps (round 1, T1, from systematic_engineer)
- ○ **T1-001** (from BASE): serve flag max_prefill_cudagraph_capture_tokens=2048.
  - Result: unofficial 645.2 (+2.2%, noise); 128K pass.
  - Conclusion: on NCCL, prefill steps are GPU-bound, not CPU-launch-bound (KB1).
  - The flag later pays off stacked on the host-staged AR (see T1-005).

## Direction R1-B: custom host-staged 2-rank all-reduce, no P2P (round 1, T1, from both brainstormers)
- ★ **T1-002** (from BASE): host-staged AR through pinned /dev/shm, a code default for TP=2 without P2P.
  - Result: official 662.8 (+5.0%).
  - Conclusion: bitwise equal to NCCL; the decode AR goes 61 -> 50 us, near its PCIe floor (KB2, KB5).
  - ○ **T1-003**: diagnosis only (in-server nsys + microbench), no new code.
    - Conclusion: the prefill-step AR skew is not caused by transport (KB3).
  - ★ ○ **T1-004**: + host-staged all_gather.
    - Result: unofficial 666.8 (~0, noise).
    - Conclusion: the logits all-gathers are small (KB6); kept as a building block.
    - ★ **T1-005**: + T1-001's prefill-graph flag.
      - Result: official 699.2 (+10.7% vs BASE).
      - Conclusion: the AR and the flag stack super-additively (KB4). Champion after round 1.

## Direction R2-A: logits diet (round 2, T1, from both brainstormers; builds on T1-005)
- ★ ○ **T1-006** (from T1-005): + use_local_argmax_reduction for the drafts.
  - Result: unofficial +1.1% (noise alone).
  - ★ **T1-007**: + FP8 draft lm_head (ruling 1).
    - Result: unofficial +3.0% vs T1-005; acceptance unchanged (KB10); 128K pass.
    - ✗ **T1-008**: + num_speculative_tokens=3.
      - Result: unofficial -6.9%. Acceptance 2.72, but TTFT p50 210 -> 792 ms and the p99 gate fails.
      - Conclusion: cause undiagnosed (KB11).
    - ★ **T1-009**: + greedy local-argmax fast path for target verification (code).
      - Result: official 730.9 (+4.5% vs T1-005, +15.8% vs BASE).
      - Conclusion: exact including ties, checked op-level (KB12). **CHAMPION** (round 2).

## Direction R3: spec-decode capacity + GDN state traffic (round 3, T1, from both brainstormers; builds on T1-009)
- ✗ **T1-010** (from T1-009, R3-A): k=3 + --kv-cache-memory 6.0 / 5.625 / 5.5 GiB.
  - Result: the server never starts (OOM in CUDA graph capture).
  - ✗ **T1-011**: k=3 + KV 5.6 GiB + prefill graph cap 1024.
    - Result: starts, then runtime OOM in the bench.
    - Conclusion: KV ~5.0 GiB is the practical ceiling; k=3 at 16 requests needs smaller per-request state, not more KV (KB16).
- **T1-012** (from T1-009, R3-B v1): GDN spec-state checkpoint + replay (write 1 state per step, not k+1), with a bitwise flush for outside readers.
  - Result: official 749.7 (+2.6% vs T1-009, +18.7% vs BASE); all gates + 128K pass.
  - Conclusion: exact (bitwise unit test with a negative control); ~2 ms/decode step saved, consistent with the microbench and the A/B (KB15).
  - **Below the 3% bar vs the champion, so not promoted; best measured result, pending the human's decision.**
