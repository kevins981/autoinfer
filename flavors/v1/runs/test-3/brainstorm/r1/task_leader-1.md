# Round 1 task for leader-1 (team T1)

## Direction
Main line: **R1-B**, a custom host-staged 2-rank all-reduce for this no-P2P PCIe box, replacing
NCCL for the TP all-reduces (and, if it pays, the logits all-gathers). First, one quick test:
**R1-A**, the one-flag prefill CUDA-graph test.

### Rationale
- R1-B: the all-reduce is ~21-23% of GPU time. NCCL gets 7.7 GB/s at decode size (491 KB) and
  13.3 GB/s at prefill size (4.6 MB). A pinned-host copy-engine microbench measured 20 GB/s in
  each direction at once, so there is ~1.7-2x headroom. NCCL env tuning was measured not to close
  the gap. A one-shot design: each rank writes its partial to shared pinned host memory, signals,
  reads the peer's partial and adds locally. a+b commutes, so both ranks get bitwise-identical bf16
  sums at baseline precision. Must be CUDA-graph capturable.
  - Go/no-go microbench target: decode size (M=48 x 5120 bf16) at <= 40 us, vs ~61 us for NCCL.
  - Main risk: flag round-trip latency through host memory at decode size.
  Details and numbers: brainstorm/r1/adjacent_field.md (Idea 1, M1/M2; its microbench is in
  brainstorm/r1/adjacent_field_mb/) and brainstorm/r1/systematic_engineer.md (Idea 2).
- R1-A first: steps that contain a prefill are ~35% of wall time. They run eager because
  `max_prefill_cudagraph_capture_tokens` defaults to 96 on non-SM100. Test it with e.g.
  `--compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'`.
  - The two brainstormers disagree on why prefill steps are slow:
    - systematic_engineer: CPU launch overhead.
    - adjacent_field: GPU 95% busy, with all-reduce rank skew.
  - This test tells us which is right, and so how much R1-B gains in prefill steps.
  - If it's a clear win, file it as its own record and request an official eval. It changes
    graph memory, so it needs the 128K-request test (task_definition.md).
  - Don't sink the round into fixing it if it breaks: note what broke and move on to R1-B.

## Budget
T1: 80 GPU-minutes this round (gpu/budget.json). Up to 2 workers. Keep ~15 min for official
evals.

## Reference to beat
CHAMPION = BASE: 631.4 tok/s at c=16, TTFT p99 1487 ms, TPOT p99 48.4 ms, GSM8K 92.8%
(champion.md). Gates: TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%.

## Scope notes
- Read task_definition.md, especially:
  - the human's rulings: env/config defaults may change in code if they are generic;
  - baseline-precision communication only, so no quantized all-reduce;
  - the memory-change test rule.
- The official eval takes only change.diff + serve_args.txt. A new all-reduce backend must be
  enabled through a code default or a serve flag, not through a job-script env var.
- The ideas you are not assigned are in tree.md (R1-*). Keep to R1-A and R1-B; mention anything
  else you notice in your report.
