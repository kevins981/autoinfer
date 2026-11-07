# Round 2 briefing (run test-3)

## The problem
Same as round 1 (brainstorm/r1/briefing.md, task/TASK.md): output tok/s at c=16 for
Qwen3.8-27B-FP8 on 2x RTX 4090 without P2P. Rules: task_definition.md. It now includes the
human's round-1 rulings:
- FP8 draft lm_head: allowed, inside the drafter only.
- Reduced-vocab draft lm_head: allowed, if built from non-benchmark text and verification stays
  full-vocab and exact.
- Generic env/config defaults may change in code.

## Facts
- CHAMPION = **T1-005**: 699.2 tok/s official (+10.7% vs BASE 631.4; 3 trials, CV 0.4%). TTFT
  p50 210 / p99 1309 ms, TPOT p50 20.9 / p99 50.6 ms, GSM8K 93.2%, MTP acceptance 2.34. It is:
  - a custom host-staged 2-rank all-reduce + all-gather through pinned /dev/shm memory (code
    default on TP=2 without P2P);
  - the serve flag `max_prefill_cudagraph_capture_tokens=2048`, so mixed prefill steps use
    piecewise CUDA graphs. Attention and GDN still run eager inside them.
  - Record and diff: bank/T1-005/. Official results: bank/T1-005/official/1/.
- Gates (vs BASE): TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%.
- Raw profiles:
  - BASE: task/baseline/ (task/PROFILING.md).
  - Champion T1-005, nsys at c=16 on 64 prompts, not analyzed beyond the AR skew numbers:
    workspaces/leader-1.hostar2/out/nsysB/ (profile.sqlite, kern_sum.csv, breakdown.txt).
  - T1-002 nsys (the host AR without the prefill flag): workspaces/leader-1.hostar2/out/nsys/.
- Insights confirmed so far (shared_kb.md, KB1-KB7), in short:
  - prefill steps are GPU-bound, not CPU-bound;
  - the decode AR is ~5 us from its PCIe floor;
  - prefill-step ARs carry ~214 us of mean rank-start skew even with graphs (source unknown);
  - large-M AR runs at ~12-14 GB/s vs a ~20 GB/s ceiling;
  - the logits all-gathers are small (~1.1 ms/step).

## What was tried (round 1)
- Records T1-001..T1-005: see reports/r1.md and reports/r1_leader-1.md.
- All round-1 ideas with scores, including the ones not selected: tree.md. Not tried yet:
  - logits diet: local argmax, FP8 or reduced-vocab draft head;
  - L2 prefetch during the AR;
  - Marlin retune for M=48;
  - GDN state writes;
  - prefill AR/compute overlap;
  - copy-engine AR for large M;
  - full (non-piecewise) mixed-batch graphs;
  - prefill steps stalling the decodes (ITL p99 ~290 ms).
- leader-1 (idle now) did the round-1 work: ask them what they saw.

## Where things are
- Engine code: /scratch/kevinsong/learning/autoinfer/.cache/tasks/qwen38-27b-fp8-2x4090/baseline/vllm
- The champion's code: workspace leader-1.hostar2. Or `ws create <you>.T1-005 --from T1-005`.
- Helper scripts: task/scripts/.
