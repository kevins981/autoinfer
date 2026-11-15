# Final report: run sweep-simple-eval (qwen38-27b-fp8-2x4090), 2026-11-15

Stopped by the human (relayed by session autoinfer2) while the official eval of T2-008 (job 73) was running; that eval is the run's final official number.

## Champions
- **Official champion: T2-008, 835.1 tok/s (+31.3% vs BASE 635.9; trials 836.3 / 834.0, CV 0.2%; GSM8K 93.6%; latency and 128K gates pass). bank/T2-008/ (official/1/results.json)**
- Previous official champions: T1-006 712.1 tok/s (+12.0%), T2-006 761.6 tok/s (+19.8%). BASE official 635.9 tok/s (unofficial ref 634.5 / 643.4).
- T2-008 is both the unofficial and the official champion (unofficial 792.8 / 789.1).

## Champion lineage (each builds on the previous)
| record | team | change | unofficial (tok/s) | official |
|---|---|---|---|---|
| T2-001 | T2 | prefill/mixed CUDA graphs up to 2048 tokens (serve flag) | 648.6 / 658.6 | via T1-006 |
| T2-004 | T2 | reduced-vocab MTP draft head (ids [0,65536)), local draft argmax, mbt 1024, graphs to 1024 | 660.0 / 664.2 | via T1-006 |
| T1-006 | T1 | target-side distributed argmax for all-greedy batches (exact; full fallback otherwise) | 676.6 / 680.7 | 712.1 |
| T2-005 | T2 | checkpoint W8A8 (dynamic group-128 e4m3 activations) via humming + 3 fixes (on T2-004) | 715.7 / 720.2 | via T2-006 |
| T2-006 | T2 | T1-006 + T2-005 | 738.8 / 747.8 | 761.6 |
| T1-008 | T1 | drafter FlashInfer plan from host upper bound + exact GPU metadata (no blocking seq_lens.cpu()) | 758.8 / 762.4 | via T2-008 |
| T2-008 | T2 | GDN replay commit in the MTP verify kernel (write only committed state; replay accepted tokens) | 792.8 / 789.1 | 835.1 |

Reverted: T1-002 (NCCL_PROTO=Simple: applied, but no in-engine effect; its +1.0% was noise).

## GPU
Pool 1000 GPU-min. Granted: T1 90, T2 120. Used: T1 56.3, T2 66.9, MGR 34.6 (BASE ref + 3 official evals). Unused: >750 GPU-min. The run ended well inside its budget.

## Team T1 (leader-1): 2 wins (T1-006 confirmed, T1-008), 1 revert
- **Mapped:** reports/T1-sweep.md + reports/T1-sweep/ (1-6, updates 1-3). Lane: communication (all-reduce/all-gather), target-side argmax, host-side stalls.
- **Tested:**
  - Negatives: T1-001 (MTP k=3 on BASE: acceptance 2.70 but latency gates fail); T1-003 (host-memory all-reduce: exact, inconclusive timing, parked); T1-004 (DBO overlap: ~50% hidden, net ~0); T1-005 (NCCL tuning: null).
  - T1-007 duplicated T2-006 and was not measured. T1-009 (humming decode tile table) was a microbenchmark only.
- **Key finding:** the framework has no Host node. Both later decode wins came from host->GPU sync edges seen in graph-level traces (pattern: plan with a host bound, exact device fix-up).
- **Left open (update-3):**
  1. Mixed-step GPU idle (10-16 ms per ~120 ms mixed step, <=3-4%).
  2. The remaining 0.61 ms decode idle.
  3. 1.58 ms of draft-side "other GEMM" outside graphs in decode.

## Team T2 (leader-2): 5 wins (T2-001, T2-004, T2-005, T2-006, T2-008)
- **Mapped:** reports/T2-sweep.md + reports/T2-sweep/ (1-6, updates 1-2). Lane: compute (GEMM precision/backends), draft head, GDN kernels, graphs.
- **Tested:**
  - T2-002 (FP8 draft head) and T2-003 (98K draft head) lost because a 4 GiB KV pool admits only 15 requests: the KV capacity is quantized by the 1,600-token hybrid page.
  - T2-007 (MTP k=3 on T2-006): 704.6 after 1 start, with 14 of 16 admitted.
  - A GEMM microbenchmark settled FP8 MMA at ~315 TF on the 4090 (SPEED_OF_LIGHT's 165 is too low).
- **Left open (update-2 + its last message):**
  1. MTP k=3, once T2-008's phase 2 frees the per-position GDN state slots so all 16 requests fit.
  2. Humming tile tuning for sm89 prefill shapes.
  3. Graph-capturing the eager GDN chunked-prefill core.
  4. The draft vocab size trade-off (64K vs 48K/80K).

## Rulings during the run
- `--no-enable-prefix-caching` allowed (manager, from principles; human may overrule). task_definition.md.

## Lessons (shared_kb.md KB1-KB6)
- Standalone microbenchmarks didn't predict in-engine gains for communication (KB2). Require in-engine evidence before promoting.
- Official gains ran ~2x the unofficial gains vs BASE for this lineage (KB4); unexplained.
