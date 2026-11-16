# leader-1 (T1) notes
- Sweep files 1-4 written (reports/T1-sweep/). Calibration: AR 46us decode / 481us prefill (5.5GB/s), prefill Marlin 46% of bf16 peak, lm_head read ~3x/step ~4ms.
- Promotion rule (human, 2026-11-16): record mean > champion mean AND every start > champion mean; gain <1% needs a 2nd harness run (all 4 starts above).
- Worker leader-1.w1 (ws t1-w1): allreduce microbench (NCCL env, FlashInfer PCIe IPC).
- Tool: agents/leader-1/step_kernels.py <sqlite> [gpu] [topN]: per-kernel per-step.
