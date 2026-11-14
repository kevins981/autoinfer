# T1 sweep report (leader-1) — summary
Framework: framework/ (6 knobs). Champion at sweep time: BASE (635.9 tok/s).
Files: [1-inputs](T1-sweep/1-inputs.md) · [2-graph](T1-sweep/2-graph.md) · [3-calibration](T1-sweep/3-calibration.md) ·
[4-knobs](T1-sweep/4-knobs.md) · [5-compare](T1-sweep/5-compare.md) · [6-ranking](T1-sweep/6-ranking.md)

Key observations (BASE profile, my analysis):
- Time = 72.5% decode steps (38.5 ms each, M=42 rows) + 27.5% mixed prefill steps (148 ms each).
- Decode step: Marlin FP8 GEMMs 18.6 ms (0.66 of DRAM BW), NCCL all-reduce 7.7 ms (135 x 58 us,
  ~7 GB/s via host SHM, no P2P), lm_head bf16 read 3x per step (target + 2 draft steps) 4.2 ms, logits
  all-gathers 1.4 ms, GDN 3.0 ms, idle 1.3 ms. Napkin at spec rates: ~24 ms.
- Mixed step: Marlin 81 ms (bf16 compute, 113 TFLOP/s), all-reduce 39 ms.
- Framework findings: every combine node sits on the serial chain (Σ, not max); the model needs per-node
  implementation-efficiency factors to rank anything here; the lm_head's missing `k` index (shared by
  target and both draft steps) is a sharing failure the Window rule can't fix (serial edges), so the fix
  is a cheaper draft head.
Ranked: (1) zero-copy host-memory all-reduce ~+10%; (2) cheaper drafter (fp8 head, local argmax,
reduced vocab) + deeper chain ~+10%; (3) AR/GEMM overlap ~+9%; (4) W8A8 for prefill ~+6.5%;
(5) Marlin decode efficiency ~+6%. Tested so far: NCCL env tuning (no gain), FlashInfer PCIe-IPC AR
(impossible: needs peer access).
