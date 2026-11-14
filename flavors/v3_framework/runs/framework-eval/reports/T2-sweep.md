# T2 sweep summary (leader-2)
Files: [1-inputs](T2-sweep/1-inputs.md) · [2-graph](T2-sweep/2-graph.md) · [3-calibration](T2-sweep/3-calibration.md) · [4-knobs](T2-sweep/4-knobs.md) · [5-compare](T2-sweep/5-compare.md) · [6-ranking](T2-sweep/6-ranking.md). Map proposal: proposals/T2-map-1.md.

- **Calibration:** decode step napkin ~25 ms against 38.2 ms measured. The gaps are Marlin at 0.66 TB/s (+6.3 ms) and NCCL ARs at ~60 us each (+4.4 ms). Mixed steps (~1/3 of wall time) are 147 ms against a ~56 ms napkin.
- **New finding:** about half the mixed-step AR time is rank skew (150 us median), i.e. CPU/lockstep, not link bandwidth.
- **Top candidates (napkin, overall):**
  1. W8A8 GEMMs for decode and prefill: 10-13%
  2. Mixed-step lockstep: 0-10%, unmeasured
  3. Custom host-memory AR: ~8% (T1 owns it)
  4. L2 prefetch during AR: 5-9%
  5. Draft flags (FP8 draft lm_head, local argmax): ~3% (T1 testing)
  6. Reduced-vocab draft: ~5%
- **Proposed split with T1:** T2 takes W8A8 (decode + prefill), mixed-step lockstep, L2 prefetch, GDN DeltaLog and Marlin efficiency.
