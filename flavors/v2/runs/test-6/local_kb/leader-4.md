# LOCAL_KB leader-4 (T4, explore)

## Direction candidate (2026-11-09): "dataflow TP layer" = hide each TP all-reduce under its CONSUMER GEMM
- Cost model (my read of KBs): trial ~97 s = rows x (bf16 tensor ~0.27 ms + AR ~0.1 ms) + fixed. Compute at roof
  (prefill 125-140 TF, decode Marlin at HBM/tensor balance). Precision levers forbidden. Comm is the one
  non-roof cost: decode AR 5.9 of 23.8 ms/step (T2-006), mixed-step AR exposed 10-43 ms/step (~7.5 s/trial).
- Blocker for overlap so far: AR -> RMSNorm(full row) -> consumer GEMM. Enabler: deferred norm scaling:
  y = inv_rms[row] * (W . bf16(h*g)) -- per-row scalar commutes with the linear map; inv_rms from sumsq
  accumulated per chunk. Consumer GEMM then needs only K-chunks of h as they arrive (in-kernel flag waits,
  no extra launches; KB24 says extra launches cost ~10 us).
- Comm kernel on a side stream (f SMs), consumer Marlin grid 128-f (T2-001: decode chain at 96 SMs only +3%).
- Per AR now ~46 us exposed + warm consumer; ideal: first-chunk latency (~10-15 us?) + cold consumer.
- Napkin payoff-if-works: decode -3..3.5 ms/step (~2480 steps) + mixed -5 s => ~+12-16%.
- Open: validity ruling (asked manager), first-chunk latency, Marlin stripe K-order (CTAs start mid-K),
  sumsq/residual production, attention layers / ba GEMM / lm_head consumers.
- RULING (manager, 2026-11-09): deferred RMSNorm scaling ALLOWED (task_definition.md end). Conditions: inv_rms fp32
  from same bf16 h, applied in fp32 epilogue; GEMM input bf16, fp32 accum; residual bitwise; record reports
  max-abs/rel diff vs stock norm+GEMM on real activations (several layers) + GSM8K + MTP acceptance (no greedy-text check).
- Worker leader-4.w1 started on premise microbench (agents/leader-4/w1_task1.md), ws t4-ovl, 8 GPU-min allowance.
- RESULT T4-001 (jobs 24-27): premise FAILED. Hiding the AR under the consumer saves <= 5-8 us/AR (est. 30-40). The AR slows 51 -> ~100 us
  beside Marlin (memory contention); the GEMM slows too. Removing ARs would be worth 24-28 us/AR (prefetch already takes ~18).
  Proposed voluntary close to the manager, ~18.5 GPU-min unspent. Waiting for its answer.
