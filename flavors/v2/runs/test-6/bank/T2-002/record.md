team:         T2
builds_on:    T2-001 (T1-042 + Marlin sms override)
performance:  "probe only (synthetic lane, results discarded): M1 gate FAILED: decode ITL +65% at f=48 (gate <= 25%), lane ~36 TF/rank (gate >= 40)"
summary:
  hypothesis: A prefill lane on a green ctx (f SMs) slows the real champion decode step by about the GEMM-only microbench's ratio (Marlin x1.31, about +5.7 ms per step).
  change:     per-thread Marlin sms (t2_set_marlin_sms thread_local; VLLM_T2_DECODE_SMS for the decode path); VLLM_T2_SHADOW_SMS=f starts a daemon thread per TP worker that loops the real per-rank linear modules of all 64 layers at M=256 on a green-ctx stream, plus 2 copy-engine D2H/H2D copies per layer (lane AR volume x0.78). +190 MiB/rank. Code: vllm/v1/worker/t2_shadow_lane.py (by leader-2.w1).
  results:    Observed (jobs 11, 13; bench 100 prompts c=16, fresh server per config; agents/leader-2.w1/{abba,f3264}):
    OFF 865.9 / 859.7 tok/s, ITL med 25.29 / 25.33 ms, TTFT p99 1233 / 1311, TPOT p99 44.4 / 44.5
    f=48 (x2): ITL med 41.71 / 41.85 ms (x1.65), TPOT med 24.5 / 25.1, TPOT p99 51.6 / 52.4, TTFT p99 1594 / 1287; lane 35.8 / 36.8 TF/rank
    f=32: ITL 36.84 (x1.46), lane 22.1 TF; f=64: ITL 46.80 (x1.85), lane 44.7 TF (single starts)
  conclusion: Interpretation: in-server interference is ~3-4x the GEMM-only forecast (+16.4 ms vs +5.7 ms per step at f=48). On these numbers the lane loses more decode time per prefill (~82 ms) than the serialized mixed step costs (68.5 ms): no gain at any f tested. Cause not yet identified (candidates: cross-rank AR sync amplifying independent jitter, PCIe contention, L2/DRAM interference with non-GEMM decode kernels); ablation in T2-003.
validity:     probe only, never a candidate. Marlin sms override changes the grid only (default unchanged when the env is unset).
