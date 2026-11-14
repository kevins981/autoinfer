team:         T56
builds_on:    T52-005
performance:  "probe only (synthetic lane, never a candidate): decode ITL x1.28 / x1.39 / x1.52 at lane f=32/48/64 (T2: x1.65 at f=48); lane under load 34 / 44.5 / 55 TF/rank"
summary:
  hypothesis: With the champion's W8A8 bs_gemm (M>96) the T2 synthetic prefill lane runs ~2x faster per SM and costs decode less than T2's bf16-Marlin lane did.
  change:     Ported T2's lane as vllm/v1/worker/t56_lane.py (hook in gpu_worker.compile_or_warm_up_model); env VLLM_T56_LANE_CTL=<file> starts a daemon thread per TP worker that creates green contexts f in {32,48,64} and, per control-file content ("0" idle | "f" | "f,512"), loops all 304 per-rank linear modules (64 layers) via m.quant_method.apply(m, x, None) at M=256 (or 512) plus 2 copy-engine D2H/H2D per layer (0.78 x AR volume). No Marlin-sms override. Marlin-lock pre-seeding kept in try/except (harmless). Dispatch check (logged once per shape): all linears are Fp8LinearMethod except in_proj_ba (UnquantizedLinearMethod, bf16); at M=256/512 > T29_W8A8_MAX_MARLIN_M=96 the t29 op takes w8a8_marlin -> quant_perm + bs_gemm_marlin (read from the code, t29_marlin_w8a8.py:_t29_marlin_fp8_linear; not nsys-verified). One server (T52-005 serve_args), phases 0,32,48,64,"48,512",0 each a `vllm bench serve` 100 prompts c=16 seed 0 (job 489, agents/leader-56.w1/full/, job.sh).
  results:    Observed (single pass per phase, one server; bench lasts only ~20 s):
    phase     tok/s  ITLmed TPOTmed ITLp99 | lane TF/rank (TP0; TP1 within ~2%) | D=ITL_off/ITL | r tok/s | cost (1-D)/r
    off(1)   1138.7  24.69  12.55   142
    f=32      928.1  31.59  15.25   157   | 33.9 loaded (57.2 lane alone) | 0.782 | 1395 | 0.156 ms
    f=48      876.6  34.37  15.83   175   | 44.5 loaded (73.6 alone)      | 0.719 | 1831 | 0.154 ms
    f=64      807.1  37.61  17.31   185   | 55.4 loaded (90.6 alone)      | 0.657 | 2280 | 0.151 ms
    f=48,M512 881.6  34.37  15.60   186   | 50.0 loaded (77.2 alone)      | 0.719 | 2058 | 0.137 ms
    off(2)   1126.6  24.71  12.47   133
    D uses ITL_off = 24.70 (mean of both off phases); r = TF*1e12/24.3e9; lane TF = mean of the last 4 complete 5-s windows of the phase (the first 3 windows precede the bench's requests, lane runs alone). Windows are noisy (+-4%) and the bench is short, so loaded rates have ~+-5% uncertainty. tok/s ratio vs off: 0.815 / 0.770 / 0.709 / 0.774.
  conclusion: Interpretation: decode slowdown at f=48 is x1.39 (T2: x1.65) and the lane runs 44.5 TF/rank (T2: 36), so the cost per lane token is ~0.15 ms across f (nearly f-independent: slowdown and lane rate both scale ~linearly with f), 0.137 ms at M=512. For comparison with the serialized mixed-step cost per prefill token the leader should use the champion's own number (T2's break-even was ~68.5 ms per ~256-row chunk = 0.27 ms/token, which would make the lane ~1.8x cheaper than serialization now; this comparison is the leader's, I did not measure serialized prefill here). Lane speed gain from W8A8 is only 1.24x under load (2x per SM was the expectation); lane-alone rate is 57/74/91 TF at f=32/48/64 (1.8 TF/SM), loaded rate ~60% of that.
validity:     probe only, never a candidate. Env unset = no thread; no change to model math or serving path.

leader review (leader-56): Verdict: NO-GO for a lane on today's decode kernels.
- Comparison: the joined mixed step costs ~0.19 ms per prompt token all-in under nsys (job 483), ~0.167 scaled to official speed.
- The synthetic lane's cost (0.137-0.156 ms/token) is GEMM-only. A real lane adds dependent ARs (~17-25 ms per prompt on copy engines), the GDN chunked core and glue (~+25%, sparring T56-1), and slot wait while the prompt sits in the slower lane (T57 closed-loop point).
- That puts the real lane at ~0.18-0.20 ms/token: break-even to a loss. The go threshold was <=0.11.
- Resized decode grids are unlikely to rescue it. T57-002 found the co-run contention is DRAM/L2 arbitration (an ideal streaming kernel on 128-f SMs keeps 880-914 GB/s but starves the lane to 12-39 TF), not SM count.
- W8A8 made the lane only 1.24x faster under load (lane-alone 1.8 TF/SM, loaded ~60% of that).
validity: probe only; no serving change (env unset = no thread).
