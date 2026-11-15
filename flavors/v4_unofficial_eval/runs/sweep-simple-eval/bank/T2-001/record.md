team:         T2
builds_on:    BASE
performance:  unofficial: 648.6 / 658.6 tok/s (mean 653.6) vs BASE 634.5 / 643.4 (mean 639.0): +2.3%
summary:
  hypothesis: mixed prefill steps (>96 tokens) run eager; capturing them in CUDA graphs removes host-launch time and the cross-rank skew that NCCL waits on (napkin up to -48 ms per mixed step, ~+13%).
  change:     serve flag only: --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}' (prefill/mixed CUDA graphs up to 2048 tokens instead of 96). No code change.
  results:    Observed: unofficial_eval --ws t2-cg --label t2-cg2048 --long (GPU job 4): 648.6 / 658.6 tok/s, 128K pass, p99 TTFT 1265 ms / TPOT 44.1 ms, MTP acceptance 2.351.
  conclusion: Interpretation: real but much smaller than the napkin (+2.3% vs ~+13%): the mixed-step fixed cost is only partly host launch; a profile of this config is needed to see what remains (T1's pairing analysis: ~300 us transfer per prefill-size AR is genuine).
validity:     leader-2 review: serve flag only, same kernels/math replayed from CUDA graphs (padding rows don't touch real rows); memory change -> 128K request tested: pass. Latency p99s within gates. Parsers, sampling unchanged. Nothing benchmark-specific (generic graph-size setting).
unofficial #1 (copied from unofficial/ws/t2-cg2048/1): 648.6 / 658.6 tok/s (200 prompts); 128K pass, p99 TTFT 1265 ms / TPOT 44.1 ms, MTP acceptance 2.351
mechanism (nsys of this config, agents/leader-2/prof_cg2048, GPU job t2-prof-cg; step_anatomy fits in agents/leader-2/cg_steps.csv):
  Observed: mixed-step NCCL fixed part fell from 40.3 ms (BASE) to 7.7 ms (per-row 0.099 ms; LL protocol), i.e. the cross-rank wait is gone; Marlin 11.6 ms + 0.158 ms/row. Mixed steps are "breakable" graphs: ~40 graph pieces, with the GDN chunked-prefill core (FLA Triton kernels, ~1,700 eager kernels per mixed step) and prefill attention run eagerly between pieces. Period fit 82 + 0.224 ms/row (vs 88 + 0.244 in BASE), but node-level tracing inflates each of the ~40 graph launches (~1-3 ms each under nsys), so the profile's 50 ms "idle" overstates real idle; the measured gain is the unofficial eval.
  Interpretation: graphs removed the skew wait (what the napkin priced) but not the host time of the eager GDN prefill pieces; remaining mixed-step costs: Marlin compute (W8A8 candidate), NCCL bandwidth in LL (T1 testing PROTO=Simple), eager GDN prefill.
