team:         T6
builds_on:    T6-004
performance:  "measurement (nsys, job 57, 2.3 GPU-min): mixed M=96 step (chunk budget 96) period 77.2 ms under nsys, kernel busy 41.8, in-step idle 34.2; decode 24.8 / 24.6 / 0.2."
summary:
  hypothesis: decompose the +30 ms of a piggybacked mixed M=96 step over a decode step into GPU kernel time vs launch-bound idle.
  change:     none (built ws t6-pol: _C 36,865,976 B, gdn_replay_op_available True; C = champion args + --max-num-batched-tokens 96; TRACE=node, 24 prompts).
  results:    Observed (rank 0 / GPU0, medians per step, 110 mixed M=90-96 steps vs 15 decode dt 36-45 steps):
    period 77.2 vs 24.8; busy 41.8 vs 24.6; idle 34.2 (many small gaps: 36 > 100 us, largest median 0.26 ms) vs 0.2; kernels 2583 vs 1041.
    Kernel time ms mixed|decode: Marlin 22.67|13.60; hostar 9.15|5.88; GDN replay 1.62|1.67; GDN chunk prefill + glue (elementwise/index/gather) ~3.5|~0.3; lmh 1.00|1.01; attention 0.52|0.27; norms 0.55|0.47.
  conclusion: Interpretation: even with zero launch gaps a 96-row piggyback step costs ~42 ms (~1.7x decode): ~0.36 ms per prefill row vs ~0.25 in whole-prefill mixed steps, so chunked-prefill piggyback cannot pay on this stack (in-situ Marlin at M=96 is +9.1 ms over M=48, above the cold microbench's +2.4). PIECEWISE mixed steps are partly launch-bound (~1500 extra eager launches: GDN prefill glue, norms, attention); nsys inflates the idle, the un-profiled gap is <= ~15 ms per small mixed step (exploit-scale for the champion, cf. T2 small items).
validity:     measurement only.
