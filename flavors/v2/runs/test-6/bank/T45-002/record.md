team:         T45
builds_on:    T37-002
performance:  "CPU only, n/a (analysis: headroom map and kills)"
summary:
  hypothesis: Before spending GPU time, map where the champion's remaining time goes in the closed loop, and kill
              radical directions with napkin math.
  change:     None to the engine. Tools in agents/leader-45/: sim/sim.py (closed-loop c=16 simulator, 400 requests,
              lognormal prompt/output lengths with means 262/227, KB111's step model, cumulative acceptance
              .774/.552/.400), sim/check.py, sim/coal.py, sim/combo.py, amx.py; trace analysis of
              agents/leader-37/out/prof_node/profile.sqlite.
  results:    Observed:
    - Calibration: sim 80.5 s / 1126 tok/s vs official T37-002 79.7 s / 1142 (-1.4%).
    - Forecast miss reproduced: sim predicts k=2->3 +4.5% (T37's forecast); measured same-job +1.34% (job 375).
      The step model gives 25.8 ms at k=3 vs ~26.2 measured; k=1 predicted -11.5% vs ~-16.8% cross-job.
    - Sensitivities (event model, vs base): ev_fixed 15->0 +7.0%; ev_tok .166->.10 +9.4%; ev_tok .166->.087 (prefill
      AR fully hidden) +12.5%; ev_fixed 15->5 +4.8%; decode row cost .10->.07 +5.2%; acceptance [.80,.60,.45] +3.2%,
      [.85,.68,.52] +8.4%; perfect k=3 drafter (4 tok/step) +19.8%.
    - Trees (T38's exact offline 4.21 tok/step at 11 nodes, drafter cost unchanged, linear row cost): row cost
      0.10 -1.4%, 0.08 +5.2%, 0.07 +8.9%, 0.06 +12.9%, 0.05 +17.1% (chain k=3 at the same row costs: 0, +3.3, +5.1,
      +6.9, +8.8%). The row-cost model is a linear fit only up to 64 rows (KB111); 11-node trees need 176 rows.
    - Prompt rows fed as fixed per-step chunks instead of events: Bp 32 -14%, 48 +1.6%, 64 +8.5%, 96 +13.6% with no
      GEMM cliff; with a 4 ms cliff above 96 rows: -3.7 / +0.2 / +6.9%. Holding a lone waiting prompt 1-5 steps for
      a partner: +0.45..+1.03%.
    - Drafter phase (k=2 node trace, GPU0): 1.25 ms/step, a kernel running 1.21 ms of it, 93 kernels (Marlin 0.39,
      hostar 0.32 with 16 CTAs, screen 0.17). Co-scheduling prefill work in the drafter phase is capped at <=+2%.
    - CPU AMX bf16 GEMM: ~6-9 TFLOP/s at 12-24 threads (agrees with KB66's ~7): CPU prefill offload dead.
    - Communication min-cut (napkin): under an even weight split, syncing at the 5120-dim hidden is the narrowest cut
      of every sublayer (GDN v / attention core 6144, MLP 17408). The only cheaper cut is an all-gather of the
      3072-dim per-GPU core output with out_proj/o_proj replicated (2.0 GB/GPU: 48x31.5 + 16x31.5 MB): mixer bytes
      0.6x, prefill AR bytes 0.8x (~+2% via the sim), decode a wash (+1.0 GB of extra weight reads per step vs
      ~0.9 ms of AR bytes saved), and 2 GB does not fit (KB16/KB82: ~0.3-0.7 GiB free).
    - Context-aware lossless AR codec (per-channel exponent prediction on top of KB43's): napkin bound ~10% fewer
      wire bytes than today's 0.768 because the 7 mantissa bits are incompressible: ~+1-2%. Not pursued.
    - Hardware: under CUDA load the memory clock sits at 10251 MHz (P2 cap) vs 10501 MHz max: the DRAM roof is
      ~2.4% below the spec sheet. Changing it needs admin rights: out of scope, recorded as a fact.
  conclusion: Interpretation: the champion is near its structural floors on the decode pass. Two parameters dominate
              what is left: the prefill path (per-token AR excess and the ~15 ms per-event fixed cost, ~+18% if both
              were gone) and the per-row cost of extra decode rows, which decides whether tokens-per-step ideas
              (trees, deeper k, budgets) can pay at all. Row cost above 64 rows has never been measured in situ in
              TP; T38 measured PP-shape GEMMs only.
validity:     No engine change; no GPU used.
