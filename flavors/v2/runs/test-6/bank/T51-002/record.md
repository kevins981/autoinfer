team:         T51
builds_on:    T45-003 / T45-004 harness (64-layer GDN-layer chain, AR as spins 39/50 us with in-AR L2 prefetch), champion kernels (workspace leader-51.pig)
performance:  "microbench only (M1 of proposals/T51-1): GATE FAILED. GEMM band 0.069-0.072 ms/row at M=192 (a: 0.072, b: 0.069), 0.089-0.090 at M=160 (vs gate <=0.05)"
summary:
  hypothesis: With the best existing kernels per shape, adding 64-144 prompt rows to a 64-row decode pass costs <=0.05 ms/row in the GEMMs (in chain, with AR prefetch).
  change: none to the server. Scripts agents/leader-51.w1/m1/{sweep.py,bench_chain.py,job1.sh,job2.sh}, outputs sweep.json, res_{a,b}_M*.json. GPU jobs 436 (sweep 1.3 min) and 437 (chain, 1.5 min), GPU0, synthetic weights, 64 distinct layers, GDN-layer proxy for all 64 layers (no attention layers; the chain cannot model them), AR = fixed spins (no per-row AR cost), gate_up on bs_gemm for M>=64 in both variants.
  results: Observed. Per layer, us (profiler means per GEMM in the chain, in-AR prefetch). Variant a = champion dispatch (t31 at M<=96 for qkvz/gdn_out/down; bs table cfg above; gate_up bs table cfg). Variant b = best (cfg, split-K 1..4) per shape and M from a 43-cfg x 4-split sweep (cold standalone, 5 timed reps).
    M    var  layer  qkvz  gdn_out gate_up down | sumGEMM  delta vs M=64 (ms/step, 64 layers)  us/added row
    64   a    342.2  37.6  21.5    82.1   53.6 | 194.8   0       -
    64   b    341.3  36.1  20.6    83.5   53.9 | 194.1   -0.04   -
    128  a    411.8  46.3  27.4   117.3   72.9 | 263.9   4.42    69
    128  b    415.9  46.9  35.3   118.7   65.1 | 266.0   4.56    71
    160  a    476.0  67.2  27.9   158.8   74.5 | 328.4   8.55    89
    160  b    477.1  71.3  27.8   154.9   75.8 | 329.8   8.64    90
    192  a    488.0  72.8  28.1   162.9   74.6 | 338.4   9.19    72
    192  b    480.1  71.3  27.9   157.1   75.9 | 332.2   8.79    69
    208  a    574.8  81.5  39.7   194.8  110.9 | 426.9   14.85   103
    208  b    562.1  83.0  40.8   189.4  100.8 | 414.0   14.03   97
    (b's chosen cfgs, M=192: qkvz 121, gdn_out 103, gate_up 103, down 103, all split 1; M=208 down = cfg 128 split-K 4; M=128 down = cfg 26 split-K 3 standalone 62.5 us vs 76.7 for the table cfg, but in the chain down is 65.1 vs 72.9, and gdn_out chose split-K 3 cfg 31 which was worse in chain, 35.3 vs 27.4.)
    Byte floors per shape (0.98 TB/s) us: qkvz 42.8, gdn_out 16.0, gate_up 90.9, down 45.5 (sum 195). Chain GEMM vs floor: M=64 1.0x; M=128 1.36x (b 1.37); M=160 1.69x; M=192 1.70x (b 1.70); M=208 2.1x.
    Per-shape, standalone best (cold, us) M=64/128/160/192/208: qkvz 49.2/52.2/73.7/73.8/85.0 (floor 42.8); gdn_out 22.5/29.7/30.7/30.7/43.0 (16.0); gate_up 99.3/122.9/160.8/162.8/194.6 (90.9); down 56.3/62.5/78.8/78.8/103.4 (45.5). In-chain values above are 15-20% lower at M=64 (AR prefetch warms L2).
    Roofline for the same layer (4 GEMMs = 191.3 M weights per rank): FP8 MMA time = 1.31 us/row/layer at 291 TF, i.e. 0.084 ms/row over 64 layers; compute equals weight-stream time (195 us) at M~149. An ideal kernel would be flat to M~149 and then rise 0.084 ms/row; against the M=64 baseline (itself at the byte floor) the ideal band is 0.029 ms/row at M=192, 0.013 at M=160, ~0 at M=128. The measured kernels reach 76% of the e4m3 roof at M=192 (221 TF).
  conclusion: Interpretation:
    - Gate (<=0.05 ms/row at M=160-192) FAILS with existing kernels: 0.069-0.072 at 192, 0.089-0.090 at 160, ~0.07 for 64->128. Variant b (best cfg + split-K) buys nothing over the champion's table in chain (<=2% of sumGEMM; M=192 -1.8%, M=208 -3%, M=128 +0.8%); split-K helps some standalone shapes but not in the chain.
    - The band is a step function of tiles, not a slope: M=128->160 costs +65 us/layer (qkvz +20, gate_up +40) as the kernels move to 3 m-tiles, then 160->192 is nearly free (+10). 128 and 192 rows are the cheap points; 160 is the worst. So B should be chosen at the plateau ends (M_total = 128 or 192 = 64+64 or 64+128 prompt rows), and 144 prompt rows (M=208) is on a new step (+87 us/layer).
    - What would make the gate: a kernel at ~1.0-1.15x floor out to M~150 (M=128 is 1.36x now, 71 us/row), or exploiting the plateau (M=192 costs 69-72 us/row from 64, but only +10 us/layer from M=160), i.e. a single mixed pass of 64+128 rows costs 9.2 ms of GEMM over a decode pass vs 8.5 ms at 64+96. At 0.07 ms/row GEMM + 0.07 ms/row AR (KB111) a prompt row costs ~0.14 ms vs 0.166 in a prefill event: the saving is ~15%, as T45-004 found at M=64->128.
    - Caveats: synthetic weights (NaN-containing random fp8 weights: split-K numerics NOT re-verified here; T38 checked them), GDN-layer proxy (attention layers have 7168x5120/3072x5120 shapes, similar), AR as fixed spins (no row-dependent AR cost), single chain measurement per cell (chain rep spread <=2%; sweep medians of 5 cold reps), no t31 beyond M=96.
validity:     Microbenchmark only. Nothing to review for serving.

leader-51 addendum (interpretation, CPU only): this attempt failed on existing kernels, not on a hard limit.
- e4m3 roof: 1.31 us/row/layer. At 85% of the roof, M=160 costs ~247 us/layer (0.035 ms/row) and M=192 ~295 (0.05);
  measured 329/332.
- Full-width shapes already reach 1.07-1.10x floor at M=128 with bs_gemm (T38 pp_gemm: gate_up 213 vs floor 194).
  TP gate_up is at 1.34x (121.9 vs 90.9), so the TP-shape gap looks like tiling/occupancy (half the N tiles), not physics.
- Sim (pig2.py) with a captured pass: today's band 0.07 + glue 3 gives +4.7/+5.4% (B=128/144); a 0.035 band + glue 3-5
  gives +4.8..+8% (B=96).
