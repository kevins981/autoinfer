team:         T1
builds_on:    T1-037
performance:  "OFFICIAL #1 929.1 tok/s (+0.7% vs T1-037 922.8; p99 TPOT 38.4, under guard); unofficial: ABBA same job (job 165), B mean 879.2 vs A 871.1 tok/s = +0.94% (both B > both A); ITL p50 25.93 -> 25.46 ms"
summary:
  hypothesis: ncu (jobs 160/161) shows the GDN replay kernel is limited by the per-SM LSU/MIO path (66-72% busy), ~60% of it warp shuffles from 32-lane row-dot reductions, not by DRAM (0.66 TB/s, 65%). Reducing each row over 8 lanes (16 values per lane; 3 xor shuffles cover 4 rows) cuts the shuffles ~7x.
  change:     ws leader-2.red8 (w23): RED8 layout in the GDN replay and flush kernels.
              - one routine red8_advance (decay, fp32 dot, delta, update) used by log replay, new tokens, the flush kernel and mixed replay; red8_token adds the q dot.
              - default VLLM_GDN_RED8=1; 0 = the T1-037 32-lane path. Modes 2/3 (register prefetch) exist, bitwise equal to mode 1, no gain, not default.
              No serve-arg or memory change.
  results:    Observed:
              Unit tests (jobs 162/164):
              - RED8=0 bitwise == stock.
              - RED8 at cap 8 / 6 / 0 and modes 1-3: bitwise identical on 27,075 checks.
              - vs stock: state max rel 2.4e-7, output 1 bf16 ulp on 0.01% of elements.
              - Negative controls detected: 5841 / 18045 / 9609.
              Microbench (us/layer alone / with GEMM): N=16 cap 8 50.3/151.6 -> 46.2/147.4; N=8 cap 8 25.9/129.2 -> 19.7/124.3.
              ncu cold: LSU 66% -> 31%, instructions -35%, mio_throttle 13% -> 2%, long_scoreboard 23% -> 42%; DRAM 0.68 TB/s (now latency-bound). Register prefetch / 3 CTAs per SM: no further gain.
              ABBA (job 165, fresh servers, 200 prompts, c=16):
              - tok/s: 1A 869.2, 2B 880.6, 3B 877.8, 4A 873.0
              - ITL p50: 25.95 / 25.43 / 25.49 / 25.91 ms
              - p99 TPOT: 34.4 / 47.0 / 35.4 / 28.9 ms
              - acceptance 2.328-2.343
              gsm8k50 88% (= BASE subset). Greedy c=1: 1/10 identical (divergence at tokens 14-278), as expected from the fp32 reorder.
  conclusion: Interpretation: +0.9%, and -0.47 ms/step in-server, more than the microbench's 0.2 ms; perhaps the MIO relief helps more next to the GEMM/AR kernels (unverified). KB48 ("HBM-bound") was wrong: the kernel was LSU/MIO-bound and is now latency-bound.
validity:     leader-2 reviewed the increment vs T1-037: red8_advance and red8_token are fp32 (h *= decay; dot += h*k; 3 xor-shuffle sums; delta = (v - dot)*beta; h += k*delta). These are the same elementwise ops as stock; only the summation order of the row dots changes (allowed reordering). No lower precision; one routine for all state-advancing paths, so lazy replay stays self-consistent (bitwise, tested). No memory, KV, comm or serving change. OK.

official #1 (2026-11-08T13:50): 929.1 tok/s at c=16 (mean of 2 trials: 931.3 / 926.8, CV 0.3%) (+46.1% vs BASE), GSM8K 94.0%, MTP acceptance 2.33, TTFT p50 178 ms, TPOT p50 15.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
