team:         T1
builds_on:    T1-037
performance:  "OFFICIAL #1 922.2 tok/s (= T1-037 922.8 within noise: no e2e gain); unofficial: lm_head GEMM microbench -9.4% at M=48 (1055 -> 956 us); e2e ABBA (job 156) unresolved: job-wide ~5% drift, B mean 818.7 vs single A 821.5, ITL flat"
summary:
  hypothesis: T1-037's lm_head GEMM runs at the HBM read roof, so a tighter lossless code (0.705 vs 0.781) cuts its time ~10% (~-0.10 ms/step, ~+0.4%).
  change:     ws leader-2.lmh3 (w24): lmh_codec v2.
              - Code: 3-bit exponent offsets (0-6 direct, 7 = escape into a per-group nibble pool); header = 1 B group max + 2-bit lane deltas; whole-group raw fallback in a trimmed side table.
              - Same load path: bitwise verify over all blocks with the GEMM's decode_group, sanity vs F.linear, raw shard freed, ERROR + cuBLAS on failure.
              - M=64 tile retuned (MT4xNT1x8 warps x3 stages).
              - L2 prefetch of the encoded prefix OFF by default (VLLM_LMH_CODEC_PF=1 turns it on). No serve-arg change.
  results:    Observed:
              - Load: 1.271 -> 0.896/0.898 GB (0.7045/0.7062); 835/1910 fallback groups; verify 0 mismatches. Synthetic stress (inf/NaN/zeros/subnormals): verify 0.
              - Kernel output bitwise == T1-037's for M 1..64; graph == eager.
              - Microbench v1 -> v2 (us): M3 1042->930, M16 1058->952, M24 1052->949, M48 1055->956, M64 1070->1000.
              - ABBA job 156 (tok/s, ITL p50 ms): 1A 821.5/28.81 | 2B PF1 814.7/28.83 | 3B PF0 823.7/28.82 | 4B PF0 816.5/28.77 | 5B PF1 819.9/28.77 | 6A killed by the timeout.
                Every arm ~5% slower than job 151, for v1 and v2 alike. The job overlapped a concurrent `ws build` (CPU load; host-staged AR is CPU-sensitive) - the probable cause.
              - gen_check 10/10 identical vs T1-037; gsm8k50 88%.
              - Argmax fusion (job 157): exact but no gain (+3-8 us), killed.
  conclusion: Interpretation: exact, ~-0.1 ms/step on the lm_head in isolation; the e2e gain (~0.4%) is below what job 156 could resolve. Official eval decides.
validity:     leader-2 reviewed the increment vs T1-037: only the lmh_codec .cu/.py change (codec format, tile, PF default). Verify still runs decode_group on the final buffers (side table trimmed before verify). Same lossless-weight rule as T1-037. Kernel fp32 order unchanged (outputs bitwise == T1-037). OK.

official #1 (2026-11-08T13:07): 922.2 tok/s at c=16 (mean of 2 trials: 922.3 / 922.1, CV 0.0%) (+45.0% vs BASE), GSM8K 93.2%, MTP acceptance 2.338, TTFT p50 173 ms, TPOT p50 15.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
