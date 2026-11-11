# Round 7 report from leader-2 (T1): mixed (prefill-containing) steps
Best solid: bank/T1-023 = T1-020 + R7-B (GDN replay on the mixed-step path) + R7-C (every-32 prefill graph grid). Official 861.5
(863.1/859.8), +1.3% vs T1-020; same-job ABBA +1.5% (C above all other arms). GSM8K 93.2%, TTFT p99 1230, TPOT p99 33.9, gates + 128K.
Highest number: bank/T1-025 = T1-023 + R7-A (hybrid DMA/hostar token-split prefill chain). Official #2 874.8 (863.7/885.8, CV 1.8%),
+1.5% vs T1-023, GSM8K 92.8%, TTFT p99 1212, TPOT p99 41.4, gates + 128K. Caveat: trial 1 equals T1-023's level; the official gain
rests on one high trial (same bimodal per-start pattern as T1-020). Same-job evidence: 2 ABBAs pooled +0.7% (B mean above every A;
3/4 B above every A). I'd promote T1-025 only on the "small but consistent same-job" rule; T1-023 is the conservative choice.
(T1-025 official #1 = infra failure: a budget-refused first attempt left its eval workspace behind; #2 is the same change.)

Experiments:
- T1-023 R7-B: spec requests in mixed steps go through the replay kernel (pre-norm variant) instead of mode-2 flush + Triton kernel.
  Unit test vs stock CUDA: 0 mismatches / ~44k checks (neg. control 6,860); vs T1-020's Triton kernel ~1 ulp (different kernel).
  Profile: 6.48 -> 2.63 ms/mixed step (est. 5; replay costs 46 us/layer in mixed steps vs 25 in decode graphs). R7-C: every-32 grid,
  +0.63 GiB/rank graph memory, capture 28 s, KV unchanged. ABBA A 807.0 / B 811.5 / C 819.3.
- T1-024 R7-A gate (microbench): DMA AR (D2H, stream-memop doorbell, H2D, add) bitwise == hostar; split must be 64-aligned (even
  splits +15% worse: Marlin remainder launch); pure DMA -7.5%/layer at M=300; hybrid (chunk0 DMA, chunk1 hostar) -13.3%; high
  priority stream hurts. In-chain copies only ~12 GB/s (vs 21-24 alone), unexplained.
- T1-025 R7-A in-server: as above, threshold 256 tokens, main model only, opt-out VLLM_DMA_SPLIT=0. Split vs unsplit ~1 ulp (Marlin
  reorder); DMA AR bitwise in-server; gsm8k50 88%, 128K pass. +0.7% same-job (est. +2-3%).

Proposed SHARED_KB:
- KB-a: GDN replay on mixed steps saves 3.85 ms per mixed step (6.48 -> 2.63 ms); replay kernel runs 46 us/layer there vs 25 in decode
  graphs (cold L2/eager suspected). Every-32 prefill graph grid +1.0% same-job, +0.63 GiB/rank graph memory. (T1-023)
- KB-b: A copy-engine AR is bitwise equal to hostar and graph-capturable, but inside a GEMM chain its copies run at ~12 GB/s (vs 21-24
  alone) -> ~85 us/AR slower than hostar at M=300. Only 64-aligned token splits pay (Marlin remainder launch). Hybrid split -13%/layer
  microbench, +0.7% in-server (most mixed steps < 256 tokens; in-server AR cheaper). (T1-024, T1-025)
- KB-c: Official trials show a bimodal per-server-start throughput pattern (~+2.5%: T1-025 863.7 vs 885.8; T1-020 ABBAs 798/810);
  cause unknown. Single official numbers carry ~±1.3% from this alone. (T1-020, T1-025)

Left open: cause of the ~12 GB/s in-chain DMA copies (both GPUs through host? first touch?) — fixing it is R7-A's remaining upside;
split threshold 128 (64/64) untested in-server; leftover no-op flush launches in mixed steps (~0.45 ms/step); replay kernel 46 vs 25 us
in mixed steps; the bimodal start-to-start variance.
Next (one): understand and remove the per-server-start bimodality (~2.5% between starts, seen in ABBAs and official trials). If a
start-time factor (e.g. memory placement/NUMA of the pinned host buffers, GPU clock state, L2 set-hashing of weight placement) makes
some starts slow, fixing it lifts every "slow" start to the fast mode (~+1-2% mean) and halves our measurement noise. Evidence:
T1-020 ABBAs (808/811 vs 798/799), T1-025 ABBA (833.5 vs 817.7) and official (885.8 vs 863.7). First: a GPU-cheap characterization
(5-6 fresh starts, AR trace + decode step period per start, pinned-buffer NUMA node, clocks).
Notes: GPU 72.5/80 (w12 30.5, w13 22.0, evals 20.0). No GPU contention. Workers stopped. Workspaces kept: leader-2.mixreplay (T1-023),
leader-2.dmaar2 (T1-025).
