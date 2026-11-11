# T50 write-up (exploit, leader-50)

GPU used: 27.8 of 100 GPU-min (jobs 430, 431, 432 + official eval of T50-002). Closed on the manager's decision (option b).

## Win
- **T50-002 is the champion at 1181.3 tok/s official** (trials 1179.1 / 1183.5; +0.67% vs T45-005 at 1173.4). All gates pass:
  GSM8K 93.6%, latency, 128K.
- What it is: T45-005 plus two exact pieces.
  - **T42-003's certified target lm_head** (leader-42 / leader-42.w1). Their delta applies cleanly with `git apply -p2`.
  - **gate_up W8A8 tile cfg 103 for M ≥ 192.** The change is in `t29_bs_gemm/configs.json`; cfg 127 is kept at M ≤ 128.
- Same-job evidence against T45-005:
  - job 432, 400 prompts, BAAB: all B > all A, +1.25%;
  - job 431, 300 prompts: mixed, +1.04%.
  - The two pieces were not measured separately.
- Lesson: exact gains that were measured but never merged (T42's was "+0.5%, not established" on its own) are worth stacking onto
  the current champion. Two of them together cleared the same-job noise.

## Measurements (bank/T50-001, job 430, 0.9 GPU-min)
- Microbenchmark of bs_gemm_marlin: all 43 configs × 5 TP shapes × M = 128..2048. Cold L2, GPU0 at 2670 MHz.
- The tile table is the best config or within 1% of it everywhere, except gate_up at M = 192-320, where cfg 103 is 3-7% faster.
- bs_gemm runs at 88-92% of cuBLASLt per-tensor FP8. The gap is the mandatory per-128-K fp32 promotion.
- The e4m3 roof is ~340 TF at 2670 MHz: cuBLASLt slightly exceeded my 329 TF roof at M ≥ 1536.
- Efficiency by M:

  | M | TF |
  |---|---|
  | 128-192 | 136-217 |
  | 256-512 | 190-271 |
  | ≥ 1024 | 280-296 |

  Mid-M efficiency is the remaining GEMM slack. It needs a kernel change, not a config; T51 is working on it.
- Split-K helps only down_proj at M = 128 (8%, job 209 data), about 0.08% end to end.

## Ruled out without GPU (from the T37-002 node trace; scripts in agents/leader-50/ana/)
- **Decode host-AR config threshold** (600 KB puts 64-row decode in the (16,4) config): no lever.
  The (16,4) time of 51.6 µs equals the (8,1) per-row slope extrapolated to 64 rows.
- **CUDA-graph padding of mixed steps:** about 11 rows on average, but padding inside an m-tile costs the GEMM almost nothing.
  The saving would be only 0.2-0.4%.
- **Row reduce-scatter + FP8 all-gather instead of AR + norm + quant:** exact, but it cuts bytes by only 15-20% and adds a sync phase.
  Dead for both decode and small mixed steps.
- **Decode bookkeeping:** about 45 tiny kernels around sampling and the drafter cost ~0.3 ms/step. A realistic saving is ≤ 0.4%.
- **Small mixed steps** (~120 rows, 42-50 ms) are GPU-bound. The idle column in steps.csv is an artefact.

## Decode all-reduce (sparring T50-1, sparring/)
- The decode AR "floor" is a plateau among the implementations tested, not a proven hardware bound.
- For small messages, the AR finishes its exchange early (6-18 µs). Its L2-prefetch blocks keep the kernel alive to about 32 µs
  (T36 per-call trace). That prefetch is useful work: per KB25, KB78 and KB62, 40-60% of any transport saving survives in the chain.
- The net prize of a full-duplex decode AR is about 1.3 ms/step (~3.5% end to end). Every path to it is closed here:
  - GDRCopy / BAR push needs a root-installed module, and there is none on this machine;
  - P2P is "chipset not supported";
  - copy-engine-only AR was no better (KB62);
  - the hybrid copy-engine-write + SM-read lost at decode size (T16-001).

## Open (for others)
- Mid-M W8A8 GEMM efficiency (stream-K or balanced grid): T51.
- Decode bookkeeping fusion, T46 host embedding, split-K down_proj at M = 128: each ≤ 0.4%. Only worth doing as a stack.
