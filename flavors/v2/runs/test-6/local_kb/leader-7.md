# leader-7 (T7, explore) working notes

## Accounting from official T1-042 data (CPU only, agents/leader-7/lens.py)
- Benchmark prompts (400, chat template): mean 261, median 127.5; 215/400 prompts < 150 tokens (17.4k tokens),
  86 in 150-400, 99 >= 400 (61.8k tokens = 59% of prompt tokens).
- Official T1-042 trial: 96.3 s, median ITL 25.4 ms (decode step), median TPOT 15.35 ms, accept 2.325.
  Steps ~2450 (400 x 228/2.325 / 16); mixed ~400 -> decode ~2050 x 25.4 = 52 s; mixed ~44 s by TPOT
  (0.163 m + 0.837 x 25.4 = 15.35 x 2.325 -> m ~ 88 ms avg).
- Mixed-step cost model from T2-006 (rank-0 GPU span): R=112 ~57 ms, R=369 91, R=776 144 (R = prompt + 45 decode rows).
  Sum over steady-state prompt distribution: ~33 s of mixed steps.
- Ideal mixed step = R x 0.183 ms (Marlin compute roof, 133 TF GPU0) + ~5-7 ms non-GEMM; AR hidden.
  Excess: small (R~130): 59 vs ~29 ms -> 30 ms x 215 = 6.5 s; R~245: 0.7 s; R~387: 0.9 s; R~612: 0.3 s. Total ~8.4 s
  (~+9.5% if all removed). SMALL PROMPTS carry most of it.
- Small-step excess vs decode (T2-006): Marlin +19.4 ms (cold M=112 29.7; graph grid pads 130 -> 160), hostAR +4.3
  (10.2 exposed total, 84 us/call), GDN +2.4, norm +2.6, idle +1.5, attn +0.8.

## Direction (A): hide prefill-size ARs with a split-half RS/AG pipeline on copy engines
- Producer row-parallel GEMM in 2 N-half launches (peer's half first) -> D2H peer half + doorbell while own half runs;
  RS add on own half; residual + bf16(h*g) own half; AG via DMA while the consumer runs its own K-half;
  consumer second K-half after the peer half lands; inv_rms in the epilogue (manager ruling 2026-11-09).
- No weight re-streaming (unlike the token split), no SMs for comm (KB33: DMA under compute-bound Marlin ~free).
- Untested at prefill sizes: T4-001 tested consumer K-chunking only at decode (memory-bound, zero-sum KB62).
- Est. at R=128: per layer GEMM ~366 us, AR 2 x 94 us serial; hidden -> ~145 us/layer, ~9 ms/small step.
- Other parts: (B) Marlin in situ at M=96-256 (T6-004 policy; padding), (C) prefill-path fixed costs (GDN glue,
  eager launches), (D) uneven MLP split for GPU0's locked clock (needs ruling: FP8 duplicate slice on GPU1).

## T7-001 (job by leader-7.w1, 4.8 GPU-min): direction A DEAD
- us/layer rank0 S/N/H: R=128 641/465/674; R=256 1151/822/1108; R=384 1646/1181/1571 -> hidden -19%/13%/16% (gate 60%).
- Half-partial (655 KB) D2H ~48 us + H2D ~48 us > own-half producer GEMM ~28 us; copies 9.5-14 GB/s beside Marlin, consumer GEMM ~2x slower.
- Recommended voluntary close to manager (2026-11-09); remaining small-step levers ~+1.5-2.5% (exploit items).
