# T7 write-up (leader-7, explore). Closed voluntarily after the seed

GPU: 4.8 of 20 used (worker leader-7.w1). Record: bank/T7-001. KB: KB65. Notes: local_kb/leader-7.md.

## Accounting: where the mixed-step time goes (CPU only)
Inputs: agents/leader-7/lens.py (the benchmark's 400 prompts, tokenized with the chat template; same sampler as
`vllm bench`), the T1-042 official results, and T2-006's rank-0 anatomy.
- **Prompt lengths:** mean 261, median 128. 215/400 prompts are < 150 tokens. 99 are >= 400 tokens, which is 59% of
  prompt tokens.
- **Wall split:** ~2450 steps per trial, ~400 of them mixed. From median TPOT 15.35 ms and ITL 25.4 ms, a mixed step
  averages ~88 ms. Mixed steps are ~40% of wall.
- **Mixed-step cost**, rank-0 GPU span, R = prompt + 45 decode rows: R~112 -> 57 ms, R~369 -> 91, R~776 -> 144.
  Summed over the prompt distribution: ~33 s of a 96 s trial.
- **Ideal mixed step** = R x 0.183 ms (Marlin compute roof, 133 TF on GPU0) + ~5-7 ms non-GEMM, with the AR hidden.
  Excess over that ideal:

| prompt bucket | steps | actual vs ideal | excess per trial |
|---|---|---|---|
| < 150 | 215 | ~59 vs ~29 ms | ~6.5 s |
| 150-256 | 29 | | 0.7 s |
| 256-400 | 57 | | 0.9 s |
| >= 400 | 99 | already ~at roof | ~0.3 s |
| **total** | | | **~8.4 s (~+9.5% ceiling)** |

**Where a small step's ~30 ms excess goes** (vs a decode step, T2-006):
- Marlin +19.4 ms. The prefill graph grid pads R~130 to 160 rows, ~5 ms of that. Cold M=112 is 29.7 ms vs a 20.5 ms roof.
  In situ it is worse than cold (T6-007).
- Host AR 10.2 ms exposed (84 us x 128). KB65 says this is NOT hideable.
- GDN prefill path +2.4, norms/act +2.6, in-span idle +1.5, attention +0.8.
- So ~20 ms/step remains outside the AR. Realistically ~+1.5-2.5% is reachable through exploit items: T6-004 policy
  in situ, less padding, graphing/fusing the GDN prefill glue.

## Direction tried: hide prefill-size ARs (T7-001, failed its gate)
- **Schedule:** a split-half reduce-scatter + all-gather on the copy engines.
  - The producer GEMM computes the peer's N-half first.
  - The consumer GEMM starts on its own K-half (RMSNorm ruling).
  - No weight re-streaming.
- **Gate:** hide >= 60% of the AR at R=128 and 256.
- **Result:** -19% / 13% / 16% hidden at R=128/256/384. The schedule is slower than stock at R=128.
- **Why:**
  - A 655 KB half-partial round trip is ~96 us (~14 GB/s each way), longer than any half GEMM (~28 us for out_proj/down).
  - Beside Marlin, the copies drop to ~9.5 GB/s and the GEMM runs ~2x slower.
  - Splitting the GEMMs costs +35-57 us/layer by itself.
- The champion's token-split path at R >= 256 was not reproduced (the reimplementation hung).

## Prefill coalescing (napkin, untested)
- **Idea:** hold a new short request until a second one arrives, then prefill both in one step.
- **Gain:** two small mixed steps (~2 x 59 ms) become one R~215 step (~70 ms) plus one decode step (25 ms). That saves
  ~24 ms per pair.
- **Cost:** completions arrive every ~7 steps, so the held slot idles ~7 steps x 25 ms / 16 ≈ 11 ms per pair.
- **Net:** ~+1% over ~100 pairs. Mean TTFT rises ~0.1-0.2 s, so p99 TTFT (gate 1609 ms, now ~1100) is the risk.
  The policy must be generic (a time/queue-based window), not keyed to c=16.

## Opinion
- Small prompts are where the mixed-step excess lives. It is GEMM and fixed-path work, not communication.
- Nothing there clears a 5% explore bar on its own. The items above belong to an exploit team.
