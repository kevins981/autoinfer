# T8 sweep 3: napkin vs measured (champion T5-001)

Measured source: the champion's own mechanism profile cited in its record (bank/T5-001/record.md notes; job 118, nsys node trace,
64 prompts, champion serve args, GPU 0; per-category kernel sums over 250 full-batch decode steps, agents/leader-5.w1/cat_node_dl.txt)
plus the harness bench (eval/1 trial1: 101.3 s, 91,036 output tokens, acceptance length 2.35). No GPU spent by T8 on calibration
(deadline). Napkin: bytes / 1.0 TB/s per GPU, AR = 0.49 MB per direction / 25 GB/s.

## Decode step (M = 48 rows), per GPU
| part | napkin (bytes -> ms) | measured kernel sum (ms) | ratio |
|---|---|---|---|
| gate_up (64 x 5120x17408 fp8 / 2) | 5.70 GB -> 5.70 | 7.08 | 1.24 |
| down (64 x 8704x5120) | 2.85 GB -> 2.85 | 3.32 | 1.16 |
| GDN qkvz in_proj (48 x 16384x5120 / 2) | 2.01 GB -> 2.01 | 1.94 | 0.96 (L2 prefetch) |
| out_proj + o_proj (64 x 6144x5120 / 2) | 1.01 GB -> 1.01 | 1.32 | 1.31 |
| attn qkv (16 x 14336x5120 / 2) | 0.59 GB -> 0.59 | 0.56 | 0.95 |
| small bf16 GEMMs (in_proj_ba) | ~0.05 | 0.50 | ~10 (launch/latency-bound) |
| GDN recurrent (state read 1.2 GB + 1 write 1.2 GB with DeltaLog) | 2.4 | 1.90 | 0.79 (writes deferred in L2, spill into next kernels) |
| all-reduce hmar (~130 x 0.49 MB / 25 GB/s) | 2.6 | 5.96 | 2.3 |
| norms/act/quant + attention + misc | ~0.6 | 1.26 | 2 |
| target body wall | 18.6 (sum) | 23.67 | 1.27 |
| target lm_head bf16 (1.27 GB/GPU) | 1.27 | 1.58 | 1.25 |
| sample/bookkeeping (incl. nccl SendRecv 0.52 ms) | ~0.1 | 0.73 | — |
| draft step x2 (0.2 GB layer + 0.64 GB fp8 head each) | 1.7 | 2.83 | 1.7 |
| **step** | **21.7** | **29.0** | **1.34** |

## Whole benchmark
- Decode steps ~ 91,036 / (2.35 x ~15.5 active) ~ 2,500 x 29.0 ms ~ 72 s (71% of 101.3 s); the rest (~29 s) is mixed prefill+decode steps
  for 104.5K prompt tokens.
- Calibration verdict: napkin is within 1.16-1.3x on the weight-streaming GEMMs (ranking among them is reliable); it is badly off on
  all-reduce (2.3x: latency/sync-bound, not bandwidth-bound) and on the draft steps (1.7x: M=16 GEMMs + serial small kernels).
  So rank AR and draft candidates with measured costs, not napkin.
