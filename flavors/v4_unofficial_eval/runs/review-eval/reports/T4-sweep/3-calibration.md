# T4 sweep, step 3: napkin cost model, calibrated against the champion

Measured run: GPU job 66 (leader-4.w1), nsys of T3-001 (workspace t4-champ, its serve args), 64 ShareGPT
prompts at c=16, node trace and graph trace. Analysis: agents/leader-4.w1/analysis.md (+ cat_step.md,
seq_dec.txt, prof_*/steps.csv). Harness reference: bank/T3-001/eval/1 (111.8 s, 90,684 output tokens,
median ITL 32.4 ms). Hardware constants: BW = 1.0 TB/s (spec), F_fp8 effective measured below, link
~25 GB/s per direction.

## A. Decode step, 16 requests x 3 rows (M = 48), per GPU
Measured: graph trace, D=16 steps median period **32.74 ms**, GPU busy 31.74, idle 0.97 (200 steps).
Node trace per-kernel sums (GPU0, median D=16 step, period 34.2 because of trace overhead).

| part | napkin (bytes or work / rate) | napkin ms | measured ms | ratio / note |
|---|---|---|---|---|
| body FP8 GEMMs (264 launches) | 12.16 GB / 1.0 TB/s | 12.2 | 14.7 | 0.83 of BW (gate_up 89 MB in 112 us = 0.80 TB/s; out_proj 15.7 MB in 25 us = 0.63) |
| target lm_head (bf16, M=48) | 1.27 GB / 1.0 | 1.27 | 1.58 | 0.80 |
| draft lm_head x2 (bf16, M=48 then 16) | 2.54 GB / 1.0 | 2.54 | 2.73 | 0.93 |
| MTP layer x2 (FP8 GEMMs + fc) | 2 x 0.19 GB + fc 2x26 MB | 0.43 | ~0.70 | small-M inefficiency |
| hmar all-reduce (129 + 6 draft) | latency ~20 us + 0.49 MB / 25 GB/s ~20 us = ~40 us each | 5.4 | 6.2 (46 us each) | 0.87; includes peer skew |
| L2 prefetch (side stream) | hidden under AR | 0 | 4.6 kernel time, 0 on the critical path | overlapped (confirmed by start/end times) |
| GDN recurrent kernel (48) | read 1 state + write 3 states per req: 16 x 6.3 MB = 100 MB/layer -> 100 us | 4.8 | 3.49 (72.7 us/layer) | 0.73: faster than the DRAM napkin -> part of the state writes is absorbed/merged in L2 (72 MB) or written back later; napkin overestimates |
| GDN conv, ba GEMM (bf16), splitK | small | ~0.4 | 0.68 | launch-size bound |
| full attention (16 layers) | KV ~0.2 GB fp8 | 0.2 | 0.40 | |
| norms / quant / act | activations, ~10 MB | ~0.1 | 0.8 | 5-6 us per kernel launch floor (~260 launches) |
| sampling, rejection, draft bookkeeping, AllGather/SendRecv | | ~0.2 | ~0.8 | SendRecv 0.46 ms once per step in draft 1 |
| GPU idle (CPU/launch) | 0 | 0 | 0.97 | graph trace |
| **total** | | **~27.5** | **32.7** | napkin 84% of measured |

The napkin reproduces the decode step within 16%, and part by part within ~20% except the small-kernel
floor (norms/quant at ~5 us per launch) and the GDN kernel (faster than its DRAM napkin). Corrected
napkin used for ranking: GEMMs at 0.83 TB/s, lm_heads at 0.85-0.93, hmar 46 us, small kernels at
measured launch cost, + 1 ms idle -> 32.5 ms.

Shares of the 32.7 ms decode step: body GEMMs 45%, hmar 19%, lm_heads 13% (draft 8%), GDN 11%,
everything else 12%.

## B. Mixed steps (a new request's prompt + ~15 decoding requests)
Measured (graph trace, 49 steps): median period **120 ms**; by prefill tokens P (prof_graph/steps.csv):
P ~ 57-91: 72-122 ms (bimodal ~78 / ~118 ms), P ~ 320-400: 106-122 ms, P ~ 600-900: 182-232 ms.
Slope ~0.22 ms per prefill token; intercept at P -> 0: ~60-100 ms, i.e. **2-3x a decode step even
with a tiny prompt**.

| part (P ~ 308 + 45 decode rows, node trace) | napkin | napkin ms | measured ms |
|---|---|---|---|
| FP8 GEMMs (M ~ 353, compute-bound) | 2 x 353 x 12.2e9 FLOP / F | at F = 165 TFLOP/s: 52; at 330: 26 | 41.2 (-> effective F ~ 210 TFLOP/s) |
| all-reduces (NCCL, M > 96 rows) | 132 x (3.6 MB / 25 GB/s ~ 145 us) | 19 | 38.7 (293 us each, incl. peer wait); step_anatomy 51 ms median over mixed steps |
| GDN prefill (chunked) + spec part | | ~3 | 7.3 |
| lm_head (2 calls), bf16 ba GEMMs, silu/norm (M ~ 353) | | ~5 | ~7.4 |
| draft steps | ~ as in decode | ~4.5 | ~4.5 |
| GPU idle | 0 | 0 | 5-68 (median 10.7 node / 7.5 graph; large for small-P steps) |
| **total** | | **~60-75** | **~120** |

The napkin does NOT reproduce mixed steps: measured is ~1.6-2x the napkin. Most of the gap is (1) the
NCCL all-reduces at ~2x their link napkin (and well above hmar's per-byte cost), (2) GPU idle in mixed
steps (piecewise graphs with eager attention/GDN between pieces, prefill metadata built on the CPU), and
(3) a P-independent excess of ~45-85 ms in small-P mixed steps whose cause is being analysed
(agents/leader-4.w1/mixed.md, task 2). This is a finding in itself: **the cost model's biggest
mismatch is in mixed steps, and mixed steps are ~32-38% of the benchmark's time**.

## C. Whole benchmark
Graph capture: decode steps 15.0 s (522 steps), mixed steps 7.0 s (49 steps): mixed = 32% of step time.
Harness (eval/1 trial 1): 111.8 s. Model: ~0.75 mixed steps per request (some steps take 2 prompts) ->
~300 mixed steps x ~140 ms mean = ~42 s; decode ~70 s / 30.6 ms mean = ~2,300 decode steps x 2.33
accepted x ~15 running = ~80K tokens + ~300 x ~35 tokens in mixed steps = ~91K tokens (measured
90,684). Consistent: **decode ~62%, mixed ~38% of the benchmark time**.

Ranking rule for step 4: a change saving x ms per decode step gains ~x/32.7 x 0.62 of throughput;
saving y ms per mixed step gains ~y/140 x 0.38.
