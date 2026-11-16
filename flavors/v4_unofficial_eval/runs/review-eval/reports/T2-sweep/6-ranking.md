# T2 sweep, step 6: ranking

Napkin gain vs BASE (143.2 s per trial, 635.9 tok/s). Overlapping items don't add (W1, P4 and
P6 all touch mixed steps). Split with T1 (to agree): T1 owns the AR transport, draft lm_head
(fp8 / reduced vocab) and k=3; T2 owns W1, W8A8 for mixed steps, argmax-before-gather and lm_head
kernel choice.

| # | candidate | napkin | wins if | riskiest assumption | cheapest test that could kill it |
|---|---|---|---|---|---|
| 1 | **W1: CUDA graphs for mixed steps** (`max_prefill_cudagraph_capture_tokens` 2048) | up to +9% | eager launch overhead ~35 ms per mixed step is real outside nsys, and the graphed GDN prefill path is captured (breakable graph) | the ITL p90 of 117 ms comes from launch overhead, not GPU work; graph memory fits with the 128K request | harness run with the flag (queued, job 4) |
| 2 | **R4: W8A8 block-FP8 GEMM for M >= ~128** (mixed steps), Marlin below | +3% (165 TF, kernel efficiency only) to +7% (330 TF) | Triton/other kernel beats Marlin per token at M=128..2048 on the real shapes | FP8 fp32-acc throughput on Ada; quant overhead | worker GEMM microbenchmark (job 3) |
| 3 | **R1: argmax before gather** (drafter always; target when all requests in the batch are greedy w/o logprobs) | +2.8% | gathers are ~1.6 ms per step | sampler/rejection paths need full logits for other features | read sampler code (CPU) |
| 4 | **R2'/R2a: lm_head kernel choice** (target 1.59 ms -> ~0.8; draft 1.37 -> ~0.75 if T1's fp8 head doesn't supersede) | +1.4% (+2.1%) | a cuBLAS/cutlass config streams 0.635 GB at >= 80% BW for M=16/48 | torch picks the kernel; forcing another needs a custom op | microbench torch.mm vs alternatives at (16|48) x 5120 x 124160 |
| 5 | P6/T1-#4: overlap AR with GEMMs on mixed steps | <= +7% before W1/AR fix, less after | mixed steps are AR-bound after W1 | NCCL and Marlin co-running | after #1 result |
| 6 | W2: host gaps in decode step (1.3 ms) | +2% | async scheduling works with MTP | — | server log / flag |
| 7 | R4': W8A8 / better kernel at decode M=48 | 0-5% | a kernel beats Marlin's 650 GB/s in-server | kernel quality at small M | same microbench (M=48 rows) |
| 8 | ReplaySSM GDN state | +2% | — | high build cost | later |
