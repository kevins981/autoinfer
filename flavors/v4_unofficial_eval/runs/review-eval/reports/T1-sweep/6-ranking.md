# T1 sweep, step 6: ranking

Napkin gain against BASE's measured 143.2 s per trial (635.9 tok/s). Gains overlap (1 and 4 both
attack the mixed-step all-reduce): they don't add.

| # | candidate | napkin gain | wins if | riskiest assumption | cheapest test that could kill it |
|---|---|---|---|---|---|
| 1 | Faster all-reduce: NCCL env (proto/channels/SHM) and/or FlashInfer PCIe-IPC AR (opt-in, decode sizes <= 96 rows) | up to +14% (decode -2.6 ms/step, mixed -47 ms/step) | a variant beats NCCL-LL at 430 KB and at 2.66 MB | NCCL/SHM is link-limited, not protocol-limited; IPC path may need P2P | AR microbench, real shapes, 2 GPUs (running: worker leader-1.w1, <=5 GPU-min) |
| 2 | Draft lm_head: FP8 (flag) -> reduced vocab + replicated | +2% (FP8) to +4.5% (reduced) minus acceptance loss | acceptance length drops < ~2% | the MTP path honours `draft_lm_head_quantization` for qwen3_5 | the FP8 flag in one harness run (acceptance + tok/s) |
| 3 | W8A8 block-FP8 (or better) GEMM for M >= ~128 (mixed steps) | +8.7% | a kernel reaches >= 70% of 165 TFLOP/s at M=256..2048 on these shapes | FP8/fp32-acc on Ada is no faster than bf16, so the gain is only Marlin's inefficiency | microbench Triton w8a8_block_fp8_matmul vs Marlin on the 4 GEMM shapes (CPU build, ~2 GPU-min) |
| 4 | DBO on mixed steps only (overlap AR of micro-batch A with GEMMs of B) | +11% before #1, ~+3% after | vLLM's ubatching overlaps TP all-reduce | DBO in this vLLM only overlaps EP all-to-all | read the code (CPU) |
| 5 | k=3 drafts with the cheap draft head | +3% now, ~+9% after #1, #2 | p3 >= ~0.35 and draft step < ~1 ms | acceptance at position 3 | harness run with k=3 (gives p3 directly) |
| 6 | argmax-before-gather (draft always; target when all greedy) | +1.7% | — | the drafter already does a local argmax | read the code (CPU) |
| 7 | GEMM_out split along d, overlap AR | +4-7% before #1 | NCCL and Marlin co-run without slowing each other | SM contention | a later microbench |
| 8 | CPU idle / async scheduling | <= +3% | — | async scheduling may already be on | read the code / server.log |
| 9 | Decode GEMM kernel (80% -> 90% BW), DeltaLog GDN | +5%, +1.6% | — | high build cost | later |

Plan: #1 (microbench now) -> harness the best AR variant; #2 FP8 flag in the same session;
then #3 microbench; #5 once #1/#2 are in.
