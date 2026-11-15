# T4 sweep, step 6: ranking (vs HEAD T3-004; wall ~99.5 s; mixed steps ~38%)

| rank | candidate | napkin gain | wins if | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| dropped (ruled NO, final) | **R4** FP8 W8A8 prefill with fp16 accumulation per K sub-block, fp32 across blocks | W8A8 GEMM time in mixed steps ~-40% => ~5-6% | ruled allowed; tensor cores reach >450 TF with f16 acc in Triton at M 128-1024; no overflow; GSM8K A/B within 2 points | that it's allowed (ruling asked); overflow (e4m3 products to 2e5 vs fp16 65504) | the manager's ruling; then a 1-GPU microbench: Triton block-fp8 GEMM with `out_dtype=f16` per 32/64-K sub-block vs current, TF/s + max-abs-diff + inf count on real activations |
| 2 | **P4** layer-level 2-micro-batch overlap in mixed steps (split the prompt rows by position into 2 halves, staggered by one layer half on 2 streams; half B's GDN/attention takes half A's state/KV of the same layer) | comm ~35 ms + compute ~25-30 ms per ~300-token mixed step -> ~max + ramp: -15..-20 ms per prompt => **~6-8%** | concurrent AR (SM-driven host AR / NCCL, few SMs) doesn't slow the other half's W8A8 GEMMs much, and halves of ~150 rows stay near-compute-bound | link/HBM/SM contention (T2-005..007 coupling, T1-012 TPOT rise) | 2-process microbench of one layer's GEMM+AR chain at real shapes: serial 300 rows vs two 150-row halves staggered on 2 streams, in a CUDA graph, both GPUs (~3 GPU-min) |
| 3 | decode W8A16 kernel > 0.9 TB/s (new design: persistent, deeper cp.async pipeline, split-K across SMs) | -1..-1.5 ms/step => 3-4% | in-situ BW reaches the isolated BW | T1's 0.77-0.90 ceiling is the kernel family, not the card | isolated microbench of gate_up 17408x5120 at M=48 vs 0.79 TB/s in situ |
| 4 | lm_head bf16 Triton GEMV/GEMM (0.80 -> ~0.92 TB/s) | -0.2..-0.3 ms/step => ~0.8% | in situ like isolated | T1-014's in-situ loss was not the kernel's fault | microbench + quick bench |
| 5 | decode column-split GEMM||host-AR overlap | <=4% (T3 SM2) | host AR and GEMM don't contend | coupling (T2) says they do | same microbench as #2, decode shapes |

Dropped with evidence (file 5): pipelined/duplex AR, k=3, tree, lazy3, AR+norm fusion, PP, W8A8 at decode M.

Plan with 77.6 GPU-min: ruling for #1 (free); worker builds the #2 microbench (gate, ~3 min) and the #1
microbench if allowed (~2 min). Only a passing gate gets engine work. #3-5 are fallbacks.
