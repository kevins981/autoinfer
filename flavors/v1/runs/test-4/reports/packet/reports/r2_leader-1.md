# Round 2 report, leader-1 (T1, overlap prefill AR with compute) — received 2026-11-07
Null on throughput; no official eval. T1-004 (from T1-003): chunked row-parallel GEMM + side-stream NCCL AR for eager steps >= 512 tokens,
Marlin SM reserve (env) of 2 SMs. Microbench: AR fully hidden (751 vs 1404 us), per-layer -23/-25% at T=1024/2048; chunked GEMM >=99.94%
bitwise, max 1 ulp. Same-job bench 400, 2 pairs: OFF 682.4/692.5, ON 686.1/688.2 = 0.0%. nsys: big prefill steps -10% (590->531 ms),
but the step mix varied (+29 mid-size steps in ON); span only -1.0%. Only down_proj (K=8704) hides its AR; o_proj/out_proj chunks too short.
Code hack-grade. KB: Marlin's persistent grid blocks concurrent kernels unless it reserves SMs; prefill ARs at PCIe floor (~14 GB/s);
a ~10% prefill-step speedup (~3-4% GPU time) is not measurable at c=16. Left open: hardened down_proj-only variant (~+1%, hard to measure).
GPU 26.9/55.
