team: T9
builds_on: T1-042 / M2-S3c (microbench only)
performance: "microbench: codec on the CE payload (S3cc) is SLOWER than S3c at every R (+108 us/layer R=384, +119 R=768, +400 R=1024); kill criterion hit"
summary:
  hypothesis: in S3c both chunks ride the CE, so shrinking the bytes to 0.781x (hostar block-exponent codec, bitwise) shortens the critical path; gate >= 40 us/layer saved at R=768.
  change: bench7.py + ce.cu + job.sh (this folder), from T9-004's harness (real GDN/attn core, FP8 Marlin, graphs, NUMA node 0, 8 layers, PC=2 pieces, 5 reps x 20 replays). ce.cu = T9-001's CE ext + the hostar codec functions copied unchanged (enc_block/dec_block) + enc_k (device encode, one thread per 32-value block, no O overflow region transferred) + dec_add_k (decode fused into the add). Payload sent = A|B|C|GH = 0.7812x. Jobs 109 (smoke), 110 (full), ~1 GPU-min.
  results: Observed (us/layer, median; r0 / r1 agree within 1%):
    R=384:  S2 1647.7 | S3c 1524.7 | S3cc 1633.1  -> S3cc - S3c = +108.4
    R=768:  S2 3054.2 | S3c 2905.9 | S3cc 3024.8  -> +118.9
    R=1024: S2 4092.5 | S3c 3830.2 | S3cc 4229.6  -> +399.4
    Bitwise: codec AR == engine DMA AR bitwise on randn payloads (rows 384 and 1024, ovf=0, ratio 0.7812). Chain check S3cc vs S3c r2/resid bitwise at R=384, 768; at R=1024 a few (~120) elements differ, caused by the omitted overflow region (3 overflow blocks counted: raw blocks beyond CEX=2 per 32-block group are lossy in this microbench since O is not copied). A heavy-tailed synthetic payload has ~1% overflow blocks (x1e3 tail test) -> not bitwise, as expected.
  conclusion: Interpretation: the encode kernel (before the D2H, on the chain stream) and the decode+add (after the H2D) add more SM time and serial latency than the 22% byte saving removes; the 1024 case is worst (+400) because the chunk is compute-bound and the codec kernels are SM-time extra. This agrees with T1-031 (compression on the DMA lane lost). Also, a guaranteed-lossless DMA codec needs an overflow path; a static-size CE copy cannot carry the sparse O region (a conditional copy / graph conditional node would be needed), which is another cost. Direction dead for the CE prefill payload; skip to B-lite.
validity: measurement only.
