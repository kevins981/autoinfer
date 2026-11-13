team:         T52
builds_on:    T52-001 (microbench only; no engine change)
performance:  "microbench parity: K-chunk cut-through AR2+qkvz 89.0-91.3 us vs champion 88.3-91.8 (NL=4 chain, both ranks; jobs 447-457, ~2.0 GPU-min)"
summary:
  hypothesis: with 1/rms deferred past the W8A8 quant (ruling 2026-11-11), the consumer GEMM can eat K-chunks as AR chunks land, so the
              AR read/add/norm/quant (~16-25 us) overlaps the GEMM: est. ~55-60 us best, 65-75 in chain, vs ~89.
  change:     agents/leader-52.w1/fused2.cu: 128-CTA t31b grid (320 thr, 168 regs); warps 0-7 GEMM, warp 8 AR writer, warp 9 AR reader
              (residual add, ssq partial, quant of bf16(x*(1+w))); C=8 x 640-col chunks (T53 granularity), interleaved order; champion
              0.78 codec; host flag per chunk -> 1 host poller + device-word relay (T53's finding); r applied to the fp32 acc in epilogue;
              options: capped L2 prefetch, writer pacing. Harness chain53.py/.sh; notes NOTES.md.
  results:    Observed: correctness vs champion chain max-abs 15.4 (max|out| 904), relF 1.5%; vs fp32 truth both equal error (champion relF
              2.45%, fused2 2.42%). Chain slot (r0/r1): champion 88.3-91.8; no stream 96.7-97.2; pf cap 5 91.4-92.7; cap5+pacing40 89.2-90.5;
              cap7+pacing40 89.0-91.3 (best); full-slab pf 97.8-98 (delays chunks ~10 us). Timeline (best): slot 0 published 18-20 us,
              peer sees flag +5-7 us, first chunk fully processed 41-47 us, last 66-72; GEMM K-loop ~45 -> 78-85 at ~2.0 us/step (cold pace).
  conclusion: Interpretation: the mechanism works but a chunk becomes usable ~20 us after publish and the first publish takes ~18-25 us,
              so the GEMM starts at ~45 us and only overlaps the last ~25 us of the AR; warming L2 during the AR delays chunks by as much as
              it saves. At decode scale (GEMMs 30-80 us) the host-staged chunk latency (~20-40 us) eats the overlap; same tail limit as
              T53's producer side. Not separated: reader compute vs PCIe read latency within the ~20 us publish->ready lag.
              Consumer-side decode fusion: two variants (T52-001 whole-row, T52-002 cut-through) at or below parity -> closed for decode
              with this transport. The same mechanism at prefill scale (GEMMs 200-400 us vs ~20-40 us chunk latency) is the pivot.
validity:     microbench only; fp reordering per the 2026-11-11 ruling (not bitwise). Not for official eval.
