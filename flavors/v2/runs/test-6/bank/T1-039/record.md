team:         T1
builds_on:    T1-037
performance:  "unofficial: KILLED at microbench: slower in every config (+15.8 us/layer at 16 req with GEMM, cap 8)"
summary:
  hypothesis: The GDN replay kernel is HBM-bound on the fp32 state read (KB48). A lossless fixed-rate re-encoding of the state (~0.885 of the bytes) would save ~3.4 us/layer, ~0.15-0.19 ms/step.
  change:     ws leader-2.gdnstc (w23), default ON via VLLM_GDN_STATE_CODEC=1. NOT suitable as-is.
              - Format: per 32-row chunk, a raw low-16 mantissa plane, sign + top-7 mantissa, 4-bit exponent offsets vs a per-32-value block emax, aux for raw blocks; overflow -> whole head raw.
              - Marker: flag bit 7. Flush modes 0-2 decode to raw; mode 3 decodes only on a NaN-magic tag match.
  results:    Observed:
              Gate (job 153, 17,328 real head-states):
              - entropy: sign 1.0 bit, exponent 3.59 bits, mantissa 22.8 of 23 bits -> bound 0.859.
              - best fixed-rate (1x32, k=4): 0.8845, 1.4% raw blocks.
              Unit tests (job 158): 0 mismatches (cap 8 / cap 0 / wide / overflow / stale-flag / mode-3 tag). Negative controls fail: 5092 / 3696 / 3696 / 32 / 1838.
              Microbench (us/layer, alone / with GEMM; real states):
              | config | raw | full codec | decode-only |
              |---|---|---|---|
              | N=16 cap 8 | 49.9 / 151.8 | 53.3 / 167.6 | 49.8 / 161.1 |
              | N=16 cap 0 | 65.4 / 168.3 | 63.1 / 188.5 | 62.3 / 173.4 |
              | N=8 cap 8 | 26.1 / 130.0 | 27.8 / 133.6 | 26.8 / 132.1 |
              Estimate: -3.4 us/layer.
  conclusion: Interpretation: reading 11.5% fewer bytes saves nothing (decode-only -0.1 us). The kernel is NOT limited by state bytes. ~25 MB in ~50 us = ~0.5 TB/s, about half of HBM peak. KB48's "HBM-bound" looks wrong; the limiter is probably latency / the per-CTA pipeline. The codec only adds work (ALU, sub-32 B stores, barriers); the larger penalty with GEMM is unexplained (no profile).
validity:     exact by test (bitwise + controls); not proposed for official eval (slower).
