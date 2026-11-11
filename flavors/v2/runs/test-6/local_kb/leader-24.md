# local_kb leader-24 (T24, explore)
## Direction candidate: warm-optimised M<=48 decode GEMM + prefetch
- Reasoning: KB79 in-situ decode GEMMs ~= Marlin L2-warm floors; warm efficiency 49-67% of HMMA (qkvz 4.03 GF/41.6us=97TF,
  gdn_out 1.51 GF/21us=72TF, down 4.28GF/45us=95TF, gate_up 8.56GF/87.8us=97TF). T17 math-only 130 TF (16 warps), padded M64.
- Napkin (assumed): GDN layer GEMMs in situ ~213 us; at ~120 TF + 3 us/kernel: ~160 -> -53 us/layer; attn layer -51;
  total ~-3.3 ms/step decode (13%) -> ~+7-8% e2e if HBM schedule holds (192 MB/layer = ~200 us HBM vs ~295 us layer).
  Needs cross-GEMM prefetch (down during gate_up) since faster GEMM leaves HBM spare.
- Risk: T8 (2-warp, 57% with stream), T17 (B stream 0.84 TB/s). Untested: M=48-specific 3 m-tiles, decoupled L2 prefetch warps.
- W1-1 (leader-24.w1): step 1 warm floor at M=48, step 2 decoupled load path. ws leader-24.k48 (= T14-004, built).
- W1-1 RESULT (job 155, 0.2 min): KILLED. gate_up M=48 no-B-loads 88.7, no-loads 80.4, B-loads-only warm 103.2 vs Marlin warm 85.8. Per-fragment fixed cost dominates; Marlin near realistic warm floor. bank/T24-001. Recommended voluntary close.
