# T29 fused producers (workspace t29-fu = t29-v2 + this)
## Code
- t29_bs_gemm/t29_norm_quant.cu: Gemma RMSNorm(+add) + group-128 e4m3 quant (+PI), mirroring flashinfer's CuTe
  FusedAddRMSNormKernel/RMSNormKernel (sm_89 path): 128 thr = 2 rows x 64; thread t holds cols b*512+t*8+v;
  h fp32 unrounded; variant 2 = sequential b-outer/v-inner fma, butterfly 1..16, w0+w1, div.rn, rsqrt.approx, (h*rstd)*(w+1).
- t29_fused.py: T29_PI_KEY (QuantKey subclass), norm_quant, silu_mul_quant (Triton), act-quant registry entry,
  capture mode (T29_CAPTURE=<dir>, eager), GEMM table compare (T29_CAPTURE_REF_TABLE).
- marlin.py: layer._input_quant_key = T29_PI_KEY; apply_weights consumes QuantizedActivation -> gemm_q.
- model.py Qwen3_5RMSNorm.forward_fused: T29 branch (same AR via all_reduce_for_unfused_norm) + capture branch.
- fused_allreduce_gemma_rms_norm.py: fallback AR factored out (same calls).
- flags: T29_FUSED=1; T29_FUSED_NORM_MIN_M (default 256); T29_FUSED_ALL_M=1 (T28, all M); T29_FUSED_NORM=0 (silu only).
## Evidence
- job 225 (probe.py): variant 2 the only bitwise of 8 vs flashinfer (y, residual, q, s; plain + PI; res + nores).
- job 226 (job2.sh): probe2 PI-store bitwise M=1..2049; capture 109 dumps -> cmp.py BITWISE PASS (cap/ dir, 1.8G);
  v2 vs T29-004 table 0 diffs on live activations (cap/gemm_v2_vs_ref.txt); fused server smoke OK (srv_fused/).
- timings: see job 226 log (fused norm loses below M=256; silu wins everywhere).
