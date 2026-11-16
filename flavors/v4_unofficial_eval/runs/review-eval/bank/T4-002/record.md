team:         T4
builds_on:    T3-007
performance:  "screen negative (no harness): quick bench, 1 start each, same job, 400 prompts c=16: fused ON 852.1 vs OFF 850.7 tok/s (+0.2%, noise); unprofiled decode D=16 step median 30.44 (ON) vs 30.07 ms (OFF), +0.37 ms."
summary:
  hypothesis: each of the 128 decode HMAR all-reduces per step is followed by a separate Gemma add-RMSNorm + dyn128 quant Triton
              kernel (~2.8 us + launch gap, one more pass over x); fusing them into the HMAR kernel saves ~0.4-0.6 ms per decode step
              (~+1%) (T4 update-1).
  change:     by worker leader-4.w1 (workspace t4-arn): hmar.cu new hmar_norm_kernel / allreduce_norm_quant (blocks own whole rows,
              <= 6 rows/block for M <= 96; sum bf16(fp32 a + fp32 b) into shared memory, residual add fp32, residual stored bf16,
              fixed-order row sum of squares, rsqrt.approx, bf16 y, per-128-group amax, div.full scale/inverse, e4m3 satfinite);
              host_mapped_all_reduce.py all_reduce_add_rms_norm_dyn128 (same L2 prefetch side-stream launch); model.py forward_fused
              hook, VLLM_T4_ARN=0 disables, fallback for M > 96 / other shapes. Serve args = T3-007's.
  results:    Observed (agents/leader-4.w1/arn.md, eval/quickbench): 2-process test M in {1,7,16,42,48,96}: AR sum, residual, scales
              bitwise equal to the unfused path; y differs in 0.0008-0.003% of elements by 1 bf16 ulp; 0.25-0.29% of fp8 codes differ
              (Triton's fp32->e4m3 rounds some ties differently; torch RN differs from Triton by the same 0.26%). GSM8K-50 ON 88.0%.
              Graph microbench, us per AR+norm pair with prefetch: M=16 unfused 20.2-20.4 vs fused 20.3-20.6; M=48 46.6-47.6 vs
              46.9-48.3; M=96 96-97 vs 97.5-102.6 (AR alone 18.0 / 43.7-45.2 / 94-96). Server: fire line in both workers (rows=16);
              per-iteration (--enable-logging-iteration-details) decode D=16 median +0.37 ms ON, mixed steps unchanged within noise.
  conclusion: Interpretation (wording narrowed per review T4-3): HMAR plus the prefetch join dominates each pair (~44 of ~47 us at M=48;
              PCIe transfer vs flag wait vs prefetch not separated); inside a CUDA graph the separate norm
              costs only ~2-3 us and no launch saving appears; the fused kernel starts row work only after its whole chunk arrives and
              serialises up to 6 rows per block, so it costs the same or slightly more. Hypothesis (not shown): a per-row pipelined variant might win
              ~0.3-0.4 ms/step (estimate). Not pursued.
validity:     leader-4: AR at baseline precision (bitwise); norm/quant same scheme (dyn128, group 128), reduction order and tie rounding
              differ (fp reordering): would need GSM8K-250 if revived. Not a candidate. Review reviews/T4-3: agrees screen negative; exactness is
              sample-specific (not every fp8 mismatch shown to be a tie / <= 1 ulp): revival needs a numeric bound and GSM8K-250.
