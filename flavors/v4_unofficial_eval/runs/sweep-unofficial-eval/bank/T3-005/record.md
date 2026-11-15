team:         T3
builds_on:    T3-004
performance:  "microbench only: no gain (job 195); not served"
summary:
  hypothesis: GDN in_proj_ba (bf16, K=5120, N=48/rank) runs cutlass split-K 27 + splitKreduce = 18.3 us x 48 layers = 0.88 ms per
              decode step in T3's HEAD profile (agents/leader-3/prof_t3004_node); a small-N Triton GEMM could do it in ~4 us.
  change:     worker t3-w1, workspace t3-w4: Triton small-N bf16 GEMM (vllm/model_executor/layers/ba_triton.py) behind
              VLLM_GDN_BA_TRITON. Never run in a server.
  results:    Observed: (a) 36,672 of 38,235 ba kernels in the profile overlap kernels on another stream: ba runs on the aux stream
              (_ba_stream) concurrently with in_proj_qkvz (_w8a16_kernel, 55.7 us at M=48), so its 18 us is inflated by sharing and
              is off the critical path. (b) isolated, graph-timed: cuBLAS F.linear 4.1-4.7 us at M=16-96; best Triton (88 configs)
              7.8-9.5 us; Triton wins only at M=3 (22.8 -> 7.8 us). Exactness: 1 bf16 ulp differences, both as close to fp32 ref.
  conclusion: Interpretation: nothing to gain; the per-kernel sum (leader-3's script) double-counted an overlapped kernel. Closed.
validity:     not proposed.
