team:         T1
performance:  "unofficial: 0.0% vs T1-003 (same-job bench 400 x2); big prefill steps -10% (nsys); not evaluated officially"
summary:
  hypothesis: Prefill-sized ARs (NCCL, link-bound, ~31% of big prefill steps, 8% of span) can be hidden behind
              the preceding row-parallel GEMM by chunking tokens: GEMM chunk i+1 on main stream || NCCL AR chunk i
              on a side stream. Estimate ~+3%.
  change:     builds on T1-003. overlap_ar.py (new): for eager steps >= 512 tokens, TP=2, not capturing: n=4 chunks
              (>= 256 tok each), pynccl AR on a side stream, join, cat; output marked already-reduced (data_ptr),
              consumed by Qwen3_5RMSNorm.forward_fused / fused_allreduce_gemma_rms_norm (skip AR once).
              linear.py: RowParallelLinear (reduce_results=False) uses it. marlin.cu: env VLLM_MARLIN_SM_RESERVE
              (getenv per call) reduces Marlin's persistent grid by R SMs; set to 2 only around chunk GEMMs.
              Env VLLM_OVERLAP_AR_CHUNKS (4; 0=off). Needs C++ build.
  results:    Observed:
              - microbench (job 18, R=0): pipeline -13..-21% of (GEMM+AR) per layer; independent GEMM||AR barely
                overlap (K8704 T1024 1400 vs 1428 us): Marlin persistent grid holds all SMs; high-prio stream no help.
              - with R=2 (mb2): GEMM||AR K8704 T1024 1404 -> 751 us (fully hidden); pipeline n=4 per layer -23% (T=1024),
                -25% (T=2048); Marlin alone at 126 SMs +2%. Chunked GEMM bitwise >=99.94%, max diff 1 bf16 ulp.
              - job 23 (bench 100): OFF 646.3/643.2, ON 649.4/654.2 (+1.1%). gsm8k50 44/50; long_request 128K OK.
              - job 25 (bench 400): OFF 682.4/692.5, ON 686.1/688.2 (-0.04%). TTFT p50 ON 244/235 vs 228/223;
                TTFT p99 ON 1059/1250 vs 1338/1411; MTP AL equal. Engage log on both ranks (T=2048, n=4).
  conclusion: Interpretation: the per-layer saving in the microbench does not reach end-to-end throughput at c=16.
              Null result; not taken to official. (profile analysis: see below when added)
validity:     leader-1 review: math unchanged except Marlin per-M config fp reordering (<=1 ulp, allowed); comms bf16
              NCCL; gating on token count only (generic). Fragile: data_ptr marker requires every consumer of a
              reduce_results=False row-parallel output to call overlap_ar.consume (true for this model's norms and MTP;
              an aux-hidden-state (eagle3) path would double-reduce). getenv-based Marlin reserve is a hack. Would need
              hardening before any official use.
profile (job 27, nsys ON vs T1-003 OFF; bank/T1-004/profiles/why.md):
  Observed: prefill steps (>60 ms) 9.87 s/52 -> 8.87 s/50 (-3.8% of profile, = napkin). Biggest step 590 -> 531 ms:
  NCCL 53% overlapped with Marlin; Marlin & NCCL each ~+8% slower when concurrent. down_proj (K=8704) hides its AR;
  o_proj/out_proj (K=3072, 75-130 us chunks vs ~370 us AR) and the last chunk's AR stay exposed. Large-AR skew p50
  190 -> 52 us. But ON run had 29 more 30-60 ms steps (+1.26 s), cancelling it: total span -1.0%. Decode unchanged.
  Interpretation: the mechanism works (~-10% on big prefill steps) but the end-to-end effect is within run-to-run
  step-mix variation at c=16; throughput gain unproven (~0 +- 0.7% in bench 400).
