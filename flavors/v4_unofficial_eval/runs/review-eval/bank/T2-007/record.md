team:         T2
builds_on:    T1-007 + T2-006
performance:  "+2.1% vs the champion (T1-007, 771.3): 787.6 tok/s (starts 792.6, 782.6), eval/1 (GPU job 38); both starts above 771.3; gain > 1%, so no additional start required under the promotion rule"
summary:
  hypothesis: T2-006 (fused dyn128 activation quant in the RMSNorm and SiLU producers, plus GEMM configs expanded to all
              graph M) and T1-007 (host-mapped all-reduce, hmar) act on different costs, so their gains should stack.
  change:     change.diff = bank/T2-006/change.diff (all files) + bank/T1-007/change.diff's three distributed files
              (cuda_communicator.py, hmar/hmar.cu, host_mapped_all_reduce.py). Checked hunk-for-hunk by leader-2 and by
              the reviewer. Workspace t2-stack (leader-2.w1); the hmar extension was prebuilt on CPU.
              serve_args.txt is identical to T1-007's (= T2-004's).
              Measured code = filed code.
  results:    Observed (eval/1, harness with GATES=long,gsm8k, GPU job 38):
              - throughput: 787.612 tok/s (starts 792.591 / 782.633; reviewer recomputed T1-007 = 771.281);
              - MTP acceptance 2.332;
              - TTFT p99 1019 ms, TPOT p99 33.4 ms (each start within the BASE latency limits);
              - 128K request passed (HTTP 200, health check OK afterwards);
              - GSM8K-250 93.6% (234/250).
              Startup log on both starts: "Using ['HMAR','PYNCCL']". hmar launch counters (ar_stats.rank0/1, not
              separated by phase) show 672,543 launches per rank over the whole job: hmar runs, but this does not give
              the dispatch rate of the fused decode calls specifically.
  conclusion: Interpretation: the stack keeps most of both gains. The T2-006 part is fusion plus the expanded GEMM configs
              (8 -> 31 M buckets); neither was isolated, and no additivity is claimed. The +2.1% is the measured stacked
              result over T1-007.
validity:     leader-2 review, with reviews/T2-6 (code and numbers) and the worker's answers:
              - Same W8A8 scheme as T2-004/T2-006: dynamic per-token-group-128 e4m3 activations, fp32 K-block
                accumulation, FP8 weights and KV unchanged.
              - The all-reduce is at baseline precision (bf16 sum, T1-007's validity carries over).
              - The fused norm path calls tensor_model_parallel_all_reduce -> GroupCoordinator ->
                CudaCommunicator.all_reduce -> hmar when eligible: bf16, contiguous, 16-B aligned, payload
                <= 983,040 B (i.e. M <= 96 rows of 5120). Larger captured mixed-step graphs use NCCL on both ranks.
                No path bypasses the communicator on sm89.
              - Disclosure (inherited from T1-007, not added here): hmar's eligibility check is per rank. For arbitrary
                inputs (e.g. a view offset by one element on one rank only) the ranks could pick different backends
                and hang. No producer of such a mismatch exists in this path: all-reduce inputs are fresh row-parallel
                GEMM outputs, and whole-row slices keep 16-B alignment.
              - Disclosure (inherited from T2-006, allowed by the manager's ruling): the SiLU producer keeps silu*up in
                fp32 and uses scale = max(amax/448, 1/(448*512)). The floor changes the scale only for groups with
                amax < ~2e-3. The error bound stated in T2-006 ("<= 2.2e-6") is wrong: FP8 code spacing varies. Example:
                at the floor scale s, x = 399*s maps to code 384, an error of ~6.5e-5. No numeric bound is claimed;
                GSM8K is the empirical check.
              - No new aliasing or race: hmar writes a separate output, and the norm updates the residual in place
                after it. Zero-row steps skip the producer launches.
review:       reviews/T2-6 (codex, 1 round): numbers, gates, diff identity, dispatch and graph-capture behaviour
              confirmed. Points (inherited hmar rank-consistency, wrong T2-006 error bound, gain attribution, wording)
              were all agreed by the worker and adopted above.
