team:         T1
builds_on:    T1-010
performance:  "unofficial: PASS +0.52% vs T1-010 (GSM8K 93.6%, 128K pass)"
summary:
  hypothesis: In eager prefill steps over 512 tokens, the NCCL all-reduce takes ~33% of a ~234 ms step. T2's profile of
              T1-005 puts these steps at ~16.8% of time, so this AR is ~5.5% of total time. Splitting each RowParallel
              GEMM into 2 row chunks and running chunk 1's AR on a side stream while chunk 2 computes can hide part of
              it. Napkin: ~1-2%. T1-006 (on T1-005) screened at 0.0% with 1 start of 100 prompts per arm, which is too
              noisy for a 1-2% effect. Its "rank skew" explanation has since been refuted (no clock asymmetry today).
  change:     T1-006's code rebased on T1-010 (worker t1-w1, workspace t1-w6): RowParallelLinear._chunked_matmul_all_reduce,
              K=2. On when an eager step has >=192 rows (VLLM_CHUNKED_AR_MIN_ROWS, default 192 here; 0 = off).
              Skipped during CUDA-graph capture, so graphed decode and graphed mixed steps of <=512 tokens are unchanged.
              Side effect: with the knob on, use_fused_allreduce_norm is off for the whole model. The AR runs inside
              the linear (host AR / NCCL, same ops), and add+norm runs separately.
  results:    Observed: unofficial #1: B 755.7 / 757.3 vs HEAD mean 752.6 (+0.52%); GSM8K 93.6%; 128K pass; p99 TTFT 1217 ms
              (HEAD 1146), p99 TPOT 40.1 ms (HEAD 36.4; gate 55.3); acceptance 2.349. Mechanism (earlier, T1-006 code): side-stream NCCL AR
              hides ~74% behind bf16 GEMMs in the microbench (t1-w1 job 20). No in-situ profile of this record.
  conclusion: Interpretation: a small gain, consistent with the napkin's lower end (part of the ~5.5% prefill-AR time hidden,
              minus the K=2 GEMM efficiency loss). The p99 TPOT rise (+3.7 ms) suggests decode rows in big mixed steps pay
              for the side-stream contention. Small; promote only on the unofficial PASS.
validity:     Exact: row chunking of a GEMM followed by a sum AR of each chunk gives identical per-row results. T1-006 had
              80 per-layer checks bitwise equal at 2048 rows. Re-check on this code (job 93, t1-w1): 2048 rows: 260/260
              bit-equal; 512 rows: 34/256 bit-equal, max abs 0.125 at |ref| up to 84 (Marlin picks a different tile/split-K for M=256
              vs M=512, so the fp32 accumulation order differs). That is fp reordering of the same products, which the contract allows.
              gsm8k-8 100%. Unofficial eval run with --gsm8k. No precision change. Off by env.

unofficial #1 (2026-11-15T09:02): PASS: +0.52% vs HEAD T1-010 (B 755.7 / 757.3 vs reference mean 752.6 over 2 starts, 0 run in this job; 200 prompts); GSM8K 93.6%, 128K pass, p99 TTFT 1217 ms / TPOT 40.1 ms, MTP acceptance 2.349 (unofficial/1/unofficial.json)

official #1 (2026-11-15T09:13): 785.3 tok/s at c=16 (mean of 2 trials: 785.7 / 785.0, CV 0.1%) (+23.5% vs BASE), GSM8K 93.6%, MTP acceptance 2.333, TTFT p50 182 ms, TPOT p50 18.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
