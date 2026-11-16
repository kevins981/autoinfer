team:         T6
builds_on:    T5-001 + T6-003
performance:  "eval/1 (GPU job 119): 900.3 tok/s, starts 895.6 / 905.1, both above the champion T5-001's 895.4 (+0.55%). Third start (eval/2, GPU job 121, TRIALS=1): 894.7, BELOW 895.4. Mean of 3 starts 898.5 (+0.34%): does not beat the champion under the rule; within noise."
summary:
  hypothesis: T6-003's exact row-split GEMM->AR overlap in mixed steps (>= 192 rows) stacks on T5-001's decode-side GDN DeltaLog: the two touch
              different steps (mixed vs decode) and different files.
  change:     T5-001's code + T6-003's split (t6_split_ar.py, the RowParallelLinear hook in linear.py, model.py/mtp.py incl. the final-norm/aux
              is_reduced fix). The 4 files were copied unchanged from workspace t6-split; T5-001 doesn't touch them (they are identical to T3-007's
              there). change.diff = ws diff t6-stack (the measured workspace); its size is T5-001's diff + T6-003's split hunks. Serve args = T5-001's.
  results:    Observed: eval/1: 900.3 (895.6, 905.1); acceptance 2.347/2.348 (T5-001: 2.343); TTFT p99 / TPOT p99 within the gate (gate_latency
              pass); 128K pass. "[t6_split_ar] first split call" appears in both starts. T5-001's own starts: 898.7 / 892.1.
  conclusion: Interpretation: +0.34% over 3 starts (895.6, 905.1, 894.7), consistent with T6-003's +0.28% on T3-007. Two codes give the same small sign,
              but each is within noise. Not a candidate. eval/2: acceptance 2.34, 128K pass, gate_latency pass.
validity:     As T6-003 (review T6-2, 2 turns): every split output is reduced exactly once (decoder norms, target and MTP final norms, aux capture);
              dyn128 scales sliced with the codes; graph capture safe for serialized forwards. Numeric check (job 116): split vs unsplit RowParallelLinear on TP=2
              with W8A8 Triton, bf16 and quantized inputs, M = 192/260/1000: bitwise identical. So the change is exact: GSM8K not rerun (T5-001's 93.2% carries
              over), and MTP acceptance is unchanged. No memory change (two events and one stream; the outputs are the same size).
