team:         T1
performance:  "unofficial: 704.9 vs same-job T1-005 696.9 tok/s (+1.1%), within noise"
summary:
  hypothesis: drafts only need argmax; use_local_argmax_reduction replaces the 2 x ~0.35 ms draft-logits all-gathers.
  change:     T1-005 + speculative-config flag use_local_argmax_reduction=true (serve_args.txt carries the full spec JSON).
              Worker leader-1.w3, ws leader-1.logitsflags (out/C1).
  results:    Observed (job 22, bench 400 c=16, same job as C0=T1-005): 704.9 tok/s, TTFT p50/p99 210/1460, TPOT p50/p99
              20.7/49.1, acceptance ~2.34, gsm8k50 88%. Spec config confirmed in server.log. Napkin +1.8%.
  conclusion: Interpretation: works, small (~1%) gain, below noise alone.
validity:     leader-1: existing vLLM flag; drafts change only proposals (greedy argmax of the same logits); target verification
              unchanged -> output identical. No memory change. OK.
