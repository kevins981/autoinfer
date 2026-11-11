team:         T1
builds_on:    T1-012
performance:  "unofficial: +5.5% same-job (A 666.7/678.4 vs B 708.2/711.6, 200 prompts c=16, A B B A fresh servers)"
summary:
  hypothesis: FlashInfer's per-step host-side planning leaves ~0.9 ms GPU idle before the target graph each decode step
              (run-1 change 6, nsys TRACE=graph); TRITON_ATTN has no such planning.
  change:     serve flag only: --attention-backend TRITON_ATTN (default picked FLASHINFER). Code = T1-012. Worker leader-2.w7,
              ws leader-2.triton.
  results:    Observed (agents/leader-2.w7/job1): tok/s A 666.7, 678.4; B 708.2, 711.6 (+5.5%); TTFT p99 A 1367/1375, B 1306/1335;
              TPOT p50 A 21.1/20.8, B 19.9/19.8 ms; acceptance unchanged (2.344/2.368 vs 2.348/2.362). KV 267,721 tokens both,
              Running 16 / Waiting 0, GDN replay on (48 layers), prefill graphs captured, no new warnings. gsm8k50 88% (= base subset).
              Estimate: 0.9 ms/~46 ms step ~ +2-3%; measured ~1.2 ms/step (+5.5%), larger than estimated.
  conclusion: Interpretation: FlashInfer planning was on our critical path too. 128K with TRITON pending (w7 job 2).
validity:     leader-2 review: a different attention kernel for the same math (fp reordering, allowed by task_definition);
              no code change; acceptance and GSM8K unchanged; KV/memory identical. 128K must pass (checked in official eval). OK.
