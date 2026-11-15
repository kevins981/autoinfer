team:         T1
builds_on:    T1-005
performance:  "unofficial: +8.41% at c=16 vs T1-005 (ABBA, 200 prompts, all B > all A), T1 GPU job (t1-triton)"
summary:
  hypothesis: ~1.2-1.9 ms of GPU idle per 34.7 ms decode step (TRACE=graph profile, worker job 28, agents/leader-1.w1/idle/)
              comes from CPU-bound attention-metadata prep that FlashInfer serializes with two host syncs
              (flashinfer.py ~1698 seq_lens.cpu() when spec tokens are used; no in-place draft decode metadata update ->
              eager rebuild + sync before draft step 2). TRITON_ATTN supports in-place draft metadata update and fp8 KV on
              sm89 and needs no .cpu() in its build -> expected ~+4-5% if its kernels are not slower.
  change:     serve flag --attention-backend TRITON_ATTN on top of T1-005 (no code).
  results:    Observed (agents/leader-1/jobs/triton/abba.log): A 681.9/680.4, B 741.8/735.1 tok/s; acceptance A 2.344/2.346,
              B 2.354/2.355; p99 TTFT/TPOT B 1212/1178 / 46.6/40.9 vs A 1090/1043 / 47.8/48.1 ms. gsm8k 50 on B: 88.0%
              (= BASE's documented 88.0%), 0 no-answer.
  conclusion: Interpretation: +8.4% is ~2x the idle estimate: Triton also removes FlashInfer's per-step planning work on the
              mixed (eager) steps, and/or its kernels are faster at these shapes. Not yet split; profile later.
validity:     Different attention kernel = fp reordering of the same softmax attention over the same fp8 KV cache (allowed:
              kernels may change). fp8 KV kept. Engine option, generic. Memory: no KV budget change; 128K checked by official eval.
