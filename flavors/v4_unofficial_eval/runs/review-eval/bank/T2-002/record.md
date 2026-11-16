team:         T2
builds_on:    T2-001 + T1-001
performance:  "server start failed (CUDA OOM during warmup), no measurement" (GPU job 10)
summary:
  hypothesis: stack T2-001 (prefill graphs to 2048) + T1-001 (fp8 draft lm_head) + two exact sampling
              flags: --enable-batch-sharded-sampling (target logits all-to-all instead of all-gather)
              and speculative-config use_local_argmax_reduction (drafter argmax before gather).
  change:     serve flags only (serve_args.txt).
  results:    Observed: OOM at server start: "Tried to allocate 20 MiB, 1.75 MiB free" in the MTP
              embed all-reduce during warmup (eval/1/trial1/server.log). Model loading 14.99 GiB vs 14.4
              (BASE/T2-001): the fp8 draft lm_head adds ~0.6 GiB; T2-001's graph pool adds ~0.8 GiB.
              Both flags logged as active before the crash.
  conclusion: Interpretation: with the 5 GiB KV budget fixed, T2-001 + T1-001 (+ these flags) don't fit.
              Needs a lower prefill graph cap, a trimmed KV budget (128K test), or dropping the extra
              draft-head copy. Not a performance verdict on any of the flags.
validity:     flags only; exact (sampling flags) / allowed by ruling 1 (fp8 draft head). Not measured.
review:       reviews/T2-1: failure recorded correctly (OOM in the first drafter prefill piecewise capture, MTP
              embedding out-of-place all-reduce, 20 MiB requested, 1.75 MiB free). The fp8 draft head copy is
              124160 x 5120 per rank = 0.592 GiB (matches 14.40 -> 14.99 GiB). The +0.81 GiB capture figure is
              from T2-001 vs BASE (net device memory over capture), not measured in this run. Defensible
              conclusion: this 4-option configuration fails at startup with the fixed 5 GiB KV budget; which
              subset is sufficient to fail is not isolated (T1's job 11, T2-001 + T1-001 alone, also OOMed).
