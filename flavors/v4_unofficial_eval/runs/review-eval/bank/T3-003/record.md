team:         T3
builds_on:    T3-001
performance:  +1.2% vs the champion (T3-001, 816.6): 826.1 tok/s = mean of 3 starts 821.2 / 821.9 (eval/1, GPU job 65) and 835.1 (eval/2,
              TRIALS=1, new GPU job 67); every start above 816.6.
summary:
  hypothesis: the drafter reads the bf16 target lm_head (1.27 GB/GPU) twice per decode step (2 x ~1.36-1.48 ms); its fp8 copy
              (draft_lm_head_quantization=fp8, ruling 1) halves those bytes. The copy (+~607 MiB/GPU) OOMs CUDA graph capture at the
              5 GiB KV budget, so KV is trimmed to 4.4 GiB (pays for the copy almost exactly).
  change:     flags only, on T3-001's code (change.diff = T3-001's, workspace t3-pf unchanged): speculative-config +
              "draft_lm_head_quantization":"fp8"; --kv-cache-memory 4724464025 (4.4 GiB). serve_args.txt.
  results:    Observed: eval/1 (job 65): 821.5 (821.2, 821.9), acceptance 2.333, TTFT p99 1044/968 ms, TPOT p99 34.6/34.7 ms,
              gate_latency pass, 128K pass, preemptions 0 in both starts. eval/2 (job 67, TRIALS=1, GATES=none): 835.1, acceptance 2.348,
              TTFT p99 963, TPOT p99 33.5. KV capacity 235,650 tokens (vs 267,721).
              Mechanism (worker leader-3.w1, agents/leader-3.w1/mb4a.log): drafter head at M=16 in a CUDA graph, cold: bf16 cutlass wmma
              1484 us vs fp8 copy (per-token fp8 quant 3 us + cutlass fp8 scaled_mm 775 us) 783 us -> ~-1.2 ms/decode step napkin (~3%).
              Quick benches: job 63 821.3 vs 811.5 (champion config, same job), job 64 824.4. Startup OOM at 5 GiB: agents/leader-3.w1/
              fail1/srv9_fp8head/server.log.
  conclusion: Interpretation: a real but smaller gain than the head microbench (+1.2% vs ~3%); part of the drafter head time is
              presumably not on the critical path or the step has other costs. T1-001's earlier +6.7% (on BASE) and T1-003's neutral
              stack (on T2-003, KV 4.25, preemptions) are consistent with a modest gain once the full-logit gather was gone.
validity:     Ruling 1: fp8 copy used only by the drafter (proposals); the target lm_head stays bf16 and verification is exact (review
              T3-4 checked share_target_lm_head/quantized_lm_head_copy: new fp8 storage, the target's bf16 storage untouched; Qwen3.5
              MTP uses its top-level lm_head via the local-argmax mixin). Disclosure: the copy uses per-row FP8 weight scales and per-
              token FP8 activations. KV budget rule: 128K request served (eval/1 long_request pass); 0 preemptions in all starts.
              Caveat (review T3-4): KV usage peaked ~95.8% in a quick run at 4.4 GiB; less headroom than at 5 GiB for longer mixes.
              GSM8K not needed: target arithmetic and exact acceptance unchanged from T3-001 (review T3-4 agrees).
