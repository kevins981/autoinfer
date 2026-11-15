team:         T1
builds_on:    T1-017
performance:  "unofficial: PASS +2.12% vs T1-017 (GSM8K 93.2%, 128K pass)"
summary:
  hypothesis: (1) Tuned W8A8 Triton configs for this 4090 are 13-26% faster than vLLM's default at every (shape, M)
              (jobs 156, 163), and W8A8 then wins from M=80 (job 158). (2) The last Marlin GEMMs, GDN in_proj_qkvz
              (8192x5120/rank, 48 layers) and attention qkv_proj (7168x5120, 17 layers), hold ~21% of the Marlin bytes.
              On the checkpoint-layout route their W8A16 is 0.85-1.02x Marlin at M<=80 and their W8A8 is faster above
              (job 163). (3) T1-018 lost (-6.1%, TTFT p99 2559) to runtime Triton JIT, because warm-up didn't cover
              every config and the M%16 variants. This record warms every config and variant up through the real
              serving path.
  change:     on T1-017 (workspace t1-w14, worker t1-w1):
              - tuned W8A8 JSONs for the 5 shapes (utils/configs/N=..,K=..,device_name=NVIDIA_GeForce_RTX_4090,...);
              - W8A8_MIN_M = 80 for all 5;
              - VLLM_W8A16_TRITON_PROJ default "out_proj,gate_up,down,qkvz,qkv", so all target/MTP fp8 linears are on
                the checkpoint layout (no Marlin repack);
              - per-shape W8A16 small-M configs from job 163;
              - warmup(): every W8A16 config, every W8A8 config at M=128/129, and every JSON key >= 80 through the
                real _w8a8 path at M=key and key+1;
              - misaligned inputs are cloned.
  results:    Observed:
              - jobs 166-179 (check8.py): 0 Triton compile events after warm-up over 168 M values x 5 shapes (odd,
                multiples of 16, offset views), with warm and cold caches; warm-up adds ~15 s per rank with a cold
                cache (~1 s warm).
              - layer check with real weights (job 163): shard/scale mapping correct; W8A16 error 2.1-3.0e-3 of
                max|out| (Marlin 2.6-3.8e-3); W8A8 2.3-3.4e-2.
              unofficial #1: B 854.4 / 847.1 vs HEAD mean 833.1 (+2.12%); GSM8K 93.2%; 128K pass; p99 TTFT 956 ms
              (HEAD 1003), p99 TPOT 31.6 ms (HEAD 45.4); acceptance 2.330. 0 "JIT compilation during inference" warnings in the server logs.
  conclusion: Interpretation: tuned W8A8 configs and the qkvz/qkv move pay as priced (~+2%), and complete warm-up removes the
              T1-018 stalls. p99 TPOT drops 30%, so mixed steps are now much cheaper.
validity:     W8A16: same math as Marlin up to fp reordering. W8A8: the checkpoint's own per-token-group-128 fp8 scheme,
              allowed per batch size, with a same-job FP8 GSM8K A/B (below). Comms bf16; target lm_head unchanged.
              Configs keyed on shape and device. Reviewed by leader-1.

unofficial #1 (2026-11-15T12:34): PASS: +2.12% vs HEAD T1-017 (B 854.4 / 847.1 vs reference mean 833.1 over 2 starts, 0 run in this job; 200 prompts); GSM8K 93.2%, 128K pass, p99 TTFT 956 ms / TPOT 31.6 ms, MTP acceptance 2.330 (unofficial/1/unofficial.json)

FP8-activation GSM8K A/B (task_definition: Activations > Accuracy), job 182, same job, same code (workspace t1-w14 = this
diff; script notes/gsmab19.sh): bf16-activation reference (VLLM_W8A8_PREFILL=0) GSM8K-250 = 93.2% (no-answer 5); W8A8 on
(default) = 93.6% (no-answer 4). Difference +0.4 points, within 2.0. PASS. Results in notes/.

official #1 (2026-11-15T12:48): 897.5 tok/s at c=16 (mean of 2 trials: 899.2 / 895.8, CV 0.3%) (+41.1% vs BASE), GSM8K 93.2%, MTP acceptance 2.32, TTFT p50 164 ms, TPOT p50 16.3 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
