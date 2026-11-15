team:         T2
builds_on:    T3-006 (champion) + reduced-vocab draft head (T2 worker leader-2.w1, task 11, ws t2-dv on T2-009)
performance:  "OFFICIAL #1: 931.8 tok/s (+1.39% vs T3-006 919.0, +46.5% vs BASE), GSM8K 93.2%, all gates pass. Unofficial: +3.14% ABBA on T2-009 (job 116)"
summary:
  hypothesis: The MTP draft's FP8 lm_head reads 124160x5120 rows per GPU per draft call (2 calls/step, ~1.44 ms/step
              in T2-006 nsys). Only the draft's proposals depend on it; the target verifies with its full bf16 head.
              Restricting the draft argmax to the tokenizer's first K ids (BPE merge order ~ frequency) plus all added/
              special tokens cuts that cost with little acceptance loss.
  change:     online/lm_head.py `_maybe_reduce_vocab`: after the FP8 draft head copy is made, keep rows for token ids
              [0, K) + [248044, 248077), re-split into contiguous halves per TP rank (padding = duplicate of the last
              row). Local row -> global id table `_dv_ids`.
              logits_processor.get_top_tokens: if `_dv_ids` is present, map the local argmax through it (no
              outside-vocab mask needed), then the same (value, id) all-gather reduction.
              Env VLLM_T2_DRAFT_VOCAB_K, default 151643 in this record (0 = off). Only the draft head copy
              (spec_decode/eagle/utils.py) goes through this path; the target lm_head is untouched.
  results:    Observed:
              - Smoke (worker job 113, bench 100, 1 run each): K=0 841.6, K=151643 864.7, K=98304 858.4 tok/s.
                Acceptance length 2.356 / 2.369 / 2.347.
              - Memory: model load 14.81 -> 14.58 GiB/rank (the full fp8 draft copy is replaced, not kept).
              - Job 116 (ws t2-dv = T2-009 + this; A K=0, B K=151643): 128K request pass on B.
                ABBA: A 886.7/891.7, B 914.7/919.5 tok/s, +3.14%. Acceptance 2.339/2.346 (A) vs 2.335/2.334 (B).
              - Coverage of ids < K on generic local text (vLLM docs + code, 1.7M tokens; agents/leader-2/vocab/cov.py):
                98K and 151.6K ids both cover 99.83%.
              Napkin was ~0.55 ms/step (~1.8%); measured more (+3.1%).
  conclusion: Interpretation: larger than the GEMM-byte napkin. The full-vocab path also pays the outside-vocab mask
              and a max over 124K columns per row, which shrink too (not separately measured). Acceptance is unchanged
              within noise.
              Port: the two files are untouched by T3-006 and were copied unchanged onto ws t2-dv6 (from T3-006), with
              the default flipped to 151643.
validity:     (leader-2) Speculative decoding stays exact: only the draft's proposals change, and the target verifies
              every token with its full-vocab bf16 lm_head (the target path is not touched). The subset is data-free:
              token ids [0, 151643) by tokenizer id order plus the 33 added/special tokens. No benchmark, ShareGPT or
              GSM8K data was used to choose it (manager's ruling, option (a)). The coverage check used vLLM's own
              docs/code only, and only to compare K values; the chosen K is the tokenizer's pre-extension Qwen base
              vocab boundary. No precision or communication change for the target. Memory down 0.23 GiB/rank.
              128K request passes.

official #1 (2026-11-15T00:34): 931.8 tok/s at c=16 (mean of 2 trials: 930.7 / 932.8, CV 0.2%) (+46.5% vs BASE), GSM8K 93.2%, MTP acceptance 2.329, TTFT p50 160 ms, TPOT p50 15.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
