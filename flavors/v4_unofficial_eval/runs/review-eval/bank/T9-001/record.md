team:         T9
builds_on:    T5-001
performance:  +1.41% vs the champion (T5-001, 895.4): 908.0 tok/s = mean of 3 starts 912.7 / 902.5 (eval/1, GPU job 126) and 908.8
              (eval/2, TRIALS=1, GPU job 127); all 3 starts above 895.4. 128K pass and gate_latency pass in both evals.
summary:
  hypothesis: the drafter's fp8 lm_head costs 876 us per draft step (2 per decode step; job 118 node profile in bank/T5-001), ~6% of
              the 29 ms decode step. The drafter only needs a greedy argmax. Restricting it to a generic token subset of K=2^17 ids,
              re-sharded evenly (65,536 rows/rank instead of 124,160), should cut the head to ~470 us: -0.8 ms/step, ~+1.9% overall,
              minus the acceptance loss (each -1% of acceptance length ~ -1% throughput). T1-002 (bf16, replicated, K=32K, on BASE) lost
              4.6% acceptance; K=2^17 keeps all Latin and CJK merges (reports/T9-sweep/6-ranking.md #1).
  change:     by leader-9 (workspace leader-9.k3; change.diff = ws diff leader-9.k3 = T5-001's diff + 2 files, md5 aec9363e).
              eagle/utils.py: build_reduced_draft_head: subset = token ids [0, K-33) (tokenizer id order = BPE merge rank) + the
              tokenizer's 33 added/special ids (<|im_end|>, think tags, ...); rows copied from the target bf16 lm_head by a zero-padded
              TP all-reduce (x + 0 = x, exact), rank r keeps subset rows [r*K/2, (r+1)*K/2), then quantized with the same online fp8
              method as the existing draft copy (per-row weight scales, per-token activation scales). ReducedDraftHead: local
              logits -> local max -> id_map -> all-gather (value, id) pairs -> argmax (same reduction as LogitsProcessor.get_top_tokens).
              speculator.py: get_draft_top_tokens uses it when present (greedy local-argmax path only; non-greedy drafting keeps the
              full fp8 head). Active by default when draft_lm_head_quantization=fp8 (VLLM_T9_DRAFT_VOCAB=0 disables, generic default,
              ruling 3). Serve args identical to T5-001's.
  results:    Observed: eval/1 (job 126): 907.6 tok/s (912.7, 902.5); acceptance length 2.325 / 2.345 (T5-001: 2.35 / 2.336; mean
              2.335 vs 2.343, -0.3%); per position start 1 77.2/55.4%, start 2 78.1/56.4% (T5-001 start 1 78.2/56.7%); TPOT p50
              15.88 ms start 1 (T5-001 16.22); TTFT p99 1029/1006 ms, TPOT p99 30.5/33.7 ms (gate_latency pass); 128K pass (19 tokens,
              82.9 s, server alive after). server.log: "T9 reduced-vocab draft head: K=131072 (ids [0,131039) + 33 added ids)" on both
              ranks. eval/2 (job 127): 908.8 tok/s, acceptance 2.332, TTFT p99 1035 ms, TPOT p99 32.7 ms, 128K pass. No microbenchmark/profile of the head itself (time: the run ends 08:15).
  conclusion: Interpretation: the head is in use (acceptance moved slightly, as expected when drafts outside the subset are lost)
              and the step got cheaper (TPOT p50 -0.34 ms per token, consistent with ~-0.8 ms per decode step at ~2.3 tokens/step).
              Gain +1.36% vs the +1.9% gross napkin minus ~0.3-0.9% acceptance loss: consistent. Mechanism not isolated by a profile.
validity:     leader-9 (the review tool's backend was unavailable for the rest of the run, manager notice): draft side only. Target
              model, target lm_head (bf16), verification (full vocab, exact rejection) and sampling unchanged, so the served output is
              what the target produces (exact speculative decoding). Ruling 2 (reduced-vocab draft lm_head): token ids chosen by the
              published tokenizer's id order + its special tokens, no benchmark or ShareGPT data, K a power of two fixed before any
              measurement (no tuning: one K measured). Ruling 1 (fp8 draft head copy): same online fp8 scheme, only inside the drafter.
              Rows copied exactly (zero-padded all-reduce of bf16). Memory: +0.34 GB/GPU (fp8 65,536x5,120) on top of the champion's
              config; 128K request passed, no OOM. GSM8K not run: output exact by construction (draft-only change). Ties in the draft
              argmax may resolve differently from the full head (draft-only, no output effect). Limitation: tp_size must divide K/64
              (asserted); non-greedy drafting falls back to the full head.
