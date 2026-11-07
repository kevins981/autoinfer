# leader-2 local KB
## R1 (head diet)
- draft_lm_head_quantization=fp8 is W8A8 (online/lm_head.py:131). Manager ruled ALLOWED (drafter only).
- Existing flag --enable-batch-sharded-sampling: target logits all-to-all, Qwen3.5 implements compute_logits_local. Testing as arm E.
- get_top_tokens (logits_processor.py:216) = vocab-parallel argmax for drafter; base for target fast path (b).
- Worker leader-2.w1 in workspaces/leader-2.headdiet; budget 30 GPU-min for worker, rest for official evals.
- Job1/2 results: C (LA+fp8 head) +2.4% (noise); depth 3 (T2-002) acc 2.757 (pos3 .413) but -3.9% tok/s and TTFT p99 2225 FAIL.
  E (C + batch-sharded sampling, T2-003) ~+3.8% vs BASE over 2 jobs; custom greedy target fastpath (T2-004) = E, no gain on top.
- p99 TPOT swings 42-57 ms between benches on the same server: gate risk is noise-driven; judge on official 400-prompt runs.
- Never let a worker edit workspace code during a GPU job (contaminated arm C in job 1).
- OFFICIAL T2-003: 653.9 tok/s (652.1/655.6), +2.8% vs BASE; GSM8K 93.2, acc 2.348 (.781/.567), TTFT p99 1341, TPOT p99 48.2; 128K pass.
  Unofficial same-job gain (+3.8%) overstated the official gain (+2.8%): short benches inflate relative gains a bit.
- Budget: used 29.6; manager moved the remaining 25.4 to T1.
## R2 plan (if T2-003 promoted): depth-3 verify-step cost
- Measured: depth 3 = +16.5% tokens/step, -3.9% tok/s => step cost ~+21% (est. +5-6%); TTFT p99 2225.
- First: nsys (task/scripts/profile_nsys.sh) of T2-003 with n=2 vs n=3 at c=16; compare per-step kernel breakdown
  (Marlin M=48 vs 64, GDN, attention, the draft pass, cudagraph padding / capture sizes, host gaps).
- Check the cudagraph capture sizes and the max_num_batched_tokens interaction with prefill chunks (TTFT p99).
