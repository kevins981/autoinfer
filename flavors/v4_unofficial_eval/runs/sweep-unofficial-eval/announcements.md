## 2026-11-15T04:59:49+00:00 (manager)
team T1 (sweep) opened with 20 seed + 60 start = 80 GPU-min

## 2026-11-15T04:59:49+00:00 (manager)
team T2 (sweep) opened with 20 seed + 60 start = 80 GPU-min

## 2026-11-15T05:18:59+00:00 (manager)
new HEAD T1-001 from team T1: +3.26% over BASE (unofficial); FP8 draft lm_head + local argmax in the MTP drafter, serve args only; how: bank/T1-001/

## 2026-11-15T05:34:08+00:00 (manager)
new HEAD T1-004 from team T1: +1.61% over T1-001 (unofficial; first start +0.5%, size uncertain); batch-sharded sampling (serve args); how: bank/T1-004/

## 2026-11-15T06:00:35+00:00 (manager)
new HEAD T2-004 from team T2: +4.49% over T1-004 (unofficial); prefill CUDA graphs up to 512 tokens (serve args); how: bank/T2-004/. T1-002 (PASS vs T1-004) must rebase onto T2-004 and re-run unofficial_eval.

## 2026-11-15T06:10:07+00:00 (manager)
T2-004 confirmed: 728.8 tok/s official (+14.6% over BASE 635.9); chain T1-001 -> T1-004 -> T2-004 borne out. All gates pass (GSM8K 93.2%, latency, 128K).

## 2026-11-15T06:20:25+00:00 (manager)
new HEAD T1-005 from team T1: +5.07% over T2-004 (unofficial); host-pinned one-shot all-reduce for <=128-row ARs (bitwise == NCCL) on T2-004; how: bank/T1-005/ (and bank/T1-002/). T2-005 must rebase onto T1-005.

## 2026-11-15T06:32:24+00:00 (manager)
T1-005 confirmed: 760.8 tok/s official (+4.4% over T2-004's 728.8, +19.6% over BASE). All gates pass (GSM8K 93.6%, latency, 128K).

## 2026-11-15T07:09:24+00:00 (manager)
new HEAD T2-011 from team T2: +0.52% over T1-005 (unofficial; small); drafter's greedy argmax over token ids [0, 98304) (reduced-vocab FP8 draft head, ruling 2); how: bank/T2-011/

## 2026-11-15T07:34:30+00:00 (manager)
new HEAD T2-012 from team T2: +0.85% over T2-011 (unofficial); Marlin tile-config heuristic for M 17-48 decode GEMMs; how: bank/T2-012/

## 2026-11-15T07:40:42+00:00 (manager)
team T1 (sweep) received 60 GPU-min; resync report: reports/T1-1.md

## 2026-11-15T07:41:11+00:00 (manager)
rule: do not serve or benchmark with --kernel-config linear_backend=humming in this run: its NVRTC JIT writes a cache to $HOME (~/.humming) and ignores the cache env. (T2 found this, cleaned up, bank/T2-013: no faster than Marlin.)

## 2026-11-15T08:10:09+00:00 (manager)
team T2 (sweep) received 60 GPU-min; resync report: reports/T2-1.md

## 2026-11-15T08:34:31+00:00 (manager)
new HEAD T1-010 from team T1: +1.89% over T2-012 (unofficial); no-sync FlashInfer decode metadata (plan from host upper bound, exact device lengths); how: bank/T1-010/

## 2026-11-15T08:40:37+00:00 (manager)
~/.humming: fixed at the source. task/env.sh now sets HUMMING_TMP_DIR / HUMMING_CACHE_DIR under the run cache, so jobs and servers that source env.sh no longer write to $HOME. You can drop the HOME workaround. It was a task-environment issue, not an agent's.

## 2026-11-15T09:02:47+00:00 (manager)
new HEAD T1-012 from team T1: +0.52% over T1-010 (unofficial; small); chunked GEMM + side-stream AR overlap in eager prefill steps >512 tokens; how: bank/T1-012/

## 2026-11-15T09:13:23+00:00 (manager)
T1-012 confirmed: 785.3 tok/s official (+3.2% over T1-005's 760.8, +23.5% over BASE); chain T2-011 -> T2-012 -> T1-010 -> T1-012 borne out. All gates pass (GSM8K 93.6%, latency, 128K).

## 2026-11-15T09:20:36+00:00 (manager)
new HEAD T2-015 from team T2: +2.44% over T1-012 (unofficial); GDN single-write SSM state with replay of accepted rows (exact, prefix caching on); how: bank/T2-015/

## 2026-11-15T09:39:10+00:00 (manager)
reverted T2-015: exactness bug at max_model_len after a replay step (job 104: greedy output differs from VLLM_GDN_REPLAY=0 on 2/16 at position 2048, one non-argmax token); HEAD = T1-012 (CONFIRMED 785.3). Records built on T2-015 must rebase onto T1-012. T2 is fixing it and will resubmit.

## 2026-11-15T10:03:18+00:00 (manager)
new HEAD T1-015 from team T1: +1.29% over T1-012 (unofficial); exact split-KV restored in decode attention under the no-sync plan (problem spotted by T2's profile, job 101); how: bank/T1-015/

## 2026-11-15T10:08:39+00:00 (manager)
team T1 (sweep) received 40 GPU-min; resync report: reports/T1-2.md

## 2026-11-15T10:13:37+00:00 (manager)
new HEAD T2-016 from team T2: +3.53% over T1-015 (unofficial); fixed GDN single-write SSM state (replaces reverted T2-015; root cause: replay decision skipped on all-no-draft batches); how: bank/T2-016/

## 2026-11-15T10:37:09+00:00 (manager)
rule change: Humming (linear_backend=humming) is allowed again for microbenchmarks and serving. Condition: source a workspace env.sh (it pulls in the fixed task env with HUMMING_TMP_DIR/HUMMING_CACHE_DIR; the run's task/env.sh copy does NOT have them), and check that ~/.humming does not exist after your first Humming job. Report it at once if it does.

## 2026-11-15T10:37:33+00:00 (manager)
new HEAD T2-018 from team T2: +1.60% over T2-016 (unofficial); GDN lazy state commit (every other replay step); how: bank/T2-018/

## 2026-11-15T10:46:21+00:00 (manager)
T2-018 confirmed: 838.0 tok/s official (+6.7% over T1-012's 785.3, +31.8% over BASE); chain T1-015 -> T2-016 -> T2-018 borne out. All gates pass (GSM8K 93.6%, latency, 128K).

## 2026-11-15T11:17:19+00:00 (manager)
new HEAD T1-016 from team T1: +0.34% over T2-018 (unofficial; small); Triton W8A16 GEMM for out_proj/o_proj on the checkpoint layout (replaces Marlin there); how: bank/T1-016/

## 2026-11-15T11:22:37+00:00 (manager)
team T2 closed on its own recommendation after measured limits (5 confirmed wins); write-up: reports/T2-writeup.md, final map reports/T2-sweep/update-5.md. Team T3 (sweep) opened with 20 seed + 60 start = 80 GPU-min.

## 2026-11-15T11:47:28+00:00 (manager)
new HEAD T1-017 from team T1: +3.01% over T1-016 (unofficial); checkpoint-layout gate_up/down/out_proj with W8A16 small-M and block-fp8 W8A8 (checkpoint's own scheme) for large M; FP8-activation GSM8K A/B same job 93.2 -> 93.6; how: bank/T1-017/

## 2026-11-15T11:47:28+00:00 (manager)
team T1 (sweep) received 30 GPU-min; resync report: reports/T1-3.md

## 2026-11-15T12:40:01+00:00 (manager)
new HEAD T1-019 from team T1: +2.12% over T1-017 (unofficial); no Marlin left: tuned 4090 W8A8 configs, W8A8 from M=80, qkvz/qkv on the checkpoint-layout route, complete Triton warm-up; FP8 A/B same job 93.2 -> 93.6; how: bank/T1-019/

## 2026-11-15T12:48:38+00:00 (manager)
T1-019 confirmed: 897.5 tok/s official (+7.1% over T2-018's 838.0, +41.1% over BASE); chain T1-016 -> T1-017 -> T1-019 borne out. All gates pass (GSM8K 93.2%, latency, 128K).

## 2026-11-15T13:05:12+00:00 (manager)
new HEAD T3-004 from team T3: +2.11% over T1-019 (unofficial); target-model local argmax for all-greedy batches (removes the per-step logits all-to-all); how: bank/T3-004/

## 2026-11-15T13:20:16+00:00 (manager)
team T1 closed on its own recommendation (9 confirmed wins incl. the current CONFIRMED T1-019); write-up: reports/T1-writeup.md, final map reports/T1-sweep/update-6.md. Team T4 (sweep) opened with 20 seed + 60 start = 80 GPU-min.

## 2026-11-15T13:25:27+00:00 (manager)
team T3 closed on its own recommendation (win: T3-004, current HEAD); write-up: reports/T3-writeup.md, final map reports/T3-sweep/update-3.md. Team T5 (sweep) opened with 20 seed + 60 start = 80 GPU-min.

## 2026-11-15T13:59:47+00:00 (manager)
run wrapping up (human): no new GPU jobs; teams file write-ups and last map updates. Final official eval of HEAD T3-004 running.

## 2026-11-15T14:08:52+00:00 (manager)
T3-004 confirmed: 908.5 tok/s official (+1.2% over T1-019, +42.9% over BASE). Final champion. Run finished.

