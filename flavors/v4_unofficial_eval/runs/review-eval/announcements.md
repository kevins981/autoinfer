## 2026-11-15T23:55:50+00:00 (manager)
team T1 (sweep) opened with 45 GPU-min

## 2026-11-15T23:55:50+00:00 (manager)
team T2 (sweep) opened with 45 GPU-min

## 2026-11-16T00:00:53+00:00 (manager)
Human ruling (task_definition.md, task/TASK.md): promotion rule. A record's score is the mean of all its harness runs; it beats the champion only if that mean is higher AND every start is above the champion's mean. A gain under 1% needs a second harness run, with all 4 starts above the champion's mean. Continue.

## 2026-11-16T00:26:24+00:00 (manager)
Promotion: T1-001 (T1) is champion at 678.4 tok/s (+6.7% vs BASE 635.9): flag only, FP8 draft lm_head for the MTP drafter (ruling 1). Record: bank/T1-001. T2-001 (689.5) is pending its GSM8K check and will be judged against T1-001.

## 2026-11-16T00:28:46+00:00 (manager)
Tool fix (human): task/harness/summarize.py now runs the latency gate (p99 vs BASE) by default; results.json written before lacked gate_latency. Re-run summarize.py on your output folders. Manager check of T1-001 and T2-001 eval/1: gate_latency pass.

## 2026-11-16T00:39:15+00:00 (manager)
Promotion: T2-001 (T2) is champion at 689.5 tok/s (+1.6% vs T1-001, +8.4% vs BASE): flag only, CUDA graphs for prefill/mixed steps up to 2048 tokens (max_prefill_cudagraph_capture_tokens). GSM8K 93.2%. Record: bank/T2-001. T1-001's FP8 draft lm_head flag is not in it.

## 2026-11-16T00:50:23+00:00 (manager)
team T2 (sweep) received 45 GPU-min; resync report: reports/T2-1.md

## 2026-11-16T00:53:36+00:00 (manager)
Promotion: T2-003 (T2) is champion at 707.0 tok/s (+2.55% vs T2-001, +11.2% vs BASE): flags only, T2-001's prefill graphs + --enable-batch-sharded-sampling + MTP use_local_argmax_reduction. Record: bank/T2-003.

## 2026-11-16T01:05:34+00:00 (manager)
Promotion: T2-004 (T2) is champion at 743.9 tok/s (+5.22% vs T2-003, +17.0% vs BASE): T2-003's flags + Triton block-FP8 W8A8 (the checkpoint's own activation scheme) with 5 tuned tile configs. GSM8K 92.8%. Record: bank/T2-004.

## 2026-11-16T01:24:24+00:00 (manager)
team T1 (sweep) received 45 GPU-min; resync report: reports/T1-1.md

## 2026-11-16T02:01:28+00:00 (manager)
Promotion: T1-007 (T1) is champion at 771.3 tok/s (+3.7% vs T2-004, +21.3% vs BASE): T2-004 + a custom exact TP=2 all-reduce through host-mapped pinned memory for decode-size all-reduces. Record: bank/T1-007. Pending candidates are now judged against 771.3.

## 2026-11-16T02:45:43+00:00 (manager)
Funding policy change (human, funding_policy.md): a team gets at most 3 grants (opening + 2 extensions, last chances included). At the resync after its third grant the team is closed and replaced, even after a win. T1 and T2 have had 2 grants each: the next extension is each team's last. Continue.

## 2026-11-16T02:52:27+00:00 (manager)
Funding policy change (human, funding_policy.md): at most 2 grants per team (opening + 1 extension). T1 and T2 have had both: at each team's next resync it is closed and replaced (write-up and last map update first), even after a win. Plan your remaining balance accordingly. Continue.

## 2026-11-16T02:55:32+00:00 (manager)
team T1 (sweep) closed (2-grant cap; 2 wins: T1-001, T1-007 = champion). Write-up: reports/T1-writeup.md. Team T3 (sweep) opened with 45 GPU-min.

## 2026-11-16T03:23:12+00:00 (manager)
Promotion: T2-007 (T2) is champion at 787.6 tok/s (+2.1% vs T1-007, +23.9% vs BASE): T1-007 + T2-006's fused W8A8 activation quantization and GEMM configs. GSM8K 93.6%. Record: bank/T2-007.

## 2026-11-16T03:29:41+00:00 (manager)
Promotion: T3-001 (T3) is champion at 816.6 tok/s (+3.7% vs T2-007, +28.4% vs BASE): T2-007 + a side-stream kernel that prefetches the next GEMM's FP8 weights into L2 during each decode-size hmar all-reduce. Record: bank/T3-001.

## 2026-11-16T03:44:51+00:00 (manager)
team T3 (sweep) received 45 GPU-min (its last grant under the 2-grant cap); resync report: reports/T3-1.md

## 2026-11-16T04:10:58+00:00 (manager)
team T2 (sweep) closed (2-grant cap; 4 wins: T2-001, T2-003, T2-004, T2-007). Write-up: reports/T2-writeup.md. Team T4 (sweep) opened with 45 GPU-min.

## 2026-11-16T04:14:55+00:00 (manager)
Run schedule (human): the run ends at 08:15 UTC today. From 07:45 UTC no new teams and no new grants; file what you have measured (records, map updates, write-ups) before 08:15. Plan GPU work accordingly. Continue.

## 2026-11-16T04:23:04+00:00 (manager)
Promotion: T3-003 (T3) is champion at 826.1 tok/s (mean of 3 starts, +1.16% vs T3-001, +29.9% vs BASE): T3-001 + FP8 draft lm_head (ruling 1) with the KV budget trimmed to 4.4 GiB to fit it. Record: bank/T3-003.

## 2026-11-16T04:57:51+00:00 (manager)
Promotion: T3-007 (T3) is champion at 853.6 tok/s (+3.3% vs T3-003, +34.2% vs BASE): T3-003 + Triton attention backend with bf16 query/probabilities (code default; FP8 Q/P stays off). GSM8K 93.6%. Record: bank/T3-007. Note: TRITON_ATTN as a bare flag (FP8 Q/P with FP8 KV) is invalid (T3-006).

## 2026-11-16T04:58:22+00:00 (manager)
team T3 (sweep) closed (2-grant cap; 3 wins: T3-001, T3-003, T3-007 = champion). Write-up: reports/T3-writeup.md. Team T5 (sweep) opened with 45 GPU-min.

## 2026-11-16T05:48:27+00:00 (manager)
team T4 (sweep) closed (no win; recommended by its leader after a closing review). Write-up: reports/T4-writeup.md. Team T6 (sweep) opened with 45 GPU-min.

## 2026-11-16T06:10:27+00:00 (manager)
team T5 (sweep) received 45 GPU-min (last chance; its last grant); resync report: reports/T5-1.md

## 2026-11-16T06:40:08+00:00 (manager)
Promotion: T5-001 (T5) is champion at 895.4 tok/s (+4.9% vs T3-007, +40.8% vs BASE): T3-007 + GDN DeltaLog (the GDN MTP decode kernel writes the committed state plus a small log instead of 3 full fp32 states; bitwise exact). GSM8K 93.2%. Record: bank/T5-001. Pending candidates are judged against 895.4.

## 2026-11-16T06:59:14+00:00 (manager)
team T5 (sweep) closed (2-grant cap; win T5-001 = champion). Write-up: reports/T5-writeup.md. Team T7 (sweep) opened with 45 GPU-min.

## 2026-11-16T07:04:31+00:00 (manager)
team T6 (sweep) closed (no win; recommended by its leader after closing review T6-3). Write-up: reports/T6-writeup.md. Team T8 (sweep) opened with 45 GPU-min.

## 2026-11-16T07:08:07+00:00 (manager)
team T7 (sweep) closed (no win; recommended by its leader). Write-up: reports/T7-writeup.md. Team T9 (sweep) opened with 45 GPU-min.

## 2026-11-16T07:15:08+00:00 (manager)
team T8 (sweep) closed (no win). Write-up: reports/T8-writeup.md; its mixed-step profile is bank/T8-002. No replacement team (under 1 h left); T9 remains open.

## 2026-11-16T07:15:33+00:00 (manager)
The review tool's backend is at its usage limit until 08:17 UTC (after the run ends): no reviews for the rest of the run. Nothing else changes; a candidate's record must carry the leader's own validity review. Continue.

## 2026-11-16T07:26:36+00:00 (manager)
Promotion: T9-001 (T9) is champion at 908.0 tok/s (mean of 3 starts, +1.41% vs T5-001, +42.8% vs BASE): T5-001 + a reduced-vocab (2^17 ids by tokenizer order) FP8 drafter head; verification stays full-vocab and exact. Record: bank/T9-001.

## 2026-11-16T07:27:37+00:00 (manager)
Run finalized: champion T9-001 at 908.0 tok/s (+42.8% vs BASE). Final report: reports/final.md.

