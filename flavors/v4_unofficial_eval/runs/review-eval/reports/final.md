# Final report: run review-eval (qwen38-27b-fp8-2x4090)

Finalized 2026-11-16 ~07:30 UTC by the manager: all teams had closed (T9 last, on its recommendation) and none
could fund a build-and-measure cycle before the human's 08:15 end; no GPU job running.

## Champion
**T9-001: 908.0 tok/s** at c=16 (mean of 3 starts 912.7 / 902.5 / 908.8), **+42.8% vs BASE (635.9)**.
- Record: bank/T9-001/ (change.diff md5 aec9363e; serve args bank/T9-001/serve_args.txt).
- Checks: latency gate pass on every start, 128K request pass, acceptance 2.33. GSM8K not needed (draft-only);
  every stack since the last math-affecting change (W8A8 fused quant, T2-007) measured 93.2-93.6% (gate 89.6%).
- Serve flags on top of BASE: max_prefill_cudagraph_capture_tokens=2048, --enable-batch-sharded-sampling,
  MTP use_local_argmax_reduction + draft_lm_head_quantization=fp8, --kernel-config Triton block-FP8 W8A8,
  --kv-cache-memory 4724464025 (4.4 GiB), --attention-backend TRITON_ATTN.

### Promotion chain (11 promotions, none reverted)
| # | Record | Team | tok/s | vs prev | What |
|---|---|---|---|---|---|
| 0 | BASE | - | 635.9 | | baseline |
| 1 | T1-001 | T1 | 678.4 | +6.7% | FP8 draft lm_head (ruling 1) |
| 2 | T2-001 | T2 | 689.5 | +1.6% | CUDA graphs for mixed steps up to 2048 tokens |
| 3 | T2-003 | T2 | 707.0 | +2.55% | batch-sharded sampling + local-argmax drafter (no full-logits all-gather) |
| 4 | T2-004 | T2 | 743.9 | +5.2% | Triton block-FP8 W8A8 (checkpoint's dyn per-group-128 scheme), configs tuned on random data |
| 5 | T1-007 | T1 | 771.3 | +3.7% | exact TP=2 all-reduce through host-mapped pinned memory (HMAR), decode sizes |
| 6 | T2-007 | T2 | 787.6 | +2.1% | T1-007 + fused norm/SiLU + dyn128 activation quant, configs for all graph sizes |
| 7 | T3-001 | T3 | 816.6 | +3.7% | L2 prefetch of the next GEMM's weights during each HMAR all-reduce |
| 8 | T3-003 | T3 | 826.1 | +1.16% | FP8 draft head fitted by trimming KV to 4.4 GiB (3 starts) |
| 9 | T3-007 | T3 | 853.6 | +3.3% | Triton attention with bf16 Q/P (code default; the flag alone = FP8 Q/P, invalid) |
| 10 | T5-001 | T5 | 895.4 | +4.9% | GDN DeltaLog: committed state + log instead of 3 fp32 states per MTP step (bitwise exact) |
| 11 | T9-001 | T9 | 908.0 | +1.41% | reduced-vocab (2^17 ids by tokenizer order) FP8 drafter head (3 starts) |

Full history with evidence: champion.md.

## Teams (9 sweep teams; pool 1000 GPU-min: 366.2 used, 633.8 unused)
| Team | GPU used / granted | Wins | Tested, not won | Left open (last map update) |
|---|---|---|---|---|
| T1 | 69.3 / 90 | T1-001, T1-007 | reduced-vocab head, k=3 (confounded), NCCL tuning (nothing), FlashInfer IPC AR (no P2P), overlap variants | fewer/cheaper decode all-reduces (132/step, ~5 ms of ~30) |
| T2 | 73.0 / 90 | T2-001, -003, -004, -007 | T2-008 null (fusion bypassed by the CUDA GDN spec-decode op) | fuse out_proj quant into that op's epilogue (~+0.5%) |
| T3 | 83.6 / 90 | T3-001, -003, -007 | 002/004 within noise, 005 screen negative, 006 invalid | per-target prefetch; prefetch under mixed-step ARs |
| T4 | 30.8 / 45 | - | recurrent GDN prefill (x2), fused HMAR+norm+quant, finer graph grid (OOM) | coarser graph grid with a memory test |
| T5 | 53.6 / 90 | T5-001 | L2 prefetch 24 MiB (weak negative) | DeltaLog for mixed-step spec rows; log outside spec slots (~2.4 GB/GPU, unblocks k=3) |
| T6 | 39.2 / 45 | - | HMAR to 768 rows (1.7x microbench, no harness number); exact row-split GEMM->AR overlap (+0.3%, a start below) | - |
| T7 | 3.2 / 45 | - | k=3 on T5-001 (-0.7%, 1 start: acceptance +15%, ITL +14%) | spec-state slots under DeltaLog |
| T8 | 2.0 / 45 | - | fp16-accumulate W8A8 (not allowed); mixed-step profile T8-002 | fixed host/sync cost per mixed step |
| T9 | 11.4 / 45 | T9-001 | - | GDN prefill host-launch-bound (~1,700 eager launches per mixed step) |

Write-ups: reports/T<n>-writeup.md; maps: reports/T<n>-sweep.md + reports/T<n>-sweep/; resyncs:
reports/T<n>-<k>.md; track records: teams/.

## Top open lever for a next run
Mixed (prefill + decode) steps: on the champion's code T8-002 shows a ~110 ms period floor with the GPU busy
~50 ms, and T9 counts ~1,700 eager launches per mixed step in the GDN prefill path (napkin +15-25% if it holds
outside the profiler). Caveat: nsys inflated mixed-step host cost before (T4); confirm with
--enable-logging-iteration-details.

## Rulings made during the run (task_definition.md, journal.md)
Human: promotion rule (mean of all starts, every start above, <1% needs a third start); phase-2 map reading
allowed; grant cap 2 per team; whole-map review after 6-ranking. Manager: SiLU-producer scale floor allowed
(disclosed); filed code must equal measured code; prefetch sizes chosen on generic evidence, never by comparing
on the benchmark; fp16-accumulate FP8 MMA not allowed (human may overrule).

## Incidents
- T1 messaged T2 its candidates during T2's phase 1; it arrived after T2's file 4 was written (mtime verified): no effect.
- T2's worker twice answered in plain text instead of SendMessage, stalling T2 (~1 h, then 19 min); fixed after
  the human / tick flagged it.
- This run's HF snapshot (c3a462c) differs from BASE's (017b9c7) only in chat template / tokenizer config; same
  benchmark input tokens (KB1).
- Review tool unavailable from ~07:04 (backend usage limit until 08:17): T7-T9 maps and T9-001 had no outside
  review; T9-001 was judged on raw files and the leader's own validity review.
- task/SPEED_OF_LIGHT.md's 165 TFLOP/s for FP8 GEMM is below the 240-275 TFLOP/s T8 measured (job 124).
