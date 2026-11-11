# LOCAL_KB leader-3 (T3, explore)
## Direction: self-distilled MTP draft head (proposals/T3-1.md). Funded: seed 20 + start 80.
- Baseline facts: T1-042 per-position acceptance 0.77/0.55 cum (official logs), E=2.325.
- MTP: 1 layer, FP8 block-quantized (weight_scale_inv), fc bf16; vLLM indexes layers[step % num_mtp_layers] (qwen3_5_mtp.py:172) -> a 2nd layer for step 2 is natively supported.
- Output of MTP forward is post-norm (qwen3_5_mtp.py:189); step 2 re-normalizes it via pre_fc_norm_hidden.
- vocab 248320; champion draft lm_head = first 98304 ids, FP8.
## Workers
- leader-3.w1 (ws t3-distill from T1-042): task 1 = seed pipeline, 12 GPU-min allowance. agents/leader-3/w1_task1.md
## Open questions
- Manager ruling: self-generated data OK (task_definition.md end); document generator, topic list, counts, split. Also report in-server step time with acceptance.
- Job 2 (w1, 10 min): cold server start 409 s; phase A done (498 prompts: 400 self-instruct, 98 Magpie, low Magpie yield); phase B timed out, nothing saved. Approved +9 (allowance 21). Lesson: cold workspace startup ~7 min; write outputs incrementally; one server start per job.
- T3-001 filed: stock head offline held-out a1 0.781 / a2cum 0.561 (on 0.761/0.533, off 0.803/0.591); matches in-server 0.77/0.55. Data: 400 prompts, 144K resp tokens, split 360/40. Team used 16.4 min, balance 83.6.
- Note: server uses new runner vllm/v1/worker/gpu/model_runner.py (hooks must go there).
- w1 task 2: pilot fine-tune, 2 lrs on 2 GPUs, 8 min allowance (agents/leader-3/w1_task2.md).
- T3-002: pilot full FT hurt held-out (lr1e-5 .778/.549, 5e-5 .768/.534 vs stock .782/.562). Task 3: dedicated step-2 layer / LoRA / lr2e-6, masked step-2 loss, bootstrap CI.
- T3-003: restricted FT (A/LoRA/lr2e-6, masked step-2) = stock within +-0.25pt CI. Task 4: KL soft labels from lm_head(stored h) + data slope + train-fit. DECISION POINT after task 4: if no gain or flat slope -> kill/pivot.
- T3-004: KL soft labels: held-out flat (+0.13pt a2 CI incl 0), train a2 +12pt (memorization). Slope ~+0.35pt/doubling. KILLED self-distill direction. Balance 75.8.
## Pivot (2026-11-09): confidence-budgeted (adaptive) verification for MTP
- Upstream infra: vllm/v1/worker/gpu/spec_decode/adaptive_verification.py (DSpark only; budget = argmax E/cost(M) over survival-ranked slots; cost tables from measured step timings). Config enable_adaptive_verification; incompatible with use_local_argmax_reduction (needs draft probs) -> we'd compute sharded max-prob (local max, sumexp, all-gather).
- gdn_attn.py:419 has adaptive (non-uniform spec len) handling. mamba abstract.py:94: num_speculative_blocks = 0 if cache_config.use_replayssm else k (memory issue KB13 for k>2 unless replayssm/T1 replay frees it).
- Cost model: Marlin cliff M>64 (KB14), ~0.28 ms/token, drafter ~0.55 ms/step.
- w1 task 5: offline sim (agents/leader-3/w1_task5.md). Manager informed of kill + pivot.
- T3-005: adaptive verification sim: K_MAX=4,B=32 (M=48) E/t 1.067; stale ranking ~1.01. Proposal T3-2 written (payoff ~+6%, central +2.5%, 30%). Risk: GDN spec blocks scale with k (KB13).
- Manager accepted T3-2 (no new grant). Validity: B must be generic -> plan B = 2*num_reqs (same verify width as k=2 at any load), capped by graph sizes; c=1/c=32/128K must work; include bitwise exactness check vs k=2 greedy in record. Bring drafter ms + ABBA to resync.
- Task 6 (held-out): K_MAX=4 B=32 raw 1.0635, iso-calibrated 1.0688; rule(d) stale-B iso 1.0717. Slots per req ragged 0..4 (raw: 22% get 0, 28% get 4).
- w1 task 7: GDN spec-slot code read + in-server drafter/verify timing (6 GPU-min).
- T3-007 (job 19): drafter 0.84 ms/step (model 0.55); verify cliff just above M=48 (48:25.9, 52:29.2); sample/reject 1.09; c=1 forward 17.7. GDN spec slots used at 1600-token boundaries (not dead); K_MAX=4 needs kernel changes. Sim w/ measured costs: best ~1.047-1.049 decode -> ~+3-4% e2e. T3-2 BELOW BAR. Balance 72.2.
## Resync #1 (2026-11-09): LAST CHANCE +20 (balance 92.2). Next last chance closes team.
- Chose: build T3-2 as exact win (K_MAX=3, B=2*num_reqs, on-GPU current ranking). Port upstream DSpark adaptive path. Task 8 phase 1 = CPU plan (agents/leader-3.w1/PLAN_T3-2.md). Kill: ragged path infeasible w/ T1 replay/graphs, or ABBA < +1.5%.
- Note: at k=3 the M=56 decode is FULL graph too; the cliff 48->52 is real compute (not graph mode). Unknown cause (Marlin 4 m-blocks?).
- Manager review reqs for T3-2 champion: (1) memory change -> show no OOM under bench, 16-req admission, 128K pass; (2) exactness also at batch>1 (ragged batches) vs k=2 greedy (note: batched text can diverge from fp reordering; use VLLM_TEST_GREEDY_IDS_CHECK-style verifier check + match rate + GSM8K); (3) repeated same-job ABBA + official per-trial numbers.
- Task 8 plan approved (agents/leader-3.w1/PLAN_T3-2.md): steps 0-5; step 0 = memory proof (KV ~5.5GiB, k=3, c=16 Running 16 + 128K). Allowance steps 0-4: 15 min. Worker estimates 3-4 days code; marginal ~+3%.
- KB61 hit us: t3-distill.dump never built -> T3-007 timings (job 19) invalid (baseline _C: no GDN replay, no Marlin tiling). Records marked. Worker told to build + remeasure in step 0 (8 min).
- Manager: if re-measured (built) costs put T3-2 payoff >= 5%, resync #1 becomes an extension (not last chance). Else decision stands.
- Step 0 (built): fwd 22.4@M48, sample 1.12, propose 1.69 (0.84/step). k=3 16 reqs needs >=5.62 GiB; OOM in graph capture at 5.65-5.75 -> no window. T3-2 ~+3.3% e2e, below bar. Proposed to manager: stop T3-2, close T3 voluntarily (balance 80.5). Worker filing T3-008.
- CLOSING: T3-008 filed; reports/T3-writeup.md written. Final: T3-2 sim built 1.046 raw / 1.049 iso decode.
