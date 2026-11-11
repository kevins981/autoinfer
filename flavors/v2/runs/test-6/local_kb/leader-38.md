# LOCAL_KB leader-38 (T38 moonshot)
## Status
- 2026-11-10 start. Seed 30 GPU-min. Champion T37-002 1142.2 tok/s. Workspace t38-champ (from T37-002).
## Measurements (own)
- job 382 (nsys node trace, champion T37-002, 80 prompts; agents/leader-38/prof_champ_node, scripts ../anat.py ../gpusteps.py ../gpusteps2.py):
  - decode step (16 req x4 rows=64): ~24-26 ms, GPU 100% busy. hostar_kernel (host-staged AR + L2 prefetch) 7.0 ms (141 calls ~50us),
    t31b 6.9 + Marlin 6.8 (weight GEMMs at ~DRAM roofline ~930 GB/s), GDN replay 1.8, target lm_head 1.0, drafter ~3x0.68 ms.
  - mixed steps (GPU-defined lm_head->lm_head): ~60 prefill tok: 42.5 ms (vs decode 23.8); ~330 tok: 96-110 ms.
    bs_gemm (W8A8 Marlin-layout prefill GEMM, fp32-acc mma) = 20.5 ms (M~120) and 61 ms (M~390) => ~155 TFLOPS.
  - mixed steps = ~29% of GPU time in the profile (6.2/21.7 s); bs_gemm ~16.6% of time per GPU.
- GEMM FLOPs: 24.3 GFLOP/token/GPU (48 GDN layers 383M params, 16 attn 372M).
- job 383 mma peak (both GPUs, ~2.52 GHz): e4m3 f32acc 330 TFLOPS, e4m3 f16acc 659, bf16 f32acc 165.
## Direction candidate A: prefill GEMM at tensor-core rate (f16 in-block accum = 2x; better tiles)
- Ruling asked 2026-11-10 (f16 accumulator within 128-K block, exact overflow guard). Manager passed to human; don't build final kernel until answered.
## Cards: per-item overhead (small prefills cost 18 ms for 60 tokens), unused structure (fp16 acc mode unused), boundaries (AR via host 27% of decode).
- job 388 (ovl/ovl_test.py): AR(64 rows, no pf) 61us || 89MB GEMM 102us => concurrent 110.5 vs serial 167. 128 rows 144 vs 218.
- spar T38-1: beliefs table (TP>PP assumed; TBO assumed; spec-tree ~break-even untested; prefill batching ~wash). Key caveat: stock Marlin K-striping breaks ordered chunk consumption -> use all-K-per-CTA GEMM.
## Proposal T38-1 (2026-11-10): stream the AR into the consumer GEMM (K-chunk flags, inv_rms in epilogue). proposals/T38-1.md
- 2026-11-10: T38-1 CLOSED before building. Manager pointers: T4-001 measured ceiling of AR-beside-consumer = 5-8 us/AR (AR slows 2x under HBM traffic). Also: T36-002 AR at wire time (host-mapped ld/st ~21-22 GB/s total per GPU), T13-001 producer-fused AR fails, T16/T20/T34 transports no gain, T22-001 L2 warm ceiling ~3.5%, T31-001/002 decode GEMM kernels near floor, T35-003 concurrent streamer fails (GEMMs already DRAM-bound in chain). => COMM + DECODE GEMMs are closed areas. Record bank/T38-001.
- Remaining multipliers: acceptance (2.67 @k=3; cond. 0.756/0.70/0.71), prefill/mixed steps (29%).
- w1 note: champion decode post-attn norm = flashinfer gemma_fused_add_rmsnorm (g = 1+w); check whether sumsq uses fp32 h or bf16 h before any inv_rms fusion.
- 2026-11-10: T38-2 (self-distilled/recursive MTP) CLOSED before GPU: T3-001..004 + reports/T3-writeup.md already measured it (offline proxy matches in-server; hard-label, LoRA/step-2, KL fine-tunes all flat on held-out; +0.35 pt a2 per data doubling). T3 also: confidence-budgeted verification ~+2-3% (memory blockers), draft step 0.84 ms vs ~0.5 ms bytes.
- T37-001: step ~ 16.9 ms + 0.10 ms/row + 0.83 ms/draft step; prefill extra = 15 ms/event + 166 us/prompt token (24% of wall); 60-token prefill step: ~1130 eager kernels, GPU idle ~24/62 ms (graph trace), but T32-003: removing host launches alone neutral. FP8 weight entropy ~6.5 bits/byte.
- My diff (diffsteps.py, job 382): 60-tok mixed vs decode: bs_gemm M~120 20.5 ms vs t31b+Marlin 13.7 (+6.8), AR +5.6 (0.1 ms/row), GDN chunk kernels +1.8, misc ~2.5.
- Remaining items I see are each ~2-4% (mid-M prefill GEMM at DRAM roofline, GDN short-prefill path, cheaper draft steps, FP8 weight codec ~11% bytes w/ fixed-rate code). No >5% structural bet identified beyond the prefill fixed cost (~7.5% ceiling).

## 2026-11-10: tree-acceptance measurement dispatched
- Task to w1: agents/leader-38/task_w1_tree.md (6 GPU-min allowance). Offline exact chain/static/dynamic tree acceptance on T3 held-out data.
- Decision rule: dynamic N=19 well below ~3.9 tok/step -> close PP+trees with evidence; >=4.1 -> report live expensive bet (PP engine rewrite).
- T38-002 filed: dynamic trees N=11 4.21, N=19 4.74 tok/step (chain k=3 2.73). Acceptance is not the limit.
- PP+tree cost model (spar T38-2 turns 7-9): per microbatch (8 seqs) 64 layers ~30.7 ms (26.2 GEMM at floor + 4.5 non-GEMM incl. GDN state 1.2GB r+w/GPU/round),
  head 1.4, drafting D=6 x 0.95 = 5.7 -> period 37.8 ms (uneven split ~39/25 layers, head+drafter on GPU1, no bubble in steady state).
  Break-even T ~4.13 decode-only; +5% decode needs 4.33. Prefill (24% wall) gains from no AR: separate +.
  Soft spots: GEMM at floor up to 96-128 rows (w1 microbench B), level-wise drafting acceptance (w1 task A), non-GEMM kernel times.
  Tree GDN: ancestor-masked UT transform -> exact, no per-node states.
- T38-003 filed: level-wise trees (K=4) N=15 D=6 4.09; PP-stage GEMM 13.4 ms @M16 .. 16.3 @128 .. 25.9 @256 (job 390).
  => PP+trees decode ~-15% today, ~even if narrow-K GEMMs + cheap draft head recovered. Upside is PP prefill (no AR; AR 93/311 us per prompt token).
- Spar T38-2 ended (turn 12): r>=0.94 suffices if prefill s=0.4. Stage 1 must be a mixed-load PP vs TP (spec off) regression, not prefill-only.
- proposals/T38-3.md sent to manager 2026-11-11; waiting for pointers before stage 1 (~12 GPU-min). Balance 40.8.
- Stage 1 (job 392, T38-004): spec-off champion code. TP prefill 205 us/prompt token (wall); PP 105 us/token GPU work per GPU -> 0.51x: PASS
  if the pipeline is full. Stock PP: GPU1 88% busy (12% bubble), decode microbatch 15.2/17.0 ms per GPU. Non-GEMM ~4 ms/microbatch (assumed 3).
  Wall model (prefill share 0.20-0.24, s=0.49): +5% needs decode r >= 0.92-0.94; full-success ~+12%. Balance 32.2.
- Stage 2 (job 393, T38-005): r = 0.91-0.92 with measured parts (pass rule 0.93) -> FAIL. Hybrid PP-prefill/TP-decode: memory fits
  barely; time ~0 gain (fill/drain + own weight reads; batching costs slots in the closed loop). Spar T38-2 ended turn 16: close supported.
- CLOSED T38-3. Write-up: agents/leader-38/T38-writeup.md.
