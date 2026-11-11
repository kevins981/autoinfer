# LOCAL_KB leader-29 (T29, explore, slot B)

## Cards (first draw)
Mismatch-with-machine + the new W8A8 ruling: the e4m3 MMA (2x bf16 rate, KB69) is the unused capability; prefill GEMMs are bf16-tensor-bound, decode sits on both HBM and bf16-tensor roofs (T10). Predictability card -> deeper spec once GEMM rows get cheap. Others (host CPU, NCCL variants): nothing new beyond T2-T27.

## Direction (draft)
FP8 W8A8 (per-token-group-128 acts) for the target's linears: stage A mixed/prefill steps (compute-bound, KB77 Marlin excess ~22 s/trial),
stage B decode M=48 (removes dequant per-fragment cost KB81, halves tensor roof) -> stage C k=3 cheaper (KB63/KB20).
Napkin: mixed Marlin at M~300 ~55 ms bf16-roof per forward/rank vs ~28 FP8 -> up to ~8-10 s/trial of ~94 s.
Asked leader-28 (T28) to split prefill vs decode FP8 to avoid duplicate proposals.

## Log
- ws t29-fp8 from T27-002 (built in bg). w1 task1: microbench Marlin vs Triton block W8A8 at M=48..1024 (agents/leader-29/w1_task1.md).
- Split agreed with leader-28: T29 owns W8A8 GEMMs at M>96 (mixed/prefill); T28 owns M<=96 + k=3 system work. Share fused quant kernel if built.
- w1 task1 (jobs 190/191, 1.7 GPU-min): Triton block W8A8 vs Marlin per forward GPU0 cold: ~0 at M=48-64, -11% M=96, -19..-26% 128-192, -30..-37% >=256. Quant 2.5-13 us/call. Rel err 1.5e-2 vs 2.3e-3. Kernel promotes per BLOCK_K=128 in fp32 (compliant; BLOCK_K must be 128).
- My calc on T19 trace (372 mixed steps, padded M p50 208): 7.4 s/trial linear-time saving -> ~+8.6% if in-situ = microbench.
- One weight layout only (no dual copy). Agreed with T28: standard [N,K] layout; T28's small-M kernel plugs into M<=96 branch of our op.
- Proposal T29-1 sent (payoff +12.5%, central +5%, 80 min). w1 task2: in-server all-M W8A8 env gate T29_W8A8 + smoke + ABBA (<=12 min).
- FUNDED: T29-1, 80 GPU-min start grant.
- w2 started: stage C faster block FP8 GEMM M 96-1024 (10 min allowance)
- T29-001 (jobs 192/195, 10.6 min): all-M Triton W8A8 in-server: +0.2% pooled (pairs -0.5/+0.95). Decode ITL +1.05 ms/step (microbench said +0.18); mixed p99 ITL -15%, TTFT p50 -11%. Implied mixed gain ~2.7 s vs 7.4 predicted (~37% transfer). gsm8k50 90%. Missing JIT variant at warmup.
  Obs->Interp: an all-or-nothing gate needs decode parity. Options: (i) fix decode overhead (nsys, w1 task3); (ii) Marlin-layout W8A8 (k-permutation within 32-slab applied to acts) so decode stays Marlin (w2 assessing); (iii) T28's small-M kernel.
- T28 suspects decode +1.05 = ~320 quant launches (microbench e2e-GEMM gap 0.92 ms at M=48). T29 owns fused quant (contract: per_token_group_quant output). Layout decision after nsys.
- w2 (job 196): Marlin-layout W8A8 FEASIBLE (2 k16 Marlin tiles = m16n8k32 e4m3 B fragment under k-perm pi within 32; perm acts at quant). Own CUDA kernel bs_gemm (standard layout) beats Triton +1..9% at M>=192, loses at M<=128; still 10-16% < cuBLASLt. Triton spends ~186 PRMT per 64 QMMA.
  DECISION: main line = Marlin layout; decode M<=96 stays Marlin bitwise; M>96 -> W8A8 kernel reading Marlin qweight + pi-permuted quant. Told T28.
- T28 builds 'Marlin-e4m3' (modified marlin_template, e4m3 MMA in Marlin pipeline) as small-M alternative; w2's bs_gemm is the other design. Shared interface = agents/leader-29.w2/gemm/PI_SPEC.md (pi, byte order, fp32 scales [N/128,K/128], perm acts).
- PI_SPEC (w2) independently confirmed by T28's worker from gptq_marlin_repack.cu (pi, byte order, n mapping). Real-op bitwise check pending in both teams' GPU jobs.
- w1 job 199 nsys (TRACE=node, 48 prompts): decode +0.93 = GEMM -0.6, quant +0.73 (264 launches), cuBLAS bf16 ba GEMMs +0.44 (unknown), AR +0.18. Mixed: GEMM kernel time -984 ms/48 prompts (~8 s/trial, matches 7.4 predicted) but periods ~unchanged under node trace (w1: launch-bound at M<=200 -- doubtful, KB46/KB61 say no GPU idle; node-trace inflation). My ITL arithmetic: mixed ~-10 ms/step -> ~4 s/trial (~55% transfer). Asked w1 to check per-token ITLs.
  Forecast for Marlin-layout (decode unchanged): ~+4% from A alone if 55% transfer.
- w1 Prometheus ITL histograms (jobs 192/195): mixed-like gaps -13% (mean 120->106 ms), ~-3.6 s/400-prompt trial = ~50% transfer of 7.4 predicted. Decode-like +1.8 s/run. T29-002 = T29-001 + warmup fix.
  Revised: A (Marlin-layout, decode unchanged) ~+3.5-4%; fused quant ~+0.6%; better GEMM ~+1-1.5%. Payoff now ~+6-7% -> near bar. Next lever to understand: where the other 50% goes (likely ARs exposed in token-split/overlap once GEMMs shrink; rank skew). Must measure on the Marlin-layout build.
- w1 fused quant (t29_act_quant.py: rmsnorm/silu_mul/plain -> fp8+scales, PERM opt) bitwise on CPU; GPU check (<=2 min) queued.
- w2 jobs 196/202: Marlin-layout W8A8 kernel (t29_w8a8.py) e2e vs Marlin per forward: -5.0/-8.8/-13.8/-18.0/-23.4/-29.4/-36.4/-77.5 ms at M=96..1024; beats Triton 4-12%; ~cuBLASLt at M<=128, 12-15% behind at >=384. Repack emulation bitwise vs real op. Triton fp32->e4m3 double-rounds via f16 (fix in w2's quant_perm; w1 to check).
- w1 task4: gate T29_W8A8=2 (Marlin layout, M>96 W8A8, decode Marlin), abba 2 pairs, 14 min. w2: verify quant fix + epilogue + think about 50% transfer.
- Balance 79.4 after job 205.
- w1 task4 wired (custom op t29_marlin_fp8_linear, M<=96 champion marlin_gemm exact, M>96 quant_perm+bs_gemm_marlin split-1, prebuilt ext, precheck bitwise at M<=96). Job queued after 205/206. Fused producers got the e4m3 rounding fix.
- T28-003: Triton all-M W8A8 + two-slot GDN + k=3 vs k=2: +2.93% all B>A. T28-004: Marlin-e4m3 M=64 warm 13.39 vs Marlin M=48 13.75. Proposed combined stack: mode 2 + T28_E4M3(M<=96) + k=3 + two-slot; one GSM8K-1319 A/B for both if manager agrees.
- w2 crit analysis (job-199 nsys, CPU): mixed GPU0 busy -1177 ms, idle +587 -> 50%. Two mechanisms: (1) small mixed steps (<=192 tok) launch-bound once GPU faster (gaps before eager GDN-prefill/elementwise; period 87->91 ms despite -16 ms GEMM) [node-trace caveat]; (2) large steps: chunk streams run concurrently, both GEMMs end together, ARs interleave on CEs -> ~650 us idle/layer (Marlin used to serialize via full SMs). Fix (a) enforce chunk order (events/priority + CE ordering) est 650->250 us/layer; (b) split qkvz per chunk.
  These are the next levers in-direction: (a) cheap; (1) needs graphing GDN prefill glue.
- Agreed w/ T28: ONE combined record (mode2 + T28_E4M3 M<=96 + k=3 + two-slot), joint GSM8K-1319, split minutes. T28 copies mode-2 code after T29-003 lands (tell them). T28 may take interim Triton-path stack to official.
- **T29-003 (job 207): mode 2 (Marlin-layout W8A8 M>96) +9.2% same-job, all B>A (1025.7/1034.2 vs 942.3/943.9, 300 prompts). Decode median ITL unchanged. Mixed-like gaps -23%. gsm8k50 92%. Memory +10 MB.** Exceeds forecast (+3.5-4%): Marlin-layout GEMM faster than Triton + in-situ Marlin worse at padded mixed M (not separated).
- I reviewed validity (promote loop ok). w1 task5: T29-004 = default mode 2, GSM8K-1319 A/B same job + 128K, then official.
- w2 ordering prototype (ws t29-ord, VLLM_T29_ORDER=1/2, scheduling-only). Funded ABBA 12 min -> T29-005. Budget plan: w1 GSM8K ~18, official ~15, w2 209 ~6, ordering ~12 -> ~10 left.
- w2 job 209: quant_perm fix bitwise verified (40 pts). Epilogue + configs_v2: -1.6..-3.4% per forward at M>=192; 1.04-1.11x cuBLASLt; ncu: remaining gap = per-128-K promotion (rule). Stop GEMM tuning. configs_v2 to adopt in a later record (with ordering). T29-005 ordering ABBA queued (in-server startup bitwise self-check).
- T29-005 ordering (job 211): -0.08% e2e (noise), mixed-like -1.8% (~0.4%); bitwise self-check 164/164. Kept as record default off. Note A(mode2) 1028-1034 in job 211 reproduces T29-003. w2 -> CPU-only plan for small mixed steps host-bound hypothesis.
- w2 small-step study (CPU): small mixed step ~1752 eager launches (~36/GDN layer); under nsys B host-bound; untraced est host-bound 0-15 ms/step; ceiling 0-2.6%. Probe T29_HOSTPROBE ~2.4 min; fixes (a) trim GDN glue (~2-3 days, exact), (b) direct triton launchers. Decide after official/resync.
- job 212 killed by w1's shell 30-min limit during A GSM8K. B: GSM8K-1319 96.7%, 128K pass. Rerun approved (no 128K), 17 min.
- job 214: GSM8K-1319 B(mode2) 96.82 vs A(bf16) 96.21, +0.61 PASS. official_eval T29-004 launched.
- **OFFICIAL T29-004: 1045.8 tok/s (+8.1% vs 967.3)**, trials 1043.9/1047.7, TTFT p99 835, TPOT p99 27.4, GSM8K 94.4, acc 2.325, 128K pass. Balance 35.7.
- CHAMPION: T29-004 promoted. Resync: +80 GPU-min (balance ~115). New bar 1098.1. Resync report reports/T29-1.md.
- Proposed KB87-90 to manager. Next big lever idea: mixed-step intercept (38.5-25.3=~13 ms over decode x 372 steps ~4.9 s ~5%) -- if host/eager-bound, trimming/graphing GDN prefill glue may recover a chunk. Probe (job 219) decides.
- t29-v2 ws ready (configs_v2 only; bs_gemm.cu unchanged). Park; bundle with host-trim/ordering into one ABBA + GSM8K-1319 later.
- PROBE job 219 (untraced): small mixed steps GPU-bound (backlog ~22 ms start, 3.7 ms end); host enqueue 63-65 ms/step hidden by decode backlog (~4 ms margin). Mechanism 1 dropped. GPU span small step: A 50.1 / B 44.3 vs decode ~25. Next: w2 attributes the ~19 ms GPU intercept (non-GEMM ~10-15 ms) -> candidate: short-prefill GDN path.
- w2 intercept anatomy (job199 B mapped to mode2): small step (pt1-80) GPU ~41.5 vs decode 23.6: GEMM +7.4, AR +5.2 (0.56 us/row, PCIe bw), glue +2.2, fla +1.7, quant +0.8, conv/state +0.6, attn +0.5, norms +0.2. Exact removable ~2-3 ms/step (~0.5%). AR now ~= GEMM per-row cost in mixed steps.
- Interp: mixed-step direction near its floors (FP8 GEMM w/ mandatory promotion, AR at PCIe bw). Remaining in-direction: bundle (configs_v2 + fused quant + glue trim) ~+1-2%. Big next payoff = combined stack with T28 (k=3 at M=64).
- Bundle plan: w1 fused quant in t29-v2; w2 glue trim in t29-glue; merge -> 1 ABBA + GSM8K-1319 -> official.
- Sequenced with T28: T28 record (e4m3+k3+two-slot on T29-004, ABBA running, ~2-3 h to official) first; T29 bundle on top, builds_on T28's. Asked manager if bitwise-equal FP8 changes skip GSM8K-1319. w1: bitwise fused producers + M<=96 pi-quant path (flag, for T28) + v2 bitwise check.
- Manager addendum: fused producers / tiling-only GEMM config = identical FP8 code if bitwise on real captured activations (all layer types, M ranges) -> bundle reuses T28's GSM8K-1319.
- T28-005 combined stack NULL (-0.06%): e4m3 M=64 in situ +2.3 ms vs Marlin M=48 + k3 fixed 1.5 ms. Bundle lands on T29-004, reuses job 214 GSM8K if bitwise. Fused all-M pi-quant now on T28's critical path.
- job 225: fused norm variant 2 BITWISE vs flashinfer CuTe (y,res,q,s); silu fused bitwise. Fused norm slower at M=128 (7.8 vs 5.5 us), faster >=512. Wiring in ws t29-fu (T29_FUSED, T29_FUSED_ALL_M, T29_FUSED_NORM). Capture+smoke job approved (~9 min).
- job 226: capture compare BITWISE PASS (109 dumps, all sites except attn-with-res -> re-capture), v2 table bitwise on live acts (0 diffs), fused smoke ok (gsm8k50 92%). Fused norm default only M>=256. Bundle job (~16 min) approved -> T29-006 (reuse job 214).
- T28 on last chance: silu-only fused at M<=96 (quant only 0.49 ms/step at decode, T28-006); one more k=3 ABBA with single-accumulator kernel; else closes, e4m3 + two-slot as assets.
- w2 glue trim done (T29_GLUE=1; 16->4 glue launches/GDN layer, est -1 ms GPU/mixed step ~+0.4%, host -5..8 ms small step), CPU-harness bitwise, in-server self-check. Merge into t29-fu; check on precheck server only.
- Human: new T29 workers will be Sonnet 5.5 (medium); w1/w2 stay Opus. Keep w1/w2 for hard kernel work.
- T29-006 bundle (job 231): +1.09% all B>A (1045.9/1034.8 vs 1027.6/1030.7), mixed-like -7.5%; capture 147 dumps bitwise; glue 192/192. My validity written. official_eval T29-006 launched.
- OFFICIAL T29-006: 1062.1 (+1.6% vs 1045.8), trials 1062.3/1061.8, TTFT p99 851, TPOT p99 30.3, GSM8K 93.6, 128K pass. Balance 83.8.
- CHAMPION: T29-006 promoted (1062.1). New bar 1115.2. Waiting w2's N-split AR verdict before close-out.
- Write-up reports/T29-writeup.md done; KB91-93 proposed; awaiting w2 N-split verdict to append; then close.
