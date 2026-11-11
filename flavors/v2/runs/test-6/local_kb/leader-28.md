# LOCAL_KB leader-28 (team T28, explore) — run test-6

## Direction (agreed split with T29, 2026-11-09)
T29: W8A8 (checkpoint FP8 scheme) for GEMMs dispatched at M > 96 (mixed/prefill steps), Triton block kernel.
T28: W8A8 for M <= 96 (decode shapes) with a custom small-M FP8-MMA kernel + k=3 exact speculation.
Shared constraint (leader-29): weights can live in ONE layout only (+13 GB/rank for a second copy). Agreed:
standard checkpoint layout: W fp8 e4m3 [N,K] K-contiguous, scales fp32 [N/128,K/128]; activations via
per_token_group_quant_fp8 (fp8 [M,K] + fp32 [M,K/128]). Marlin goes away entirely once both paths are W8A8.

## Why k=3 pays once the decode GEMM is FP8-MMA (numbers)
- T5-004 (fixed k=3, Marlin): step 25.22 -> 29.55 ms (+4.3) for E 2.345 -> 2.69 (+15%) => -3.5% e2e.
  Decomposition (T5-003/006, KB63): Marlin 3->4 m-block cliff +3.0-3.4 in situ; 3rd draft step +0.8; AR +0.6; GDN/attn/norm +0.2.
- FP8 MMA (KB69): 2x bf16 rate; tensor time at M=64 = 64*24.4 GFLOP/291 TF = 5.4 ms << byte floor 12.9 ms (12.25 GB @0.95 TB/s).
  So an FP8 W8A8 GEMM at M=64 should cost ~ Marlin M=48 (byte-bound), removing the cliff: step ~26.2-26.8 => 2.69/26.5 vs 2.345/25.22 = +8-10%.
- T29 microbench (agents/leader-29.w1/mb/summary.md, GPU0 cold read-flush, per-forward linear totals, ms):
  Marlin M=48 16.35, M=64 17.69; Triton W8A8 e2e (quant+GEMM) M=48 16.53, M=64 17.27. Triton GEMM only: M=48 15.61, M=64 16.36.
  Quant kernel ~2.9-3 us per GEMM x ~320 GEMMs/forward = ~0.9 ms/forward: launch-bound; fusing quant into producers is a ~3% item.
  Note: this cold microbench gives Marlin 16.35 vs in-situ 13.0 ms (KB77/KB79, L2 prefetch + warm arm). Relative numbers only.
- Stock Triton at M=64 already gives k=3 ~ +4% on paper (step ~27.75). Custom kernel target: M=64 GEMMs <= Marlin's M=48 in-situ time.

## Kernel design notes (for the worker)
- Standard [N,K] fp8 layout is the natural B operand for mma.sync.m16n8k32 e4m3 (B col-major fragment = 4 consecutive k bytes per n per
  thread): no repack, register-direct loads possible (T26's swap-AB loop hit 93% HMMA peak with weights in registers).
- k is a summation index: any fixed k-permutation inside a 32-k slice (within the 128-group) can be applied to A at quant time to match
  whatever fragment layout the B loads produce.
- Per 128-K block: FP8 MMA accumulate (f32 acc, ~13-bit window, allowed within a block), then acc_main += blk * a_s[m] * b_s[n] in fp32.
- Small N shapes (down N=5120, out/o_proj N=5120): need split-K (Triton stock has none) to fill 128 SMs at small M.
- Earlier failures (T8/T17/T24/T26) were bf16-dequant kernels: tensor pressure + dequant per fragment. FP8 MMA halves MMA issue and
  removes dequant; the open question is purely the load path at the HBM roof cold and from L2 warm (KB39/KB81/KB83).

## Inspiration cards (what they made me think)
Imbalance/Generality: decode is byte-bound and the FP8 tensor cores are idle 60% of the time: spend tensor to buy tokens (k=3) is the
imbalance trade. Boundaries: the quant kernel is a boundary crossing (~0.9 ms/forward of launches) worth fusing into producers.
Predictability: deeper speculation is exactly guess-and-verify; its price here is per-token AR + draft bytes, not FLOPs any more.

## Open questions
- In-situ cost of Triton small-M W8A8 vs Marlin at M=48 (T29 measuring in-server now).
- k=3 memory: T5-004 fit 16 req with KV 5.75 GiB + cap 1536 (stock GDN state); two-slot GDN (T5-005/T25-003) not needed at k=3.
- Does the third draft step still cost 0.8 ms with the champion's screened FP8 draft head (T27-002: 336 -> 115 us lm_head)? Probably ~0.6.

## Status log
- 2026-11-09: proposal T28-1 FUNDED (80 GPU-min start grant + 20 seed). Manager notes: state KV capacity change vs champion
  for k=3/4 and pass 128K; GSM8K-1319 same-job A/B vs bf16-activation reference needed for any FP8-activation record (can share
  one run with T29 when the code is the same); verification stays exact.
- Workers: leader-28.w1 = small-M FP8 W8A8 CUDA kernel kill test (<= 6 GPU-min, ws t28-k3). leader-28.w2 = in-situ k=3 on T29's
  env-gated W8A8 path (T29_W8A8=1, ws t29-fp8 to port from), AB 2 starts (<= 9 GPU-min).
- Budget plan: seed 20 -> w1 6 + w2 9 (+5 reserve). Grant 80 -> kernel iterations 12, ABBAs 36, GSM8K-1319 20, official 12.
- T28-001 (job 193): custom swap-AB register-direct FP8 kernel v1 loses (M=64 22.48 vs Marlin M=48 16.47 cold; 0.56 TB/s); stock Triton
  M=64 16.63 cold ~ Marlin M=48, warm 14.84 vs 13.75 (+1.1). w1 task 2: smem-staged activations, 32 rows/warp, Triton split-K; target warm.
- T29-001 (jobs 192/195, in situ, k=2): all-M Triton W8A8 = +1.05 ms/decode step vs champion (ITL median 25.9 vs 24.9); mixed p99 ITL -15%;
  e2e +0.2%. Likely ~320 quant launches/forward (~0.9 ms in microbench). T29 owns the fused quant (RMSNorm/silu_mul/GDN+attn producers ->
  fp8 [M,K] + fp32 [M,K/128]), one implementation for both teams. Layout decision (standard vs Marlin-repacked) deferred to T29's nsys.
  Hook: T29_W8A8=1 in kernels/linear/__init__.py init_fp8_linear_kernel; configs in quantization/utils/configs/ (RTX_4090).
- T28-002 (job 197): custom v2 still loses to Triton warm at M<=64 -> PARKED; wins at M=80/96 (~2 ms/forward warm): candidate for k=4.
  Stock Triton beats Marlin at M=48 cold AND warm in isolation => T29-001's +1.05 ms/step is outside GEMM time (quant launches,
  AR-window prefetch order on [N,K]). w1 task 3: prefetch pointer/order under T29_W8A8 + Triton-with-16MB-prefix bench + M<=96 config table.
- Forecast update: cliff at M=64 gone in microbench (Triton M=64 vs Marlin M=48: -0.23 cold / +0.36 warm). k=3 payoff now hinges on
  (a) removing the k=2 in-situ penalty (shared with T29), (b) in-situ k=3 step growth ~+1.4 ms (w2 measuring).
- LAYOUT CHANGE (leader-29): shared layout is now Marlin's repacked qweight (decode k=2 stays bitwise Marlin). Interface for both teams'
  W8A8 kernels: Marlin qweight; fp32 copy of weight_scale_inv in standard [N/128,K/128] order; activations fp8 [M,K] π-permuted within
  each 32-slab (π: slot 4t+i <-> k = 2t+{0,8,1,9}[i], +16 upper half; authoritative spec agents/leader-29.w2/gemm/PI_SPEC.md) +
  fp32 [M,K/128] scales. T29's kernel: agents/leader-29.w2/gemm/bs_gemm.cu (new kernel). T28's: "Marlin-e4m3" inside marlin_template.h
  (fp8 A tile, raw B words as k32 fragment, per-128-k fp32 rescale, rest unchanged). Whichever wins per M regime is used.
- w2 (job 194/198): A = T29_W8A8 k=2: 974.0 tok/s (300 prompts), E 2.335, TPOT p50 14.74, ITL med 25.89, KV 83.8% at 16 running.
  B = k=3 OOMs at KV 5.75 (graph capture) and 5.5 GiB (runtime, GDN forward): memory tighter than T5-004's era (screen + W8A8 path).
  Fix: port T25-003's two-slot GDN (VLLM_GDN_TWO_SLOT) and run k=3 at KV 5.0 (same as A) + 128K. KB13 arithmetic: 13 blocks/req
  x 16 = 5.8 GB > 5.0 GiB without two-slot.
- T28-003 (job 201 vs 194): k=3 + W8A8(Triton) + two-slot: 991.1 vs 974.0 (+1.8%, unpaired); step +2.6 ms (Marlin +4.3) for E +14.4%;
  KV pool 267,721 -> 276,547 tokens, peak 84% -> 59%, 128K pass, gsm8k50 90%. Residual ~1.2 ms over forecast = Triton's own M=48->64 warm
  growth (+1.0, T28-002) => Marlin-e4m3 at the byte floor is worth ~1 ms/step more (~+4% decode). Next: w2 same-job ABBA k=2 vs k=3 (both
  W8A8 + two-slot); w1 Marlin-e4m3 first measurement. GPU used 10.4 / 100.
- ABBA (w2, out4): k=3 vs k=2 both W8A8+two-slot: ALL B > ALL A +2.93% (983.0/989.1 vs 960.1/955.8); step +2.65 ms, E +14.4%, TPOT p50 -2.1%,
  TPOT p99 32.8 (gate 55). Record T28-003 updated. Next: ABBA vs the pristine champion (t28-champ from T27-002) -> if all B > all A, record is
  a champion candidate pending GSM8K-1319 A/B (asked manager whether one W8A8 run can serve T29 and T28 since k/two-slot are exact).
- T28-004 (jobs 203/206): Marlin-e4m3 WORKS; π validated on real repack. M=64 warm 13.39 <= Marlin M=48 13.75 (success), cold 17.71 vs 16.59
  (gate_up cold 0.65 TB/s). M=48 warm 12.74 (-1.0 vs Marlin). tmb=4 spills: use tmb 2/3 + parallel slabs. w1 integrating into ws t28-e4
  (T28_E4M3=1, T28_E4M3_MAX_M=96; standalone π-quant kernel for now; idea: in-kernel A quant to kill the ~320 quant launches).
- w2 task 4: ABBA t28-k3 (W8A8 Triton + k=3 + two-slot) vs t28-champ (pristine champion). t28-champ built.
- Manager ruling (task_definition.md clarification): one GSM8K-1319 same-job A/B may serve T28's and T29's records if the FP8 activation
  code is identical (same quant kernel, same GEMM kernels + config table at every M the record runs incl. M=64, same layers); k=3 and
  two-slot may differ. Record must cite the shared job and list differences. Once Marlin-e4m3 replaces Triton at M<=96: own A/B.
- PLAN (agreed with leader-29): combined stack = T29_W8A8=2 (Marlin layout; M>96 -> T29's bs_gemm_marlin via torch.ops.vllm.t29_marlin_fp8_linear)
  + T28_E4M3 at M<=96 (e4m3 in that op's M<=96 branch) + k=3 + two-slot GDN. ONE combined record (both teams cited) with ONE GSM8K-1319
  A/B (minutes split). Interim: if the Triton-path stack (T28-003) ABBA vs champion is clearly positive (>= ~+2.5%, all B > A), consider
  its own GSM8K-1319 (~25 min) + official (~12) to lock a champion now; tell leader-29 first.
- w1 next: integrate e4m3 (structured to move into T29's op), sweep cold gate_up (bps=2, tmb=4 diet, slab-inner), large-M bench vs bs_gemm.
- T29-003: +9.2% all B > A vs champion (job 207): Marlin layout, M>96 -> bs_gemm_marlin, decode bitwise Marlin. T29-004 = default-on
  (T29_W8A8 default "2"), op signature + M<=96 branch frozen. Our interim Triton-path record is DROPPED. New base: t28-comb = T29-003 +
  two-slot GDN port + e4m3 at M<=96 (w1 building). w2: finish/cancel the ABBA vs old champion, then prep the k=3 ABBA script on t28-comb.
- T29-004 record exists (default mode 2; t29_act_quant.py out of tree; bs_gemm ext from hash-fixed source copy). Its GSM8K-1319 A/B
  = job 212 (FP8 vs T29_W8A8=0 + 128K). t28-comb to be based on T29-004; our e4m3 ext must follow the hash-fixed-source pattern.
- ABBA (out5): Triton stack (k=3+W8A8+two-slot) vs pristine champion: +4.13% all B > A (981.8 vs 942.8). BUT decode step 24.95 -> 28.59
  (+3.64) for E +14%: decode tokens/ms NEUTRAL on the Triton path; the gain is mixed steps. => for the combined stack, k=3's decode
  payoff depends entirely on e4m3 in situ at M=64 (warm 13.39 vs 13.75) and on killing the ~320 quant launches (fused/in-kernel quant).
  Expected: step ~26.9 with standalone quant (+5.9% decode, ~+3.5% e2e over T29-004); ~26.0 with fused quant (~+5.5% e2e).
- T29-004 GSM8K-1319 A/B = job 214: FP8 (M>96) 96.82% vs bf16 96.21% (+0.61, PASS); 128K pass (job 212); official eval launched.
  => FP8 activations at M>96 cost no accuracy; our combined-stack A/B will add M<=96 (decode) FP8 rows.
- NEW CHAMPION T29-004: 1045.8 official (+8.1%). Our bar now 1098.1 (+5%). Decode share of wall rises (~65%): k=3 on e4m3 with
  standalone quant ~+6% decode -> ~+4% e2e; with fused/in-kernel quant ~+9.6% decode -> ~+6% e2e. Payoff-if-it-works still ~5-6%:
  the in-kernel quant (or T29's fused quant) is now REQUIRED for the bar, not optional.
- t28-comb smoke PASSED (job 222, k=2): 825.6 tok/s/64 prompts, acc 2.40, gsm8k50 90%, 14.45 GiB/rank. Fix needed in diff: marlin_e4m3.cu
  own gencode "8.9" (stock arch list compiles the body out). Record T28-005 opened (diff + serve_args k=3). w2 job6 = k=3 ABBA vs T29-004.
  w1 task 5 = nsys attribution of the k=3 decode step (e4m3 GEMMs vs 13.0 ms Marlin; quant share) + two-slot log line + large-M bench.
  w1's view on in-kernel quant: restructure (128-k amax across two tk=64 stages), +9-12 KB smem, 136x redundant quant per N-tile; measure first.
- Sequencing agreed with leader-29: step 1 = T28 record (e4m3 + k=3 + two-slot on T29-004) -> GSM8K-1319 + official; step 2 = T29 bundle
  on top (v2 configs, fused producers incl. π-quant at M<=96 for our op, GDN glue trim), builds_on ours. Fused RMSNorm->quant is NOT yet
  bitwise (1 bf16 ulp on ~24% of elements vs flashinfer's norm) -> would need its own A/B unless fixed; asked manager about bitwise rule.
- Manager addendum (task_definition.md): fused producers / config tables count as identical FP8 code if BITWISE equal (fp8 bytes, fp32
  group scales, consumed bf16 outputs) on real captured activations over all layer types and the M range; tiling-only / split-K-on-K-block
  config changes count too. Relayed to leader-29. Our GSM8K-1319 A/B (once run) can serve T29's bundle under those conditions.
- job6 ABBA (t28-comb k=3+e4m3+two-slot vs T29-004): NULL -0.06% (B 1017.5/1037.0 vs A 1028.5/1027.2); step +3.83 for E +13.7%.
  => e4m3 path at M=64 in situ costs ~+2.3 ms over Marlin M=48 (quant launches ~0.9 + GEMMs near cold microbench). Wait for w1's nsys.
  Ceiling of k=3: max ~+7% decode (~+4.5% e2e) with perfect GEMM + no quant launches. Budget used ~48/100.
- T29 building fused π-quant producers at ALL M behind T29_FUSED_ALL_M (CUDA norm+quant mirroring flashinfer's layout for bitwise; Triton
  silu_mul+quant already bitwise). Will plug into our M<=96 path. w2 CPU task: does the AR prefetch fire/match on the e4m3 path?
- T28-006 (nsys job 224, k=3 M=64): e4m3 GEMMs 15.06 ms vs Marlin M=48 13.0 (+2.0); quant 0.49; AR 7.05 (+~1.0); draft 0.65+1.05.
  Cause of GEMM shortfall: block-local accumulators double frag_c -> tmb=4 spills -> tmb=2/3 parallel slabs re-stream B. Fix: flush per k32
  (mma with C=0, FMA into the single main acc with a per-(m,n) s_a*s_b product per 128-group) -> stock Marlin register budget -> tmb=4 single pass.
  HONEST RE-FORECAST: k=3 fixed costs ~2.4 ms (+9.6%) vs E +13.7% => max ~+3.7% decode (~+2.4% e2e; ~+3.8% with fused quant). Below the
  5% bar even if everything works. -> resync with the manager: one last cheap test (single-pass kernel + fused quant ABBA), else close.
- w2 prefetch notes (agents/leader-28.w2/prefetch_notes.md): AR-window prefetch fires on the e4m3 path (same Marlin buffer; hostar_allreduce_pf;
  only for ARs <= 1 MB i.e. M <= 102; cap 88 MiB not 16 MB; stripe walk => only ~19-25% of CTAs start in any prefix: same for stock).
  Real difference: e4m3 table picks tmb=2 (two M-slabs) at M=48/64 for gate_up etc. -> B read twice, second read partly misses 72 MB L2.
  w1 task 6: single accumulator (per-k32 flush) -> tmb=4 one slab; retune; bench (<= 2 GPU-min). Resync sent to manager (payoff below bar,
  propose one last ~15-min test or close). w2 preparing job7 on t28-comb.b (+ optional T29_FUSED_ALL_M=1).
- RESYNC 1 DECISION: LAST CHANCE from existing balance (51.6): single-pass kernel bench (<=2) + one k=3 ABBA vs T29-004 (<=14).
  If all B > all A: own GSM8K-1319 A/B + 128K + official (fits in ~36; top-up possible). If null/loss: write reports/T28-writeup.md
  (k=3 cost decomposition, e4m3 kernel asset + spill cause, two-slot port) and return the rest. No exploit team: any reliable win is promoted.
- 2026-11-10: T29 delivered fused π-quant producers (ws t29-fu; flags T29_FUSED, T29_FUSED_ALL_M, T29_FUSED_NORM; bitwise job 226 incl.
  M=3..67 + MTP). For us only silu-only pays (~2.4 us + launch per down GEMM; fused norm +3 us/call). w2 wiring it into t28-comb.b's
  M<=96 branch (QuantizedActivation via _input_quant_key); B arm of job7 adds T29_FUSED=1 T29_FUSED_ALL_M=1 T29_FUSED_NORM=0.
  Build coordination: w1 owns csrc + cfg table, w2 owns Python wiring; builds serialized by message.
- T28-007 (jobs 227/229): single-slab tmb=4 impossible in Marlin's template (regs; stock also spills) -> k=3 GEMM penalty stays -> k=3 CLOSED.
  e4m3 beats Marlin from M=128 up but bs_gemm_marlin is far better above 96. Last-chance ABBA re-targeted: e4m3 at k=2 (M=48) vs T29-004,
  expected <= +1-2%; manager informed. If not all-B > all-A: close + write-up.
- Manager: a reliable ~+1% win (all B > all A) IS worth GSM8K-1319 + 128K + official (top-up available); thin margin -> second ABBA in
  reversed order first. KB candidate for write-up: Marlin's M=64 "3->4 m-block cliff" = its own stack spill at tmb=4 (255 regs; frag_c 128).
  w2: smoke of fused-quant port (<=3 min) then k=2 e4m3 ABBA (<=12 min), two-slot off.
- NEW CHAMPION T29-006 1062.1 (T29-004 + v2 tiling + fused producers + GDN glue trim, bitwise). Our job8 ABBA is vs T29-004 (decode
  unchanged in T29-006 so the decode delta transfers). If B wins: rebase e4m3 M<=96 (+ ALL_M silu) onto T29-006 and run the reversed-order
  second ABBA vs T29-006 as the record's evidence. Bar now 1115.2.
- T28-008 (job 234): e4m3 at k=2 + fused silu quant vs T29-004: LOSS -1.84% all B < A (step +0.65 ms). TEAM CLOSED: write-up final,
  38.5 GPU-min returned. Workers stopped.
