# LOCAL_KB leader-31 (team T31, explore, slot A). Started 2026-11-10.

## Inspiration cards (T31-1): what they made me think about
- "Timing" + "Mismatch with the machine": the decode step has 137 all-reduce windows (~5.7 ms/step) in which HBM is nearly idle; the champion already fills them with L2 prefetch (KB25) but cashes only ~16 MB per window because Marlin's L2-warm data path is as slow as HBM (KB79/KB81). The mismatch is Marlin's bf16-dequant path vs e4m3 tensor cores (2x rate, no dequant, KB69). That became the direction.
- "Excess guarantees": bf16 activations at decode are a stronger promise than the checkpoint's own W8A8 scheme (allowed since 2026-11-09); the drafter already runs FP8 pieces.
- "Large L2": 72 MB vs 42-89 MB per decode GEMM: the prefetch budget per AR window (~40 MB at HBM rate) fits; the open question is the L2 read rate at 40 MB working sets (T26 saw 1.29 TB/s on 42 MB vs 3.7 on 16 MB: capacity effect or kernel effect? w2 measures).

## Trial model used for forecasts (champion T29-006, 1062.1 tok/s)
- 400 x 228 = 91.2K output tokens -> 85.9 s/trial. Mixed steps ~370 x ~55 ms = ~20 s; decode ~2690 steps x 24.5 ms = ~66 s (77% of wall).
- Decode step kernel sum 23.6 ms (KB91 table): GEMM 13.1, host AR 5.6-5.7 (137 calls x ~42 us), GDN 1.6, lm_heads 1.2, drafter cuBLAS 0.7, norms 0.5, attention 0.3, act 0.2.
- Per-rank weights per step: gate_up 89 MB x64, down 44.6 x64, qkvz 42 x48, gdn_out 15.7 x48, attn_qkv 21 x16, attn_o 15.7 x16 = ~12.25 GB.

## Direction: e4m3-MMA decode GEMM that cashes the AR-window L2 prefetch (proposal T31-1)
- Prize arithmetic: prefetchable bytes/step ~5.7 GB (qkvz/attn_qkv in AR#2 window, 40 MB of gate_up in AR#1 window, gdn_out/attn_o during the latency-bound GDN/attention core) x (1/0.95 TB/s - 1/BW_L2). At BW_L2 = 3 TB/s: 4.1 ms/step. Costs: ~190 extra quant launches x 1.9 us = ~0.37 ms (KB95), AR stretch unknown (w2).
- Why previous decode-GEMM teams lost (my reading): T8/T17/T24/T26 kept bf16 MMA + dequant (floor = Marlin's dequant path, KB81/KB83); T28 used e4m3 MMA but inside Marlin's smem/ldmatrix template (warm M=16 only 7% faster than Marlin: data-path bound at ~1.3 TB/s equivalent, ASSET.md table). Untested: e4m3 + register-direct weights + activation operand staged once per CTA.
- Workers: w1 kernel prototype (6 GPU-min), w2 L2/prefetch budget (5 GPU-min). Reserve ~9 of the 20 seed.

## 2026-11-10 T31-1 APPROVED (manager): start grant 80 released when S1 seed results are sent, unless they falsify the premise
- Falsifiers: kernel warm rate not clearly below Marlin's warm floor; L2 read rate at ~40 MB stuck near 1.29 TB/s with no workaround.
- Manager asks: (1) S1 must say why a register-direct path escapes what sank T28-008 (+0.65 ms/step in situ despite -1.0 ms/forward warm standalone; quant launches 0.49 ms); measure warm standalone AND an AR-prefetch + GEMM chain arm. (2) decode W8A8 = new FP8 code -> own GSM8K-1319 same-job A/B.

## S1 results so far (T31-001, w2, jobs 236/239)
- L2-warm read ceiling ~3.4 TB/s up to ~72 MB working sets (plain-read warm); T26's 1.29 TB/s was its kernel, not L2. Premise part (ii) holds.
- prefetch.global.L2::evict_last hints are DROPPED under load: reader after a champion-style prefetch only ~1.5 TB/s, ~40% of 64 MB resident. Real AR window (~40 us) lands ~17-26 MB at 24-40 requested, ~29-47 at 64; AR stretch <= +0.4 us up to 64 MB (8 blocks), +10 us at 88 MB. W2-2: try real loads (ld with evict_last policy) as the prefetch.
- Marlin chain: gate_up 111.7 (no pf) -> 84-85 (>=64 MB pf) = its floor, matching in-situ KB79; so the new kernel's gain is strictly below Marlin's floor: 0.66 us per resident MB (1/1.05 - 1/3.4).
- Forecast check: 45 MB/window x 137 windows... per-step resident bytes realistically ~3.5-5 GB -> 2.3-3.3 ms/step from residency alone (before the kernel's own MMA floor). Still consistent with central +5%, payoff ~+10%.

## Kernel v1 (w1, job 243; record T31-002 pending W1-2)
- Correct (1 ulp vs T29 bs_gemm, deterministic, 0 spills). Warm M=48: qkvz 32.9 vs Marlin 40.5, down 40.4 vs 45.4, gdn_out tie, gate_up not testable warm (89 MB > L2). Cold 10-30% slower. Chain arm (prefetch hints) no gain: hints dropped.
- Ablation: fixed stream-K reduction overhead 12-19 us/launch; MMA+overhead-only arm 54 us gate_up (ideal 29): 1 warp/scheduler + barrier per block. Design causes -> W1-2 (one CTA per slab, 8 warps in-CTA K-split, no per-block barrier, real-load-resident chain test, ncu).
- Marlin warm gate_up swings 96-108 us between jobs (KB81's 85.8 not reproduced in job 243).
- T30 funded (65) for two-wave layer-pipelined mixed steps: no overlap with us.

## Re-forecast sent to manager (from T31-001 + job 243), kernel at its floors
- Scenario A (hints, ~25 MB/window): ~2.0 ms/step net -> +6.7%. Scenario B (real-load prefetch ~40 MB/window): ~2.85 ms -> +9.8%; + GDN-core site for gdn_out -> ~+11%. Central +4%, odds >=5% ~30%, conditional on W1-2 (qkvz warm <= 26, cold within 5% of Marlin).
- Rule for the kernel: on non-prefetched shapes (down, gdn_out, attn_o) it must match Marlin cold, else the gain is eaten.

## 2026-11-10 Start grant 80 RELEASED (balance ~93). Manager: get to a same-job ABBA vs T29-006 EARLY (T28-008 lesson), own GSM8K-1319 A/B.
- Budget plan: W1-3 gate_up + resident chain ~4.4; W2-2 ~2; S2 smoke (first in-server run of the kernel at M<=96, gsm8k-50, decode-step probe) ~6; early ABBA (2x2 starts) ~14; prefetch tuning ABs ~10; GSM8K-1319 A/B ~15; official ~12; reserve ~25.
- Integration split: w1 = csrc op + config table; w2 = Python dispatch in t29_marlin_w8a8.py M<=96 branch, quant producers (T29_FUSED_ALL_M silu; norm+quant; new quant for GDN-core and attention outputs), real-load prefetch in hostar, prefetch sizes per consumer.
- W2-2: real-load prefetch = 32 MB resident per AR window at +2 us (16 blocks), 100% resident; hints 20-30 MB at 8 blocks. Residency budget per window ~32 (loads) to ~45 (hybrid, untested). Scenario between A and B: ~+8-9% if the kernel holds in situ.
- Assigned: W2-3 (chain run with w1's plugin, 3 min; then Python integration), W1-3 (gate_up layout, csrc op). Next after those: in-server smoke, early ABBA.
- W1-3: gate_up with contiguous residency is a wash vs Marlin (R48 86.1 vs 87.8); qkvz -31% at R32. Cause (my reading): MMA-bound warm phase serialized with HBM-bound cold phase. Lever: interleaved residency along K (W1-4). Bonus not in forecast: cold wins on down/gdn_out/attn_o (-6 us each, 128 calls/step = ~0.8 ms).
- Revised arithmetic: without interleaving ~1.3 ms/step net (+4% e2e); with interleaving working ~3.5 ms/step (+11%). Fixed ~7 us/launch x 264 = 1.85 ms/step is another lever.

## Chain results (w2 jobs 277/279) and W1-4 (job 278)
- Interleaved residency does NOT help gate_up (ours ~86 at any pattern, Marlin flat ~88 from 30-58 MB resident). Weights-only arm 62.8 us at 44 MB resident: cold stream wall; L2 footprint 89 MB > 72 so the stream evicts the resident set (hypothesis; W1-5 tests evict_first on cold loads). gate_up stays on Marlin (T31_SKIP) unless W1-5 surprises.
- Chain (hint 48 MB, Marlin on gu): e4m3 saves 16.8 us/layer (qkvz -9.4, down -4.0, out -3.4) = ~1.07 ms/step gross, ~0.8 net of ~2 quant launches/layer -> ~-3% decode, ~+2.5% e2e expected in situ. Real-load prefetch LOSES in the chain (AR2 prefetches two ranges: +26-36 us stretch); hint stays default; useful hint cap ~48-64 MB on AR1.
- Honest status: measured in-chain payoff ~+2.5%; levers left: fixed ~7 us/launch (x264), gate_up (W1-5), quant fusion. Payoff if all work ~+8%; central ~+2-3%. W2-4 in-server smoke A/B (8 min) is the real test; then ABBA, then resync with these numbers.

## Validity review of the T31 kernel math (csrc/t31/t31_gemm_e4m3.cu, lines 150-176): 4 e4m3 m16n8k32 MMAs accumulate into a fresh fp32 tile per 128-K block, then one fmaf per element with As[row]*Bs[block] into the fp32 accumulator; tile zeroed per block. Matches the W8A8 ruling (accumulator inside one 128-K block, fp32 across blocks). Quant = T29's quant_perm (vLLM's math, bitwise). Record T31-003 filed (smoke +1.3%, 1 pair); ABBA W2-5 pending; W1-6 fixed-cost split pending.
- ABBA job 285: +2.25% all B > all A (1064.2/1064.1 vs 1039.3/1042.2). In-situ transfer vs chain prediction (~3% of decode step gross -> ~2.3% e2e): ~1:1. W2-6 GSM8K-1319 A/B + 128K running (17 min). Then official. Balance after: ~50.
- Manager (Addendum 2 in task_definition.md): a strict M-subset variant (T31_MAX_M=48) may reuse job 288's GSM8K-1319 A/B if kernel/quant/configs are identical and outside the subset the path is bitwise Marlin; cite job 288 and the subset in the record. If both variants win, submit the one with better same-job evidence; if W2-7 is null, official on T31-003 as is.
- W1-7: dispatch host cost 12.1 -> 3.3 us/call (ws requirement recomputed per call: 16 pick + 16 ws_numel per linear). ~2.3 ms/step of host time on eager mixed steps; plausible TTFT cause. W2-7 = ABBA of the shipping candidate (MAX_M 96 + patch) vs champion; MAX_M=48 ABBA only if the TTFT shift persists. W1-8: ASSET.md hand-over (CPU).

## OFFICIAL T31-003: 1088.9 (+2.5% vs 1062.1), both trials above the champion's. Write-up reports/T31-writeup.md final. Recommending close; ~31 GPU-min returned.
