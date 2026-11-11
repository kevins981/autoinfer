# T28 write-up (leader-28, explore). Closed after resync 1 (last chance): the last ABBA lost

GPU: 61.5 of 100 used (seed 20 + start 80); resync 1 = last chance from the existing balance; 38.5 returned. Records: bank/T28-001..008. Workers: leader-28.w1 (kernels), leader-28.w2 (in-situ A/Bs, ports). Notes: local_kb/leader-28.md.
Champion during the team: T27-002 967.3 -> T29-004 1045.8 (T29's W8A8 at M>96 on the Marlin layout).

## Direction: FP8 tensor cores at decode sizes, to make exact deeper speculation (k=3) pay (proposals/T28-1.md)
- Premise: decode (60% of wall) is byte-bound at 12.25 GB/step; the only way past a byte floor is more accepted tokens per weight pass.
  Fixed k=3 had lost (T5-004: -3.5%) because Marlin pays a "3->4 m-block cliff" at M=64 (+3.0-3.4 ms/step, KB63). The new W8A8 ruling
  allows e4m3 MMA (2x the bf16 rate, KB69), so an FP8 GEMM at M=64 should be byte-bound and cost about Marlin's M=48 time.
- Split with T29 (agreed, dispatch by M, one shared Marlin-repacked weight layout): T29 M>96 (mixed steps), T28 M<=96 + k>=3.
- Forecast: payoff if it works +8.5% (over 967.3), central +3.5%, odds >= +5% ~40%.

## What was measured
1. **Stock Triton W8A8 vs Marlin, small M** (T28-001/002, with T29's harness): Triton at M=64 already costs ~Marlin's M=48 cold
   (16.63 vs 16.47 ms/forward) but +1.1 ms warm; a custom standard-layout kernel (swap-AB register-direct, then smem-staged) lost to
   Triton warm at M<=64 and won only at M=80/96. Parked.
2. **Marlin-e4m3** (T28-004): Marlin's own template with an fp8 A tile, Marlin's raw B words as the e4m3 k32 fragment under a fixed
   in-slab k-permutation π (derived independently, matches T29's PI_SPEC, validated on the real repack), per-k32 flush into one fp32
   accumulator with As*Bs. Standalone: M=64 warm 13.39 ms/forward vs Marlin M=48 13.75 (success line met warm), cold 17.71 vs 16.59
   (gate_up cold 0.65 TB/s). M=48 warm 12.74 vs 13.75. Beats Marlin bf16 from M=128 up, but T29's bs_gemm_marlin is far better above
   M=96 (T28-007), so the M>96 split stands.
3. **k=3 in situ on the Triton path** (T28-003): k=3 vs k=2, both W8A8 + two-slot GDN, same-job ABBA: ALL B > ALL A, +2.93%;
   vs the old Marlin champion +4.13% all B > A, but the decode step grew +3.64 ms for E +14%: decode tokens/ms NEUTRAL; the gain was the
   mixed steps (which T29-003/004 then captured better, +9.2%).
4. **Combined stack** (T28-005: T29-004 + Marlin-e4m3 at M<=96 + two-slot GDN + k=3) vs T29-004, same-job ABBA: NULL, -0.06% mixed;
   step +3.83 ms for E +13.7%.
5. **Attribution** (T28-006, nsys job 224, M=64 decode step 28.83 ms): e4m3 GEMMs 15.06 ms (vs Marlin M=48 13.0, +2.0); π-quant
   launches 0.49 ms (launch floor, NOT the 0.9 feared); host AR 7.05 (+~1.0 for 16 more tokens); 3rd draft ~0.7; GDN 2.23.
   w2's code read: the AR-window L2 prefetch fires on the e4m3 path (same buffer), so prefetch was not the cause.
6. **The cause and the hard limit** (T28-007): the e4m3 table runs M=64 as two M-slabs (B streamed twice; gate_up's 89 MB doesn't fit
   the 72 MB L2) because the one-slab config (tmb=4) spills 1.5-1.8 KB at 255 registers (frag_c alone is 128 regs at tmb=4). Stock bf16
   Marlin stack-spills at tmb=4 too: that is what its "3->4 m-block cliff" really is. Forcing the scale FMAs after each MMA or
   constant-folding the scales changes nothing. A byte-floor M=64 kernel needs a different warp tiling (smaller per-warp accumulator or
   accumulators in smem): a new kernel, of the kind T26 showed loses to Marlin's load path.

7. **Last ABBA, e4m3 at k=2** (T28-008, job 234, vs T29-004, + T29's fused silu quant, bitwise-checked in the server): LOSS, all B < all A,
   -1.84% (1015.7/1012.5 vs 1031.2/1035.2); decode step +0.65 ms with E unchanged. The standalone -0.33 cold / -1.0 warm ms/forward did not
   transfer: in situ Marlin's single tmb=3 slab at M=48 is the better data path than the e4m3 table's configs (plus ~0.35 ms of quant launches).

## k=3 cost decomposition (in situ, 16 requests, rank 0; T28-005/006 + T5)
| item | ms/step | source |
|---|---|---|
| GEMM at M=64 vs M=48 (e4m3 two-slab) | +2.0 | T28-006 nsys vs KB77/79 |
| host AR (64 vs 48 tokens/AR) | +0.7..1.0 | T28-006 (7.05 vs ~6.0-6.4) |
| 3rd draft step (screened FP8 head) | +0.6..0.7 | T5-003, KB84 |
| π-quant launches (256 x 1.9 us) | +0.5 | T28-006 |
| GDN/attention/norm glue | +0.2 | T5-006 |
| **total** | **+3.8** (measured +3.83) | |
E 2.34 -> 2.66 (+13.7%). Even with a perfect M=64 GEMM (+0), the fixed ~2.4 ms (+9.6%) caps the decode gain at ~+3.7% (~+2.4% e2e).
**Deeper exact speculation on this stack is capped by per-token AR bytes (PCIe floor, KB23) and the byte-bound drafter (KB80), not by
the GEMM any more.** The same arithmetic kills k=4 (M=80: +0.6 draft, +0.6-1.0 AR per level, E +~8%).

## Forecast vs outcome
| part | forecast | measured |
|---|---|---|
| E at k=3 | 2.68 | 2.66-2.67 (ok) |
| cliff -> 0 with FP8 MMA | assumed | standalone warm yes (13.39 vs 13.75); in situ NO (+2.0 ms): register-bound two-slab |
| k=3 step cost | +1.4 ms | +3.8 ms (AR +1.0 not +0.6; GEMM +2.0; quant +0.5) |
| e2e | +4.6 s/trial (central +3.5%) | 0 (null) |
My main error: I took the standalone warm column as the in-situ regime and did not check the config the table would actually pick at
M=64 (two slabs) against L2 capacity; and I under-costed the AR growth per token (T5's +0.6 was measured at a different AR path).

## Assets (documented, env-gated, in workspaces/t28-comb.b; diff in bank/T28-005/change.diff + later)
- **Marlin-e4m3 kernel** (csrc/quantization/marlin/marlin_e4m3*.{cu,h,inc}; own gencode block "8.9": the stock arch list compiles the
  body out and leaves an empty stub, which crashes at launch). Reads the champion's Marlin weights + fp32 standard-order scales +
  π-permuted group-128 fp8 activations (the same inputs as T29's bs_gemm). Flags T28_E4M3 / T28_E4M3_MAX_M / T28_E4M3_DRAFT /
  T28_E4M3_CHECK_QUANT. Hand-over: agents/leader-28.w1/ASSET.md, k/PI.md, k/DESIGN.md. Best use: M=49..96 warm standalone (beats Marlin and
  Triton); NOT at M=48 in situ (T28-008: +0.65 ms/step vs Marlin) and not at M=64 (T28-006: two-slab re-stream). Any reuse must re-check the config walk against L2.
- **Two-slot GDN state on the champion** (T5-005/T25-003 port: fused_gdn_decode_kernel.cu, _custom_ops.py, gdn_attn.py, replay.py,
  qwen_gdn_linear_attn.py, envs.py; VLLM_GDN_TWO_SLOT, log line; KV pool 267,721 -> 276,547 tokens, peak KV 84% -> 59% at k=3; 128K pass).
  Prerequisite for any k>=3 work (stock GDN spec state needs 13 blocks/req at k=3: 16 requests don't fit 5.0 GiB, KB13).
- **T29's fused π-quant producers wired to the M<=96 branch** (w2 port of t29-fu; QuantizedActivation via _input_quant_key).
- Tooling: agents/leader-28.w1/mb/bench*.py (per-shape cold/warm/per-forward tables incl. Marlin, Triton, e4m3, bs_gemm), prof/an.py
  (per-step kernel breakdown from nsys sqlite), agents/leader-28.w2/job1/job*.sh (abba recipes), prefetch_notes.md.

## KB candidates
1. Marlin's M=64 "3->4 m-block cliff" (KB63) is a register/stack-spill effect of tmb=4 (255 regs; frag_c 128), not tensor time: stock
   bf16 Marlin stack-spills 8-48 B at tmb=4, an e4m3 variant 1.5-1.8 KB; T6's slab policy is the workaround, and it re-streams B through L2
   (gate_up 89 MB > 72 MB L2) in situ. (T28-007, T28-006)
2. In situ at k=3 (M=64) the step grows +3.8 ms = GEMM +2.0 + AR +1.0 + 3rd draft +0.7 + quant +0.5 + glue; with a perfect GEMM the
   fixed ~2.4 ms vs E +13.7% caps k=3 at ~+3.7% decode: deeper exact speculation is AR/drafter-bound on this stack. (T28-005/006, T5)
3. Per-token-group FP8 quant launches at decode cost 1.9 us each in CUDA graphs (0.49 ms/step for 256 linears), not the ~3 us
   microbench figure. (T28-006)
4. Standalone warm tables don't predict in-situ GEMM time when the chosen config streams B twice: check the tile walk against L2
   capacity (89 MB gate_up vs 72 MB L2) before trusting a warm number. (T28-006/007)

5. In situ at M=48 (k=2) the Marlin-e4m3 kernel + fused silu quant is +0.65 ms/step vs bitwise Marlin bf16 (-1.84% e2e, all B < all A): no
   FP8-MMA decode GEMM derived from Marlin's template beats Marlin at decode sizes on this stack. (T28-008)

## Opinion
- Within this direction nothing left clears the bar: k=3/4 are AR/drafter-bound; the e4m3 kernel loses at k=2 in situ.
- What could still move decode: fewer AR bytes per token (the AR is at the PCIe floor per byte; only algorithmic byte reduction helps,
  and the 0.78 codec is already in), or a drafter that proposes several tokens per weight pass (data-limited, KB60).
