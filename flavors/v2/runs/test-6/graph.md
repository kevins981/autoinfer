# GRAPH (run test-5, resumed from test-3): what was tried, and how the champion came about
Root: whole-engine speedup of output tok/s at c=16 (focus.md). BASE = 631.4 tok/s official.
Champion: **T1-042, 942.2 tok/s official (+48.2% vs BASE)** (round 14).
Legend:
- ★ = on the champion's path.
- ✗ = failed or worse.
- ○ = null result (within noise) or diagnosis only.
- A child builds on its parent.
- Scored but untried ideas are in ideas.md.

## Direction R1-A: CUDA-graph mixed prefill steps (round 1, T1, from systematic_engineer)
- ○ **T1-001** (from BASE): serve flag max_prefill_cudagraph_capture_tokens=2048.
  - Result: unofficial 645.2 (+2.2%, noise); 128K pass.
  - Conclusion: on NCCL, prefill steps are GPU-bound, not CPU-launch-bound (KB1).
  - The flag later pays off stacked on the host-staged AR (see T1-005).

## Direction R1-B: custom host-staged 2-rank all-reduce, no P2P (round 1, T1, from both brainstormers)
- ★ **T1-002** (from BASE): host-staged AR through pinned /dev/shm, a code default for TP=2 without P2P.
  - Result: official 662.8 (+5.0%).
  - Conclusion: bitwise equal to NCCL; the decode AR goes 61 -> 50 us, near its PCIe floor (KB2, KB5).
  - ○ **T1-003**: diagnosis only (in-server nsys + microbench), no new code.
    - Conclusion: the prefill-step AR skew is not caused by transport (KB3).
  - ★ ○ **T1-004**: + host-staged all_gather.
    - Result: unofficial 666.8 (~0, noise).
    - Conclusion: the logits all-gathers are small (KB6); kept as a building block.
    - ★ **T1-005**: + T1-001's prefill-graph flag.
      - Result: official 699.2 (+10.7% vs BASE).
      - Conclusion: the AR and the flag stack super-additively (KB4). Champion after round 1.

## Direction R2-A: logits diet (round 2, T1, from both brainstormers; builds on T1-005)
- ★ ○ **T1-006** (from T1-005): + use_local_argmax_reduction for the drafts.
  - Result: unofficial +1.1% (noise alone).
  - ★ **T1-007**: + FP8 draft lm_head (ruling 1).
    - Result: unofficial +3.0% vs T1-005; acceptance unchanged (KB10); 128K pass.
    - ✗ **T1-008**: + num_speculative_tokens=3.
      - Result: unofficial -6.9%. Acceptance 2.72, but TTFT p50 210 -> 792 ms and the p99 gate fails.
      - Conclusion: cause undiagnosed (KB11).
    - ★ **T1-009**: + greedy local-argmax fast path for target verification (code).
      - Result: official 730.9 (+4.5% vs T1-005, +15.8% vs BASE).
      - Conclusion: exact including ties, checked op-level (KB12). **CHAMPION** (round 2).

## Direction R3: spec-decode capacity + GDN state traffic (round 3, T1, from both brainstormers; builds on T1-009)
- ✗ **T1-010** (from T1-009, R3-A): k=3 + --kv-cache-memory 6.0 / 5.625 / 5.5 GiB.
  - Result: the server never starts (OOM in CUDA graph capture).
  - ✗ **T1-011**: k=3 + KV 5.6 GiB + prefill graph cap 1024.
    - Result: starts, then runtime OOM in the bench.
    - Conclusion: KV ~5.0 GiB is the practical ceiling; k=3 at 16 requests needs smaller per-request state, not more KV (KB16).
- ★ **T1-012** (from T1-009, R3-B v1): GDN spec-state checkpoint + replay (write 1 state per step, not k+1), with a bitwise flush for outside readers.
  - Result: official 749.7 (+2.6% vs T1-009, +18.7% vs BASE); all gates + 128K pass.
  - Conclusion: exact (bitwise unit test with a negative control); ~2 ms/decode step saved, consistent with the microbench and the A/B (KB15).
  - ★ Champion after round 3 (promoted by the human at the resume; the 3% bar that held it back no longer exists).

## Direction R4: stack prior/baseline-run-1's untried wins on T1-012 (round 4, T1/leader-2, from the prior run; no brainstorm)
- ★ **T1-013** (from T1-012): draft lm_head over the first 98304 token ids (N from repo-text coverage, ruling 2).
  - Result: unofficial +1.0% same-job; acceptance unchanged.
  - ○ **T1-015**: k=3 vs k=2 at c=14 (all admitted), the gate for R3-B v2.
    - Result: unofficial +0.1% (acceptance +14.5%, step +14%). Conclusion: R3-B v2 no-go (KB20).
- ★ **T1-014** (from T1-012): --attention-backend TRITON_ATTN (flag only).
  - Result: unofficial +5.5% same-job; removes ~1.2 ms/step of FlashInfer host planning (KB18).
- ★ **T1-016** (T1-013 + T1-014 + Marlin {64,128,128} decode tiling, run-1 patch 0005; Marlin alone +0.45%, noise).
  - Result: official 805.7 (806.1/805.2; +7.5% vs T1-012, +26.7% vs BASE); all gates + 128K pass.
  - ★ Champion after round 4.
- ○ --enable-batch-sharded-sampling: not run (it disables our greedy verify path, KB21).

## Direction R5: the decode all-reduce (round 5, T1/leader-2's next; no brainstorm; builds on T1-016)
- ○ **T1-017** (from T1-016): test-only per-call AR trace (default off).
  - Conclusion: AR is pure PCIe transport at its floor (~21+21 us), ~0 wait, skew ~3 us (KB23).
  - ✗ **T1-018**: N-chunked GEMM + AR pipeline (microbench). Projected ~1% at best (~10 us per extra launch): no-go (KB24).
- ★ **T1-019** (from T1-016 + T1-017's trace): L2 prefetch (evict_last) of the next GEMM's FP8 weights from extra AR-kernel blocks.
  - Result: official 837.0 (+3.9% vs T1-016); ABBA +4.5%; step 29.9 -> 27.7 ms (KB25).
  - ★ **T1-020**: + out_proj/o_proj and lm_head-prefix prefetch.
    - Result: official 850.3 (850.7/849.9; +5.5% vs T1-016, +1.6% vs T1-019 cross-job, +0.8% same-job pooled, bimodal B); all gates + 128K pass.
    - ★ Champion after round 5.

## Direction R6: Marlin decode GEMM efficiency (round 6, T1/leader-2's next; no brainstorm; builds on T1-020)
- ○ **T1-022** (from T1-020): roofline + config sweep + new instantiations + Triton W8A16 (microbench).
  - Result: cold Marlin at 86-92% of the HBM roof; best new config ~0.28 ms/step; Triton 1.4-1.7x slower. No-go (KB28).
- ✗ **T1-021** (from T1-020): prefetch-issue variants (stop-when-AR-done, denser, 256B lines).
  - Result: microbench -15 us/layer, in-server +0.1 ms/step worse. No-go (KB29).

## Direction R7: mixed (prefill-containing) steps (round 7, T1/leader-2; from brainstorm r7: SE + AF + leader-2; builds on T1-020)
- ★ **T1-023** (from T1-020): R7-B GDN replay on the mixed-step path + R7-C every-32 prefill graph grid.
  - Result: official 861.5 (863.1/859.8; +1.3% vs T1-020; ABBA +1.5%); gates + 128K pass. Mixed GDN 6.48 -> 2.63 ms/step (KB34).
  - ○ **T1-024**: R7-A gate (microbench): hybrid DMA/hostar 64-aligned 2-way token split, -13.3%/layer at M=300.
  - ★ **T1-025** (T1-023 + T1-024's design): in-server hybrid split for mixed steps >= 256 tokens.
    - Result: official #2 874.8 (863.7/885.8; trial 1 at T1-023's level); same-job pooled +0.7% (2 ABBAs, B mean above every A); gates + 128K pass. In-chain DMA copies only ~12 GB/s (KB35). (official #1 = infra failure.)
    - ★ Champion after round 7.

## Direction R8: per-server-start bimodality (round 8, T1/leader-2's next; no brainstorm; builds on T1-025)
- ★ **T1-026** (from T1-025): NUMA self-bind of TP workers (generic; opt-out VLLM_NUMA_SELF_BIND=0).
  - Result: official 880.9 (878.4/883.4, CV 0.4%; +0.7% vs T1-025); gates + 128K pass. Cause found: AR slots on the far NUMA node in 7/10 unbound starts (KB37).
  - ★ Champion after round 8.
  - ○ **T1-027**: measurement: split threshold 256 vs off vs 128 on T1-026: +2.0% vs off; 128 worse (KB38).

## Direction R9: Marlin warm-L2 decode restructure (round 9, T1/leader-2's next; no brainstorm; builds on T1-026)
- ✗ **T1-029** (from T1-026): 2-warps/scheduler Marlin prototype (microbench). Correct (1 ulp) but its loads alone are slower than Marlin's full kernel warm: no-go (KB39).
- ○ **T1-028** (from T1-026): skip no-op GDN flush launches. Exact; same-job +0.2%, official 880.0 (879.9/880.1) = T1-026 within noise. Not adopted (no measurable gain) (KB40).

## Direction R10: AR bytes + Marlin remainder waste (round 10, T1/leader-2; from brainstorm r10: SE + AF + leader-2; builds on T1-026)
- ★ **T1-030** (from T1-026): R10-A lossless host-AR payload compression (decode + prefill sizes).
  - Result: official 897.3 (895.1/899.4; +1.9% vs T1-026; same-job +2.0%); gates + 128K pass (KB43).
  - ★ Champion after round 10.
  - ✗ **T1-031**: + compression on the DMA lane of the token split. Bitwise, but chain -1.4%/layer at M=300: dropped.
  - ○ **T1-032**: R10-B Marlin single launch for any M. Exact; kernel -3..-7% at odd-32 M (missed the gate by 1-2 ms); in-server +0.4% (noise). Every-16 grid to 2048 OOMs (KB44).

## Direction R11: stack of small exact wins (round 11, T1/leader-2's next; no brainstorm; builds on T1-030)
- ○ **T1-033** (from T1-030 + T1-028 + T1-032): + every-16 grid over 128-512. Exact, 128K pass; 2 same-job ABBAs pooled -0.07%: null, not adopted (KB45).

## Direction R12: bytes moved per step (round 12, T1/leader-2; from brainstorm r12: SE + AF; builds on T1-030)
- ✗ **T1-034** (from T1-030): R12-B Marlin ALU-slack probe for PFOR FP8 weights: net ~0 (warm GEMMs have no ALU room) (KB49).
- ★ **T1-035** (from T1-030): R12-A lazy GDN checkpoint. Official 906.3 (903.6/908.9; +1.0%); bitwise by construction (KB48).
  - ○ **T1-036**: GDN kernel at 1 wave: null; GDN is at its floor.
  - ★ **T1-037**: + lossless codec on the bf16 target lm_head, decoded inside a custom GEMM.
    - Result: official 922.8 (925.9/919.8; +1.8% vs T1-035, +2.8% vs T1-030); same-job +1.2%; gates + 128K pass (KB50).
    - ★ Champion after round 12.

## Direction R13: GDN state traffic and lm_head, continued (round 13, T1/leader-2's next; no brainstorm; builds on T1-037)
- ○ **T1-038** (from T1-037): lm_head codec v2 (0.705). Kernel -9.4%; official 922.2 = no gain (KB52).
- ✗ **T1-039** (from T1-037): lossless GDN fp32 state codec. Exact but slower (+15.8 us/layer): killed (KB51).
- ★ **T1-040** (from T1-037): RED8 GDN row-dot reduction over 8 lanes (fp32 reorder only).
  - Result: official 929.1 (931.3/926.8; +0.7% vs T1-037); same-job +0.9% (both B > both A); gates + 128K pass; p99 TPOT 38.4 (KB51).
  - ★ Champion after round 13.

## Direction R14: GDN latency hiding, then the largest gap in the step anatomy (round 14, T1/leader-2's next; no brainstorm; builds on T1-040)
- ★ **T1-041** (from T1-040): GDN replay mode 5 (batched prologue + warp-private staging), bitwise. Official 934.1 (+0.5%); premise mostly false, ~1.6 us/layer hideable (KB54).
  - ★ **T1-042**: DMA AR lane pipelined in 4 pieces (bitwise). Official 942.2 (941.4/942.9; +0.9% vs T1-041); same-job +1.3% (KB55).
    - ★ Champion after test-5 round 14 (until T9-005).
    - ✗ **T1-043**: residual add + RMSNorm fused into the host-AR epilogue. ABBA -0.6%: killed (KB56).

## Direction R15: GDN ba-projection skinny kernel (round 15, last; T1/leader-2's next; builds on T1-042)
- ○ **T1-044** (from T1-042): killed before building: the ba GEMM already overlaps the qkvz Marlin; upper bound ~1.8 us/layer (KB58).
- ○ **T1-045** (from T1-042): token-split threshold 192/128 vs 256: -0.5%/-0.6%; keep 256.

# Run test-6 (v2: explore teams T2, T3, T4)

## T2 (explore, leader-2): spatial prefill/decode overlap — CLOSED 2026-11-09 (reports/T2-writeup.md)
- ○ **T2-001** (from T1-042): green-context co-run microbench; pure Marlin decode x1.31 at f=48.
- ✗ **T2-002** (from T1-042): in-server synthetic lane: decode ITL x1.65 at f=48; kill gate failed.
- ○ **T2-003, T2-004**: ablations: interference is SM-proportional, zero-sum (KB59).
- ✗ **T2-005**: anatomy on an unbuilt workspace: INVALID (KB61).
- ○ **T2-006** (from T1-042): critical-path anatomy of decode and mixed steps; remaining exact mixed-step items +1-2% total.

## T3 (explore, leader-3): draft tokens per step — CLOSED 2026-11-09 (reports/T3-writeup.md)
- Direction T3-1: self-distilled MTP head — CLOSED by the team's kill criterion.
  - ○ **T3-001**: offline pipeline matches in-server acceptance.
  - ✗ **T3-002, T3-003, T3-004**: fine-tunes (hard-label, LoRA/step-2 layer, KL distill): no held-out gain (KB60).
- Direction T3-2: confidence-budgeted MTP verification (K_MAX=3, B=2*num_reqs) — stopped below the bar (~+3%); design kept.
  - ○ **T3-005/006**: offline E +8.4% (K_MAX=3) at M=48; confidence ranks acceptance well.
  - ✗ **T3-007**: in-server step costs, but on an UNBUILT workspace: costs invalid (KB61); acceptance stands.

## T4 (explore, leader-4): AR hidden under consumer GEMMs — CLOSED 2026-11-09 (reports/T4-writeup.md)
- ✗ **T4-001** (from T1-042): premise microbench: AR/GEMM overlap saves <= 5-8 us/AR (go bar 15); KB62.
  - ○ **T3-008**: built-workspace costs; K_MAX=3 OOMs in graph capture at every viable KV size; ~+3-3.5% e2e.

## T5 (explore, leader-5): deeper exact speculation on the built stack — CLOSED 2026-11-09 (reports/T5-writeup.md)
- ○ **T5-001** (from T1-042): k=3 vs k=2 same memory +3.1%; cap-1024 penalty ~4%.
- ○ **T5-002, T5-003**: Marlin 4-m-block cliff +3.0 ms in situ; tile override -0.76 ms cold.

## T6 (explore, leader-6): Marlin slab re-streaming at M=49-256 — CLOSED 2026-11-09 (reports/T6-writeup.md)
- (T5) ✗ **T5-004**: fixed k=3 + tile override -3.5%; cap penalty not real. ○ **T5-005**: two-slot GDN, bitwise (asset). ✗ **T5-006** (from T6-004): budgeted K_MAX=3 at M=48 E +5.6% / step +3.8%; M=64 cliff +3.4 ms in situ (KB63).
- (T6) ○ **T6-001..004**: Marlin slab policy, M=72-120 -21..-35% microbench (KB64). ✗ **T6-005**: in-server null at k=2; piggyback -30%.

## T7 (explore, leader-7): hiding prefill-size ARs in small mixed steps — CLOSED 2026-11-09 (reports/T7-writeup.md)
- ✗ **T7-001** (from T1-042): split-half RS/AG on copy engines: AR hidden -19%..16% (gate 60%); KB65.
- (T6) ✗ **T6-006** INVALID (unbuilt). ○ **T6-007**: mixed M=96 kernel floor ~42 ms -> piggyback cannot pay.

## T8 (explore, leader-8): 2-warps/scheduler decode GEMM — CLOSED 2026-11-09 (reports/T8-writeup.md)
- ○ **T8-001**: PCIe page size / interference (KB66). ○ **T8-002**: Marlin loop 68.4% vs 91.5% peak at 2 warps/sched (KB66).

## T9 (explore, leader-9): copy-engine AR overlap in mixed steps — CLOSED 2026-11-09 after its win (reports/T9-writeup.md)
- ○ **T9-001** (from T1-042): NUMA-verified bench: 86-96% of AR hidden beyond champion split at R>=256 (KB67, corrects KB65).
- (T8) ✗ **T8-003**: 2-warp kernel 129.5 vs Marlin 87.8 us warm; M1 gate failed (KB68).

## T10 (explore, leader-10): survey; no direction — CLOSED 2026-11-09 (reports/T10-writeup.md)
- ✗ **T10-001**: e4m3 MMA accumulator ~13 bits: FP8-MMA/Ozaki cannot be exact (KB69).

## T11 (explore, leader-11): prefill lane in decode AR windows — CLOSED 2026-11-09 (reports/T11-writeup.md)
- ✗ **T11-001**: AR-window prefill slabs: chain +20..33% (KB70).

## T12 (explore, leader-12): stacked package, cheaper drafter — CLOSED 2026-11-09 (reports/T12-writeup.md)
- ✗ **T12-001**: CPU kill: free n-gram drafting E +0.7..2.7%. ✗ **T12-002**: freq-ranked 32K vocab + FP8 fc -7.3% (KB71).

## T13 (explore, leader-13): opened 2026-11-09, direction not yet chosen
- (T9) ○ **T9-002**: real GDN core: 39-58% hidden beyond champion split. ○ **T9-004**: in-segment variant S3c bench.
  - ★ **T9-005** (from T1-042): S3c in-segment copy-engine AR overlap in mixed steps. Official 948.9 (947.9/949.8; +0.7%); ABBA #1 +1.2%, ABBA #2 +0.7%, all B > all A.
    - ★ Champion (test-6) until T14-004.
- (T9) ✗ **T9-006**: decode CE AR +23 us/AR vs hostar. ✗ **T9-007**: CE payload codec slower at every R.

## T13 (explore, leader-13): producer-fused AR — CLOSED 2026-11-09 (reports/T13-writeup.md)
- ✗ **T13-001**: posted writes slow the GEMM ~+9 us; split-K outputs land at the end (KB72).

## T14 (explore, leader-14): mixed steps at compute roof; landed Marlin policy on T9-005 — CLOSED 2026-11-09 after its win (reports/T14-writeup.md)
- ✗ **T14-001/002** (from T9-005): G1 failed (Marlin critical path -0.59 s/trial vs 1.5 gate).

## T15 (explore, leader-15): survey; no direction — CLOSED 2026-11-09 (reports/T15-writeup.md)
- ✗ **T15-001**: HMMA+FFMA do not co-issue. ○ **T15-002**: PCIe duplex; hybrid CE+SM AR floor ~21-25 us (KB73).

## T16 (explore, leader-16): hybrid CE/SM decode all-reduce — CLOSED 2026-11-09 (reports/T16-writeup.md)
- ✗ **T16-001**: hybrid AR +11 us/AR vs hostar; CE start latency ~20 us in graphs (KB74).
- (T14) ○ **T14-003**: partial ABBA (job 119) +0.2/+0.5%.
  - ★ **T14-004** (from T9-005 + T6-004): T6's Marlin slab policy on the champion. Official 956.6 (955.8/957.5; +0.8%); ABBA job 120 +0.9% all B > all A.
    - ★ Champion (test-6) until T27-002.

## T17 (explore, leader-17): roofline decode GEMM M<=64 + k=3 — CLOSED 2026-11-09 (reports/T17-writeup.md)
- ✗ **T17-001**: M0 killed; write-flush artifact behind the premise (KB76, method correction).

## T18 (explore, leader-18): CPU-only survey — CLOSED 2026-11-09 (reports/T18-writeup.md; KB75)

## T19 (explore, leader-19): new-request path — CLOSED 2026-11-09 (reports/T19-writeup.md)
- ○ **T19-001** (on T14-004): in-situ step model; lifecycle <=1%; coalescing ~0 (KB77).

## T20 (explore, leader-20): fence-free LL decode AR package — CLOSED 2026-11-09 (reports/T20-writeup.md)
- ✗ **T20-001**: LL AR exact, in-chain only 2-3 us/AR (KB78).

## T21 (explore, leader-21): hybrid prefix-state cache + prompt-as-draft feeding — CLOSED 2026-11-09 (reports/T21-writeup.md)
- ○ **T21-002**: generic prefix-state cache works, +0.5% n=1 (KB86).
- ○ **T21-001**: stock fine-grained prefix flags hit 144/104,718 tokens (validity PENDING RULING).

## T22 (explore, leader-22): HBM-continuous decode (in-kernel prefetch) — CLOSED 2026-11-09 (reports/T22-writeup.md)
- ○ **T22-001**: decode GEMM warm floors (KB79). ✗ **T22-002**: down-prefix prefetch ABBA -0.9%.

## T23 (explore, leader-23): stacked exact overhead items — CLOSED 2026-11-09 (reports/T23-writeup.md)
- ○ **T23-001**: CPU attribution (drafter at floor; KB80). ✗ **T23-002**: dead-block skip <=1.0 ms/step. ✗ **T23-003**: remainder re-stream, T6 choice holds.

## T24 (explore, leader-24): L2-warm M<=48 decode GEMM — CLOSED 2026-11-09 (reports/T24-writeup.md)
- ✗ **T24-001**: premise failed (KB81).

## T25 (explore, leader-25): GPU1-aware MLP rebalance package — CLOSED 2026-11-09 (reports/T25-writeup.md)
- ○ **T25-001**: per-GPU Marlin tables. ✗ **T25-002**: uneven split ABBA -0.4%. ✗ **T25-003**: decode-even split ABBA -0.7% (KB82).

## T26 (explore, leader-26): swap-AB register-direct decode GEMM — CLOSED 2026-11-09 (reports/T26-writeup.md)
- ✗ **T26-001..003**: loop 93% peak, full kernel slower than Marlin (KB83).

## T27 (explore, leader-27): budgeted K=3 + PCA-screened draft lm_head — CLOSED 2026-11-09 after WIN (reports/T27-writeup.md)
- ✗ **T27-001**: budgeted K_MAX=3 decode-only -2.6% (KB85). ✗ **T27-003**: penalty localised to adaptive plumbing.
- ★ **T27-002** (on T14-004): PCA-screened FP8 draft lm_head, official 967.3 (+1.1%) — CHAMPION (KB84).
  - ★ **CHAMPION** (test-6, promoted 2026-11-09). Path: BASE → … → T1-042 → T9-005 → T14-004 (kernel T6-004) → T27-002 (screen ruling R1; PCA data from T3).

## Resume (2026-11-09): T28 (slot A), T29 (slot B) opened; directions pending proposals.
- T29 (explore, slot B) direction: checkpoint W8A8 for mixed/prefill steps
  - T29-001/002 Triton all-M W8A8 (wash) ; T29-003 Marlin-layout W8A8 M>96 (ABBA +9.2%) -> T29-004 1045.8 (builds_on T27-002) -> T29-006 1062.1 (exact bundle) ; T29-005 chunk ordering (noise)
  - champion path: ... T14-004 -> T27-002 -> T29-004 -> T29-006 -> **T31-003**
- T28 (explore, slot A, CLOSED) direction: small-M W8A8 + exact k=3/4 ; T28-005 combined k=3 (null) ; T28-007 spill-fix (failed, structural) ; T28-008 e4m3 at k=2 (loss -1.84%)
  - T28-001/002 custom standard-layout kernel (lost to Triton at M<=64, parked); T28-003 k=3 + two-slot GDN on W8A8 (ABBA vs k=2 W8A8 +2.93%); T28-004 Marlin-e4m3 kernel (M=64 warm 13.39 vs Marlin M=48 13.75 ms)
- T30 (explore, slot B, CLOSED): two-wave layer-pipelined mixed steps
  - T30-001 AR-off floor ; T30-002 attribution (AR fully exposed) ; T30-003 finer DMA pieces (null) ; T30-004/005 2-wave (in-server -7.5%, host-bound) + k-wave host-free (k=3 best, 26 ms above floor) -> closed
- T31 (explore, slot A, CLOSED): register-direct e4m3 decode GEMM cashing AR L2 prefetch
  - T31-001 L2/prefetch budget ; T31-002 kernel v2 (gate_up wall) -> **T31-003 CHAMPION 1088.9** (builds_on T29-006)

## Resume 3-4 (2026-11-10): external teams T32/T33, last explore team T34 — all CLOSED, no wins (champion stays T31-003)
- T32 (external, slot C, CLOSED; reports/T32-1.md): scout deck T32-1 -> no card clears +5% (T32-1/2); sparring revised to "no measured path" (T32-3).
  - T32-001 (RS+AG FP8 input vs AR: ~8% slower, inconclusive) ; T32-002 Stage 0 GDN prefill CUDA-graph capture (bitwise 15/15, host 0.51 -> 0.03 ms/layer, GPU +0-3%) ; T32-003 captured GDN in-server (bitwise, host -33 ms/mixed step, ABBA +0.29% neutral) — closed: wave-eligible steps are only 14-17% of GPU time (ceiling +1.6-2.0%).
- T33 (external, slot C, CLOSED; reports/T33-1.md): deck T33-1 -> card 13 (lossless FP8 weight code) tested: 0.8925 bytes bit-exact, loses where AR prefetch makes weights L2-resident; overlap-only decode GEMM stack ~+1-1.5% (T33-001..007; KB101-103).
- T34 (explore, slot A, CLOSED; reports/T34-writeup.md): hybrid CE-write + SM-read mixed-step AR reaches the PCIe duplex floor but saves 3-39 us/AR (champion transport within ~11% of floor; KB104-106). Job 345 crash required a server reboot (KB106).
- ★ CHAMPION unchanged: T31-003 1088.9 (path: ... T27-002 -> T29-004 -> T29-006 -> T31-003).

## Moonshots (2026-11-10): T35, T36 (closed, no wins), T37
- T35 (moonshot, slot A, CLOSED): run-ahead L2 weight streamer — refuted (T35-003; KB108). T35-001/002 screen negatives (small token budgets, k=3 cross-job).
- T36 (moonshot, slot D, CLOSED): confidence/load-chosen speculation depth — withdrawn after past records; T36-001 (K sweep, c=14 pair), T36-002 (decode AR anatomy, KB109).
- T37 (moonshot, slot D): static row-budgeted step -> redefined stage 1: two-slot GDN state + k=3 with 16 running (T37-001 measurements; boundary defect found and fixed, KB110) -> ★ **T37-002 CHAMPION 1142.2** (builds_on T31-003 + T28-005's two-slot code).
  - champion path: ... T29-006 -> T31-003 -> **T37-002**

- T38 (moonshot, slot E, CLOSED 2026-11-11; reports/T38-writeup.md): T38-1 (decode AR non-barrier, withdrawn pre-build; T4-001 ceiling) -> T38-2 (self-distilled draft head, withdrawn; T3 records) -> T38-3 (PP=2 engine + draft trees): PP prefill 0.51x TP (T38-004) but decode r=0.91-0.92 < 0.93 needed (T38-005); no win (KB112).

## Research teams (2026-11-11): T42-T49
- T42 (CLOSED): certified exact target lm_head (7-bit plane + enclosure): exact, ~0.8% of decode, KB121. T43 (CLOSED): row-selective AR; decision depth/oracle bound, KB114. T44 (CLOSED): SM PCIe duplex when paced (KB115), no target on champion (KB116); CE-AR reorder slower (KB117 superseded).
- T45 (research): row-pricing rule (KB118); T45-003/004 gate_up band bench (KB120) -> ★ **T45-005 CHAMPION 1173.4** (builds_on T37-002): gate_up at 48<M<=96 via w8a8_marlin.
  - champion path: ... T31-003 -> T37-002 -> **T45-005**
- T46-T49 (research, second set, all CLOSED):
- T46 (CLOSED): host as a memory tier: exact host embedding (frees 1.27 GB/GPU), link adds ~2.4% weight bandwidth, ceiling ~1% (KB125). T47 (CLOSED): asymmetric waves failed its gate (KB128); DRAM saturates from 16-32 SMs, KB59 zero-sum is a kernel property (KB129). T48 (CLOSED): compression-ratio rule (KB126), k=4 recompute (KB127). T49 (CLOSED): PP one microbatch (KB123), compressible memory (KB124).
- T50 (exploit, opened 2026-11-11): (CLOSED) T50-001 gate_up cfg 103 + T42-003 lm_head -> T50-002 CHAMPION 1181.3 (builds_on T45-005 + T42-003). champion path: ... T37-002 -> T45-005 -> **T50-002**
- T51 (exploit-long) CLOSED: T51-001..003 priced piggyback prefill in a mixed pass; existing kernels fail the GEMM-band gate (0.07 vs 0.05 ms/row); funded 25 GPU-min for a TP-shape kernel prototype.
- T52 (exploit-long, megakernels CLOSED direction set by the human, opened 2026-11-11): running.
- T53 (exploit-long, megakernels direction set by the human, opened 2026-11-11): CLOSED (decode megakernel ceiling too low, KB132).
- T52 (exploit-long, megakernels CLOSED): T52-001/002 decode fusion at parity; T52-004 mixed-step AR exposure is a scheduling effect -> ★ **T52-005 CHAMPION 1212.4** (builds_on T50-002 + T30-004/005 waves + T32-003 captured core). champion path: ... T45-005 -> T50-002 -> **T52-005**
- T54 (research-handoff, memo research_handoff_1113.md, opened 2026-11-13): CLOSED after triage (KB135).
- T55 (research-handoff, memo research_handoff_1113.md, opened 2026-11-13): CLOSED after triage (KB134).
- T56 (sweep, framework/, opened 2026-11-13): CLOSED, no win (KB136).
- T57 (sweep, framework/, opened 2026-11-13): CLOSED (KB137).
- T58 (sweep, updated leader_sweep prompt, opened 2026-11-13): CLOSED (KB137).
- T57-004 ★ CHAMPION 1225.5 (builds_on T52-005 + T56 fc-table idea): drafter package (fc table, packed argmax, glue tiles). champion path: ... T50-002 -> T52-005 -> **T57-004**
- T58-001 ★ CHAMPION 1229.6 (builds_on T57-004): L2-prefetch size gate 1 MiB -> 2.7 MB for mid-size hostar ARs. champion path: ... T52-005 -> T57-004 -> **T58-001**
