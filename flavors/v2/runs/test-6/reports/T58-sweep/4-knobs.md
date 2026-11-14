# T58 sweep, step 4: the knob-by-knob sweep

Phase 1 (no run history). Inputs: 2-graph.md (sites), 3-calibration.md (measured times).

**Reference times.**
- Trial: **75.9 s** (job 494, 1225 tok/s).
- Decode step: **24.7 ms** at 64 rows (16 requests x 4).
- Decode-like time per trial: ~55 s median-based, ~62 s with the tail. Prefill-path time: ~10-14 s. Unexplained tail/idle: up to ~12 s.
- A decode-step saving of 1 ms is worth ~2.5 s per trial, about +3.3%.

**Gain** is napkin speedup of the whole trial vs. the champion. Every row gives:
- the site (node / edge / index);
- the napkin;
- the condition under which it wins;
- a verdict: **C** = candidate, **P** = pruned, with the reason.

## Knob 1: Keep / Recompute (data nodes across steps)

| # | Site | Options (02's table) | Napkin | Verdict |
|---|---|---|---|---|
| K1 | k,v[r,l,j] of 16 attn layers | keep (champion) / recompute | 16 KB per token per rank, ~0.13 GB at c=16: keep wins by 02 §3 | P (champion is the winner row) |
| K2a | GDN state S[r,l]: lazy checkpoint cap | champion: Row B (ReplaySSM) with log cap 8 tokens, so S is stored every ~2nd step (pending ~2.7 + accepted ~2.7 + 4 new > 8). Cap 16-24: stored every ~4-6 steps | writes 0.57 -> ~0.2 GB per step = **-0.35 ms**. Extra log reads (+~0.08 GB, ~0.08 ms) and replay FMAs (~6 tokens x 65 kFLOP per head, ~0.1 ms spread). Net **~-0.2 ms per step = +0.6-0.8%** | C (small). Wins if the replay-kernel time stays memory-bound. Log fits: (512+256+8) B per token per head, so ~80 tokens fit in the 64 KB slot S[1] |
| K2b | S: Row C (keep raw k,v history, no S) | crossover at prefix ~d_k = 128 tokens; contexts here are 260-500+ | Row C reads 250 KB vs S 64 KB per (r,l,head) | P (dominated for t > d_k) |
| K2c | S in lower precision / compressed | fp32 required (exactness); fp32 mantissas are incompressible | — | P (contract) |
| K3 | target logits, hidden states | no future readers (02 Rule 1) | — | P |
| K4 | cross-request keep: shared prefix (system prompt ~45 tokens x 400 requests = ~18K of 104.5K prefill rows) | recompute (if the hybrid model's prefix cache can't match 45 tokens: mamba-state checkpoints only at block boundaries) / keep S + KV at the prefix boundary | 18K rows x ~50-110 us per row = **-0.9 to -2.0 s = +1.2-2.7%** | C. Wins if the champion does not already hit the prefix (enable_prefix_caching=True, but hybrid-model match granularity unknown). Ruling 2026-11-09 allows a generic fine-grained mechanism, not tuned to 45 tokens |
| K5 | drafter pass-1 inputs (target hidden at all 4 rows) | already kept within the step | — | P |
| K6 | drafter fc on embeddings: table T[v] = norm(E[v]) W_fc,e (Keep a v-indexed product; Rewrite by distributivity) | 248K x 5120 bf16 = 2.5 GB/rank, host only | saves half of the fc (13 MB/pass, x3) = ~0.04 ms; gather from host adds latency | P (< 0.2%) |

## Knob 2: Rewrite (exact algebra)

| # | Site | Rewrite | Napkin | Verdict |
|---|---|---|---|---|
| R1 | AR -> residual add -> RMSNorm -> W8A8 quant (130 sites/step) | fuse the add+norm+quant into the AR reader's epilogue. It already holds both partials; the 1/rms-past-linear ruling allows the per-row scale to be applied late | removes ~130 norm launches (0.42 ms) + quant passes (~0.3 ms) of the measured 0.92 ms: **-0.5 to -0.7 ms/step = +2-2.8%** | C. Needs a per-row reduction across the AR's element-chunked blocks (a row spans blocks): either row-aligned chunks or a second tiny pass. The quant producer must stay bitwise (rulings) |
| R2 | weight bytes W[l] (12.18 GB/step): exact entropy re-encoding of the FP8 weights (generalizes the champion's lm_head codec) | Huffman/ANS or block-exponent codes on e4m3; ratio maybe 0.85-0.9 (unmeasured) | if GEMMs were DRAM-bound: -10-15% of 12.5 ms = **-1.3 to -1.9 ms (+5-7%)**. But decode GEMMs are latency/occupancy-bound cold (60-70% of DRAM) and lean on L2 prefetch in situ, so fewer bytes only help if the decoding GEMM keeps its occupancy | C (speculative). Riskiest: the compression ratio of e4m3 weights (cheap to measure offline, no GPU) and in-register decode cost |
| R3 | GDN at 4 rows per request (verify) | recurrent vs chunk/dual form (03 §4.2): M=4 << d=128, so the recurrent/attention form wins. Champion uses the recurrent form | — | P (champion is the winner row) |
| R4 | split K instead of N for a GEMM so a norm can run on half rows (removing an AR) | the rms needs the full row (a link round trip per layer anyway), and qkvz partials would need their own AR | more link ops, not fewer | P |
| R5 | start the next GEMM on (x_prev + own partial) before the AR finishes; add W·(peer partial) after (distributivity over the AR's sum) | W8A8 dynamic quant of the sum != sum of quants (contract); bf16 path: 2nd GEMM (+46 us hot) to hide 1 AR (~50 us) | break-even at best | P |

## Knob 3: Speculate

| # | Edge | Gates | Napkin | Verdict |
|---|---|---|---|---|
| S1 | x[t+1] chain depth k=3 -> 4 at full batch | exact (champion's MTP) | extra row per request: +16 rows. GEMMs +~6% (0.8 ms), AR bytes +25% (0.65 ms), GDN rows (+0.4), one more drafter pass (+0.55), lm_head (+0.2): +2.6 ms (+10.5%) for +0.28 tokens/req (p4 ≈ 0.39 x 0.71) = +10.4% tokens | P at full batch (break-even). Also needs the GDN kernel's kMaxLogTokens 4 -> 5 |
| S2 | adaptive depth by load: k up to 5-6 when few requests run (end-of-run tail and low-D steps: D <= 36 ~5 s per trial, D 37-56 ~8 s) | exact; allowed (clarification 2026-11-10) | at 1-9 requests the step is 19-20.5 ms and nearly flat in rows (~0.15 ms per extra row). k 3 -> 6 adds +0.63 tokens/req (+23%) for +2-5% step time. Gain on ~5 s of tail: **~-0.8 to -1.0 s = +1.1-1.3%** | C. Needs the GDN replay kernel for more than 4 tokens per step and drafter graphs at those sizes |
| S3 | per-request confidence-based depth / tree at full batch (EAGLE-2-like dynamic slots) | exact | an extra row pays if it brings > ~0.27 accepted tokens. Rows chosen where the drafter is confident (P(reach pos 4) ~0.5 x p4 ~0.8 = 0.4) beat the uniform 4th row (0.28) | C (+1-3% if drafter confidence is calibrated). Riskiest: calibration of drafter confidence; variable rows per request in captured graphs |
| S4 | better drafter (retrain MTP on self-generated data, ruling 2026-11-08) | Gate 2 fine; ML layer | conditional acceptance 0.71-0.77 -> 0.8: tokens/step 2.69 -> ~2.95 = **+9%** | P for this team (training GPU time >> 100 GPU-min). Noted for the run |
| S5 | high-cardinality edges: AR outputs, layer outputs, hidden states | Gate 2: p ≈ 0 | — | P (04 §3) |
| S6 | draft-of-draft: run drafter passes 2 and 3 in one window on a guessed input (drafts need no verification) | Gate 1: shares the W_mtp read (0.21 GB); Gate 2: no good guess for x̂[t+2] before pass 2 | saves ≤ 1 pass (~0.5 ms, 2%) at a large acceptance cost | P |
| S7 | drafter pass 1 run on the drafts (known before verification) in the window of the target lm_head | same resource (DRAM) on both sides: no Mechanism-2 overlap | — | P |

## Knobs 4/5: Split / Merge and Place

| # | Index / placement | Napkin | Verdict |
|---|---|---|---|
| P1 | PP=2 instead of TP=2 (layers placed by halves) | 1 microbatch: two serial stages each reading 12.2 GB at full layer width: >= 25.4 ms + GDN + small, vs TP 24.7 including all ARs. 2 microbatches: each GPU reads its weights twice per step (Shared-Read Rule): ~25 ms per GPU. Decode GEMMs are only 60-70% DRAM-efficient cold, so the real PP step is ~35 ms | P. Flips only if the AR cost per step exceeds the weight time per GPU (~12 ms; now 7.3 ms) |
| P2 | replicate the GDN mixer (qkvz + out) to drop its 48 ARs | +2.76 GB weights per GPU (needs KV budget cut 5 -> ~2.3 GiB; 128K needs ~2.1 GB KV/rank) -> +2.9 ms DRAM vs -48 x 56 us = -2.7 ms | P (break-even/loss) |
| P3 | replicate attention qkv/o (drop 16 ARs) | +0.84 GB -> +0.88 ms vs -0.9 ms | P (break-even) |
| P4 | asymmetric TP split α ≠ 0.5 (05 §4.1.3) | measured GPU1/GPU0 ratio 0.99-1.01 in situ, no rank skew; prefill compute 283 vs 290 TF (2%) -> α* ≈ 0.494, gain ~0.1 s | P (nothing to balance; the ledger's "15% slower" does not show at these clocks/loads) |
| P5 | split r into 2 groups at decode, staggered so one group's AR runs under the other's GEMM (DualPipe, 06 §3) | Shared-Read Rule: the second group re-reads weights (L2 hot only for slices < 72 MB). Measured GEMMs are per-launch latency-bound (cold qkvz 70 us at M=16 and at M=64; hot 46-52): 2 groups ≈ +200 us GEMM per layer to hide ~110 us of AR | P (decode GEMMs latency-bound) |
| P6 | split d (output columns) of out_proj/down and stream each column chunk's AR under the next chunk's GEMM (Split d + Window, Link || HBM) | per AR saves ≤ its bandwidth part (~20 us at 64 rows); the fixed 25 us remains on the last chunk; loses part of the champion's prefetch-during-AR (~1.6 ms per step of DRAM work) | C (marginal): napkin -2.6 ms + 1.6 ms = **-1 ms (+3-4%)**, high build cost (GEMM and AR sharing SMs or DMA lanes) |
| P7 | merge prefill rows into decode steps (chunked prefill, 06 §2.1) with a token budget near the ridge (~172 rows W8A8) | steps with P <= 191 cost the same host-side time as decode steps (24.5 vs 24.7 ms). P 192-767 adds ~+15 ms per step and P >= 768 adds ~+100 ms (host-side, calib). If GPU-side marginal costs hold (napkin 80-95 us/row: GEMM ~40-50 below the ridge + exposed AR bytes ~43), chunking the big prefills into decode-riding pieces saves where current steps cost more per row | **C, pending GPU-side mixed-step times** (3-calibration §4; idle.md). See §"Mixed-step candidates" |
| P8 | place the drafter on one GPU (no TP for the drafter: no drafter ARs/all-gathers) | drafter weights 2x per GPU: +0.54 ms GEMM vs -0.39 ms of ARs/gathers | P |
| P9 | context-parallel / KV offload | no capacity pressure at c=16 | P |

## Knob 6: Window (including overlaps the framework prices with a latency-free link)

| # | Site | Napkin | Verdict |
|---|---|---|---|
| W1 | AR protocol fixed latency: 130 ARs x ~25 us = **3.3 ms per decode step (13%)** | The framework's link node costs Link_bytes/Link_BW only: **it has no per-transfer latency term**, so it cannot see this. Copy-engine memcpy has ~3 us latency (calibrated). An AR at ~10 us fixed + bytes/24 GB/s would save ~15 us x 130 = **-2 ms/step = +6-8%** | **C (top)**. Riskiest: what the 25 us is. Candidates: arrival skew between ranks (per-call difference ±5 us, so unlikely the main part); the L2-prefetch blocks' issue time being counted inside the AR kernel; __threadfence_system + flag polling over PCIe; graph launch. Cheapest test: 2-process microbench of hostar_kernel alone (prefetch on/off, R=4..64), then the same with a DMA/mem-op protocol |
| W2 | L2 prefetch during the AR (Link ‖ HBM, champion) | prefetch amount per AR is bounded by AR time x DRAM BW (~45 MB per 50 us). If W1 shortens the AR, less prefetch fits, and GEMMs fall back toward 60-70% DRAM efficiency (cold) | coupling note: W1's net gain = AR time saved - prefetch lost. Napkin net ~**-1.0 to -1.5 ms** if the prefetch is not replaced. Pairing W1 with better cold-GEMM DRAM efficiency (W3) recovers it |
| W3 | decode GEMM DRAM efficiency (cold: 580-670 GB/s = 60-70%) | if the decode GEMMs streamed at ~90% of DRAM without help, 12.18 GB = 14.1 ms cold vs ~20 ms cold now. In situ they already get 12.5 ms via prefetch. Alone it frees the AR's prefetch role, so it pays only with W1 (makes W1's -2 ms real) | C (enabler for W1). Riskiest: whether a 64-row FP8 GEMM can keep enough bytes in flight (deeper cp.async pipelines, split-K across all 128 SMs) |
| W4 | GDN replay kernel (DRAM: state) ‖ GEMM (DRAM: weights) | same resource | P |
| W5 | piecewise decode graphs with eager attention in between (16 gaps/step) | measured GPU idle 0.16-0.22 ms/step | P (≤ 0.9%) |
| W6 | host/GPU idle | **measured GPU-side (calib/idle.md): utilisation 98.0-98.4%, idle 1.6-2% (~1.4 s), 64% of it kernel-to-kernel gaps < 0.1 ms; no gap >= 0.5 ms in decode steps.** The "12 s tail" of the host-side step log was mixed-step GPU work misattributed by async scheduling | **P** (<= +2% even if all idle vanished) |
| W7 | eager wave steps (P >= 768, ~33 per trial, 125 ms host-side each = 4.4 s) | host-bound per the champion's own record (host 120 vs GPU 131 ms at the threshold). Capturing the wave steps as graphs, or replacing them by chunking (P7), removes host time | C (part of P7/W6) |

## Prefill GEMM throughput (mixed steps, ~10-14 s per trial)

| # | Site | Napkin | Verdict |
|---|---|---|---|
| G1 | FP8 MMA with f16 accumulation within each 128-K block (657 vs 329 TF measured) | prefill GEMM time ~96 us/row x 104.5K rows ≈ 10 s; halving the MMA time is worth maybe -3 to -5 s = **+4-6%** | C, **needs a ruling**. task_definition: "accumulation within each 128-element K-block may use the FP8 tensor cores' own accumulator". Is the f16 accumulator that? Risks: products up to 448² = 2e5 overflow fp16 (65504), so an exact-overflow fallback is needed; 11-bit accumulation within a block. GSM8K FP8 A/B required |
| G2 | in-situ bs_gemm efficiency (152-222 TF in mixed steps vs 255-290 microbench) | 20-25% of prefill GEMM time ≈ 2-2.5 s if fully recovered | C (unclear cause: contention with DMA ARs, waves, smaller M per wave) |

## Lm_head

| # | Site | Napkin | Verdict |
|---|---|---|---|
| L1 | target screen plane 7 -> 4 bits (0.57 -> 0.32 GB) | -0.25 ms/step screen; candidate sets grow (host gathers) | C (small, +0.5-1%) |

## Summary of surviving candidates (napkin, vs. 75.9 s trial)

| Rank | Candidate | Napkin gain | Confidence |
|---|---|---|---|
| 1 | W1 (+W3) lower-latency AR protocol (+ GEMMs that don't need the AR's prefetch) | +4-8% | medium: the 25 us is measured; its cause is not |
| 2 | W6 host/GPU idle tail | 0 to +8% | unknown until idle.md |
| 3 | P7/W7 prefill chunking / big-step path (>= 768 waves cost ~3x per row) | +2-4% | low: host-side times |
| 4 | G1 f16-accumulate FP8 MMA in prefill | +4-6% | needs a ruling |
| 5 | R1 AR epilogue fuses add+norm+quant | +2-2.8% | medium |
| 6 | P6 column-streamed AR under GEMM | +3-4% | low (overlaps with W1/W2) |
| 7 | R2 lossless FP8 weight compression | 0 to +7% | low |
| 8 | K4 fine-grained prefix keep of the shared system prompt | +1.2-2.7% | depends on the champion's current hit rate |
| 9 | S3 confidence-based dynamic draft slots | +1-3% | low |
| 10 | S2 load-adaptive depth in the tail | +1.1-1.3% | medium |
| 11 | K2a larger lazy-checkpoint cap | +0.6-0.8% | high (small) |
| 12 | L1 4-bit screen plane | +0.5-1% | medium |

Framework notes (where it could not express something or was wrong for this machine):
- **No latency term on transfer nodes.** The cost formula prices Link_bytes/Link_BW only. Here a fixed ~25 us per AR x 130 per step is the single largest item the napkin missed (W1).
- **GEMMs below the ridge are not at DRAM BW.** The framework's "333-row rule" assumes the HBM-bound side runs at BW. Decode-size FP8 GEMMs on this GPU run at 60-70% of DRAM cold. This decides P5 (row-split overlap) and couples W1 with W2/W3.
- **Ridge for this machine is ~172 rows** (W8A8, f32 accumulate, measured 329 TF), not 330. The accumulation precision rule (G1) is a contract term that changes F by 2x; the framework treats F as a hardware constant.
- **Host time is not a resource in the window model.** Eager steps, scheduling and request admission cost GPU idle (W6). The framework has no node for them.

## Addendum (still phase 1, before reading any history): GPU-side mixed-step decomposition

Source: calib/mixed_vs_decode.md (node trace, GPU1). 5 mixed steps (P 60-68, D 60; span 43 ms) vs. 5 neighbouring decode steps (span 24.5-26.6 ms). Extra +18.3 ms:

| Item | Extra per mixed step | Mechanism (read from the code) |
|---|---|---|
| gate_up | +4.4 ms (80 -> 152 us/call) | M = 124-128 > 96 selects bs_gemm instead of t31 for ALL rows; and the L2 prefetch inside the AR is off: `VLLM_HOSTAR_L2PF_MSG_MAX` = 1 MiB, while a 128-row AR is 1.31 MB |
| other body GEMMs | +4.3 ms (t31 34 us -> bs_gemm 72 us/call) | same: path switch at M > 96 + no prefetch |
| hostar ARs | +5.0 ms (52 -> 88 us/call, 141 calls) | twice the rows: **~0.56 us per row per AR = ~79 us per row per step**, vs napkin 43 (compression or bandwidth lower for prefill payloads?) |
| GDN chunked prefill + eager torch glue | +2.7 ms (387 + 741 eager launches) | the prefill request's GDN goes through the eager chunked path (T32's captured core is only used in wave steps?) |
| idle, attention, norms, memcpys | +1.8 ms | |

So the "fixed 12.5 ms" of a mixed step (3-calibration §4b) is mostly the **M > 96 path switch**: GEMMs leave the tuned decode path and lose the AR prefetch. On top come the eager GDN prefill glue and the AR bytes.

New candidates from this (priced on the GPU-side table, 3-calibration §4b):

| # | Site | Napkin | Verdict |
|---|---|---|---|
| M1 | AR-time L2 prefetch for mid-size ARs (MSG_MAX 1 MiB -> ~2.6 MB, i.e. up to 255 rows). The 128-row AR lasts 88 us: room to prefetch ~80 MB | Decode shows what prefetch does: gate_up 138 cold -> 80 us; qkvz 70 -> 35. At M=128, maybe -40 to -60 us per layer, so **-2.5 to -4 ms per small mixed step**. 184 such steps (P <= 191) per trial: **-0.5 to -0.7 s = +0.7-1%** | **C (cheapest test of all: one env var, then an A/B)** |
| M2 | GEMM path for 97-255 rows: decode-tuned t31-style kernel (or bs_gemm configs tuned for the L2-warm regime) instead of cold bs_gemm | cold microbench bs_gemm M=128: gate_up 158 us vs 138 at M=64; with prefetch, warm configs ~100. Combined with M1: mixed-step GEMM extra 8.7 -> ~3 ms. **-0.8 to -1.0 s = +1-1.3%** | C |
| M3 | GDN chunked-prefill glue in non-wave mixed steps (741 eager torch launches, 1.2 ms + 1.5 ms of chunk kernels) | capture or fuse (T32's captured core exists for waves; extend it to all mixed steps): **-1.5 to -2 ms per mixed step x 349 = -0.5 to -0.7 s = +0.7-0.9%** | C |
| M4 | mixed-step AR per-row cost (79 us per row per step at 128 rows, vs 43 napkin) | if mixed ARs reached the napkin 43 us per row: -36 us x 64 rows = -2.3 ms per small mixed step, ~-0.4 s; for big steps the DMA split chain hides part | C (small; tied to W1) |

Revised summary ranking (gains on the 75.9 s trial, GPU-side pricing):

| Rank | Candidate | Napkin gain | Cheapest test |
|---|---|---|---|
| 1 | W1 AR fixed latency (130 x ~25 us per decode-like step x 2393 steps = **7.8 s per trial**), with W3 so the prefetch's role is not lost | +3-5% (net of lost prefetch) | 2-process hostar microbench: prefetch on/off, R=4..64 |
| 2 | G1 f16-accumulate FP8 MMA for prefill rows (~10 s of prefill GEMM) | +4-6% | ruling first; then a microbench + numeric diff |
| 3 | M1+M2+M3 mixed-step path (fixed ~12.5 ms x 349 = 4.4 s) | +2-3% together | M1: env var A/B |
| 4 | R1 AR epilogue fuses add+norm+quant | +2-2.8% | count of launches/time in decode (have: 0.92 ms) |
| 5 | K4 shared-prefix keep (18K rows x ~150 us = 2.7 s if nothing hits today) | 0 to +3.5% | check the prefix-cache hit counters in a server log |
| 6 | S3 confidence-based dynamic draft slots | +1-3% | offline: drafter confidence vs acceptance from a logged run |
| 7 | S2 load-adaptive depth in the tail | +1% | |
| 8 | R2 lossless FP8 weight re-encoding | 0 to +7% | offline entropy of the e4m3 weights (no GPU) |
| 9 | P6 column-streamed AR under GEMM | +3-4% minus overlap with W1 | |
| 10 | K2a lazy-checkpoint cap 8 -> 16+ | +0.6-0.8% | |
| 11 | L1 4-bit lm_head screen plane | +0.5-1% | |
| — | W6 idle | pruned (2%) | |
