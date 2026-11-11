# leader-50 (T50, exploit) working notes

## Inspiration cards (first set)
Coordination/Boundaries/Excess guarantees/Idleness/Unused results + host CPU, transfer fixed cost, PCIe paths, L2, FP8 MMA.
Made me think about: the AR's two phases (write then read) as a boundary with a guarantee nobody relies on
(full bf16 partial both ways, when the consumer quantizes to e4m3 anyway -> RS over rows + FP8 all-gather is
exact but only -15..-20% bytes and adds a sync phase: napkin-dead for decode and small mixed steps), and the
L2-prefetch blocks keeping the AR kernel alive (idleness) - already known coupling (KB29/KB78).

## CPU findings (existing traces; T37-002 node trace agents/leader-42.w1/out/s2b/A, my scripts agents/leader-50/ana/)
- Decode step 26.1 ms (64 rows): hostar 6.6-7.0 ms (128+ ARs, 51.6 us median at 64 rows), t31b 6.9, Marlin(gate_up then) 6.2,
  GDN replay 1.8, lm_head 1.03, cutlass small bf16 (ba proj) 0.5, add+rmsnorm 0.45, quant_perm 0.2.
- Decode AR vs rows (8,1 config, same trace): 4-24 rows ~36 us, 40 r 43.0, 48 r 46.6, 56 r 49.2; 64 rows (16,4 config) 51.6.
  => the 600 KB threshold putting 64-row decode into the (16,4) config is NOT a lever (matches the (8,1) slope). Dead, 0 GPU.
- T36 per-call AR trace (agents/leader-36/artrace, k=2 era): comm done (rd_end) 6 us at 30 KB, 15 us at 160 KB, 37 us at
  480 KB; kernel ends ~32 us minimum because the L2-prefetch blocks keep it alive. Known coupling (KB29, KB78): not waste.
- Mixed steps: ~120 rows (60-token prompts, ~1/3 of mixed steps) 42-50 ms, GPU-bound (no idle in lm_head-to-lm_head spans;
  steps.csv 'idle' column is an artefact). Excess over decode: bs_gemm +7.7 ms (20.8 ms vs ~13 byte floor), hostar +6.3.
- Big mixed steps: bs_gemm effective ~200-217 TF at token-split chunk M ~270-460, ~240-265 TF at ~1900 tokens
  (inferred: tokens matched by sort order, not exactly). Raw e4m3 mma roof 329 TF (KB69). bs_gemm ~16% of wall.
- Capture-size padding: mean 11 padded rows per mixed step now, 6.7 with 80 sizes; but padding inside an m-tile is ~free
  for GEMMs, only AR bytes -> ~0.2-0.4% e2e. Not worth a test.

## Plan
1. (job: bsweep) bs_gemm efficiency at M=128..2048 for the 5 shapes, all 43 configs, vs cuBLASLt per-tensor and 329 TF roof.
   If configs leave >10% on the table at chunk M -> retune table (cheap win); if the kernel caps at ~65-75% of roof ->
   consider a bigger kernel change.

## Results
- T50-001 (job 430, 0.9 min): bs_gemm table ~optimal except gate_up (cfg 103 better by 3-7% at M=192-320). bs ~88-92%
  of cuBLASLt per-tensor; mid-M (128-512) at 190-250 TF vs ~295 at M>=1024; e4m3 roof ~340 TF at 2670 MHz.
- T50-002 (building): T45-005 + T42-003 delta (certified target lm_head, applies clean with -p2) + gate_up cfg 103.
  ABBA vs T45-005 running (job abba1). Expect +0.5-1%.
- T50-002 ABBA: job 431 mixed +1.04% (300p), job 432 all B>A +1.25% (400p BAAB). Filed; official_eval T50-002 launched.
  Manager unreachable via SendMessage at 14:30 -> resend report (incl. T42 tensor-core-bound assumption flag) later.
- OFFICIAL T50-002 #1: 1181.3 (1179.1/1183.5), +0.67% vs T45-005, gates pass (GSM8K 93.6, 128K pass). 27.8 GPU-min used.
  Manager still unreachable (test-6-manager not in `agent list`) at ~14:50: report pending -> resend when it is back.
- Next candidates: mid-M (128-512) W8A8 GEMM efficiency (190-250 TF vs ~295; stream-K / tile change; token-split chunks live
  there); T46 host embedding (~0.2%) as another small exact stack item.

## After promotion (T50-002 champion 1181.3)
- Split-K in bs_gemm: only down@M=128 gains (8%, job 209 data; T51 chain -8 us/layer) -> ~0.08% e2e. Not worth it.
- T51 owns the M=128-208 GEMM band + decode-pass piggyback (told them my bsweep data; they asked me to stay out).
- Decode bookkeeping (lm_head->level1 and combine->next GEMM): ~45 tiny kernels, ~0.3 ms/step total (1.1%); realistic
  saving <=0.15 ms (~0.4%) with fiddly vLLM spec-decode fusion.
- Sparring T50-1 (ended; sparring/): decode AR "floor" = plateau among tested implementations, not a proven bound. Net prize
  of a duplex decode AR ~1.3 ms/step (prefetch keeps 40-60% of transport savings: KB25/KB78/KB62). Paths: GDRCopy/BAR
  push unavailable (no /dev/gdrdrv, needs root); P2P "CNS"; CE-only no better (KB62); hybrid CE-write+SM-read lost at
  decode (T16-001, ~20 us CE branch latency in graph). -> closed for exploit purposes.
