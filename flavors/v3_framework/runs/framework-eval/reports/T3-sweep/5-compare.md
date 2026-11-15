# T3 sweep, step 5: my map against T1's and T2's maps and the run's history

Read after files 1-4 were written: reports/T1-sweep/* (incl. update-2), reports/T2-sweep/* (incl.
update-1), reports/T1-writeup.md, T1-1/T2-1 resyncs, the bank (T1-001..017, T2-001..008).
shared_kb.md is empty.

## Status of my candidates
| my candidate (4-knobs) | status | on what code | note |
|---|---|---|---|
| GDN DeltaLog (state writes 3 -> 1) | **built, champion** (T2-008, +4.4% official) | current | my napkin 1.2-2.4 ms/step; T2 measured ~1.2 ms |
| DeltaLog with checkpoint every 2 steps | untried | - | T1 lists "ReplaySSM ~+2%" generically; nobody priced L=2 |
| Reduced-vocab draft head | tested, negative (T1-006, mixed -1.57%) | T1-001 era (fp8 draft head + local argmax already in) | T1's run was a flawed first version (vocab-parallel split by id left rank 0 with all its rows: no saving) and used the tokenizer id order as the frequency proxy (8.9% of doc tokens >= 32K). The balanced version never ran. Verdict: the saving is real (draft lm_head 1.53 ms/step measured) but a 64K subset only saves ~1.1 ms while out-of-vocab (~3%) costs ~2.5% of accepted tokens: **≈0 net with id-order; needs a better generic frequency source to be worth it** |
| k=3 drafts | tested twice, negative (T1-003 -3.6%, T1-010 -5.8%), k=1 also negative (T1-016) | T1-009-era | agrees with my per-row link pricing; dead |
| Tree drafts | untried; everyone drops it | - | agreed dead on this link |
| Split-d GEMM -> AR overlap (coarse, chunk kernels) | tested, negative (T1-007, microbench, Marlin) | Marlin era | T1's key observation: GEMM and the SM-driven AR only overlap ~5 us when on two streams |
| Fused tile-level GEMM -> AR (my largest decode napkin, ~3.8 ms) | untried | - | T1-007's observation weakens it (the consumer kernel spins on SMs); differs in having no per-chunk launch/AR fixed cost. Risky |
| L2 prefetch during AR | tested, negative (T2-005, microbench: 0-5.5 us/layer) | Marlin era | my own test was inconclusive; T2's result settles it for now |
| Decode GEMM efficiency (W8A8 Triton 0.68-0.88 of peak) | partly: T2 tuned configs for the 5 shapes (JSONs in T2-004); T1-017 showed (for Marlin) that an in-situ vs isolated gap exists that isn't the kernel, L2 or clocks | Marlin era for T1-017 | open. My data: gate_up 131 us in the server vs 107 isolated back-to-back (job 75); official clocks are steady (2.5-2.67 GHz SM, 10.25 GHz mem) |
| Hostmem AR for messages > 1.31 MB (mixed steps) | **untried**. T1-002 set the cap from isolated microbenches ("neutral at 260-375 rows, slower above"); T1-015 forced NCCL on the row-split side stream because "NCCL beats the spinning hostmem AR there" | - | my calibration: in the server, NCCL's 2.7 MB ARs take ~650 us (vs 242 isolated): ~4 GB/s. Testing now (ABBA job 79) |
| Chunk size <= 512 for mixed steps | tested by analysis (T2 update-1: >512-row steps already GPU-bound; graphing them gains nothing) and T1-015 (row-split for eager >=512 steps: official -0.7%) | current-ish | agree: not worth a separate test |
| Small-kernel fusion (norm/act + FP8 quant) | T2 lists it (~1.3%); T2-007 showed the torch.compile route is unavailable (breakable graphs); needs hand-written fused kernels. T1-011 (fused AR + add + RMSNorm): +0.26%, not significant | current | low priority, open |
| PP / asymmetric splits / drafter placement | dropped by all three maps | - | agreed |

## What my sweep found that the other maps lack
1. **The link's practical floor for the AR**: D2H+H2D at once on one GPU gives 22.6 GB/s total
   (job 75), so the decode AR (480 KB each way) has a floor of ~42 us + latency; it measures 48.6.
   T1/T2 priced the AR against 25 GB/s per direction (or measured ~15-17 GB/s per direction each
   way, T2 jobs 2/8). Consequence: the decode AR is at ~87% of its floor; further AR-kernel work
   on decode sizes has little left. (T2's 15-17 GB/s each way agrees with my 22.6 total within
   method differences.)
2. **The per-row link price of verify rows** (~0.12 ms per row per step): the clean reason k=3 and
   trees lose here. T1 explained it with a compute ridge (~80 rows for Marlin, ~165 for W8A8); at
   48 rows with W8A8 the GEMMs are still weight-bound, so the link term is the bigger one.
3. **NCCL in the server is ~2.7x slower than isolated for the row-split halves (> 1.31 MB)**:
   per-step AR exposure up to 130 ms in 480-890-token mixed steps (job 72). Nobody measured
   hostmem in place for those sizes; T1's "NCCL beats hostmem on the side stream" came from the
   row-split context (concurrency with GEMMs), and T1-015's forced-NCCL arm didn't win end to end.
4. Pricing the GDN state passes from the kernel source (1 read + 3 writes) gave the same DeltaLog
   candidate T2 built, independently.

## What they have that my sweep missed
1. **The host CPU as a resource** (T1's +8.4% TRITON_ATTN win came from a host sync; T2-001's
   +6.4% from graphing mixed steps). My napkin had no CPU row either; my calibration's 0.7 ms of
   decode gaps is the residue after those fixes. Framework lesson shared by all three maps.
2. **Rank skew / lockstep** in mixed steps (T2: half the mixed-step AR time was waiting for the
   peer rank). My "AR exposed" numbers include that wait; I attributed it all to NCCL's link rate.
   The ABBA will tell which matters: if hostmem's spin-wait shows the same exposure, the cause is
   skew, not NCCL.
3. **SM contention of the spinning AR** (T1-007, T1 update-2): the framework's Window max() is
   wrong when the "link" node is SM-issued. I listed it as a framework limit for L2 only.

## Where I disagree
- **Reduced-vocab draft head: "dropped (low value)" (T1-006) is too strong.** The run never
  measured a balanced implementation; the 1.53 ms/step draft-head time is now ~5% of a decode
  step (was a smaller share when lm_heads were bigger elsewhere). It is still marginal with the
  tokenizer id order as the frequency source; it is a candidate only with a better generic
  frequency table (e.g. a public generic-text token count, not ShareGPT).
- **Copy-engine AR "~+1%" (T1)**: agreed small for decode (the link floor holds for CE too), but
  in mixed steps it is the only AR form that doesn't take SMs from the overlapped GEMM. It becomes
  interesting only if the hostmem-in-mixed-steps test shows SM contention, not link rate, is the
  limit.
