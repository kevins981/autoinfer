# T3 sweep, step 5: my map against T1's and T2's maps and the run's history

Read after files 1-4: reports/T1-sweep/ (1-6, update-1..4), T1-writeup, T1-1..3, reports/T2-sweep/ (1-6, update-1..5),
T2-writeup, shared_kb.md, announcements.md, the bank (all records' summaries; T1-016/T1-017 in full). HEAD is now
T1-017 (W8A8 prefill route), promoted while I wrote this.

## 1. My candidates: built / tested / untried
| my id | candidate | status in the run | on what code | my reading |
|---|---|---|---|---|
| B | faster host AR (more blocks, write||read pipelining), extend to large sizes | **tested (T1 jobs 40-46, M=48 only)**: write 22 / read 22 / concurrent 30 us, full AR 45-48; pipelined/wider designs within noise; T1-009 (16-block fused) within noise. Larger sizes: T1-006/T1-012 chose chunked overlap instead; host AR capped at 128 rows | T1-005-era code; HEAD's kernel is the same | **decode part: closed** (realistic -1 ms/step, attempted). Mid sizes (129-512 rows, NCCL at 11 GB/s; 105 rows at 147 us in mixed steps vs ~97 serial floor) **not measured**: my worker t3-w1 measures it (task 1). Expected small (duplex floor at 3.6 MB ~225 vs 329 us) |
| R7 | read weights once at M>64 / W8A8 for mixed steps | **built: T1-017 = HEAD** (gate_up/down/out_proj; W8A8 >= 128-160 rows, W8A16 Triton below) | HEAD | done for 3 of 5 GEMM kinds. **GDN in_proj and attention qkv stay on Marlin** (64-row passes, re-read at every M>64): untried; T1 is tuning W8A8 configs (T1-018) and is the natural owner |
| R6 | dedup per-layer index ops of the eager GDN mixed path (CPU) | **tested in spirit: T2-014** (FLA fast launch, CPU -1/3, no end-to-end gain; nsys overstates CPU gaps ~10x per T1/T2) | T2-012-era | **demoted**: the mixed step is not CPU-bound outside nsys. My calibration used nsys CPU time; that part of my calibration was wrong |
| W1/W2 | L2 prefetch of next weights / GDN state during the AR | **tested: T2-005..007**: +2.3% on NCCL AR, negative (-1.5..-1.8%, 1 start) on the host AR | T1-005 | dead on this box: the host AR's PCIe traffic and HBM are coupled (T2 update-5) |
| SM2 | decode column-split GEMM || AR overlap | ranked by T2 (W2, ~4%), **untried** | — | same coupling risk as W1 (host AR stalls when HBM is busy?) — the T2 evidence is for a prefetch kernel, not a GEMM; untested |
| SM3' | chunked AR || GEMM in mixed steps | **built: T1-012** (eager steps > 512 tokens) | HEAD | graphed steps <= 512 rows don't use it: untried there (T1-006 on T1-005 code: no gain, 1 noisy start) |
| S1 | 3 drafts | **tested 3x negative** (T1-003, T1-008, T2-008): acceptance +14-17% but step +13%, TTFT p99 gate fails | T2-011 / T1-005 | closed for now |
| S2 | tree drafts | **measured T2-020**: best +3.5-4.1% vs >= 7% cost of a 4th verify row | T2-018 | closed |
| K lazy commit K>2 | GDN | **tested T2-019 (-0.59%)** | T2-018 | closed |
| R4 | AR + add + RMSNorm fusion | **tested T1-007 (-1.6%, 4 blocks), T1-009 (16 blocks, within noise)** | T1-005 | closed-ish |
| G1 | Marlin decode BW | T2-017: Marlin 0.93-0.96 of the achievable read ceiling in isolation; T1: 1.46x in situ in GDN layers (write-back drain, unsolved) | T2-018 | kernel closed; in-situ excess open (T1's) |
| G3 | sync points (SendRecv 455 us + 5 AllGather per decode step) | not in either map | HEAD | untried; small (<=2% of decode), first find what they wait on |
| — | **prefill admission batching** (see 3) | **not in either map, no record** | — | untried; my top new candidate |

## 2. What the other maps have that mine missed (findings about my use of the framework)
- **Host dispatch as a resource** (T2 W1 -> T2-004 +4.5%): I saw it in nsys but priced it with nsys CPU times; T1/T2
  both measured nsys inflating CPU gaps ~10x. Correction: price dispatch with an unprofiled A/B.
- **Deferred writes are billed to later kernels** (T2: GDN state write-back drains into the next GEMMs). My
  calibration treated GDN as "at floor" from kernel time alone.
- **Link and HBM are coupled** for the host AR (T2 PL1). My W1 assumed orthogonal resources (framework 06 §1
  Mechanism 2). The framework's Window cost (max over resources) is wrong here: the zero-copy PCIe traffic goes
  through the same memory system / SMs.
- **Accept term for Speculate x Rewrite** (reduced-vocab draft head) — HEAD has it.
- **Exactness gate** (c=1 OFF/ON/OFF at a length cap) — adopt for any state/scheduling change.
- T1: **CPU as a device with sync edges** (T1-010: blocking .cpu() worth ~2%).

## 3. What my sweep found that the other maps lack
**The admission dimension (Window/Merge along r over time).** Both maps treat the arrival of a new request as given:
each prompt is prefilled in the first step after it arrives (one prompt per mixed step in steady state). My
calibration (job 155, graph trace) shows a mixed step costs ~92 ms + ~0.1 ms per prompt token, i.e. a **fixed
~65 ms above a decode step** however small the prompt (57-token prefills: 91-98 ms; 231 tokens: 101 ms; decode step
27-29 ms). The start-up step with 10 prompts / 1,906 tokens took 373 ms = 37 ms per prompt, against ~94 ms each when
prompts are admitted one per step. The framework's Window knob (06 §2.1) says rows that read the same weights
should share a window; the framework puts "who to admit when" in the Policy layer, which is why a pure Structure
sweep (theirs and my phase 1) didn't list it. Priced in 6-ranking.md (A1).
Second gap: both maps price the mixed step from its kernels; neither separates fixed per-step cost from
per-token cost. That split is what makes merging prefills pay.

## 4. Where I disagree
- T2 update-5 "Mixed <=512 AR growth: may be rank skew": agree it's likely skew (147 us at 105 rows vs ~97 serial
  floor), which also means part of the fixed mixed-step cost is waiting, not transferring: merging prefills
  reduces how often ranks desynchronize.
- SPEED_OF_LIGHT and both maps' "prefill is compute-bound": at these prompt sizes (~60-400 tokens + 45 decode rows)
  the mixed step is dominated by fixed costs, not FLOPs (3-calibration.md §2-3).
