# T1 map update 1 (resync 1)
Champion now: T2-002 757.6; T1-009 = T2-002 + TRITON_ATTN official 786.4 (+3.8%), candidate.

## Candidates tested (6-ranking numbering) and results
| # | candidate | result | evidence |
|---|---|---|---|
| 1 | zero-copy host-memory all-reduce | **win, built**: 43 vs 53-60 us at decode sizes (napkin 25 us: concurrent store+load to host only ~15-17 GB/s/dir, T2 job 8); neutral at >=260 rows -> threshold 128 rows. +4.35% unofficial | T1-002, job 13 |
| 2a | fp8 draft lm_head + local draft argmax (engine flags) | **win**: +3.08% unofficial | T1-001, job 9 |
| 2+1 | combination | **official 707.6 (+11.3%)**, champion until T2-002 | T1-004 |
| 2b | reduced draft vocab (tokenizer merge-rank ids) | **dropped**: small K loses acceptance (8.9% of doc tokens outside 32K), large K saves <1%; vocab split by TP id range unbalanced the head | T1-006, job 17 |
| 2c | draft depth k=3 | **screen negative** (-3.6%, 1 pair); acceptance 2.735 as napkin'd, step cost +20% (napkin +13%); TTFT p99 2079 ms fails the gate | T1-003 |
| 3 | AR/GEMM overlap (split d + side stream) | **dead**: only 5-11 us real concurrency, chunk fixed costs; overlapped is 8-14 us slower than serial | T1-007, job 26 |
| 6 | target-side logits gather | partly done via engine flag --enable-batch-sharded-sampling (all-to-all of half the logits, each rank samples half): **+1.78%** | T1-005; in champion T2-002 |
| 7 | CPU/idle in decode steps | **big win**: TRITON_ATTN backend (in-place draft metadata, no seq_lens.cpu() sync) **+8.41%** on T1-005 | T1-008; T1-009 official pending |
| 4,5,8,9 | W8A8 prefill, Marlin efficiency, mixed-step microbatching, ReplaySSM | handed to T2 by agreement (W8A8, Marlin, DeltaLog, L2 prefetch, prefill lockstep) | T2's map |
| — | NCCL tuning, FlashInfer PCIe-IPC | no gain / impossible (no peer access) | job 1 |

## What changed in the map
- **New dimension: host CPU / launch path.** The framework's window cost has no CPU resource. Here it mattered most: mixed
  steps ran eager (T2's capture-size find, +7%), and decode steps lose ~1.2-1.9 ms to CPU-bound attention-metadata prep
  serialized by host syncs (T1 Triton result, +8.4%). Both are invisible to FLOPs/bytes/link pricing. Rule for the map:
  price every step as max(GPU work, CPU prep not overlapped) and look for host syncs in the step.
- **Correction: the link is not an orthogonal resource here.** Without P2P/copy engines in the path, all-reduce traffic is
  issued by SM loads/stores, so Window "Mechanism 2" (overlap link with HBM) fails against an all-SM GEMM (T1-007). A
  copy-engine (cudaMemcpyAsync) AR is the only form where the link is truly separate.
- **Calibration, now (T1-005 profile, agents/leader-1/jobs/prof_t1005):** decode step 34.9 ms (nsys): Marlin 18.1,
  host-mem AR 5.6 (41.6 us x 135), GDN 3.0, target lm_head 1.45, fp8 draft heads 2 x 0.77, idle ~1.8 (graph trace).
- **Speculation depth is gated by step cost and TTFT p99**, not acceptance: re-test k=3 only on the champion with graphed
  mixed steps and no CPU idle (both changed since T1-003).

## Remaining untried in my map (napkin vs current champion)
- k=3 re-test on the new champion (now that mixed steps are graphed and prep idle is gone): maybe +3-5%, TTFT p99 risk.
- AR fixed cost (~12-15 us of the 41 us): fuse AR + residual add + RMSNorm; fewer flag round trips; ~+1-2%.
- copy-engine AR variant for the mixed steps (NCCL still 294 us at 375 rows): ~+1%.
- Marlin decode efficiency gap (0.66 in-situ vs ~0.8 isolated): with T2.
