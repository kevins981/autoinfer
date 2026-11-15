# Final report: run framework-eval (v3_framework, task qwen38-27b-fp8-2x4090)
Ended 2026-11-15 by the human (after a pause). Four sweep teams over the run (T1-T4), two open at a time.

## Champion
**T3-009: 951.5 tok/s at c=16, +49.6% over BASE (635.9).** Official trials 955.0 / 947.9. GSM8K 93.2% (gate 89.6%),
TTFT p99 ~805 ms, TPOT p99 ~29 ms, 128K request passes. Code and flags: `bank/T3-009/` (change.diff + serve_args.txt).

What is in it (the lineage, each step an official win; see champion.md):
| step | record | change | official tok/s | gain |
|---|---|---|---|---|
| 1 | T1-004 | FP8 draft lm_head + local draft argmax (flags) + host-memory all-reduce through pinned memory, no P2P (T1-002) | 707.6 | +11.3% vs BASE |
| 2 | T2-002 | + batch-sharded sampling (T1-005) + CUDA-graph capture sizes up to 512, so mixed steps run graphed (T2-001) | 757.6 | +7.1% |
| 3 | T1-009 | + Triton attention backend (removes 2 FlashInfer host syncs per decode step) | 786.4 | +3.8% |
| 4 | T2-006 | + W8A8 block-scaled FP8 Triton GEMMs (T2-004) | 819.9 | +4.3% |
| 5 | T1-014 | + target-side greedy local argmax (T1-012) + row-split of mixed steps with side-stream all-reduces (T1-013) | 842.7 | +2.8% |
| 6 | T2-008 | + GDN DeltaLog: row-0 state + draft log, lazy replay (exact) | 880.1 | +4.4% |
| 7 | T3-003 | + L2 prefetch of the next GEMM's weights by extra CTAs during the PCIe-bound all-reduce | 897.4 | +2.0% |
| 8 | T2-009 | + fused add-RMSNorm + FP8 quant, fused SiLU-mul + quant | 906.9 | +1.1% |
| 9 | T3-006 | + next layer's in_proj prefetch (had been silently disabled) | 919.0 | +1.3% |
| 10 | T2-011 | + reduced-vocab draft head (ids < 151643 + specials; data-free) | 931.8 | +1.4% |
| 11 | T3-009 | + fused norm+quant per row half, restoring the row-split overlap that T2-009 had silently disabled | 951.5 | +2.1% |

Rulings made during the run (task_definition.md, confirmed by the human): FP8-activation A/B carried over to combinations that don't
touch FP8 paths; fused norm + FP8 quant counts as fp reordering when the quant stage is bitwise exact (same-job GSM8K on both arms).
Reduced-vocab subset sources were answered from the existing ruling 2.

## What the teams mapped (the shared map.md was dropped mid-run by the human; each team's map is in reports/<team>-sweep/)
- **Measured dead or at a floor:** NCCL env tuning; FlashInfer PCIe-IPC and vLLM custom AR (no peer access); PP / sequence parallel (napkin);
  AR/GEMM overlap with the old Marlin GEMMs (T1-007); draft depth k=1 (T1-016, -9.2%) and k=3 as implemented (T1-003, T1-010, T2-013:
  -3.6 to -11.7%, TTFT gate broken); copy-engine AR in the server (T3-004, T3-011); 3-4 chunk splits (T3-014); row-split threshold 128
  (T3-010); prefetch 56 MB (T3-012); host-memory AR at all sizes (T3-001); DeltaLog 2-step checkpoint (ceiling ~0.33 ms/step, T3-007).
  The host-memory AR is at its PCIe floor (SM-issued duplex ~32 GB/s); DRAM measured at ~870 GB/s.
- **Inconclusive:** T2-012 (AR release/acquire handshake: +0.17% official, mixed same-job), T3-016 (row-split threshold 448: -1.45%, mixed).
- **Open leads, with ready material:**
  1. **k=3 done properly** (T3-015, profile job 142): most of k=3's +29% step cost is fixable: GEMM tile configs for M=56/64 (+4.8 ms) and
     DeltaLog limited to 2 logged rows (+2.6 ms). Napkin +5-8% overall; the TTFT p99 gate must be re-checked. Setup: agents/leader-3.w1/task9_status.md.
  2. **CUDA graphs beyond 512 tokens** (T4): 76 prompts' steps still run fully eager (~210 ms each); napkin +5-6%. Script: agents/leader-4/e1_job.sh (~20 GPU-min).
  3. **Small mixed steps CPU-bound in the eager GDN core?** (T4): +9-11% if real, <1% if not; settled by step logs + agents/leader-4/buckets.py.
  4. Mixed-step exposed all-reduce is still ~7% of wall time (T3 spar).
- **Method lessons** (T2, T3 write-ups): grep the "optimization active" log lines in every A/B (two wins came from fixing silently disabled
  paths); confirm microbench wins in a server profile; the framework's FLOPs/bytes/link model needed a CPU/launch resource and
  per-kernel efficiency factors on this machine.

## GPU time (pool 1000 GPU-min; 561.9 used, 438.1 unspent)
| team | leader | granted | used | official wins (sole / joint) | closed |
|---|---|---|---|---|---|
| T1 | leader-1 | 180 (start 100 + ext 80) | 163.8 | T1-004, T1-009, T1-014 / T2-002, T2-006 | own recommendation, after spar + closing tests |
| T2 | leader-2 | 220 (100 + 60 + 60) | 202.8 | T2-008, T2-009, T2-011 / T2-002, T2-006 | own recommendation, after its spar's checks |
| T3 | leader-3 | 245 (100 + 60 + 50 + last chance 35) | 190.7 | T3-003, T3-006, T3-009 | run end (on its last chance) |
| T4 | leader-4 | 100 | 4.5 | none (sweep done; its first test E1 never ran because of the pause) | run end |

Write-ups: reports/T1-writeup.md, T2-writeup.md, T3-writeup.md, T4-writeup.md. Resync reports: reports/T1-1, T2-1, T2-2, T3-1, T3-2, T3-3.
Incidents: vLLM socket-path length (fixed by the human in env.sh); a usage-limit stop (resumed); one exit-75 refusal judged a teardown race.
