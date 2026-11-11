# T25 write-up (leader-25, explore). Closing: S1 rebalance null in situ, package below the bar

GPU: 43.7 of 50 granted used (seed 20 + start tranche 30). Balance 6.3 returned.
Records: bank/T25-001..T25-003. Worker: leader-25.w1. Notes: local_kb/leader-25.md. Proposal: proposals/T25-1.md.

## Direction
Proposal T25-1 was a stacked exact package. Its lead item, S1, was a GPU1-aware MLP rebalance:
- GPU0's SM clock is locked at 2235 MHz; GPU1 runs at 2520-2685 MHz.
- In compute-bound mixed steps, rank 1 waits for rank 0.

S2 (small-step glue) and S3 (lifecycle, drain k) were never started.

## What was measured
- **T25-001 (G1 microbench, 0.4 GPU-min).** Per-layer gate_up+down, Marlin, cold read-flush:
  - GPU0 is +4% slower at M=48 and +11..19% slower at M=128..640.
  - The balance point is f0/f1 ~.94/1.06 (.92/1.08 at M=640). There it saves 5-9% of MLP time at M>=256.
  - Weighted over T19's mixed-step buckets, the prediction was +1.2% of wall.
- **T25-002, S1(a): uneven MLP split everywhere** (GPU0 8064 / GPU1 9344 columns).
  - Memory: rank 1 needed +0.53 GiB, and OOMed at runtime at the champion's KV 5 GiB (job 158). So the champion has <0.5 GiB headroom on GPU1.
  - Fix: port T5-005's two-slot GDN (it applied cleanly) and set KV 4.3 GiB. The pool is 237,658 tokens vs the champion's 267,721.
  - Job 160 ABBA (300 prompts): A 939.0 / 930.9, B 929.6 / 932.4, -0.4%. No stalls or preemptions; 128K pass.
  - TPOT p50 rose ~0.3 ms. That is the decode penalty of GPU1 carrying 1.073x the columns at M=48 (G1: 175 vs 169 us/layer). My forecast had costed decode at f=1.05.
- **T25-003, S1(b): even split in decode, extra slice on GPU1 for mixed steps.**
  - Implementation:
    - GPU1 holds [8064, 17408).
    - A Marlin `ldb_n` row-pitch argument makes column-sliced views possible (unchanged for contiguous tensors).
    - A planar silu_and_mul.
    - The path is chosen by padded M (<128 decode); each CUDA graph captures one path.
  - Decode is bitwise equal to the champion on both ranks for all M (offline). Mixed steps differ by <=2 bf16 ulp. Memory per rank: model 14.37 / 14.90, graphs 2.50 / 2.47 GiB. 128K pass.
  - Job 164 ABBA: A 941.6 / 935.0, B 936.8 / 926.2, -0.7%. All-B > all-A fails.
  - B's TPOT p50 is back at A's level (15.49 / 15.31 vs 15.58 / 15.43), so the decode penalty was fixed. The remaining mixed-step gain still doesn't show.

## Forecast vs outcome
| item | forecast payoff / central | outcome |
|---|---|---|
| S1 rebalance | +2.0% / +1.0% (G1-revised +1.2%) | null: (a) -0.4%, (b) -0.7%, both within noise |
| S2, S3 | +2.8% / +1.2% | not started; alone below the bar |

Why the microbench didn't translate (Interpretation, not isolated):
- G1 measured standalone Marlin. In situ, mixed steps are run by:
  - the token split (two concurrent Marlin streams, which already over-count by 11-15%, T14);
  - the copy-engine ping-pong (T9);
  - a cudagraph-padded M.
- Rank 1's "wait" in mixed steps overlaps CE AR traffic, so shortening rank 0's Marlin moves the critical path to the AR/skew elsewhere.
- (b) also adds two N=8064 calls on rank 0 plus a planar activation in mixed steps. That overhead isn't measured separately.
- Lesson (KB candidate): a standalone per-GPU Marlin gap doesn't predict an in-situ mixed-step gain. Measure a rank-0 critical-path change in situ (nsys) before building.

## Reusable
- **Two-slot GDN ported onto the current champion** (t25-s1 / t25-s1b). It applies cleanly. Pool at KV 4.3 GiB: 237,658 tokens. At c=16, peak KV usage drops from 83.8% to 68.3% at equal admission (16 running, 0 waiting, 0 preemptions over 300 prompts × 4 starts). That is ~0.7 GiB/rank of headroom for anyone who needs memory.
- **Marlin `ldb_n` row-pitch argument** (T25-003): column-sliced weight views with bitwise-identical output when the layout is natural. Marlin's reduction order depends on column position, so a permuted layout is not bitwise.
- **G1 per-GPU tables** (agents/leader-25.w1/G1/): Marlin MLP times by M and N fraction on both GPUs.

## Opinion
I agree with T14/T18/T19/T23: the remaining items are ~1% each. Without S1, the T25-1 package can't clear the bar. I am closing rather than asking for a last chance to re-forecast S2/S3.
