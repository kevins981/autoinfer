# leader-47 (T47, research_ownership) working notes

## Inspiration cards (first set)
Excess guarantees, Skew, Mismatch, Predictability, Repetition; HW: host memory, SM layout, FP8 MMA, no P2P, host common ground.
They pointed me at "what promise does the joined mixed step keep?" (all rows in lockstep through every GEMM/AR) -> asymmetric
waves. Predictability (speculate the AR / pipeline drafter) and Repetition (cross-row AR codecs) died on napkin math.

## Facts I found (CPU, existing traces)
- Champion T37-002 decode step (node trace s2b/A, 173 steps): target pass 23.0 ms; after lm_head: 0.16 ms bookkeeping,
  draft level 1 0.66 ms (64 rows), levels 2/3 0.54 ms each, 0.14 ms to next pass. Drafting total ~2.0 ms/step.
  A steady level: MTP Marlin ~187 us, bf16 fc gemv 58, 4 hostar ~89, draft lm_head path ~100, misc ~60.
  => drafter is ~0.15 ms/level above a realistic floor; KB80 roughly holds. (agents/leader-47/ana/lvl.py)
- Memory (T37-002 server log): weights 14.46 GiB, KV/state pool 5.0, graphs 2.48, ~0.9 free; pool likely ~2x oversized
  for either the 128K gate or 16 req at k=3 -> 1-2 GiB reclaimable (memory veto on replicated out_proj is implementation).
- Narrow-cut: all-gather FP8-quantized out_proj input (3.2 KB/row) vs bf16 AR (8 KB/row codec) with replicated out_proj
  (+1.0 GiB): decode a wash, prefill ~+2.5%. (T45 had the bf16 version.)
- GDN trees: branch state = scalar*S0 + rank-n update -> one shared state read; trees not blocked by state traffic, but
  sibling rows yield ~0.08 tokens < row price ~0.16: chains dominate at this acceptance.

## Sparring T47-1 (3 turns)
- memory veto: weak (confirmed above). - SM-count zero-sum: partly a bf16-gate_up compute effect; pure DRAM-vs-SM-count
  unmeasured (open, cheap microbench). - GDN state per branch: weak (low-rank), but economics still kill trees.

## Proposal T47-1: asymmetric waves (decode graph covers prefill AR). proposals/T47-1.md

## Stage 1 result (bank/T47-001, jobs 423/427, 5.5 GPU-min): GATE FAILED
- Measured J - P (marginal cost of 60 decode rows inside a joined mixed step) = 10-16 ms; D alone 22-29 ms. My model assumed
  ~23+15 ms -> wrong: the joint weight stream already makes decode rows ~half price in mixed steps; the "15 ms/event" is not theirs.
- Split must hide >= ~0.5 of D to break even; measured overlap 0.43-0.54 (Dh, host-AR decode wave) -> +0.65/+2.7 ms at P=256/512,
  negative below. Two copy-engine-AR waves serialize completely (overlap 0.00; suspected CE queueing, untested).
- Single graph with two branches never overlaps; separately launched graphs do; stream priority irrelevant.
- Rule for future teams: any "split the mixed step" design competes against decode rows that cost ~0.17-0.27 ms/row in the
  joined step, not a full decode pass; it needs overlap > J-P / D.

## T47-002 (jobs 428/429, 0.3 GPU-min): DRAM saturates from ~16-32 SMs
- Streaming read reaches 950-957 GB/s at 16 SMs (cp.async ring) / 32 SMs (ldg). KB59's zero-sum is kernel-made, not DRAM.
- Real decode GEMMs under green contexts slow in STEPS when their grid stops fitting one wave (qkvz at 112, down at 64,
  Marlin at 112 and 48); champion gate_up path robust to 48 SMs (<=+20%). At 128 SMs they are 1.32-1.41x byte floor (cold).
- Open (best next step for this program): decode GEMMs sized for 32-64 SMs (persistent, deep ring) -> frees SMs for
  comm/prefill lanes without losing bandwidth; re-test T2's spatial lane and T4's overlap on those kernels.
