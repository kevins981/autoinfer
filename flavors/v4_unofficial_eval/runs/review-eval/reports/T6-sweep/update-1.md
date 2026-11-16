# T6 map update 1 (leader-6, ~06:35 UTC)

## Tested
| candidate | result | evidence |
|---|---|---|
| #0 HMAR up to 768 rows (new after the job-101 microbench) | isolated: 1.7x faster than NCCL at 192-512 rows, bitwise-equal; server: 843.4, 1 quick start (cross-job, -1.2%): no gain shown | T6-001, jobs 101, 105 |
| #1 row-split GEMM->AR overlap in mixed steps | first build INVALID (double all-reduce at the final norms: acceptance 2.294); fixed version 856.0 (+0.28%; starts 859.8 / 852.2), all gates pass, GSM8K 93.2%; within noise, not a candidate | T6-002 (job 109), T6-003 (job 113), reviews T6-2 |

## Verdicts changed
- Mixed-step all-reduce: faster transport (HMAR768) and partial overlap (split) both shorten the AR in isolation or in step bins, but neither moves
  the whole run beyond noise. The peer-skew reading (T1) holds: at the run level, the mixed-step cost is not the transport.
- My 3-calibration "mixed AR = 12% of wall" over-states what is removable. Pricing the AR at kernel time includes waiting that moves elsewhere
  when the kernel speeds up. This is the framework gap T1 also noted: the cost model has no term for rank skew or synchronization.
- Lesson on validity: a "skip the downstream AR" flag must cover every consumer, including those outside the decoder layer (final norm,
  drafter, aux capture). The MTP acceptance drop (2.29 vs 2.34) was the cheap signal that caught it before GSM8K could.

## Current ranking (for whoever continues)
1. GDN DeltaLog (T5).
2. Row-split overlap stacked with other mixed-step work: exact, +0.3% measured, may matter more if mixed steps get faster elsewhere.
3. k=3: memory-blocked.
4. Decode GEMM efficiency (76% of DRAM peak): untried, hard.
