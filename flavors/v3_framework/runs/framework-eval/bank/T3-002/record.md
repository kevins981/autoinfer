team:         T3
builds_on:    T2-008
performance:  microbench only (no server change): copy-engine AR overlaps with W8A8 GEMMs; NCCL and the hostmem AR don't
summary:
  hypothesis: In mixed steps the row-split's side-stream ARs (NCCL > 1.31 MB, hostmem below) are slow because they
              compete with the main stream's GEMMs for SMs. An AR whose data moves with the copy engines (D2H into
              mapped shm, stream flag, H2D of the peer slot, small add kernel) should overlap almost fully.
  change:     microbench only: ce_bench.py + ce_ext/ce_ar.cu (eager, host epochs). 2 ranks, R rows x 5120 bf16;
              "gemm" = W8A8 Triton gate_up (N=17408,K=5120) + down (N=5120,K=8704) at M=R; "X||gemm" = AR X on a
              side stream concurrently with gemm on the main stream (per iteration, us; GPU job 84).
  results:    Observed (rank 0; rank 1 within ~3%):
              rows  MB   gemm   ce    ce||gemm   hm(32 CTA)  hm||gemm   nccl  nccl||gemm
              96    1.0  179   104    243        85          250        91    248
              128   1.3  208   127    232        112         295        118   235
              192   2.0  265   182    298        161         402        159   319
              256   2.6  337   230    348        216         503        210   391
              512   5.2  575   438    589        417         908        379   690
              CE result bitwise equal to the hostmem AR at all sizes.
  conclusion: Interpretation: alone, CE is ~10-15% slower than hostmem/NCCL (copy fixed costs), but concurrent with
              GEMMs it costs only +3-14 us over the GEMM alone at >=128 rows, vs +27-115 us for NCCL and +87-333 us
              for hostmem (whose 32 spinning CTAs steal SMs). Server version (graph-safe) is T3-003.
validity:     microbench, no served change.
