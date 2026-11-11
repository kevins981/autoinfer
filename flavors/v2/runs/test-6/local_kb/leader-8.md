# leader-8 (T8, explore) working notes

## Orientation (CPU only)
- Read T2-T7 write-ups, KB1-KB65. Five+ teams concluded: decode at measured floors (GEMM at the M=48 ridge:
  warm Marlin tensor-pipe bound, cold ~HBM roof; AR at PCIe serial floor; GDN latency-bound), mixed-step AR
  not hideable (KB62, KB65), spatial sharing zero-sum (KB59), E capped by drafter (KB60, KB63).
- Common thread with no mechanism: every comm/compute overlap attempt is "zero-sum": host AR 51 -> ~100 us beside
  Marlin (T4-001), DMA copies 14 -> 9.5 GB/s beside Marlin (T7-001), 655 KB DMA only ~14 GB/s even alone.
- Own trace analysis of T2-006's nsys (agents/leader-8/tl3.py): token-split mixed steps (R>=256) spend 15-23% of
  span with comm only (both chunks' ARs run at the same time, lockstep, nothing to overlap). Worth ~3% of the
  trial at most (split steps ~18% of wall) -> exploit item, not a direction.

## Hypothesis H-PG (testing): GPU MMU page walks on 4 KB-page pinned host memory cause the interference
- If sysmem is mapped with 4 KB GPU pages, every new page = TLB miss + page walk (page tables in vidmem);
  under HBM saturation walks are slow -> DMA / SM sysmem traffic slows; walks also pollute shared TLBs -> GEMM slows.
- Host: Xeon Silver 4410Y, 2 sockets, both GPUs on node 0, own root ports, Gen4 x16; no IOMMU (/sys/class/iommu empty).
  THP: enabled=madvise (anon OK), shmem_enabled=never (so /dev/shm can't be huge; sharing huge-page memory across
  the two worker processes would need a single-process 2-GPU runtime or another trick).
- Test: agents/leader-8/pg/pg.cu (cudaHostAlloc vs anon 4K vs anon THP vs /dev/shm; H2D / D2H / SM round trip;
  384 KB / 655 KB / 4 MB; fixed vs rotating slot; alone vs beside an HBM-streaming kernel).

## Results so far
- H-PG refuted (T8-001, job 61). Copy engines coexist with HBM streaming (<=12%); SM host traffic x1.45 beside HBM.
- T8-002 (job 62): Marlin-like M=48 loop 1 warp/sched 68.4% of peak (= Marlin warm), 2 warps/sched 91.5%; L2 cp.async 4.2 TB/s.
- CPU kills: prompt-lookup oracle +0.9% E (ng/sim.py on T3 data); AMX prefill ~7 TF/12 cores (cpu/mm.py).
- Proposal T8-1 sent (2-warp decode GEMM, payoff +5.6%, central +2.5%, 60 GPU-min). Waiting for funding.
