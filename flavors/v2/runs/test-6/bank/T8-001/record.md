team:         T8
builds_on:    BASE (standalone microbench, no engine change)
performance:  "measurement only: host page size does not matter; copy-engine traffic coexists with HBM streaming, SM-driven host traffic does not"
summary:
  hypothesis: the "zero-sum" comm/compute overlap (KB59/62/65) is GPU MMU page-walk/TLB thrash from 4 KB-page pinned host
              memory; 2 MB (THP) pages would remove it.
  change:     pg.cu (this folder), 1 process per GPU: host buffers = cudaHostAlloc / anon 4K + cudaHostRegister / anon THP
              (madvise, 64 MB AnonHugePages confirmed) / /dev/shm (champion style). H2D, D2H (cudaMemcpyAsync) and an
              SM round trip (16 CTAs write sz to host then read sz, like hostar) at 384 KB / 655 KB / 4 MB, fixed or rotating
              slot, alone vs beside an HBM-streaming kernel (2 GB read, 952 GB/s alone). Job 61 (~0.05 GPU-min).
  results:    Observed (GPU0; GPU1 same pattern):
              - Page type: no effect anywhere (all four within ~1 us per op).
              - DMA alone: 384 KB H2D 17-20 us (20-23 GB/s), D2H 16-19 us; 655 KB 27-30 us (22-25 GB/s); 4 MB 160-170 us (25-26 GB/s).
              - DMA beside the HBM kernel: copies x1.01-1.12; HBM kernel x1.04 (H2D) / x1.00 (D2H) while copies cover ~80% of it.
              - SM round trip: 384 KB 37 us alone -> 55 us beside HBM (x1.45-1.50), 4 MB x1.43; the HBM kernel slows x1.06-1.19.
  conclusion: Interpretation: H-PG refuted (no TLB effect). Copy engines and HBM-bound kernels share the GPU almost for free,
              single GPU, near-node buffers. SM-driven host loads/stores (hostar style) are the ones that slow under HBM load
              (and slow the kernel). T7-001's 9.5-14 GB/s DMA rates are far below these single-GPU numbers; its 2-process
              microbench may have had far-node slots (KB37/38: both GPUs writing far-node -> 14.7 GB/s), not re-checked.
              Corollary: the champion's decode hostar runs beside its own L2-prefetch traffic (HBM-saturating), the case
              where SM host traffic is x1.45 slower here; a DMA AR would not suffer that (not measured in-server).
validity:     measurement only.
