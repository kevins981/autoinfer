team:         T44
builds_on:    BASE (standalone microbenchmark, no engine change); starts from bank/T15-002/duplex2.cu
performance:  n/a (measurement only; jobs 397, 398; 1.0 GPU-min)
summary:
  hypothesis: T15-002 found SM-driven host writes break PCIe duplex for any inbound read (SM-W+CE-H2D as bad as
              SM-W+SM-R) and KB109 read "~22 GB/s per GPU whatever the direction mix". Hypothesis: this is not a fixed
              property of SM traffic but oversubscription: when the writers' offered rate exceeds the upstream link,
              read requests queue behind posted writes (PCIe ordering: reads don't pass posted writes) and starve.
  change:     agents/leader-44.w1/pcie.cu (+pcie_v1.cu, job.sh, NOTES.md; out1/ = job 397, out2/ = job 398). One process,
              node-0 pinned buffers, fused SM writer/reader kernel with per-CTA globaltimer spans, CE copies on streams.
              A: store variants (cs/wb/wt/volatile/u64 relaxed.sys/u32/4KB-per-warp), writer-CTA count, paced writer;
              B: host memory type (shm+register, cudaHostAlloc, WC, THP); C: idle-gap latency + sysfs link speed log;
              D: CPU LLC/DDIO (GPU-written vs clflushed vs CPU-hot buffers); E: writer x reader CTA sweep.
  results:    Observed (GPU0, medians of 200 reps; GPU1 within 1-3% for A):
              - Store cache modifiers / widths don't change anything alone. 12 MB: W alone 481 us, W+SM-R 678, W+CE-H2D 915
                (serialized), CE-D2H+CE-H2D 562 (duplex reference), R alone 506.
              - Writer CTAs (64 readers), 12 MB W+R tot: nw4 835, 8 763, 16 662, 32 570, 64 678, 128 784. E sweep best
                nw32/nr64 568; nr32/nw32 743. 1.5 MB best 78.8 = CE+CE 79.2. 384 KB: ~28.7 everywhere (CE+CE 26.2).
              - Paced writer at 12 MB: pace 20 GB/s -> +SM-R 553, +CE-H2D 555; pace 22 -> 582 / 576; pace 24 -> 606 / 712;
                26 -> 648 / 883. Pacing restores CE-H2D beside an SM writer (915 -> 576).
              - Memory type, page size (THP), WC: no throughput effect (within ~1 us). THP shortens W issue span only.
              - Link stays at 16 GT/s from context creation on, including 1 s idle gaps; idle-probe latency 2.8 -> 3.4 us
                after >=1 s host idle. No speed changes during any phase (sysfs 1 ms samples).
              - DDIO/LLC: SM reads 24.4-24.9 GB/s regardless of where the data sits, 0.5-16 MB, no knee.
  conclusion: Interpretation: SM-driven PCIe traffic is NOT inherently half-duplex. Duplex collapses when the writers'
              offered rate exceeds the upstream link (~26 GB/s): reads (SM or copy-engine) then wait behind the posted
              write backlog. Holding writers at ~20-23 GB/s (pacing) or ~1:2 writer:reader CTAs gives CE+CE-level duplex
              (~21.5 GB/s each way) at >=1.5 MB. Corrects KB5/KB73 ("SM writes break duplex") and KB109's "~22 GB/s
              total whatever the direction mix" for prefill-size messages; decode-size (384 KB) ARs are latency/ramp-
              bound and gain nothing. Host memory type, LLC placement, link power states: ruled out as levers.
              Not directly confirmed (no PCIe counters): that the queue is the GPU's upstream posted-write queue.
validity:     measurement only; no served change.
