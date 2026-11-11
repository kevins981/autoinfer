team:         T13
builds_on:    T9-005 (standalone microbench on the built champion workspace; no engine change)
performance:  "measurement only: producer-fused AR premise fails (both gates)"
summary:
  hypothesis: Row-parallel producer GEMMs (out_proj/o_proj/down, Marlin FP8, M=48) could write their partial
              tiles to pinned host memory as tiles complete (posted SM writes overlapped with the GEMM), leaving
              only the peer read + sync of each decode AR on the critical path (FLUX-style, via host). Needs
              (A) posted writes not to slow the GEMM (gate: <=5%, writes >=18 GB/s) and (B) output tiles to
              complete spread over the kernel (gate: >=50% of tiles by 70% of the kernel).
  change:     agents/leader-13.w1/t001/ (mb.cu, bench.py, job.sh): real Marlin op, real per-rank shapes
              (out_proj/o_proj K3072 N5120, down K8704 N5120, gate_up K5120 N17408), cold weights (256 MB
              flush), CUDA graph, side kernels gated to start with the GEMM, 100 reps, medians, GPU0 + GPU1.
              Jobs 111, 112 (smoke), 113 (final). ~0.8 GPU-min.
  results:    Observed:
              - GEMM alone (GPU0): out_proj 35.7, down 80.0, gate_up 144.9 us.
              - 383 KB SM host writes beside the GEMM run at 21-22.5 GB/s (alone 22.3-23.4): writes not slowed,
                but the GEMM slows by ~+9 us roughly independent of its length: out_proj +26-29%, down +11-12%,
                gate_up +6-8% (8 or 16 writer CTAs). Serial write after the GEMM costs ~16 us.
              - Cost scales with write volume: 96 KB ~0, 192 KB ~+2.5 us, 383 KB ~+9 us; 0 B / 4 KB side kernels ~0.
              - 1-2 writer CTAs: down/gate_up +0..3 us at 16-22 GB/s; out_proj still +5.4-6.4 us.
              - Reads beside GEMM: +4.4..7 us, 15-18 GB/s. Writes+reads: +15-16 us (= serial write cost).
              - Copy-engine D2H 383 KB beside the GEMM: ~+0.5 us (agrees with T8-001).
              - Tile completion (64-col chunks, 1-CTA poller, which itself adds +8-12 us): 0% of chunks done
                by 50% of the kernel; median chunk at 0.69-0.85 of the perturbed kernel; absolute median
                completion 31 us (out_proj) / 75 us (down), i.e. ~87% / ~94% of the unperturbed kernel (estimate).
              - GPU0 process in job 113 was not NUMA-bound (pages on N1); GPU1 bound and verified on N0; bound
                GPU0 smoke runs gave the same numbers.
  conclusion: Interpretation: Gate A fails (out_proj +26%, down +11%) and gate B fails: Marlin's split-K
              reduction makes outputs appear almost together near the end, leaving ~4-5 us of overlappable
              tail against ~17 us of write time, while the writes cost the GEMM ~9 us. Net <= 0. Producer-side
              fusion of the decode AR through host memory is dead on this stack. Side facts: posted SM writes
              themselves keep full PCIe rate beside HBM streaming (the x1.45 of T8-001 is the GEMM's/reads' side);
              <=~100 KB of SM host writes beside a GEMM is ~free; CE is free (T8-001) but loses on per-op
              latency at decode size (T9-006).
validity:     measurement only (no engine change).
