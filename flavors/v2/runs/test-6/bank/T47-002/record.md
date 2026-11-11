team:         T47
builds_on:    T45-005 (kernels from workspace t47-waves2; nothing serves, no engine change: change.diff is empty)
performance:  "microbench (GPU0 only, jobs 428/429, 0.3 GPU-min): DRAM read BW saturates at 16 SMs (950+ GB/s); champion M=64 decode GEMMs slow in grid-quantized steps, not proportionally"
summary:
  hypothesis: the decode pass's proportional slowdown when SMs are taken away (KB59: 25.3 ms x 128/(128-f)) is either a DRAM/hardware
              limit (bandwidth needs all SMs) or a property of today's kernels (grids and in-flight bytes sized for 128 SMs).
  change:     none to the engine. Microbench bw.cu + bench.py (copied here; job.sh sets CUDA_VISIBLE_DEVICES=0).
              Part 1: a streaming read of a 1 GiB buffer, one persistent block per SM (forced by 80-96 KB dynamic smem; occupancy API
              confirms 1 block/SM), grid = N blocks, N = 16..128. Each thread folds its loads into an XOR/sum that is only written if it
              equals a magic value (as in KB76, so the loads can't be eliminated). Variants:
                (a) __ldg uint4, U independent loads per thread unrolled: 1024 thr x U=8 (128 KB/SM in flight), 512 x U=4 (32 KB), 1024 x U=16;
                (b) cp.async.cg 16-B ring, per-thread own slots: 512 thr x 6 stages (96 KB/SM), 256 x 4 (64 KB), 512 x 3 (48 KB).
              Best of 5 reps after a warm-up. Cross-check: the 128-block grid confined to a 32/64/96-SM green context.
              Part 2: M=64 cold, timed with CUDA events on the stream. Each rep runs a READ flush (256 MB, the part-1 ldg kernel on all SMs)
              + cudaCtxResetPersistingL2Cache() first, and rotates over 3 weight copies (KB76, KB103). Kernels:
                qkvz t31b (t31_gemm_e4m3 cfg = t31_cfg.pick(8192,5120,64,'cold'))
                down t31b (5120x8704, same pick)
                gate_up champion path at 48<M<=96 since T45-005: w8a8_marlin = quant_perm + bs_gemm_marlin
                gate_up plain bf16-activation Marlin (the pre-T45 path, reference)
              Grid limiting: GREEN CONTEXTS (torch.cuda.green_contexts, num_sms = 112/96/80/64/48/32), not grid edits.
              t31b and Marlin size their own grids from cfg/shape, so they can't be cleanly limited by grid.
              128 = default stream, no green context. Median of 15 reps (spread <= 3 us except where noted in the logs).
  results:    Observed:
    Part 1 (job 428), GB/s at N SMs:
      | variant                        | 16  | 32  | 48  | 64  | 80  | 96  | 112 | 128 | >=95% of best at |
      | ldg 1024thr U8 (128 KB/SM)     | 900 | 951 | 955 | 956 | 955 | 956 | 956 | 957 | 32 SMs (16: 94%) |
      | ldg 512thr U4 (32 KB/SM)       | 909 | 953 | 955 | 956 | 956 | 956 | 957 | 956 | 16 SMs           |
      | ldg 1024thr U16                | 912 | 952 | 954 | 956 | 953 | 956 | 956 | 956 | 16 SMs           |
      | cp.async 512thr S6 (96 KB)     | 953 | 955 | 952 | 956 | 955 | 953 | 956 | 957 | 16 SMs           |
      | cp.async 256thr S4 (64 KB)     | 952 | 956 | 953 | 956 | 955 | 956 | 956 | 955 | 16 SMs           |
      | cp.async 512thr S3 (48 KB)     | 952 | 955 | 955 | 955 | 955 | 955 | 956 | 957 | 16 SMs           |
      Green-context cross-check (128-block grid in n SMs): ldg U8 952/952/953 at 32/64/96; cp.async S6 935/952/952.
    Part 2 (job 429), cold us (median), "x floor" = time / (weight bytes / 957 GB/s), "x128" = time / time at 128 SMs:
      | SMs | qkvz t31b (41.9 MB)  | down t31b (44.6 MB)  | gate_up champion (89.1 MB) | gate_up Marlin ref   | KB59 model 128/n |
      | 128 | 57.6  1.32x  1.00    | 63.5  1.36x  1.00    | 126.8  1.36x  1.00         | 131.1  1.41x  1.00   | 1.00 |
      | 112 | 79.9  1.82x  1.39    | 63.4  1.36x  1.00    | 131.1  1.41x  1.03         | 193.3  2.08x  1.47   | 1.14 |
      |  96 | 82.1  1.88x  1.43    | 63.4  1.36x  1.00    | 127.0  1.36x  1.00         | 188.4  2.02x  1.44   | 1.33 |
      |  80 | 80.9  1.85x  1.40    | 63.5  1.36x  1.00    | 151.5  1.63x  1.19         | 188.4  2.02x  1.44   | 1.60 |
      |  64 | 82.9  1.89x  1.44    | 97.3  2.09x  1.53    | 138.2  1.48x  1.09         | 187.4  2.01x  1.43   | 2.00 |
      |  48 | 96.0  2.19x  1.67    | 99.3  2.13x  1.56    | 152.6  1.64x  1.20         | 264.3  2.84x  2.02   | 2.67 |
      |  32 | 115.7 2.64x  2.01    | 125.8 2.70x  1.98    | 196.6  2.11x  1.55         | 331.8  3.56x  2.53   | 4.00 |
    GPU0 at 2670 MHz SM / 10501 MHz mem after both jobs.
  conclusion: Interpretation:
    - Not a DRAM/hardware limit. On this 4090, 16 SMs read DRAM at 94-100% of the 128-SM rate (957 GB/s). A 48-96 KB cp.async ring
      gets 99.6% at 16 SMs, and plain ldg with 32 KB/SM in flight is enough from 32 SMs. Streaming bandwidth needs only 1/8 of the GPU.
    - It is a property of today's kernels, and it comes in steps rather than proportionally. Each kernel's cost holds flat, then jumps
      when its fixed grid no longer fits in one wave:
      - qkvz t31b: +39% from 128 to 112 SMs, then flat down to 64.
      - down t31b: flat to 80, then +53% at 64.
      - plain Marlin: +47% at 112, flat to 64, x2 at 48.
      - The champion's gate_up path is the most robust: <= +20% down to 48 SMs, non-monotone (80 worse than 64), which suggests
        tile/split choices tuned for 128 SMs.
      Even at 128 SMs every kernel sits at 1.32-1.41x its byte floor, so the gap is latency/occupancy within the kernel, not bandwidth.
    - Below the 128/n model at most points, above it at 112-96 for qkvz and Marlin: the step shape doesn't match KB59's smooth curve
      either way. KB59 measured the whole in-server step with prefill on the other SMs. That adds DRAM sharing with the concurrent
      prefill and the non-GEMM kernels, which this isolated bench doesn't include. So this bench says why the GEMMs themselves don't
      degrade gracefully, not the whole KB59 number.
    - Implication: a decode kernel built for a reduced SM budget (persistent grid sized to the green context, a 48-96 KB cp.async ring
      per SM) could in principle keep near-full DRAM rate on 32-64 SMs. Whether that makes spatial sharing non-zero-sum also depends
      on what the co-runner does to DRAM (a compute-bound prefill reads little). Not tested.
    Doubts: synthetic random weights and constant scales (timing only); green-context partitioning also limits which SMs/GPCs are used,
    so step positions may differ from a hand-sized grid; single job per part (spreads are small, <=3%).
validity:     no engine change, nothing served; timing-only microbench (exactness irrelevant, per the leader's brief).
