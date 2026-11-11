team:         T42
builds_on:    T42-001 (CPU study); measured against T37-002's lm_head kernel (lmh_codec, T1-037)
performance:  "unofficial, standalone, GPU0, M=64 per rank, cold L2: champion lmh_codec 1178 us; certified screen +
              host rescore (R-HOST) 759 us (0.64x); with on-GPU bf16 rows (R-GPU) 682 us (0.58x). 0 decision mismatches
              on 188,068 real rows + 1,728 adversarial rows. Go line (<= 0.65 ms) NOT met; kill line (> 0.8 ms) not hit."
              (DRAFT by leader-42.w1; validity review pending: leader-42)
summary:
  hypothesis: A 7-bit integer plane of the lm_head plus a rigorous per-(row, vocab) error bound can certify a small set
              of candidate ids that provably contains the baseline greedy decision. Only those candidates need exact
              bf16 logits, which can come from a host copy, so the plane can REPLACE the 0.993 GB encoded shard in HBM.
  change:     No engine change yet: a standalone torch extension, code/cs.cu, plus a test, code/cs_test.py (jobs 399/400/402).
              - Plane (per rank shard, 124,160 x 5120): q = round(w/s), 7 bits; s = per (row, 128-group) bf16 scale,
                rounded up from max|w|/63. Packed as int8 m16n8k32 B fragments in 912 B records per (n8 tile, k128):
                nibble, 2-bit and 1-bit planes plus 8 scales. 0.5662 GB vs 0.9933 GB for the champion's lossless code.
                The decode was verified bitwise against q (0 bad words).
              - hprep (per row): h ~ s_h(128a + b) with a, b int8. ||h||_2 and ||h - h_q||_2 are bounded rigorously
                (rounded up).
              - Screen kernel: the int8 MMA does two exact int32 passes per 128-group, then an fp32 combine with the
                group scale. Fused with U/L bounds, a running per-row LB (atomicMax), and emission of (id, U)
                where bf16_ru(U) >= bf16_rd(LB_now).
              - Bound = Cauchy-Schwarz with per-row scalars: hn*A_v + hqe*B_v. Per-group Hoelder was never tighter
                (CPU, code/cpusim3.py). CS needs no per-group bound arithmetic in the kernel.
              - finalize: filters against the FINAL LB = max over both ranks' LBs (a 64-float cross-rank exchange
                in-server; emulated here, its cost NOT measured). It dedupes vocab rows shared between rows (37%
                of pairs).
              - rescore: exact logits via the champion's own mma.m16n8k16 chain (same fragments, K order and
                single bf16 rounding). A tile with A row g = h_m and B col g = w_v gives the logit on its
                diagonal. Argmax by (value, smallest global id) via 64-bit atomicMax.
              - R-HOST: candidate rows are gathered from a pinned, mapped host copy of the bf16 shard into an HBM
                staging buffer, then rescored. R-GPU rescores straight from a bf16 shard in HBM, for comparison only.
              Enclosure proof: code/PROOF.md (summary in "validity" below).
  results:    Observed (job 402 log: logs/job402_log.txt; job 400 is the same before two speed tweaks, same verdicts):
              CORRECTNESS (decision = argmax over the champion kernel's bf16 logits, both shards, first index on ties)
              - Real rows: all 188,068 rows of T3's dumps (agents/leader-3.w1/data/seq, post-final-norm hidden;
                144,489 response + 43,579 prompt positions), batches of 64.
                Global-LB decisions: 0 mismatches, 0 fallbacks (CAPR = 1024 kept per row, ECAP = 65536 emitted per row).
                Local-LB (per-rank) decisions: 0 mismatches, 42 fallbacks (kept > 1024).
                Every decided value is bitwise equal to the reference logit (valmism 0).
              - Candidates per row, both ranks together, global LB:
                  all:    mean 3.65, p50 2, p90 8, p99 26, p99.9 62, max 384
                  resp:   mean 3.61, p50 2, p99 24, p99.9 50, max 347
                local LB only: mean 24.8, p99 203, max 3050 (resp mean 28.6)
                emitted during the screen (running LB): mean 825, p99 4227, max 19,669
              - Adversarial, 1,728 rows, 0 mismatches:
                  gaussian at 0.25/1/4x the real |h|, one-hot, two-hot (exact ties), scaled vocab rows, 97%-sparse
                  real rows, zero rows.
                  These had 1,100-1,560 candidates per row; they ran with CAPR raised to 65536.
                  Exact top-1 ties occurred in 23 one/two-hot rows (CPU count, code/tiecount.py); all resolved to the
                  first index.
                  Zero rows (every id ties) fall back by design (64 fallbacks).
              - Tensor-core check, 8,192 random real (row, id) pairs: the rescore's bf16 equals the champion's bf16
                bitwise (0 diffs). Measured max |c - exact| / sum|h w| = 2^-19.4; the proof assumes 2^-11.
              TIMING (GPU0, nvidia-smi 2520 MHz; CUDA graph replays, median of 30; "cold" = 256 MB READ flush +
              cudaCtxResetPersistingL2Cache before each rep; rank-0 shard only; decode-like batch of 16 sequences x
              M/16 consecutive response rows). All numbers in us:
                                       M=64            M=48            M=16
                                       cold (warm)     cold (warm)     cold (warm)
                champion lmh_codec     1178 (1181)     1055 (1056)     1041 (1042)
                hprep                     8              8               8
                screen                  647 (696*)      613             606
                finalize (+unmap)         5              5               5
                gather host->HBM         80 (185 rows,   48 (107 rows,   24 (45 rows,
                                             1.89 MB)        1.10 MB)        0.46 MB)
                rescore                  17             16              15
                FULL R-GPU              682 (687)       639             631
                FULL R-HOST             759 (763)       685             651
                (* warm screen medians are noisy: min 658)
              - Screen variants at M=64, cold: 4 warps x NT2 x 3 stages = 647-657; 8 warps NT2 = 660; 8 warps NT1 = 684;
                exact I2F vs magic-number int->float = same within noise.
              - Screen at M=16 = 606 us for 0.566 GB = 0.934 TB/s, i.e. at the read roof.
                M=64 is ~40 us above that (int8 MMA, epilogue).
              - The gather runs at 23.7 GB/s (PCIe roof).
              GPU used: 0.8 GPU-min (jobs 399, 400, 402).
  conclusion: Interpretation:
              - The certificate works on real data and holds where it matters. 0 mismatches and 0 fallbacks over
                188K real rows. The proof leaves ~256x margin on the only modelled quantity (tensor-core error).
                Two things that were not obvious before:
                (a) The cross-rank LB is essential. With per-rank LBs the rank that does not hold the winner keeps
                    ~25 candidates per row (p99 ~200, max 3050), 7x the global count. That rules out host rescoring.
                    In-server this needs one 64-float exchange between the screen and finalize.
                (b) Power-of-two scales cost ~10x in candidates (CPU sweep). Cauchy-Schwarz over all of K beats
                    per-group Hoelder for every config tried.
              - Speed: the full R-HOST path is 0.64x the champion at M=64 on GPU0, 759 vs 1178 us, i.e. ~420 us
                saved per rank per step before the exchange cost. It misses the 0.65 ms go line. GPU0 runs the
                champion at 1.18 ms, not the 1.07 ms the line was set from. Scaled to the line's ratio (0.61x),
                it still misses by ~0.03x.
                Remaining costs: the screen sits at the plane's read roof plus ~40 us of compute at M=64; the PCIe
                gather costs ~80 us at M=64.
                Further gains need fewer bytes, from a smaller plane or from fewer host bytes. Candidates:
                - an HBM cache of exact rows in the 0.43 GB the plane frees (~42K raw or ~53K codec rows);
                  needs a non-benchmark-specific policy, e.g. LRU;
                - fusing finalize + rescore;
                - prefetching the plane into L2 during the final-norm AR, as the champion does for its encoded prefix.
              - Memory: R-HOST lowers HBM by 0.427 GB per GPU (plane 0.566 vs encoded 0.993). Add ~35 MB of buffers
                (emission 33.5 MB at ECAP 65536, staging 84 MB at 8192 rows; both can shrink) and 1.27 GB of pinned
                host memory per rank.
                R-GPU keeps a bf16 or encoded copy: memory grows (ruled out, KB111).
              - Not measured: the cross-rank LB exchange; in-server overlap and L2 effects; GPU1.
              - Fallback in R-HOST would mean streaming the full shard from host (~50 ms). It never triggered on real
                rows, but a stage-2 design should keep a bounded second path, e.g. raise CAPR, or rescore in chunks
                from host for that row only.
validity:     (leader to review) Decision equality is by construction: a rigorous enclosure (code/PROOF.md) plus exact
              rescoring with the champion's arithmetic. The plane is derived from the checkpoint's own lm_head at load.
              No data-dependent tuning: the caps only decide fallback, and fallback is exact. Only assumption: the
              per-mma-step fp32 error of tensor cores is <= 2^-11/320 of the step's magnitude (published behaviour:
              ~2 ulp; measured 2^-19.4 overall).
              Per task_definition "certified screening" ruling: allowed if the reference argmax is provably in the set
              (yes, a bound), candidates are computed exactly (yes, bitwise checked), the argmax uses the baseline
              tie-break (first index; to be matched to the in-server cross-rank combine in stage 2), and memory is not
              grown (R-HOST: -0.43 GB/GPU; R-GPU would grow it and is for comparison only).

leader review (leader-42, 2026-11-11): VALID as a standalone study. PROOF.md read: the enclosure covers screen fp32
rounding (E1), plane/h quantization via Cauchy-Schwarz on exact stored residual norms (E2), and the champion kernel's
accumulation error (E3) under an explicit tensor-core assumption (<= 2^-11 relative per chain, measured 2^-19.4: 2^8
margin); bound and LB use directed rounding; ties are covered (all ids equal to the max are candidates); rescoring
reproduces the champion logit bitwise. Remaining for in-server: (1) cross-rank combine tie-break must equal the
champion's (local first-index argmax per rank, then its cross-rank rule); (2) cost of the extra 64-float LB exchange;
(3) bounded R-HOST fallback. Gate result: 759 us vs 0.65 ms go line (missed by 9%), under kill 0.8; champion 1178 cold
in this job -> 0.64x. Proceeding to stage 2 (decision: the gate was a proxy for "net saving >= ~0.4 ms/step", met: -419 us).
