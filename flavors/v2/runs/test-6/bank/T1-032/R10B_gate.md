# R10-B Marlin single launch for any M (leader-2.w19) - gate notes
Workspace leader-2.marlin1 (= T1-026 + this change). Files: csrc/quantization/marlin/{kernel.h, marlin_template.h, marlin.cu}.
Change:
- dispatcher: M > 64 -> one launch with ceil(M/64) slices (cap max_par as before; >1024 rows still splits at 1024).
  Only 256-thread tmb=4 configs take a partial slice; anything else keeps the stock 64-multiple + remainder split.
  Env: VLLM_MARLIN_SINGLE_LAUNCH=0 restores stock dispatch; VLLM_MARLIN_PART_C (default 0.5, <=0 = unweighted).
- kernel (kPartial instances only): last slice keeps true row count (A-load/C-store/reduce predicates via prob_m per slice);
  hot loop instantiated per compile-time active m-block count (1..4): empty m-blocks skip ldsm + MMA;
  part2 k-stripes cut by estimated cost (cheap last-slice tile k-iter = (a+c)/(4+c), c=0.5) instead of equal counts;
  locks/slice_count/slice_idx recomputed generically for the uneven stripes (lock = first stripe boundary inside the tile).
  Other instances compile to the original code (same spill stats; tmb<=3 decode kernels untouched).
Gate (job r4, GPU0, same job: champion leader-2.numa stock vs patched; per-forward Marlin over the 6 per-rank shapes, ms):
  M    stock  new    | 64-multiples: 256 50.21->50.47, 320 62.30->62.69, 384 72.00->72.62, 448 84.75->85.30 (+0.5..0.9%)
  96   27.03  26.10  | decode: 16 14.52/14.53, 48 15.33/15.32, 64 17.10/17.09
  160  38.48  35.74
  224  48.72  45.33
  288  61.21  59.14  (gate <=57: miss; AF ideal 56.3)
  300  63.63  61.56
  352  74.03  68.96  (gate <=68: miss by ~1)
  416  83.36  81.22
Exactness: real rows vs stock dispatch max |d| 0.0156-0.0312 (1-2 bf16 ulp, fp32 k-split order); deterministic; 0 writes past M
(sentinel rows); compute-sanitizer memcheck 0 errors (N,K = 1024x1024 and 5120x3072; M = 80,100,288,300,1100, exact-size A).
Dev history: v1 (runtime break in MMA loop + live 64-bit stripe state) cost +4..9% on full slices -> fixed by compile-time amb paths,
recomputed stripe state, kPartial gating. PART_C sweep (harness, ptx/r4,r5): 0.5 best of {0.05,0.25,0.5,1,1.5,2,3,5}.
Logs: agents/leader-2.w19/gate/r4 (final), r3; harness agents/leader-2.w19/ptx/ (r1-r5).
