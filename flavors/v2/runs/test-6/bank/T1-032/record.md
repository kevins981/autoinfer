team:         T1
builds_on:    T1-030
performance:  "unofficial: +0.4% same-job (A 853.3/848.2 vs 4 runs 852.9/854.9/852.3/856.7; within noise); not evaluated alone"
summary:
  hypothesis: Marlin's second (remainder) launch for M not a multiple of 64 re-streams all weights (~5-7 ms per forward at odd-32 M,
              ~half the mixed steps); one launch with a masked last slice removes it (R10-B / AF r10 idea 1).
  change:     csrc marlin (kernel.h, marlin.cu, marlin_template.h): for M > 64, 256-thread tmb=4 configs take ceil(M/64) slices in
              one launch; last slice uses its true row count in A-load/C-store/reduce predicates; hot loop instantiated per active
              m-block count (empty 16-row blocks skip ldsm+MMA); k-stripes cut by estimated cost ((a+c)/(4+c), c=0.5) with rewritten
              lock/slice logic; all other instances unchanged. Opt-out VLLM_MARLIN_SINGLE_LAUNCH=0; VLLM_MARLIN_PART_C. Worker
              leader-2.w19, ws leader-2.marlin1 (gate), leader-2.marlin2b (on T1-030).
  results:    Observed: gate (per-forward Marlin, 6 shapes, GPU 0, same job) stock -> new ms: 96 27.03->26.10, 160 38.48->35.74,
              224 48.72->45.33, 288 61.21->59.14 (gate 57: miss), 352 74.03->68.96 (gate 68: miss), 416 83.36->81.22; 64-multiples
              +0.5..0.9%; decode unchanged. Real rows 1-2 bf16 ulp vs stock (k-split order), deterministic, no writes past M,
              compute-sanitizer 0 errors. In-server job 131: A 853.3/848.2 vs marlin2b 852.9/854.9/852.3/856.7 (+0.4%), ITL flat,
              graph memory 2.41 -> 2.35 GiB, gsm8k50 88%. Every-16 grid to 2048: CUDA OOM in graph capture (job 132; C arm also
              contaminated B2 via orphan bench clients; only B1 848.4 valid). Estimate ~+0.3%; measured +0.4% (noise).
  conclusion: Interpretation: exact, small, unresolvable alone; every-16 full grid doesn't fit memory (partial range untested).
validity:     leader-2 review: same per-row math, fp32 k-split reorder only (KB22 class); predicates checked by sentinel rows and
              compute-sanitizer; decode path unchanged. OK. Not evaluated alone.
