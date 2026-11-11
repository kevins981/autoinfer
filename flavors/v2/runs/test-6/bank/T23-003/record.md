team:         T23
builds_on:    T14-004 (ws t23-mb, policy knobs only: AI_POLICY=manual)
performance:  "n/a: microbench, gate failed (<=37 ms/trial vs 600 ms gate)"
summary:
  hypothesis: T6's policy keeps stock Marlin (main + remainder launch) for down/qkv at M>128, chosen on write-flush numbers (KB76); slab may be faster under a read flush.
  change:     none kept. Microbench job by leader-23.w1 (agents/leader-23.w1/W1-3/: bench.py, job.sh, bench.json, mdist.py), ~0.6-1.5 GPU-min, GPU0.
  results:    Observed: (a) Marlin M in mixed steps is the cudagraph-padded size (steps of 32), not pt+dt; at padded N>=256 T14-004's token split runs each GEMM twice per layer (c0=64*ceil(N/128), c1=N-c0) = 8 launches/layer; 6 launches/layer at padded 160 = stock remainder launches. (b) Read-flush cold and warm, real shapes, M up to 918: stock is never meaningfully slower than slab for down/qkvz/attn_qkv; slab is 5-35 us slower at 160/224/288/416, wins only at isolated points (<=14.5 us cold, ~0 warm). Remainder launch is cheap (down 149 us at M=160 vs 153 at M=192). Slab vs policy max abs diff 2-8 (bf16 scale).
  conclusion: Interpretation: T6's stock choice for cls1/2 stands under read flush. Best per-(shape,M) cherry-pick saves <=37 ms/trial cold, 2 ms warm. Dead.
validity:     n/a (nothing kept).
