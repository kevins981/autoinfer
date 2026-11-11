team:         T44
builds_on:    T44-003 (T37-002 + T44 knobs); env T44_DMA_REORDER (default off)
performance:  "gate kill: reordered DMA AR is bitwise but 1.05-1.10x SLOWER at 192-300 rows, neutral at 700-1100; (b)+(c) alone 0.99-1.00" (jobs 406, 409; 0.6 GPU-min)
summary:
  hypothesis: T44-003: the doorbell write-value on the D2H stream blocks the next D2H; a separate doorbell stream gated by an
              event (a), resets after the last H2D (b), join before the ack (c) cut the split-step AR ~30% at 300 rows.
  change:     (a)+(b)+(c) in dma_staged_ar.cu behind T44_DMA_REORDER; v3 = (b)+(c) only. Bench agents/leader-44.w1/s4/.
  results:    Observed (rank 0, exposed us, old/new interleaved 3x, same job):
              job 406: R=192 148.5 vs 165.9 (1.117); 300 210.9 vs 228.4 (1.083); 448 295.8 vs 309.2; 700 445.4 vs 449.7;
              1100 686.2 vs 682.0. job 409: 300: old 208.9, new 220.2 (1.05), v3 207.9 (0.995); 192 v3 0.99; 700 v3 1.00.
              Bitwise eager + graph capture (60 replays) OK; negative control: delayed producer clean for old/new, broken
              doorbell (gated before the D2H) detected (rank 1 5/5 mismatches).
  conclusion: Interpretation: T44-003's attribution was WRONG. The D2H "end" event marks the copy engine finishing issue, not
              the data landing. The 13-60 us before the flag completes is the posted data draining over the link; the flag
              must follow it on any stream (the PCIe ordering of T44-001). Moving the doorbell adds a cross-stream hop and
              gains nothing. The copy-engine AR is near its real floor; the "~165-185 us required path" doesn't hold.
              Process lesson: cudaEvents after a D2H copy don't time host-side arrival. Validate a timeline attribution
              with a cheap intervention before forecasting from it.
validity:     knob off by default; nothing shipped.
