team:         T32
builds_on:    BASE (standalone microbench, no engine change)
performance:  "measurement only: transport microbench; harness latency artefact, slopes only"
summary:
  hypothesis: replacing a mixed-step host-staged DMA all-reduce (bf16, 10 KB/token/direction) by reduce-scatter (bf16 half) + owner-side add/quant + all-gather of the next W8A8 GEMM's e4m3 input (+fp32 group scales) cuts link bytes 24% and the exposed AR time in proportion (TokenWeave, adapted to PCIe + FP8 consumer).
  change:     agents/leader-32/rsag/rsag.py: one process, 2 GPUs, pinned host buffers (affinity node 0), copy-engine D2H/H2D per token piece, cross-device CUDA events as signalling, R=128..1024 x pieces 4/8/16, 10 reps. Jobs 296 (timing bug: per-stream spins) and 297 (gate event; used).
  results:    Observed (job 297, max over GPUs, us): AR 4 pieces R=256/512/734/1024: 1680/1830/1948/2115; RSAG 1830/1979/2115/2288. Slopes 256->1024: AR 435 us for 7.86 MB (55 us/MB = 18-21 GB/s, matches KB/T30), RSAG 458 us for 5.96 MB (77 us/MB). Constant ~1.5 ms offset in both (also at R=128) and +~65 us per extra piece: cross-device event waits in one process are slow; not representative of the champion's device-flag signalling. More pieces hurt RSAG more (2 dependent hops per piece).
  conclusion: Interpretation: in this harness the two-phase RS+AG does not turn its 24% byte cut into time; its dependent second hop is latency-bound here. Not a hardware verdict: the champion signals via host flags polled on device (~5 us round trip, KB5), not cross-device events. Byte-cut ceiling stays a model number (<= ~0.24 x 10.7% of GPU time = ~2.5% e2e), unproven.
validity:     microbench only; no served-path change.
