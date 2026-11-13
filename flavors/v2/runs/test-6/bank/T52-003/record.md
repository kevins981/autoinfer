team:         T52
builds_on:    T50-002 (microbench only; no engine change)
performance:  "microbench gate: producer chunk hooks +0.2..+1.2% (bitwise, pass); consumer K-chunk hooks +3.4..+10.3% (fail 3% gate, v4) (jobs 460-465, 3.8 GPU-min)"
summary:
  hypothesis: chunk-major producer tile order and K-chunk-gated consumer cost <=3% GEMM time at mixed-step M (256-512) for T29 bs_gemm W8A8.
  change:     agents/leader-52.w1/g1/bsc.cu (bs_gemm copy with compile-time hooks: per-chunk tile counter + globaltimer; per-K-chunk flag poll,
              coherent A-scale loads, 1/rms epilogue); g1.py/g1.sh; results g1v2.json/g1v4.json; NOTES.md.
  results:    Observed: bs_gemm already orders tiles n-major and walks K ascending (code read). Producer P vs A: out +0.2-1.2%, down -0.4..+2.1%
              (P vs hooks-off copy <=0.5%), gate_up/qkvz <=0.5%; bitwise. Chunk release (us, M=384 C=8): out [31x4, 50x4] end 50; down [80x4,130x4]
              end 129; gate_up M=512 progressive 84..331; wave-quantized: first chunk at ~60-75% of GEMM time. Consumer: v1 runtime hooks +6-10% even
              off; v2 per-thread acquire +20-33%; v3 scales via cp.async stage +8-24% (occupancy 3->2 CTA/SM); v4 tid0 volatile poll one iteration
              ahead + ld.cg scales: gate_up +6.8..+8.6%, qkvz +8.1..+9.6%, out +9.6-9.9%, down +3.4..+10.3%. Q8 with r=1 bitwise; with r max rel
              diff 0.0077 vs bf16(A)*r (bf16 rounding). Job 463 lost 3.3 min to an out-of-range flag index spin (bench bug, fixed).
  conclusion: Interpretation: producer side is free; chunks release per wave (natural chunk = one wave, C=2 at M=384). Consumer cost suspected
              in the A-scale loads (L1 __ldg -> L2 ld.cg for coherence); ablation pending (W1-4). Leader's timeline model (local_kb/leader-52.md)
              still hides ~365 of ~550 us exposed per layer at M=384 with the consumer at +8%.
validity:     microbench only. Not for official eval.
