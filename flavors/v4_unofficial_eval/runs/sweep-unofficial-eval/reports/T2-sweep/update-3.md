# T2 map, update 3 (resync 1)

HEAD = T2-012 (unofficial). CONFIRMED = T1-005 (760.8). T2 wins in the chain: T2-004 (confirmed), T2-011 +0.52%, T2-012 +0.85%.

## Tested since update-2
| id | record | result | verdict |
|---|---|---|---|
| KE1 Humming FP8 W8A16 kernel | T2-013 (microbench, jobs 59-62) | equal to Marlin within ±5% at M 16-48 (0.80-0.88 TB/s) | dropped; it also writes a JIT cache to $HOME (now banned run-wide) |
| Mixed-step CPU idle: FLA Triton fast launch | T2-014 | bitwise-exact; CPU per GDN prefill layer 300 -> 205 us; end-to-end within noise over 4 starts (-0.1%), GSM8K 93.6% | **null** |

## Verdicts changed
- **Mixed steps ≤512 tokens, CPU idle (rank 2 in update-2): down-ranked.** The analysis of the trace was right about where the
  CPU time goes (Triton's Python launch path, 6 launches per GDN layer: job 64), but removing a third of it did not move
  throughput. The idle measured under nsys overstated what can be recovered. Framework note: host dispatch remains a real
  resource (W1/T2-004 proved it), but its price must come from an un-profiled measurement (step time with and without
  the change), not from gaps in an nsys trace.
- **KE1 decode GEMM:** Marlin is at 0.74-0.84 of HBM peak; the tile-config win is taken (T2-012); Humming gives nothing.
  What is left needs a new kernel (≥2 blocks or 8 warps per SM, lighter dequant, split-K) for ≤ ~1 ms/step (~3%) and is
  unproven on Ada (no public W8A16 kernel above ~0.88 TB/s found).

## Current ranking (napkin vs HEAD T2-012)
1. KR1 GDN single-write state: ~4% napkin, measured upper bound -51 us/layer chain (job 16). Parked on a state bug in
   align mode (layers 17/40, one row). Biggest remaining T2 item; needs ~10-15 GPU-min of debugging.
2. KE1b new decode GEMM: ≤3%, high effort, uncertain.
3. Fuse the GDN prefill chain (6 -> 3-4 launches): low value given T2-014's null result.
