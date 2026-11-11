# T13 write-up (leader-13, explore). Recommending a voluntary close after the seed

GPU: ~0.8 of 20 used (jobs 111-113, worker leader-13.w1). Record: bank/T13-001. Notes: local_kb/leader-13.md.

## Direction tried: producer-fused AR (decode, via host memory)
- **Idea:** the row-parallel producer GEMMs (out_proj, o_proj, down; Marlin FP8 at M=48) write their partial tiles to pinned host memory as each tile completes. The writes are posted and overlap the rest of the GEMM. Only the peer read, the sync and the add+norm stay on the critical path, and the separate AR kernel boundary goes away.
- **Napkin payoff:** ~+4.5-5.5% if it works, before the byte cost of encoding in the epilogue.
- **Gap it targeted:** T8-001 (x1.45) and T4 (~100 us AR) measured round trips or whole ARs. Nobody had measured posted writes on their own.

## Measured kill (T13-001): both premises fail
- **A. Interference:**
  - Posted SM writes themselves keep their PCIe rate beside the GEMM (21-22.5 GB/s, vs 22-23 alone).
  - But 383 KB of them cost the GEMM ~+9 us, whatever its length: out_proj +26-29%, down +11-12%, gate_up +6-8%. A serial write costs ~16 us.
  - The cost scales with write volume: <= ~100 KB is ~free.
  - Reads cost +4-7 us and run at only 15-18 GB/s. Writes and reads together cost about the same as the serial write.
- **B. Tile timeline:**
  - Marlin's split-K reduction makes the outputs appear almost together at the end: no chunk is done by 50% of the kernel, and the median chunk finishes at ~87% (out_proj) / ~94% (down) of the unperturbed kernel.
  - That leaves ~4-5 us of overlappable tail, against ~17 us of write time.
- **Net:** <= 0. A Marlin variant without split-K reduction would spread out tile completion, but at M=48 it would lose more than the AR's write half (the reduction is what keeps 128 SMs busy on 5120 columns).

## Side facts for others
- Posted SM host writes are not slowed by HBM streaming. The x1.45 in T8-001 is the GEMM slowing and the reads, not the writes.
- Copy-engine D2H beside the GEMM: ~+0.5 us (agrees with T8-001). It loses at decode size on per-op latency (T9-006).
- Small (<= ~100 KB) SM host writes can run beside a GEMM nearly for free. This is a possible exploit-sized angle: split a decode AR's write so ~100 KB of it overlaps the producer's tail. At most ~4-5 us/AR before the lost prefetch window, so well under 1%.

## Opinion
I agree with T2/T4/T10/T11/T12: the decode step is pinned by coupled HBM, bf16 tensor and PCIe floors, and small mixed steps (R<256) are what's left of the mixed-step excess. Those items are each 0.5-2.5% and need an exploit owner. I found no bar-clearing (>= 996 tok/s) direction with honest odds, and I'm returning the seed (~19 GPU-min).
