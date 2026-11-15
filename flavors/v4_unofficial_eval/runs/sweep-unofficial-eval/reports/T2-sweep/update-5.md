# T2 map, update 5 (closing)

HEAD = CONFIRMED = T2-018: 838.0 tok/s official (+31.8% over BASE 635.9). T2 wins in the confirmed chain: T2-004, T2-011,
T2-012, T2-016, T2-018.

## Tested since update-4
| id | record | result | verdict |
|---|---|---|---|
| KE1b new decode GEMM | T2-017 (microbench, jobs 121-122) | Marlin gate_up 840-871 GB/s; pure-read ceiling 903 GB/s; Triton prototype no better; a second weight layout doesn't fit in memory | **closed** (≤7% of GEMM time even in theory) |
| GDN kernel headroom | t2-w1 Task 16 (GPU-free) | 53 MB/layer in 38 us = 1.4 TB/s apparent, so L2 absorbs the write-back | kernel **closed**; bytes are the lever |
| Lazy commit K=2 | **T2-018** | c=1 gate 16/16 bitwise; PASS +1.60%; **confirmed 838.0** | **won** (napkin +0.6%: the write-back drain into the next GEMMs, T1 job 127, was underpriced) |
| Lazy commit K=3 | T2-019 | c=1 gate 16/16 bitwise; negative -0.59% | **dropped**: the replay work (3.5 -> 4.7 rows/step) outweighs 1/6 more writes |

## Framework notes (final)
1. **Host dispatch is a resource** (W1/T2-004, +4.5%), but price it with an unprofiled A/B: nsys gaps overstated the
   recoverable idle about 10x (T2-014 null; T1-010 saw the same).
2. **Deferred writes are billed to later kernels.** The GDN state write-back drains from L2 into the next GEMMs (+30 us
   per gate_up before KR1, +13 after: T1 job 127). Byte-based napkins priced at the writing kernel underestimate the
   saving: KR1 and lazy commit both beat their kernel-local napkin.
3. **Link and HBM are coupled on this box** (the host-memory AR uses SMs and HBM): Window-knob overlap must price both
   (PL1).
4. **Speculate x Rewrite interact**: reduced-vocab draft heads need an acceptance term (SM4).
5. **Exactness testing**: c=1 OFF/ON/OFF with logprobs, at a length cap, is the decisive test. Shadow checks that feed
   both kernels the same state miss stale-state bugs, and unit tests that make up their own flags miss bugs in how the
   server derives them.

## Remaining map (not built; for a future team)
- All-reduce, 5.7 ms/step (host one-shot AR at PCIe floor per T1); T1's area.
- Mixed ≤512 steps: AR +4.3 ms, small torch ops +1.7 ms vs T1-005 (job 101); cause unexplained (T1 refuted the
  AR+norm deferral theory).
- GDN state bytes: K=2 is the sweet spot; a lower-precision state is not exact (excluded).
- Decode GEMM: Marlin at 0.93-0.96 of the achievable read ceiling; nothing left.

## Addendum after the closing sparring session (sparring/ T2-1)
Two closing beliefs were assumptions; both are now checked:
- **"All-reduce at the PCIe floor"**: true only for the serial write-then-read design. T1 had already measured duplex
  traffic (T1 jobs 40-46: write 22, read 22, concurrent 30 us; pipelined variants within noise). Realistic floor
  ~35 vs 43 us, ~1 ms/step, already attempted. Closed (I should have read T1's update-2 first).
- **Tree speculation (top-2 at draft position 2)**: not ruled out by GDN state cost (the replay design branches without
  extra state writes). Priced in T2-020: a 4th verify row costs Marlin +10-18% per GEMM (M=64 switches tile list),
  ~+2.2 ms/step (~7%), so break-even needs position-2 top-2 coverage of ~0.75-0.8 vs 0.55 top-1. **Coverage
  untested** (two failed runs); a fixed logger is in workspace t2-w1.tree. One ~1.8 GPU-min run settles it: the
  cheapest open test on this map.
- Note: the logger fix in t2-w1.tree is itself untested on GPU.
- Mixed ≤512 AR growth: may be rank skew rather than slower communication; needs per-call traces from both ranks.
- **Tree coverage measured** (T2-020, job 151, 7410 rows, greedy, c=16): position 1 top-1 0.789 / top-2 0.886;
  position 2 given d1 accepted top-1 0.711 / top-2 0.815. Best-case gain +3.5% (branch at 2) or +4.1% (branch at 1),
  vs >=7% for the 4th verify row (Marlin alone). **Tree speculation: no.** (My "0.55" was the unconditional rate
  0.789 x 0.711; the conditional baseline is already 0.71.)
