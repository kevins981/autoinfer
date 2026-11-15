# T3 map, update 3 (closing). HEAD = T3-004 (unofficial +2.11% over T1-019; awaiting confirmation)

## Since update-2
| item | record / job | result |
|---|---|---|
| GDN in_proj_ba small GEMM | T3-005, job 195 | null: runs on an aux stream under qkvz (off critical path); cuBLAS already ~4 us isolated |
| unprofiled mixed-step cost | job 198 (client token timestamps) | decode 28.2 ms; mixed = +~28 ms fixed + ~0.17 ms/prompt token; mixed steps 27% of wall |
| copy-engine full-duplex AR | T3-006, job 200 | no-go: copy engines 10-16 GB/s/direction, no D2H||H2D overlap; ~1.8x worse than host AR |

## Final ranking of what's open (for the next team)
1. **Prefill communication** (~0.10 ms/token of a ~0.17 ms/token mixed-step slope): all overlap/duplex attempts lost
   (T2-005..007, T1-006/012, T3-001, T3-003, T3-006). Untested: SM-driven duplex with *one* flag per large chunk
   (T3-001's pipelined kernel regressed its own c=1 path; a fair same-job test vs the old kernel was never run), and
   the /dev/shm-registered buffer with copy engines.
2. **Mixed-step fixed cost** (~28 ms unprofiled; ~1,700 eager launches in the GDN prefill path): a full CUDA graph for
   mixed steps or fewer launches; nsys overstates it 2x, T2-014's CPU cut showed no gain.
3. Admission batching works on ShareGPT (+1.1%) but not on generic data: only revisit with a principled, non-benchmark
   way to set the hold (e.g. derived online from measured arrival rate and step costs).

## Framework notes
- A reduction whose consumer needs only its result (argmax) should be pushed before the collective (Rewrite across a
  Split index). This was the one T3 win.
- Per-kernel time sums must exclude kernels overlapped on other streams (T3-005).
- Price CPU/dispatch from client timestamps, not nsys (job 198 halved the nsys estimate).
- The Window knob's "orthogonal resources" assumption fails for PCIe-through-host traffic on this box (6 attempts).
