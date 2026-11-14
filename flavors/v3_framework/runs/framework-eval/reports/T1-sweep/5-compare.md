# T1 sweep, step 5: my map vs the shared map and the history

Read at 2026-11-14 after files 1-4: `map.md` is **empty**, `shared_kb.md` empty, the bank holds only
BASE, no proposals, no other reports than the two sweep folders (T2's not read yet, by rule until my
6-ranking exists). The journal says the run starts from the stock baseline with an empty map.

## Status of my candidates
Every candidate in 4-knobs.md is **untried** in this run, except what my own team measured:
| candidate | status |
|---|---|
| NCCL tuning (PROTO Simple/LL/LL128, SHM_USE_CUDA_MEMCPY, NTHREADS/MIN_NCHANNELS) | **tested, T1 job 1** (worker leader-1.w1, microbench on BASE env): nothing beats default; default already uses Simple at n>=42; LL/LL128 20-100% worse. NCCL moves ~7.3 GB/s of payload at large n, 53.5 us at n=42 |
| FlashInfer PCIe-IPC all-reduce (opt-in env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC) | **tested, T1 job 1: impossible here** (workspace construction fails: "peer access is not supported between these two devices") |
| custom zero-copy AR via pinned host memory (E3) | in progress (T1 worker microbench) |
| all others | untried |

## Corrections to my files 1-4 found after writing them (not edited in place)
- 3-calibration's AR model: the worker's isolated microbench gives AR(n) ≈ 53.5 us at n=42 and a slope of
  0.72 us/row (≈7.3 GB/s payload), matching both the decode (58 us in-situ) and mixed-step (294 us at
  375 rows) numbers. So no in-situ slowdown of the AR; the napkin's 25 GB/s link is not what NCCL's SHM
  path delivers.
- The engine already contains two of my Rewrite candidates as off-by-default options of the spec-decode
  config (found while checking feasibility): `use_local_argmax_reduction` (draft-side local argmax,
  = my R1 for the 2 draft steps only; the target-side gather remains) and `draft_lm_head_quantization:
  "fp8"` (= my R3a). Both are serve-flag tests, i.e. the cheapest possible.
- Async scheduling is on by default in this engine unless incompatible (vllm/config/vllm.py), so the
  "W2: async scheduling" part of my Window candidate is probably already in BASE; what remains is the
  eager draft metadata rebuild (FlashInfer) and the 1.3 ms of idle.

## Differences with the shared map
The map is empty, so everything in my sweep is new to it. My first map proposal (proposals/T1-map-1.md)
therefore proposes the map's skeleton: the dimensions (step kinds and their time shares), the knob
sites, how to price them (calibrated factors), and the status of each site.
Nothing in the map to disagree with yet; I will compare with T2's sweep after this file and 6-ranking.
