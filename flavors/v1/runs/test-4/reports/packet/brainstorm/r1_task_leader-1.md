[manager, manager of run test-4] Round 1 task for leader-1 (team T1)

## Direction: R1-A, a host-staged all-reduce for 2 GPUs without P2P
Build a 2-rank all-reduce that goes through pinned host memory shared by both worker processes
(e.g. a /dev/shm segment with cudaHostRegister Mapped|Portable on each side; flags with
system-scope fences and an epoch counter so it can be replayed in a CUDA graph). Plug it into
vLLM as an all-reduce backend ahead of PyNCCL (see cuda_communicator.py), at least for the decode
and verify sizes, with NCCL as the fallback. Comms must stay bf16: a sum in the baseline's
precision, ideally bitwise-identical to NCCL's for 2 ranks.

## Rationale (both brainstormers, measured; details in brainstorm/r1/adjacent_field.md #1 and systematic_engineer.md #2)
- All-reduce is ~23% of GPU wall time at c=16. NCCL RING_LL takes ~60-65 us at T=48 (491 KB),
  while the PCIe link moves those bytes in ~20 us (25 GB/s write, 23 GB/s read; AF job 1).
  In-server cost: ~6-8 ms of a ~35-40 ms decode step; ~65 ms of a ~195 ms prefill step.
- vLLM's custom AR and the FlashInfer PCIe-IPC AR both need P2P, which this machine lacks, so
  only PyNCCL runs.
- NCCL env knobs were measured useless at T<=96 (AF). NCCL_PROTO=Simple at prefill sizes
  (T=300-2048) was NOT measured (SE's job hung): check it first, 1-2 GPU-min. If it's much
  faster there, an env var helps the prefill steps for free (generic, allowed).
- Estimate (assumed, from both): +6-15% depending on the achieved per-AR latency (target <40 us,
  ideally 25-30 us, at T=48). The polling/flag latency over PCIe is unmeasured.
- Both brainstormers' scripts are reusable: agents/bs-r1-adjacent_field/ (ar_bench.py,
  pcie_bench.py) and agents/bs-r1-systematic_engineer/.

Suggested go/no-go (yours to change): a standalone 2-process microbench in a CUDA graph at
T=16/48/96 (+ prefill sizes) vs NCCL, plus a bitwise compare, before integrating.

## Budget: 55 GPU-minutes for T1 this round (gpu/budget.json); keep ~11 per official eval.

## Champion to beat: BASE, 635.9 tok/s official (bank/BASE/official/1/results.json).
TTFT p99 1341, TPOT p99 46.1 ms, GSM8K 94.4%, MTP acceptance 2.345.

## Scope notes
- Read task_definition.md (global). Memory-affecting changes (buffers, graph sizes) need the 128K
  test plus a run under benchmark load before an official eval.
- Team T2 is working on the drafter/lm_head/logits path (R1-B) in parallel. Don't overlap with it.
- Report to manager when done: records, findings, proposed SHARED_KB entries, what's left open,
  and your single "next" for round 2.
