# T2 sweep, step 5: my map vs the shared map and the history

State of the run when read (2026-11-14): `map.md` empty, `shared_kb.md` empty, bank holds only BASE,
no proposals, journal only records the run opening. (T1's sweep files exist but, per the sweep rules,
I don't read another sweep team's analysis before my own 6-ranking.md.)

## Status of each of my candidates
Every candidate is **untried** in this run (no record but BASE). Measured so far, by T2 only:
- C1 (all-reduce): NCCL protocols/channels exhausted as a lever (T2 job 2); FlashInfer PCIe-IPC AR
  unusable (needs peer access). Custom host-memory AR: primitive bandwidth test in progress (worker).
- W4b: the BASE profile shows 150 us median rank skew per AR in prefill steps (analysis of the shipped
  profile, no GPU job).

## Differences both ways
- What my sweep adds that the map lacks: everything (the map is empty). The dimensions I'd add to the
  map's "what the space is" section: (1) a per-(op, M) kernel-efficiency factor and the
  **compute dtype** of GEMMs (bf16-on-dequant vs FP8xFP8), which the framework's knobs don't list but
  the contract opens; (2) the **link implementation** (NCCL ring through host vs custom one-shot through
  pinned host memory), since bytes/Link_BW is off by 2-4x here; (3) **CPU/launch-boundness and rank
  lockstep** of eager (mixed prefill) steps, which the window model has no resource for; (4) the
  **L2 as a memory tier** (72 MB) that a Window can fill during link-bound phases; (5) the
  **combine tag of Sample** (argmax) as a split-reduction choice.
- What the map has that I missed: nothing (empty).
- Disagreements with the map: none possible yet.

## Notes on the framework (for the map's method section)
- The framework's window cost max(F, BW, Link) presumes perfect overlap inside one kernel; Marlin at
  M=45 shows it doesn't hold near the ridge (0.66 TB/s achieved). Calibration is needed before ranking.
- No "host CPU" resource: prefill steps here look partly launch/lockstep-bound, invisible to the
  FLOPs/bytes model.
- The framework's GDN Keep/Recompute row B (DeltaLog) maps directly onto this engine (it writes 3
  states per request per layer per step) — a framework-derived candidate; but its payoff is small here
  because measured GDN time is already below the byte napkin.
