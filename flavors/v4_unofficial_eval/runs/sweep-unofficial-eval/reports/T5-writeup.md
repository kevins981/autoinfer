# T5 write-up (leader-5, worker t5-w1)

Map: [T5-sweep.md](T5-sweep.md), files 1-6 and update-1. GPU: 80 granted, ~10.3 used (jobs 201, 202, 206 + a re-dump).
No win: the run ended while my one candidate was at its first gate.

## What I did
- Phase 1 map of HEAD T3-004 (jobs 201 nsys node+graph trace, 202 NVTX-only 200 prompts). Decode step 27.6 ms:
  W8A16 GEMMs 15.9 ms (0.79 TB/s), host AR 5.7 ms (135 x 42 us), lm_head 1.6, GDN 1.8.
- Phase 2 found most of my phase-1 list already tested negative (pipelined/duplex AR, in_proj_ba, k=3 drafts).
- New dimension: **lossless encoding of the all-reduce payload (C1)**. Ruled allowed by the manager. Step A on real payloads
  (T5-001): 0.77 of raw bytes, bitwise round-trip. The kernel (step B) was not built.

## What I learned
- Check the bank before ranking: three of my top phase-1 items were already in it. Reading the history is what made C1 stand
  out, because every earlier comm attempt was an overlap, and all six failed on this host's coupled, non-duplex PCIe.
- NVTX start-to-start step periods put the previous step's GPU tail inside a mixed step that blocks on a sync. Binning them
  by kind overstates mixed-step cost (by ~20 ms here). Use client token timestamps (T3's method) or account for the
  following short step.
- Exclude kernels on side streams before pricing them (in_proj_ba).
- The framework prices link bytes as fixed once lossy formats are ruled out. A lossless re-encoding is a Rewrite of a data
  node's format with p = 1 and no accuracy gate, and it applies wherever a link floor is bandwidth-bound and the data has
  low-entropy fields (bf16 exponents). It's worth adding as a generic row.
- The framework also needs the host CPU as a resource and PCIe up/down as one shared resource on this machine (also found by
  T1/T2/T4).

## Open for the next team
See update-1.md: C1 step B first (drop-in for T4's P4 HostAllReduce), then the decode W8A16 kernel redesign.
