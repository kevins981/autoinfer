# leader-48 (T48, research) working notes

## Cards (first set + one redraw)
Holding, Representation, Rework, Per-item overhead, Unused structure; Trade-offs, Imbalance, Unused results,
Repetition, Predictability. HW: copy engines, address space control (compressible memory!), transfer fixed cost.
Takeaway: "Holding" -> the 5 GiB KV reservation is NOT idle at k=3 (GDN spec-state pages fill much of it, KB13/KB111),
so replica-in-reserved-memory ideas are small. "Address space" card -> hardware generic compression never checked in run.

## Wall-time accounting (my own, from official T37-002 numbers)
- pure-decode rate would be 16*2.67/26.2ms = 1630 tok/s; actual 1142 -> ~24 s of 80 s is prefill events + ramp/drain.
- physics of a 260-token mixed step: ~26 + 12 (compute beyond mem time) + 12 (AR if serial) = ~50 ms vs measured ~84 ms
  -> ~34 ms/event excess, ~13 s/trial (UNVERIFIED estimate; T37-001 fit 15 ms/event + 166 us/token).

## Killed at 0 GPU-min (napkin)
- rounding-aware AR (send only bits that can change bf16(h+s)): ambiguity prob ~2*delta/ulp(h); needs ~all bits unless |s|<<|h| by 2^8.
- PP single-microbatch + prefill in bubbles: decode 0.65x TP (KB112) -> 86 s > 80 s.
- CPU (AMX) prefill of short prompts: closed loop charges slot idle time; net ~+2%, huge build.
- replicated mixer out_proj + FP8 all-gather of 3072-wide pre-projection: ~+3% ceiling, needs ~1 GB/GPU (embedding offload could fund); not novel (known layout choice; T45 min-cut, T46 spar).
- spatial prefill lane: zero-sum on HBM bytes (lane re-reads weights at ~0.5 TB/s), not only SMs.
- drafter replication / DP drafter: wash on bytes.

## GPU work
- T48-001 (jobs 419/421/424, 1.3 GPU-min): 4090 hardware compression: works; quantized ratios; base+narrow-delta rule;
  real data 1x; no exact layout beats dense packing. T49-002 got the same headline one job earlier.
## Spar T48-1 beliefs (end table): replication bounded ~1-2% (estimate); 15 ms/event fixed cost UNVERIFIED (7.4 ms in
  an older fit); deeper drafting for high-acceptance requests untested (cheap: k=6 per-position log).
## Outcome: proposal T48-1 = candid seed report, recommend close.
- Closing spar (T48-1 turn 7): weakest reason = KB114 bound not recomputed. Recomputed on the CPU: offline chain k4/k3
  +10.2%, break-even marginal row cost 0.115 ms; T45-005's W8A8 gate_up may get there -> k=4 retest pointer to the manager.
