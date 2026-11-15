# DeltaLog L=2 (checkpoint every other step): design note (leader-3.w1)

## Today's scheme (T2-008, read from csrc/gdn/*)
- Per request the step has W=3 slots idx[0..2] (same slots every step). A step with n<=3 rows writes the full state after row 0 to idx[0]
  and a log of rows 1,2 into the 784-byte head region of idx[1], idx[2] (k fp32[128], v bf16[128], decay, beta) plus a NaN-payload marker.
- Next step, with `accepted` = a: if a>=2 and slot idx[a-1] carries the marker, source = idx[0] (full state after the previous row 0) and the
  kernel replays the a-1 logged rows in registers (same fp32 op order as the token loop), then its own rows. Otherwise source = idx[a-1] (a full state).
- Marker trust rule: only looked at when a>=2, which implies the previous step really wrote it (recycled blocks can hold stale markers; a>=2 makes them unreachable).
- Hooks: `gdn_deltalog_materialize` (mixed steps, before the Triton spec path) and `gdn_deltalog_materialize_all` (block-boundary copies) replay the log
  into a full state at idx[a-1] (and slots bias..a-1 at boundaries), destroying the marker.

## L=2 scheme
Non-checkpoint ("N") step: no state write at all. Checkpoint ("C") step: as today.
State representation = base B (full state in idx[0]) + ordered entry list, all kept in ONE head region (idx[1], up to 5 entries x 784 B + 16 B header = 3936 B,
inside chunk 0 of the state, so the existing "write after chunk-0 smem copy" trick still works).
Header: magic0, magic1, n_conf (entries already confirmed), skip (1 if the writer was a C step: its row 0 is folded into B, 0 for an N step).
Entries [0,n_conf) = confirmed rows since B; entries [n_conf, n_conf+n_own) = the writer's own rows (C: rows 1..n-1; N: rows 0..n-1).
Next step with accepted a: replay count = n_conf + (a - skip) (a prefix of the entry list); this prefix is also the new confirmed list,
so an N step only writes the header and its own entries after position n_replay (the confirmed prefix is already in memory).
Parity is implicit: marker valid and skip==1 (previous was C) -> this is an N step; otherwise (no marker, or previous was N) -> C step.
Max entries: N step after C: (a-1)<=2 confirmed + 3 own = 5. C step after N: n_conf<=2 + a<=3 = 5 replayed, +3 own rows computed (8 row updates).

## Exactness argument
Replays use exactly the same fp32 ops, in the same order, as stepping would (the log holds the post-normalisation k, v bf16, decay, beta that the
step used; row j's update only depends on those and the running state). Replaying N rows from the checkpoint is the same op sequence as N stepwise
updates, so the state is bitwise the same (T2's test verifies this for 1-2 rows; I extend it to 5 rows and N/C alternation with random acceptance).

## Correctness hazards I need to settle (reason I'm sending this note)
1. Marker trust. Today the marker is only consulted when a>=2. With L=2 the parity decision needs the marker even for a=1 (previous step C -> now N).
   A stale marker in a recycled block at idx[1] (previous owner's last log) would then be trusted on a request's very first decode step
   (a=1) and could trigger a garbage replay. Options: (a) zero the marker region of idx[1] when the prefill/first step writes the state (needs a hook I
   haven't found yet), (b) pass num_computed_tokens and store it in the header (positional continuity check; collision is improbable but not impossible),
   (c) keep the a>=2 rule and make the first step after a=1 always a C step: then an N step is only reachable after a C step with a>=2 (46% of steps?),
   which reduces the saving. I prefer (c) if a cheap check shows enough N steps remain; it needs no new hooks.
2. Materialize hooks: all three need the generalized replay (multi-slot output in one pass, since idx[1] holds the log and idx[0]=B is the source, then the marker
   zeroed). materialize_all mode 0/1 also need the bias slots.
3. Cost: replay compute grows (avg row-updates per step 4.35 -> ~5.5). If the GDN kernel is not memory-bound the extra compute can eat the saving.

## Cheap ceiling check before building (job below)
`log_mode=2` (benchmark-only): T2's log kernel with the row-0 write skipped = the best an N step can do. Sequence timed:
GDN kernel -> out_proj GEMM -> 48 us AR-window spin (+ 40 MB gate_up prefetch) -> gate_up GEMM, 16 req x 3 rows, 24 value heads.
Saving of L=2 <= 0.5 x (mode1 - mode2) x 48 layers, minus the extra replay compute. If that ceiling is < 0.3 ms/step the idea stops here.
