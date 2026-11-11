# LOCAL_KB leader-51 (T51, exploit-long)

## Cards (first set)
- Trade-offs (separate by time/place) + Mismatch with the machine: decode passes are byte-bound with idle tensor cores,
  prefill events are compute-bound with an idle-ish weight stream. Separate-by-time is today's design; the card made me
  ask for the opposite: blend them in every pass (piggyback). Skew: prompts arrive one at a time -> each pays a fixed event cost.
- Graph card: the piggyback path only works if mixed steps are fully captured (host enqueue ~63 ms/mixed step today).

## Direction (2026-11-11): piggyback prefill ("fill every weight pass", T37's original idea, re-priced)
- Prefill = ~28% of the trial (event model: 15 ms/event + 0.166 ms/prompt token; KB111). Pure decode would be ~1640 tok/s.
- Sim agents/leader-51/sim/pig.py (T49's lengths model, event baseline 80.0 s vs ~78-80 official):
  piggyback with today's GEMM band (+3.2 ms/step at M=128 standalone, +4.35 chain) needs per-step glue <=~2 ms for +5%;
  glue 1 -> +7.4%, 3 -> +4.9%, 6 -> +1.2%. GEMM band ~+1.5 ms (1.1-1.25x floor at M=128) -> +8..+13%.
- Prior evidence against: T35-001 budget 128 -19% / 192 -2.6% (k=2, older champion, 1 start each): marginal 0.26 ms/token
  at 128 => ~10 ms extra per piggyback step, cause unattributed. T45 arithmetic "piggyback ~ event cost" ignored the
  event's fixed 15 ms but used today's band.
## Spar T51-1 (ended)
- Supported: closed-loop accounting in the sim. Assumed/untested: host starvation as cause of T35's loss; in-situ GEMM
  band; glue 1-2 ms achievable; AR 0.09 ms/prompt row at decode-size messages. Cheapest test = budget sweep with step
  logger + GPU-busy probe + nsys breakdown (worker w1, task agents/leader-51/task_w1.md).
- Reconciled with T45's sim (agents/leader-45/sim/sim.py, which killed piggyback at "+0.2% with a 4 ms cliff"): both sims
  agree at Bp=64 (+0.7..+2.4%). T45 only evaluated Bp<=96 prompt rows/step; with Bp=96-128 the M>96 cliff is amortized
  over more rows: T45's own cost model gives +7.4% (96) / +9.9% (128); mine +6.3 / +7.4 (today's GEMM band, glue 1 ms).
  So the kill was a budget-range artefact in the sim; the in-situ glue/host cost is the real unknown (T35's -19% at 128).
## Seed results (bank/T51-001, jobs 432/434) and proposal
- Mixed pass ~51 ms, flat in P from 1 to 144 (decode 24.9). Excess: GEMM +9.5 (t31b -> bs_gemm, no prefetch), AR 0.07/row,
  glue ~5, plus a wall floor above the GPU span (host? unproven). Sim pig2.py calibrated (-3.8/-11.7 vs -1.9/-10.2).
- proposals/T51-1.md: payoff if it works +10-12%, central +4%; ~80 GPU-min. M1 (GEMM band chain) given to w1.
- Manager address test-6-manager unreachable (not in agent list) at 2 tries; resend when it appears.
- M1 (T51-002): gate failed, band 0.07 ms/row @M=192, 0.09 @160 with best existing kernels (steps at m-tiles).
  The full-width bs is 1.10x floor at M=128 vs TP 1.34x -> TP-shape tiling is the cause to attack. Revised: payoff +8%
  (new kernel at 85% roof), central +3%, P(>=5%) ~25%. Next gate: TP kernel prototype <= 0.04 ms/row.
- w1 CPU analysis (agents/leader-51.w1/m1/tp_gap.md): TP gate_up loss = wave quantization (cfg 127: 544 CTAs on 384
  slots = 2 waves at 71%; M=160 3 waves). down = missing K-parallelism (80-120 CTAs). Fixes: stream-K (medium risk,
  est gate_up M=128 123 -> ~105-110 us), 4-CTA/SM config (cheap), down split-K. Cheap variant test given to w1 (<=3 min).
- T51-003 (job 438): 4-CTA/SM + split-K: no gain; occupancy not the limiter, cause unknown. Config tuning exhausted.
  Forecast: payoff +8% if a new kernel works, central +2%, P(>=5%) ~15%. Asked for a bounded kernel step (~25 min) or close.
- 2026-11-11: manager funded 25 GPU-min (balance 32.5); manager reachable at uds:/run/user/1001/cc-socks/554381.sock (not test-6-manager). Kernel step: w1 plan first, <=15 GPU-min, gate <=0.04 ms/row; keep ~10 for resync.
- w1 kernel_read.md: 64x64 tiles likely L2->SM bound (~3 TB/s implied; A re-read per n-tile). SK (one m-tile, pad-free, persistent) est. 0.055-0.065 ms/row at M=160. GO staged: ncu + pad-free configs first; gate relaxed to <=0.055 (sim: 0.05 + glue 3 => ~+7.6%).
- T51-004 (job 439): ncu refutes L2 bound; big pad-free tiles lose; kernel ~74% of e4m3 roof, tensor-issue limited at
  M>=160. Best realistic band 0.06-0.075 -> sim +2.5..+5% only if a captured pass hits glue 3-5 (unbuilt, 40-60 GPU-min).
  Spar T51-1 turns 6-8: closing supported as a funding decision. Decision: recommend close; write-up reports/T51-writeup.md.
