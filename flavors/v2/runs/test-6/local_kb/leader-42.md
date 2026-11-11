# leader-42 (T42, research) notes

## Cards (first set)
- "Unused results" + "Unused structure" -> greedy verification consumes only argmax of the target lm_head; the
  full 248K-row bf16 projection (1.27 GB/GPU, ~1.03 ms/step, T38-001) is produced for a discrete decision.
  Certified (exact) screening = produce only the part asked for.
- "Distance"/"Host memory": AR payload side-information coding (send only bits that survive downstream rounding)
  -> killed on napkin: fp32 unrounded norm input (KB92) needs ~g=12+ guard bits, so savings only if partial is
  >2^12 below residual; rare.
- Graph/compile/Python cards: nothing new vs run history.

## State of run (2026-11-11): champion T37-002 1142.2; decode step ~26 ms (k=3, 64 rows), mixed ~30% of wall.

## Findings (bank/T42-001, CPU only)
- Final-norm hidden states near-isotropic (r=512: 67% energy). Low-rank + block-CS exact screen keeps ALL vocab ids. Dead.
- Integer plane + Hoelder bound certifies the argmax: b7 G128 mean 4 cands (max 51); b6 G128 mean 24 (max 769); b<=5 fails.
  sweep_res.json 'bytes_per_w' field buggy (16/G instead of 2/G).
- Killed on napkin: AR side-information coding (fp32 unrounded norm input), link-power keep-alive (no ASPM attrs in sysfs; link gen1 only at idle).
- Sparring T42-1: supported = integer-plane counts on 400 rows; to prove = full GPU enclosure (screen-center fp32 error,
  baseline kernel error, bf16 rounding, ties); assumed = AR volume floor (partition argument only), value of freed HBM.
- Proposal: proposals/T42-1.md (stage 1 ~6-8 GPU-min in seed). Per manager rule this team stops after its first proposal.

## 2026-11-11 reopened: T42-1 funded (+20, balance 40). Ruling in task_definition.md (line ~149): bitwise decision,
provable bound, fallback to full exact lm_head, memory counted (no +0.6 GiB), host-backed must not add decode latency.
- ws t42-cs (from T37-002) built. Worker leader-42.w1 on stage 1 (agents/leader-42/w1_task1.md), budget <= 8 GPU-min,
  record T42-002. Deployment target: R-HOST (plane replaces 0.993 GB encoded shard; exact rows from pinned host).
- Stage 1 done (bank/T42-002, job 402, 0.8 GPU-min): 0 mismatches on 188K real + 1.7K adversarial rows; R-HOST 759 us
  vs champion 1178 cold at M=64 (GPU0). Key: GLOBAL LB across ranks (cands mean 3.65 vs 25 local); CS over all K beats
  per-group Hoelder; int8 MMA exact screen; bf16 scales rounded up. Proof bank/T42-002/code/PROOF.md (TC assumption 2^-11, measured 2^-19.4).
- Stage 2 assigned (w1_task2.md, 22 GPU-min, record T42-003).
- Stage 2 (T42-003, jobs 407/410/412, 15.9 min): exact in-server (c=1 10/10, gsm8k50 92, 128K pass, 0 fallbacks,
  HBM -0.34 GiB). ABBA job 410 +0.53% MIXED; ITL p50 -0.23 ms (expected -0.42). Stage 2b assigned: nsys + fuse + re-ABBA (<=13 min), T42-004.
- Stage 2b (T42-004, job 414, ~5 min): in situ lm_head section 1065 -> 796 us (-0.27 ms), decode period -0.20 ms. Below the
  0.3 stop line -> stopped. Launch overhead negligible (CPU 18-51 ms ahead). Champion gains 153 us in situ from prefix
  prefetch; plane prefetch (hint) doesn't land. Balance 18.1. Asked manager: write up vs fund ABBA+official (~20 min).
- CLOSED: write-up reports/T42-writeup.md; worker stopped.
