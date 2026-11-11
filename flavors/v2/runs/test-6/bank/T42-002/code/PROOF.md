# Enclosure proof: the baseline argmax is always in the candidate set (T42-002, leader-42.w1)

Notation, per rank shard (N = 124,160 rows, K = 5120). h = the bf16 lm_head input row (exactly the
champion kernel's input). w_v = the bf16 weight row v. t_v = h.w_v (real number).
c_v = the champion kernel's fp32 accumulator for (h, v); l_v = rn_bf16(c_v) = the baseline logit.
Baseline decision = argmax_v l_v over the full vocabulary (both shards), first index on ties.

## Plane (built once at load, from the checkpoint's own weights)
Per (row v, group g of 128 along K): s_vg = bf16 value rounded UP from max|w_g|/63 (s > 0);
q_i = round(w_i / s_vg) clamped to [-64, 63] (7-bit). Define what_i = s_vg q_i (a real number).
Per row, computed in fp64 from the actual stored s and q, then rounded UP to fp32 with x(1+1e-9):
  rn_v = ||w_v - what_v||_2,  wh_v = ||what_v||_2,  wn_v = ||w_v||_2.
  A_v = ru((rn_v + 2^-11 wn_v + 2^-18 wh_v)(1+1e-9)),  B_v = ru(wh_v (1 + 2^-18)(1+1e-9)).
No assumption on the rounding of q is needed: rn_v is the exact residual norm of whatever q was stored.

## Per step, per row (hprep kernel)
s_h = max|h| / 16256 (fp32; any positive value works). qh_i = round(h_i/s_h) (integer, |qh| <= 16319),
split exactly qh = 128 a + b with a, b int8. h_q = s_h qh (real).
  hn  >= ||h||_2         (fp32 partial sums of <= 20 squares, inflated by (1+1e-5) + 1e-30, fp64 reduce, round up)
  hqe >= ||h - h_q||_2   (|rn(h - s_h qh)| (1 + 1.2e-7) >= |h - s_h qh|, same summation)

## Screen value and its error
I_vg = sum_{i in g} qh_i q_i is computed EXACTLY by int8 MMA (int32: 128 |a.q| + |b.q| < 2^28).
approx_v = fl( s_h * fl( sum_g s_vg * fl(I_vg) ) ) (fp32 FFMA chain over 40 groups, fp32 multiply).
Exact value of what this approximates: X_v = sum_g s_vg s_h I_vg = h_q . what_v.
Rounding: I2F (rel 2^-24), 40 FFMA steps (recursive-summation bound gamma_40 on sum_g |s_vg I_vg|), final multiply
(2^-24): |approx_v - X_v| <= 43 * 2^-24 (1+eps) * s_h sum_g |s_vg I_vg| <= 2^-18 * sum_i |h_q,i| |what_i|
                         <= 2^-18 ||h_q|| ||what_v|| <= 2^-18 (hn + hqe) wh_v.          (E1)

## Distance from the screen to the real dot (Cauchy-Schwarz, all coordinates)
t_v - X_v = h.(w_v - what_v) + (h - h_q).what_v
|t_v - X_v| <= ||h|| rn_v + ||h - h_q|| wh_v <= hn rn_v + hqe wh_v.                        (E2)

## Champion kernel's fp32 error vs the real dot (tensor-core model)
c_v is produced by 320 chained mma.m16n8k16 bf16->fp32 steps (k16 each, in ascending K). bf16 x bf16 products
are exact in fp32. Assumption (TC): every mma step returns its fp32 result with error at most
2^-11/320 * (|C_in| + sum of |products| of that step), i.e. ~23 fp32 ulps of the step's magnitude, far looser than
the published behaviour of NVIDIA tensor cores (products exact, aligned and truncated/rounded once, <= ~2 ulps per
step; Fasi, Higham, Mikaitis, Pranesh 2021, PeerJ CS 7:e330). Summing over the chain:
|c_v - t_v| <= 2^-11 sum_i |h_i w_vi| <= 2^-11 ||h|| ||w_v|| <= 2^-11 hn wn_v.            (E3)
Measured (job log "TC check"): max |c - t| / sum|h w| = 2^-19.4 over 8,192 real pairs: 2^8 below the assumed bound.

## Bound and candidate rule (screen/finalize kernels)
bnd_v = ru( hn * A_v + ru(hqe * B_v) )  >=  E1 + E2 + E3       (all ops round up; A, B as above)
U_v = ru(approx_v + bnd_v) >= c_v ;  L_v = rd(approx_v - bnd_v) <= c_v.
LB = max over BOTH shards of L_u (global; the "local" variant uses the own shard only and certifies the LOCAL argmax).
Candidate: bf16_ru(U_v) >= bf16_rd(LB).

## Theorem: the baseline decision v* is a candidate, and every id tied with it is too.
Let u achieve LB. l_v* >= l_u (v* is an argmax). rn_bf16 is monotone, so
  bf16_ru(U_v*) >= rn_bf16(U_v*) >= rn_bf16(c_v*) = l_v* >= l_u = rn_bf16(c_u) >= rn_bf16(L_u) >= bf16_rd(L_u) = bf16_rd(LB).
The same chain holds for every v with l_v = l_v* (ties), so all ids sharing the max value are candidates.
Rescoring computes l_v for each candidate with the champion's exact arithmetic (same mma.m16n8k16 instruction,
same fragment layout, same chain order over K starting from 0, one rn_bf16 at the end); the element (g, g) of a
tile whose A row g is h and B column g is w_v gets the identical per-element computation (verified bitwise on
8,192 pairs and on every decided value). The argmax over candidates with (value, then smallest global id) is then
the baseline's decision. Emission during the screen uses a running LB_now <= LB, so the emitted set is a superset
of the candidate set; finalize filters with the final LB.

## Fallback
A row falls back to the full exact lm_head if: its emission buffer overflows (> ECAP entries), its kept count
exceeds CAPR, or the pair list overflows. Then the candidate set may be incomplete, so the certificate is not used.
