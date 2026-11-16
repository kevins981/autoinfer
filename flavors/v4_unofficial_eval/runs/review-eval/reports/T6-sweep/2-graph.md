# T6 sweep, step 2: the graph

Indices: r (request, 16 live), l (layer 0..63, type GDN or ATT; MTP layer m), i (positions in this step: M rows = 16 x 3 at decode: 1 verified + 2 drafts), j (past positions), t (step), d=5120, m=17408 (TP-split -> 8704/GPU), h/g heads (TP-split), v (vocab, 248,320), p (TP rank 0/1).

## Per decoder layer l (rows x[r,i,d], residual res[r,i,d])
```
n1[r,i,d]   = RMSNorm(x + res)                   (fused: + AR of previous layer, + dyn128 quant -> xq, sx)
-- GDN layer (48 of 64) --
qkvz[r,i,·] = xq · W_qkvz[l]  (FP8 W8A8; no r,i on W)       ba[r,i,·] = n1 · W_ba[l] (bf16, small)
conv        = causal_conv1d(qkvz, convstate[r,l])          KEEP convstate[r,l] (3 past rows)
S[r,l,h]    = g·S[r,l,h] + β k^T(v - S k)  per row i (sequential over i within r)   KEEP S (fp32, 64 KB/head)
o[r,i,·]    = q·S ; gated RMSNorm(o, z) -> quant
y_attn      = o · W_out[l]  (FP8, K split over p -> partial sums)  ==> ALL-REDUCE(p)   [Rule 2: K reduced]
-- ATT layer (16 of 64) --
qkv[r,i,·]  = xq · W_qkv[l] ; RoPE(25%) ; write k,v[r,l,i] -> KV cache (fp8)          KEEP k,v (fp8)
a[r,i,h]    = softmax_j(q k[r,j]) v[r,j]  (j over prefix ∪ this step's causal rows)
y_attn      = (a*gate) · W_o[l]  ==> ALL-REDUCE(p)
-- both --
n2          = RMSNorm(y_attn + res) (+ quant)
h[r,i,m]    = SiLU(n2·W_g) * (n2·W_u)   (m split over p, no comm, Rule 1)  (fused act+quant)
y_mlp       = h · W_down[l]  (m reduced) ==> ALL-REDUCE(p)
```
So 2 all-reduces per layer = 128 per target forward, each M x 5120 bf16.

## Step-level graph (decode, MTP chain)
```
Target forward over M = 16 x 3 rows (verify 2 drafts + bonus)         reads all W[l] once (N = 48)
logits[r,i,v] = h_L · W_lm   (v split over p; batch-sharded sampling: rows split over p? argmax per row)
Rejection/accept (greedy: compare draft == argmax)   -> accepted count a_r in {0,1,2}, next token
MTP draft step 1: x_m = f(h_L[r, last accepted], emb(next token)) -> MTP layer (attn + MLP) -> fp8 lm_head -> argmax -> d1
MTP draft step 2: same with d1 -> d2                                   (M = 16 rows each, serial edge)
next step: target over [next, d1, d2]
```
Serial edges: draft1 -> draft2 -> next target step -> accept -> draft... (the step is a chain of 3 forwards: target(48 rows), mtp(16), mtp(16)).

## Missing-index (sharing) table
| tensor | lacks | shared by | consequence |
|---|---|---|---|
| W_qkvz, W_qkv, W_o, W_out, W_gu, W_down [l] | r, i, t | all M rows of a step | reuse N = M = 48 at decode (<< ridge ~330): weight-bandwidth bound |
| W_lm (target, bf16 1.27 GB/GPU) | r, i, t | all rows | N = 48; 1.3 ms floor per step |
| W_mtp, W_lm_draft | r, t (but drafts are serial in t) | 16 rows per draft step | read twice per step (2 draft steps): candidate Merge across draft steps impossible (serial edge), but W_lm_draft read 2x |
| S[r,l] (GDN state, fp32 1.5 MB per request-layer-GPU?) | i | rows of the same request within a step | read+written once per request per layer per step (not per row) if kernel processes the 3 rows in one pass |
| k,v[r,l,j] | — has r | — | per-request, tiny at ~500-token contexts |
| AR inputs | — | — | 128 dependent link round trips per target forward + per MTP forward 2 |

GDN state size check: 48 v-heads/2 per GPU = 24 heads x 128 x 128 x 4 B = 1.57 MB per request-layer per GPU; x48 layers x16 req = 1.21 GB per GPU (read) + 1.21 GB (write) per target step. The MTP layer is a full-attention layer (no GDN state).

## Edges that cross devices
Only the 128 (+2 per MTP step) all-reduces and sampling (batch-sharded: each rank samples half the rows, then a small gather of token ids).
