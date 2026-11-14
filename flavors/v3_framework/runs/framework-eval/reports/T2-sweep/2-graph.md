# T2 sweep, step 2: the graph

Indices: r (request, 16), l (layer, 64; type A = full attn l%4==3, type G = GDN), i (row position within
the step: the verified token + k=2 drafts, so 3 rows/request in decode; ~215 prompt rows for a prefill
request), j (past position), t (step), d (hidden 5120), m (MLP 17408), h/g (heads), v (vocab 248,320),
c (TP rank, 2), s (draft depth 1..k).

## Nodes of one layer l (per row (r,i)); TP rank c owns a slice of the column-split index
```
x̂[r,i,d]      = RMSNorm(x[r,i,·]) [d]                           (needs full d: after all-reduce)
GDN layer:
 u[r,i,e]      = Σ_d x̂[r,i,d] W_in[l,d,e]   e∈{q,k,v,z,a,b}, split over c (column)   W_in has no r,i,t
 conv[r,i,e]   = Σ_{τ<4} w_conv[e,τ] u[r,i-τ,e]               keeps conv state (last 3 u) per r
 S[r,l,hv,·,·] (i) = decay(i)·S(i-1) + β(i)·k(i)ᵀ(v(i) - S(i-1)k(i))   fp32 128x128 per head; recurrence over i
 o[r,i,hv,·]   = S(i)·q(i); gated RMSNorm with z
 y[r,i,d]      = Σ_{e∈c} o[r,i,e] W_out[l,e,d]   -> partial sum over c  ==> ALL-REDUCE (+) #1
Attention layer (type A):
 q,gate,k,v    = x̂ W_qkv (column split over heads: 12 q heads, 2 KV heads per rank)
 k,v[r,l,j]    kept (fp8 KV cache, append-only); s = q·k over j∈Anc(i); softmax_j; a = Σ_j α v
 y             = Σ_{h∈c} (a·σ(gate)) W_o   ==> ALL-REDUCE (+) #1
x' = x + y; x̂' = RMSNorm(x')
MLP: g,u = x̂' W_gu (column split over m);  h = silu(g)·u;  y2 = Σ_{m∈c} h W_down ==> ALL-REDUCE (+) #2
x[l+1] = x' + y2
```
Head of the step: emb[r,i,d] = E[tok,d] (vocab-split -> ALL-REDUCE #0); after layer 64:
logits[r,i,v] = Σ_d x̂ W_lm[d,v] (v split over c) -> ALL-GATHER over v (currently full logits);
sample: greedy (T=0) -> argmax_v; verification (rejection sampler) compares target argmax with drafts.
Draft (MTP), s = 1..2, serial: h_s = fc([RMSNorm(emb(tok)); RMSNorm(hidden)]) -> 1 attention-type layer
(+MLP) -> lm_head (same W_lm, bf16) -> all-gather -> argmax -> token for s+1.

## The serial edges
1. layer l -> l+1 (64 layers, 2 all-reduces each: 128 link round trips per target forward).
2. the step edge t -> t+1 through Sample (low-cardinality: a token id) — already speculated by MTP.
3. draft s -> s+1 (2 serial MTP passes per step, each with a full lm_head read).
4. GDN recurrence over i inside a step (3 rows per request) and over t across steps.

## Which tensors lack which index (what can be shared)
| tensor | has | lacks -> shared across | consequence |
|---|---|---|---|
| W_in, W_out, W_qkv, W_o, W_gu, W_down (FP8) | l, c-slice | r, i, t, s | one read per window serves all ~45 rows (N=45): memory-bound (ridge ~82 rows bf16-MMA) |
| W_lm (bf16, 1.27 GB/GPU) | v-slice | r, i, t, **s** | read 3x per step (target + 2 drafts) because drafts are serial in s |
| MTP layer weights | — | r, i, s | read 2x per step (s serial) |
| KV cache k,v | r, l, j | — | per-request; small (16 layers, ~500 tokens avg) |
| GDN state S | r, l, hv | j (fixed size) | per request, **overwritten** each row i: 1.57 MB/req/layer/GPU; read 1 + written 3x per step today |
| conv state | r, l | j | tiny |
| partial sums y (pre all-reduce) | r, i, d, c | — | 45x5120 bf16 = 460 KB per all-reduce in decode; 2.6 MB in prefill steps |
| logits | r, i, v | — | 45x124160 bf16 = 11 MB per rank, gathered across c (only argmax needed at T=0) |

## Reductions and their combine tags (for Split)
- Σ_d / Σ_e / Σ_m inside GEMMs: `+` (split over c -> all-reduce).
- softmax_j (attention): online-softmax merge (FlashInfer already merges states over j chunks).
- **argmax_v (greedy sampling): combine tag = (max value, index)** — a 2-number merge, not a concat.
  Today the graph splits v over c but combines with a full all-gather (the concat tag), then argmax.
- GDN recurrence over i: not a commutative reduction (ordered scan), but its update is associative as
  a composition of affine maps (chunked/WY form in prefill kernels).
