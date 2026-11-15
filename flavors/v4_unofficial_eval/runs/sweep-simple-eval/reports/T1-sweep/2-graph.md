# T1 sweep, step 2: the graph

Indices: r request (16), l layer (64 + 1 MTP), i/j positions, t step, d hidden (5120), m MLP
(17408), h/g heads (attention: 24 q / 4 kv; GDN: 16 key / 48 value heads), v vocab (248,320),
s draft slot (0..k, k = 2 MTP drafts), p TP rank (2).

## One target layer, per row (r, i)
```text
GDN layer l (48 of them):
  xn      = RMSNorm(x[r,i] + res)                               (fused add+norm)
  qkvz    = Σ_d xn[d] Wqkvz[l,d,·]       ba = Σ_d xn[d] Wba[l,d,·]      (W: no r,i,t)
  q,k,v   = causal_conv1d(qkvz, conv_state[r,l])   conv_state: kept, overwrite, carries r
  S[r,l,h,i] = g[i] S[r,l,h,i-1] + β k^T (v - S k)               (delta rule; S fp32 [128x128], carries r)
  o[r,i,h] = q S[r,l,h,i]  ; o = gated_norm(o, z)
  y       = Σ_{h,e} o W_out[l,(h,e),d]                           <- reduction over h: TP split => AllReduce #1
  x2n     = RMSNorm(y + x)
  u       = SiLU(Σ_d x2n Wg[l,d,m]) * Σ_d x2n Wu[l,d,m]
  y2      = Σ_m u[m] Wd[l,m,d]                                    <- reduction over m: AllReduce #2
Full-attention layer l (16):
  q(gated),k,v = Σ_d xn Wqkv ; qk-norm ; partial RoPE(i)
  k,v[r,l,j]   kept (FP8 KV cache, append-only, carries r,j)
  a[r,i,h]     = Σ_j softmax_j(q[i]·k[j]) v[j]   (j ∈ ancestors: past + earlier draft rows)
  a           *= sigmoid(gate) ; y = Σ_h a Wo     <- AllReduce #1, then MLP as above (#2)
Head:
  logits[r,i,v] = Σ_d RMSNorm(x)[d] W_lm[d,v]     (v split across p => all-gather of logits)
  x[t+1]       = argmax_v logits                  (temperature 0)   THE SERIAL EDGE
MTP drafter (s = 1..2, serial):
  h_s = MTPLayer(fc([embed(x̂_s) ; hidden_{s-1}]))  (one full-attn layer + MLP, 2 AllReduces)
  x̂_{s+1} = argmax(W_lm · RMSNorm(h_s))           (shares W_lm => 1.27 GB/GPU read per draft step)
Verify: target forward on rows [x[t], x̂_1, x̂_2] per request (M = 3 rows/request), accept the
longest matching prefix (exact, greedy).
```

## Data nodes and their indices (what lacks which index = what can be shared)
| tensor | indices | size (per GPU, TP=2) | lacks | consequence |
|---|---|---|---|---|
| W (all linear weights) | l, d, m/heads | 12.2 GB FP8 | r, i, t | every row of a step shares one read: N = rows in the window |
| W_lm (lm_head) | d, v | 1.27 GB bf16 | r, i, t, **l, s** | shared by target and both draft steps (but they are in different windows => read 3x/step) |
| W_embed | v, d | 1.27 GB bf16 | r, i, t | gathered rows only |
| k,v cache | r, l(16), j, g | ~1 KB/token/layer/GPU fp8 | t | append-only, rollback-free for MTP |
| GDN state S | r, l(48), h, 128x128 | 1.57 MB/layer/req fp32 | j | in-place overwrite: MTP verify needs per-row states / rollback |
| conv_state | r, l, 3 x channels | small | j | in-place |
| activations x, y | r, i, d | 10 KB/row bf16 | — | the all-reduce payload: 10 KB x rows x 128 per forward |
| logits | r, i, v | 0.5 MB/row/GPU bf16 | — | all-gathered (V/2 per GPU) every forward and draft step |

## Edges that cross devices at TP=2 (Split rules of 05 §2)
- d-reduction after W_out / o_proj and after W_down (h and m split: Rule 2, `+` combine) => 2
  AllReduces per layer, 130 per decode step (128 target + 2 per MTP step).
- v split in lm_head: the consumer (argmax) is a reduction over v with combine `argmax-merge`
  (max, index) -- **the baseline instead gathers the whole logits row** (all-gather), i.e. it treats
  argmax as if it needed the full tensor.
- Nonlinearity barriers (Rule 3): RMSNorm and SiLU sit right after the reductions, so the combine
  must happen before them (no deferring the AR across the norm).

## Serial edges (Speculate candidates)
- token edge x[t] -> x[t+1] (low cardinality: already speculated, MTP k=2).
- layer edges y[l] -> x[l+1] (high cardinality, W carries l: no exact speculation).
- GDN state edge S[i-1] -> S[i] (high cardinality).
- draft step edge x̂_s -> x̂_{s+1} (serial inside the drafter).
