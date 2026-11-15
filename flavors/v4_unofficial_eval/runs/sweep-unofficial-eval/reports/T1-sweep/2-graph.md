# T1 sweep, step 2: the graph

Indices: r request (16), l layer (64 + 1 MTP), i query position (per step: 1 verified + 2 drafts
per request in decode; ~260 rows for a new prompt), j past position, t step, d hidden (5120),
m MLP inner (17408), h query head (24 attn / 48 GDN value heads), g KV head (4) / GDN key head (16),
v vocab (248,320), p TP rank (2: the device index created by Split).

## Data nodes
| node | indices | size | lacks (=> shareable across) |
|---|---|---|---|
| W_in[l] (GDN qkvz,ba), W_qkv[l] (attn), W_o[l], W_gu[l], W_dn[l] | l,d,(h or m) | 372-383 MB/layer fp8 | r, i, t: shared by every row in a window |
| W_lm (lm_head) | v,d | 2.54 GB bf16 | r,i,t; also shared by the target pass and both draft steps (same tensor) |
| W_emb | v,d | 2.54 GB bf16 | read only by row lookup (gather) |
| W_mtp (MTP layer + fc) | d,m | ~0.4 GB fp8 | r,i,t |
| x[r,l,i,d] residual stream | r,l,i,d | 10 KB/row bf16 | - |
| k,v[r,l,j,g] (16 attn layers) | r,l,j,g | 1 KB/token/layer/GPU fp8 | h (GQA: 6 q heads per g) |
| S[r,l,h] GDN state (48 layers) | r,l,h (t-free: overwritten in place) | 128x128 fp32 = 64 KB/head; 1.57 MB/layer/request/GPU | i: one state serves all positions of a request |
| conv_state[r,l,c] | r,l | 3 x (q,k,v channels) bf16, small | |
| logits z[r,i,v] | r,i,v | 0.5 MB/row bf16 (full vocab) | - |
| sampled/draft token ids x[r,t+1..t+3] | r,i | 4 B | (the low-cardinality choke-point) |

## Compute nodes per layer l (one step, rows i of all requests r)
```
GDN layer (48x):     x -RMSnorm-> in_proj (W_in, split h by p) -> conv1d(conv_state) -> [q,k,v,z,b,a]
                     -> delta-rule update S[r,l,h] over the step's rows i (sequential in i, reads+writes S)
                     -> o[r,i,h] -> gated norm (z) -> out_proj (W_o, reduce over h => split by p => ALL-REDUCE over p)
                     -> + residual
Attn layer (16x):    x -RMSnorm-> qkv+gate proj (split h,g by p) -> qk-norm + partial RoPE -> write k,v[j=i]
                     -> softmax_j(q k) v over j in Ancestors(i) (reads kv cache, split g by p)
                     -> sigmoid gate -> o_proj (reduce over h => ALL-REDUCE over p) -> + residual
MLP (64x):           x -RMSnorm-> gate_up (split m by p) -> SiLU*up -> down (reduce over m => ALL-REDUCE over p) -> + residual
Head:                final norm -> lm_head (split v by p) -> ALL-GATHER z over v -> argmax_v / rejection vs drafts
MTP draft (x2):      [emb(x̂), h_last] -> fc (2d->d; split + all-gather/reduce) -> 1 attn layer + MLP (2 all-reduces)
                     -> lm_head (shared W_lm, split v) -> ALL-GATHER z -> argmax -> next draft
```
## Dependency edges that matter
- Serial chain over l within a step (each all-reduce is a barrier: rows can't proceed to the next
  layer until the sum over p arrives). 128 all-reduces per target pass (2 per layer) + 2 per draft
  step + embedding all-reduce (vocab-parallel embedding: 1 per pass) -> ~135 per decode step (matches the profile's 135).
- Serial edge t -> t+1 through the sampled token (choke-point, cardinality V): what MTP speculates.
- Draft step 1 -> draft step 2 (serial through the draft token).
- GDN: S[t-1] -> S[t] serial in i within a request (in-place overwrite keep); attention: append-only keep of k,v.
- Reductions and their combine tags: sum over h / m across p (+, all-reduce); softmax over j (online-softmax merge,
  flashinfer splits j inside a GPU: merge-states kernel); **argmax over v across p (combine = (max, argmin index))**:
  currently implemented as an all-gather of the full z then argmax, i.e. the combine is not exploited.

## Missing-index observations (what can be shared)
1. All W lack r,i,t: every row in a window shares the weight read (decode M=48 << ridge 82: memory-bound).
2. W_lm lacks the "which pass" index: target pass and the two draft passes read the same 1.27 GB/GPU, but in 3 different
   windows (serial dependency between them), so it is read 3x per step. Shared-read only if merged into one window (impossible: draft 1 depends on target sample... see 4-knobs).
3. z (logits) has v; the consumer (greedy argmax / rejection at temperature 0) only needs (max, argmax) per row:
   a v-free summary. All-gathering z moves 0.5 MB/row/GPU when 8 bytes/row suffice.
4. S lacks i: all rows of a request in one step share one read+write of S (already exploited by the fused MTP decode kernel).
5. k,v lack h: 6 q heads share one KV head read (exploited by flashinfer GQA).
6. The all-reduce input partial sums o_p[r,i,d] carry p; the sum is needed on both devices (replicated residual stream).
   Alternatives: reduce-scatter + all-gather (sequence parallel) moves the same bytes over the link.
