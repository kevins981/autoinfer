# T3 sweep, step 2: the graph

Indices: `r` request (16 live), `t` generation step, `i` position in the step's window (decode
verify: 3 rows per request = 1 accepted/bonus + 2 drafts; prefill chunk: up to ~260 rows), `j`
past position, `l` layer (64), `d` hidden 5120, `m` MLP 17408, `h` q/value head, `g` KV head /
GDN key head, `c` channel of the GDN conv, `v` vocab 248320, `p` TP rank (2), `k` draft depth (2).

## Per-layer template (one target layer `l`, per row `(r,i)`)
```text
# entry: combine of the previous layer's partial sums  (TP rank p holds a partial)
x[r,i,d]        = Σ_p y_part[p,r,i,d]  + res[r,i,d]            COMBINE (+): all-reduce over p
xn[r,i,d]       = RMSNorm_d(x) ; xq = Quant_g128(xn)            reduction over d (local, full d on each p)

# GDN layer (48 of 64)
qkvz[r,i,c]     = Σ_d xq W_qkvz[l,d,c]                          W has no r,i  -> shared by all rows
ba[r,i,h]       = Σ_d xn W_ba[l,d,h]
u[r,i,c]        = conv1d_k4( qkv[r, i-3..i, c] )                READS conv state (last 3 rows of r) -> keep
S[r,l,h,t]      = α(ba) S[r,l,h,t-1] + β(ba) k^T (v - S k)      RECURRENCE over t/i  (in-place keep, fp32)
o[r,i,h,:]      = q[r,i] · S[r,l,h,i]                           per row, needs S after row i
og[r,i,·]       = RMSNorm(o) * silu(z)
y_part[p,r,i,d] = Σ_{h∈p} og W_out[l,h,d]                       split h over p -> partial sum

# full-attention layer (16 of 64)
qkvg[r,i,·]     = Σ_d xq W_qkv[l,d,·] ; q,k <- RoPE(pos)
k,v[r,l,j]      KEPT (fp8)                                       append-only keep
a[r,i,h]        = softmax_j(q[r,i,h]·k[r,l,j,g(h)]) v           reduction over j (softmax-merge)
y_part          = Σ_{h∈p} (a * σ(gate)) W_o[l,h,d]

# MLP (every layer)
x2              = Σ_p y_part + x  -> RMSNorm -> Quant            COMBINE (+): all-reduce #2
h[r,i,m]        = silu(Σ_d x2q W_g[l,d,m]) * Σ_d x2q W_u[l,d,m]  split m over p (Rule 1: no comm)
y_part[p,r,i,d] = Σ_{m∈p} hq W_d[l,m,d]                          partial -> next layer's COMBINE
```
Output (after layer 63): `logit[r,i,v] = Σ_d norm(x) W_lm[d,v]` (bf16, v split over p),
`tok[r,i] = argmax_v` (combine: argmax-merge of (value,id) pairs over p).
Serial edge: `x[r, i+1] = Embed(tok[r,i])`.

## Speculation sub-graph (MTP, chain of k=2)
```text
draft step 1: e1[r] = MTPLayer( fc[ Embed(tok[r,t]) ; h_target[r,t,d] ] )   (1 layer, + its own KV)
              x̂[r,t+1] = argmax_v Σ_d e1 W_lm_fp8[d,v]
draft step 2: e2[r] = MTPLayer( fc[ Embed(x̂[r,t+1]) ; e1 ] ) ; x̂[r,t+2] = argmax(...)
verify:       target forward on rows (tok[t], x̂[t+1], x̂[t+2]) -> argmax at 3 rows -> accept prefix
```
Draft steps carry `k`: step 2 depends on step 1 through the discrete token `x̂[t+1]` and the
continuous `e1`.

## Data nodes and their indices (what is shared, Missing-Index Rule)
| Data node | Indices | Size per GPU | Missing indices -> shareable across |
|---|---|---|---|
| Body weights `W[l,·,·]` | l, d, m/c/h | 12.2 GB total, ~190 MB per layer | r, i, t, k -> all rows of a step share one load |
| lm_head `W_lm` (bf16) | d, v | 1.27 GB | r, i, t -> shared by all logit rows of a step |
| Draft lm_head (fp8) | d, v | 0.64 GB | r, i, t; **also shared by draft step 1 and step 2 only if both are in one window - they aren't (serial edge)** |
| MTP layer weights | d, m | ~0.19 GB | r, i; same across k (draft steps 1 and 2 read the same MTP weights!) |
| Embedding | v, d | 1.27 GB (rows only gathered) | - |
| KV cache `k,v[r,l,j]` | r, l, j, g | ~0.1 GB live | i (the 3 verify rows of r share r's KV) |
| GDN state `S[r,l,h]` | r, l, h (and t: overwritten) | 1.57 MB x 48 x 16 = 1.2 GB | i: the 3 verify rows walk the same S sequentially |
| Conv state | r, l, c, 3 rows | small | - |
| Activations `x[r,i,d]` | r, i, d | 10 KB/row | - |

## Edges that matter for the sweep
1. **Layer chain** `l -> l+1` (serial, through a 2-rank COMBINE). 128 all-reduces per target
   forward (2 per layer) + the MTP layer's 2 per draft step.
2. **Within a layer**: norm -> in_proj -> (conv, recurrence S / attention) -> out_proj ->
   COMBINE -> norm -> gate_up -> act -> down -> COMBINE. Every edge serial for a given row; rows
   `(r,i)` are independent everywhere **except** (a) the GDN recurrence and conv (rows of the
   same r in order), (b) attention (rows read earlier rows' k,v of the same r). So GEMMs, norms,
   all-reduces are row-independent: splitting `r`/`i` into row groups is free of combine nodes.
3. **Step chain** `t -> t+1` via the sampled tokens (low cardinality: speculated by MTP).
4. **Draft chain** `k=1 -> k=2` within a step (discrete token + continuous e1).
5. **Verify -> next draft**: draft step 1 for step t+1 needs the target's hidden at the accepted
   position of step t (the continuous `h_target`), so the drafter cannot run ahead of the verify.
6. **CPU edges**: scheduler -> input prep -> graph launch -> sample -> output processing; vLLM
   async scheduling may overlap the CPU part of step t+1 with GPU step t.

## Which tensors lack which indices (sharing opportunities)
- Body weights lack r, i, t: merging more rows into one window raises reuse N (batching,
  speculation, chunked prefill already do this). At M=48 rows, N=48 << ridge (~165): one more row
  of verify costs only its compute and activation bytes, until ~165 rows.
- Draft weights (MTP layer, draft lm_head) lack k: draft steps 1 and 2 read them twice because the
  edge k=1 -> k=2 is serial. (A tree/parallel draft that put k=1,2 in one window would read them
  once, but MTP's step 2 depends on step 1.)
- GDN state S lacks i: the 3 verify rows share one read of S per request (the champion's MTP
  kernel already reads S once and walks the 3 rows; whether it also writes S once or 3 times is a
  calibration question).
- All-reduce payloads carry r, i: a COMBINE on half the rows is independent of the other half's
  GEMM (what the champion's row-split exploits in mixed steps; decode steps don't use it).
