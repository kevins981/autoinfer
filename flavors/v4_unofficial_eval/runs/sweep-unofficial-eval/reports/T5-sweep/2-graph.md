# T5 sweep, step 2: the graph

Indices: r request (16 live), t step, i position in step (decode: 3 rows per request = 1 real + 2
drafts; prefill: chunk rows), j past position, l layer (64 + MTP), h q/v head, g kv head / GDN k-head
group, d hidden (5120), m MLP inner (17408), v vocab (248320), p TP rank (2).

## 1. One target layer, per scalar index (GDN layer, l not in {3,7,..})
```
x̂[r,i,d]      = RMSNorm(x[r,l,i,·])[d]                                  (residual add fused)
P[r,i,c]      = Σ_d x̂[r,i,d] W_qkvz[l,d,c]   c ∈ q(2048)|k(2048)|v(6144)|z(6144)       FP8 W
ab[r,i,h]     = Σ_d x̂ W_ba[l,d,h]                                                  (tiny)
u[r,i,c]      = Σ_{τ<4} w_conv[l,c,τ] P[r,i-τ,c]  -> silu         CONV STATE: P[r,i-3..i-1] kept
k̃,q̃           = L2norm(k), L2norm(q) per head
γ[r,i,h]      = exp(-softplus(a+dt_bias)·exp(A_log)) ;  β[r,i,h] = sigmoid(b)
S[r,l,h,i]    = γ S[r,l,h,i-1] + β k̃ᵀ(v - S[i-1]ᵀk̃)       (delta rule; S: 128x128 fp32, t-free, in-place)
o[r,i,h,·]    = S[r,l,h,i]ᵀ q̃ · scale
ô             = RMSNormGated(o, silu(z))
y[r,i,d]      = Σ_c ô[r,i,c] W_out[l,c,d]                      reduce over c = heads -> ALL-REDUCE (TP)
x'            = x + y ;  x̂' = RMSNorm(x')
gu[r,i,m]     = Σ_d x̂' W_gu[l,d,m] ; h = silu(g)·u
y2[r,i,d]     = Σ_m h W_down[l,m,d]                             reduce over m -> ALL-REDUCE (TP)
x[r,l+1,i]    = x' + y2
```
Full-attention layer: replace the GDN block by
```
qg,k,v        = Σ_d x̂ W_qkv[l,d,·]  (q 6144 + gate 6144 | k 1024 | v 1024), q/k RMSNorm, partial RoPE(pos i)
KV[r,l,g,j]   = fp8(k,v)             (append-only keep, paged, 16 layers)
s,α,a         = softmax_j (q·k[j]) over j ∈ Ancestors(i) (prefix + earlier rows of the step: causal)
y             = Σ_c (a·sigmoid(gate))[c] W_o[l,c,d]              -> ALL-REDUCE
```
Head and sampling:
```
logit[r,i,v]  = Σ_d x̂_L[r,i,d] W_lm[d,v]          (bf16, vocab split over p -> local argmax + tiny gather)
tok[r,i]      = argmax_v logit                     (temperature 0)
accept        = longest prefix with draft == tok   (greedy verify; low-cardinality choke point)
MTP drafter   : x_d = fc([emb(tok), h_L]) -> 1 attention+MLP layer -> argmax over v<98304 (FP8 head);
                run once on all accepted rows, then 1 more chained step (2 drafts)
x[r,1,i+1]    = Embed(tok)                         THE SERIAL EDGE
```

## 2. Data nodes, their indices, and what lacks which index
| data node | indices | missing (=> shareable across) | size (per GPU) | kept across t? |
|---|---|---|---|---|
| W_qkvz, W_out, W_gu, W_down, W_qkv, W_o | l, d, c/m | **r, i, t** (batch merge) | 12.2 GB body | static |
| W_lm (bf16) | d, v | r, i, t | 1.27 GB | static |
| W_draft head (fp8, v<98304) | d, v | r, i, t | 0.25 GB | static |
| W_mtp layer | d, m | r, i, t, **l** (one layer reused for both draft steps: shares weights across draft steps -> a "looped" weight, framework 04 §5.2) | 0.19 GB | static |
| GDN state S | r, l, h | i, j | 1.57 MB per (r,l); 75.5 MB per r; **1.2 GB @16** | in-place overwrite (HEAD: lazy commit + pending rows) |
| conv state | r, l, c, τ<3 | | ~0.1 MB per (r,l) | in-place |
| KV cache | r, l(16), g, j | i | fp8: 2x4x256 /2 B per token per layer per GPU = 1 KB/token/layer -> 16 KB/token; ~260+228 tokens x 16 req ~ 125 MB | append-only |
| activations x, x̂, P, h | r, i, d | l-chained | M x 5120 x 2 B ~ 0.5 MB @M=48 | no |
| logits | r, i, v | | M x 124K x 4 B: avoided by local argmax | no |

## 3. Edges that matter
- **Layer chain** l -> l+1 (serial, with 2 all-reduces per layer on it: 128 exchanges / step).
- **Serial sampling edge** step t -> t+1 (crossed by speculation: MTP with 2 drafts, verify window M = 3 rows / request).
- **Recurrence edge** S[i-1] -> S[i] inside GDN (sequential over i within a request; parallel over r, h).
- **Drafter chain**: target final hidden -> draft step 1 -> draft step 2 (serial; the draft's weight W_mtp lacks the draft-step index).
- **CPU edges** (not in the framework's graph): scheduler -> input prep -> graph launch -> sample -> detokenise; with async scheduling these overlap the GPU except where a step needs a device->host sync.

## 4. Sharing opportunities read from the missing-index table
1. Weights lack r, i, t: batch merging (already: 16 requests x 3 rows = M 48 per window).
2. Weights lack t: speculation (already: chain of 2). More rows per window are almost free while M < N_ridge (~80 W8A16): 16 x 4 = 64 rows still below ridge.
3. W_mtp lacks the draft-step index: both draft steps could share one weight load only if they were in one window, but step 2 depends on step 1 (serial) -> no without guessing.
4. W_lm lacks i: target head over 48 rows is one load (1.27 GB/GPU, ~1.3 ms): the largest single tensor per step after the body.
5. S lacks i: the verify window (3 rows) can read S once and write once (HEAD does, plus lazy commit every 2 steps).
6. KV lacks i: the 3 verify rows share one KV read (FlashInfer does).
7. Prompt prefill rows of different requests share weights (chunked prefill mixed with decode rows).
