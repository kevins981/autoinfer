# T1 sweep, step 2: the graph

Indices: `r` request (16), `i` position in the current window (decode: 3 rows per request = 1 real +
2 drafts; mixed step: + ~260 prompt rows of one request), `j` past position, `l` layer (64 + 1 MTP),
`d` hidden 5120, `m` MLP 17408, `h` q head (24), `g` KV head (4), `hk` GDN key head (16), `hv` GDN
value head (48), `c` conv tap (4), `v` vocab (248,320), `k` draft depth (1..2), `t` engine step.

## 2.1 Data nodes (what exists, which indices it carries)
| node | indices | size per GPU (TP=2) | kept across t? |
|---|---|---|---|
| `W_in[l]` (GDN qkvz) | l, d, (hk/hv split) | 42 MB fp8 | weight |
| `W_ba[l]` | l, d, hv | 0.5 MB bf16 | weight |
| `W_out[l]` (GDN out / attn o) | l, hv or h, d | 15.7 MB | weight |
| `W_qkvg[l]` (attn) | l, d, h/g | ~37 MB | weight |
| `W_gu[l]`, `W_dn[l]` | l, d, m | 89 + 45 MB | weight |
| `W_lm` | v, d | 1.27 GB bf16 | weight (shared by target and draft) |
| `W_emb` | v, d | 1.27 GB bf16 (rows gathered) | weight |
| `W_mtp` (fc 2d->d, 1 attn layer, 1 MLP) | d, m | ~0.2 GB | weight |
| `x[r,i,l,d]` residual | r,i,l,d | 10 KB/row | no |
| `kv[r,j,l,g,·]` (16 attn layers) | r,j,l,g | fp8, 2x4x256 B/2 GPUs = 1 KB/token/layer/GPU | yes (append-only) |
| `S[r,l,hv,128,128]` GDN state | r,l,hv (no j!) | fp32: 24 hv x 64 KB = 1.57 MB per request-layer per GPU; x48 x16 = 1.2 GB | yes (**in-place overwrite**) |
| `conv[r,l,c,·]` conv state | r,l | 3 taps x (qkv dims) bf16, small | yes (overwrite) |
| `logits[r,i,v]` | r,i,v | 42 x 124,160 x 2 B = 10.4 MB local; all-gathered to 20.8 MB | no |
| draft tokens `x̂[r,t+k]` | r,k | ints | across t (proposal) |

## 2.2 Compute nodes per layer (decode step, M = r x i = 42-48 rows)
GDN layer l:
```
n1[r,i]       = rmsnorm(x[r,i,l])                      (+ fused residual add)
qkvz[r,i,·]   = Σ_d n1[r,i,d] W_in[l,d,·]              # column split over hk/hv: no comm
ba[r,i,hv]    = Σ_d n1 W_ba                            
qkv'          = conv1d_c(conv[r,l], qkv[r,i])          # carries r,i, recurrence over i
S[r,l,i]      = g(a)·S[r,l,i-1] + β(b)·k^T(v - S k)    # delta rule: recurrence over i (and t)
o[r,i,hv]     = q[r,i]·S[r,l,i]  -> gated rmsnorm(z)
p[r,i,d]      = Σ_hv o[r,i,hv,·] W_out[l,hv,d]          # row split over hv: Σ over hv REDUCED
x[r,i,l+½]    = x + AllReduce_+(p)                     # COMBINE NODE #1 (link)
n2            = rmsnorm(x[l+½])
gu[r,i,m]     = Σ_d n2 W_gu[l,d,m]                      # column split on m: no comm
h[r,i,m]      = silu(gate)·up
y[r,i,d]      = Σ_m h W_dn[l,m,d]                        # row split on m: REDUCED
x[r,i,l+1]    = x + AllReduce_+(y)                       # COMBINE NODE #2 (link)
```
Attention layer l: same, with `qkvg = n1 W_qkvg`, q/k norm + partial RoPE, `kv[r,j=new]` appended,
`a[r,i,h] = softmax_j(q k[r,j]) v[r,j]` over j ∈ ancestors (paged, fp8), gate·a, `p = a W_o` (row
split over h, AllReduce).

Step tail:
```
logits[r,i,v_local] = Σ_d n_final W_lm[v_local,d]      # vocab split: v NOT reduced (Rule 1)...
logits_full          = AllGather_v(logits)              # ...but vLLM gathers all v (10.4 MB/GPU)
accept / reject      = greedy compare argmax_v(logits) with x̂         # argmax over v = reduction (max-merge)
draft k=1..2:  e = fc([emb(x̂_k), h_prev]) -> AllGather/AR -> MTP layer (attn + MLP, 2 AR)
               -> logits_draft = n W_lm (FULL vocab, bf16) -> AllGather_v -> argmax -> x̂_{k+1}
```
Mixed (prefill) step: the same graph with M ≈ 260-300 rows; GDN uses the chunked form (FLA chunk
kernels) for the prompt rows; attention uses the prefill path.

## 2.3 Dependency edges that matter
- Serial chain inside a step: l -> l+1 (128 combine nodes on the chain, each a link round trip).
- The serial sampling edge t -> t+1 through `x[r,t+1]` (the token): low cardinality, already speculated (MTP).
- Draft chain k=1 -> k=2: serial through `x̂`.
- GDN recurrence over i and t through `S` (in-place overwrite: the verify of 3 rows must produce
  states for every prefix to allow rollback; the kernel writes them per row, "mtp" variant).
- Attention: `kv[r,j]` append-only (rollback is free).

## 2.4 Missing-index observations (what can be shared)
- All `W[l]` lack r, i, t, k: shared across the 48 verify rows (already merged in one window) and
  across steps t (spec decode exploits this).
- `W_lm` lacks r, i, t **and** k: target lm_head (step tail) and draft lm_head (k=1, k=2) read the same
  1.27 GB tensor in three different windows -> 3 x 1.27 GB per step (2 of them for draft rows only).
  Draft k=1 and k=2 cannot share a window (serial edge), and the target tail cannot share with draft k=1
  (draft input depends on the accepted token).
- `W_mtp` lacks k: draft steps k=1,2 read it twice (small, 0.2 GB).
- `S[r,l]` lacks j: per-request fixed state, read+written every step (1.2 GB read + up to 1.2 GB
  written per GPU per step) — it carries r, so no sharing across requests (16x traffic of one).
- `argmax_v` is a reduction over v whose combine op is (max, argmax) — 8 bytes per row, while the
  current implementation moves the whole v axis (Rule 2 of 05: the combine tag says only (max,idx)
  must cross the link).
- RMSNorm's per-row scale `rsqrt(mean_d x²)` lacks d: it commutes with the next GEMM's Σ_d
  (Distributivity): `(s·x⊙γ)W = s·((x⊙γ)W)`. This makes the next GEMM independent of the full
  all-reduce result only through s — i.e. the GEMM over d-chunks can start on partial (chunked)
  all-reduce output and apply s at the end.
- Activations x carry r,i: TP needs them replicated (all-reduce); PP would need only 1 crossing per
  forward instead of 128.
