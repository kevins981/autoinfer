# T1 sweep, step 2: the graph

Indices: r request, l layer (64 + MTP layer), i/j position, t step, d hidden (5120), m MLP inner
(17408), h q-head (24), g KV head (4), hv GDN value head (48), hk GDN key head (16), dk/dv (128),
v vocab (248,320), p TP rank (2, a split of h/hv/m/v), k draft depth (1..2).

## Data nodes (per layer l unless noted)
| node | indices | missing indices (sharing) | size / notes |
|---|---|---|---|
| x (residual) | r,i,d | l-free across... no (carries l) | bf16, 10 KB/row |
| W_qkvz_gdn, W_ba (GDN in-proj) | l,d,(hk,hv) | r,i,t | FP8 ~43 MB/GPU (+ small bf16 ba) |
| W_qkv_attn, W_o, W_gate_up, W_down | l,d,m / l,h,d | r,i,t | FP8; per GPU per layer: 16 (o) + 91 (gate_up) + 46 (down) MB |
| conv_state[r,l,hv..,3] | r,l | i | tiny |
| S[r,l,hv,dk,dv] GDN state | r,l,hv | i,j (in-place overwrite keep) | fp32, 24 heads/GPU x 64 KB = 1.57 MB per (r,l) per GPU |
| k,v[r,l,j,g,256] KV cache | r,l,j,g | t | fp8, 16 layers only, 2 KB/token/layer/GPU (2 KV heads/GPU) |
| logits[r,i,v] | r,i,v | — | bf16; vocab-parallel: each GPU v/2, then all-gather |
| W_lm[v,d] | v,d | r,i,t,l | bf16 1.27 GB/GPU (shared by target and the 2 MTP steps) |
| W_mtp (1 layer + fc) | k-free | r,i,t,k | ~0.2 GB/GPU FP8 |

## Compute nodes per layer (TP=2, rank p holds half of h/hv/m and of v)
```
GDN layer l (48x):  y1 = x W_qkvz[p] (+ba)  -> conv1d(conv_state) -> gated delta rule (S[r,l] read+write)
                    -> gated norm -> o_p = (.) W_out[p]            -> AR_1 (+ over p)   -> add+rmsnorm
Attn layer l (16x): y1 = x W_qkv[p] -> q/k norm, rope, gate -> KV write -> attn over j (FlashInfer)
                    -> o_p = (.) W_o[p]                            -> AR_1 (+ over p)   -> add+rmsnorm
MLP (64x):          h = silu(x W_gate[p]) * (x W_up[p]) -> o_p = h W_down[p]  -> AR_2 (+ over p) -> add+rmsnorm
Head:               logits_p = x W_lm[p] -> AG over v -> sample/verify (argmax at T=0)
MTP (x2 per step):  fc([emb(x̂), h]) -> 1 attn layer + MLP (2 ARs each) -> W_lm[p] -> AG -> argmax -> x̂[t+k]
```
Per decode step: 64 x 2 = 128 ARs (body) + 2 x 2 (MTP layer runs, 2 steps) = ~132-135 (nsys: 134.8/step),
~5 all-gathers (nsys 4.9/step: target logits + draft logits + metadata).

## Dependency edges that matter
- l -> l+1 serial (W carries l; edges carry high-cardinality x: no exact speculation, framework 04 §3 row 4).
- Inside a layer: GEMM_in -> mixer -> GEMM_out -> AR -> norm -> next GEMM: strictly serial per row.
  But *not* serial across rows r/i, and not across output columns d of GEMM_out (AR is elementwise
  over d: Rule 1 of 05 §2 -> AR can be split along d and pipelined with GEMM_out's other d-chunks).
- Step t -> t+1 through Sample (low cardinality token id): speculated by MTP (k=2 chain).
- S[r,l] in-place overwrite: verify of 3 rows must not commit rejected rows (the MTP kernel keeps
  per-row states; rollback handled by the kernel).

## Missing-index table (what can be shared / merged)
| tensor | missing | consequence |
|---|---|---|
| all W (body, lm_head, MTP) | r, i, t, k(for W_lm) | merge rows of all requests and of the 3 spec rows into one window: done (M~42-48). W_lm is read 3 times per step (target + 2 draft steps) because the draft steps are serial (k -> k+1 edge) |
| S[r,l] | i | the 3 spec rows of a request share one state read (MTP kernel does this) |
| k,v | t | standard KV cache |
| AR payload | — (carries r,i,d) | bytes scale with M; split along r/i or d is free of extra combine |
| logits | carries v | vocab split across ranks needs a gather for sampling, *unless* the reduction (argmax) is done per rank first (argmax over v is a reduction with combine tag max -> gather only (max, idx) pairs: Rule 2 of 05 §2) |

## Index sizes vs the 2 GPUs
h=24, g=4, hv=48, hk=16, m=17408, v=248,320: all split evenly by 2. No index is too small.
