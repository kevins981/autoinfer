# T4 sweep, step 2: the graph

Indices: r request, l layer, i/j position (t = current step position), d/d' hidden (5120), m MLP inner (17408),
h query head, g KV head (full attn), hk GDN key head (16), hv GDN value head (48), c GDN channel (k/v dim 128),
s draft step (1..2), V vocab (248,320), u TP rank (device).

## 1. Per-layer graph (one layer l, one token row (r,i))

### Shared prologue (every layer)
```
x[r,i,d]        = residual stream (bf16)                         input of layer l
n1[r,i,d]       = RMSNorm_d(x)[d] * (1+w1[l,d])                   reduction over d (row scalar rstd[r,i])
qa[r,i,d]       = Quant128(n1) (FP8 codes + scale per 128-group)   [W8A8 path]
```
### GDN layer (48 of 64)
```
p[r,i,n]        = Σ_d qa[r,i,d] Wqkvz[l,d,n]                      n over 16384 (q,k,v,z); W has no r,i,t
ab[r,i,hv]      = Σ_d n1[r,i,d] Wba[l,d,hv]                        decay a, beta b
cq,ck,cv[r,i,c] = conv1d_4( p[r,i-3..i, qkv] )                    reads 3 past rows: KEPT conv state[r,l]
qn,kn           = L2norm_c(cq), L2norm_c(ck)                       reduction over c (128)
S[r,l,hv,t]     = exp(a_t) S[r,l,hv,t-1] (I - b_t k kᵀ) + b_t v kᵀ   RECURRENCE over t (fp32, 128x128), KEPT across steps
o[r,i,hv,c]     = S[r,l,hv,i] q[r,i,hk(hv)]                         
og[r,i,hv,c]    = RMSNorm_c(o) * w * silu(z[r,i,hv,c])              gated norm, reduction over c
y_part[u,r,i,d] = Σ_{(hv,c)∈rank u} og[r,i,hv,c] Wout[l,(hv,c),d]   PARTIAL over TP rank u
y[r,i,d]        = Σ_u y_part[u,r,i,d]                               COMBINE (+): all-reduce #1
```
### Full-attention layer (16 of 64)
```
qkvg[r,i,n]     = Σ_d qa Wqkv[l,d,n]                                n = q(24x256), gate(24x256), k(4x256), v(4x256)
q,k             = RMSNorm_256(q), RMSNorm_256(k), partial RoPE(pos i)
k,v[r,l,j,g]    = KEPT (fp8 KV cache, paged)
s[r,i,j,h]      = Σ_c q[r,i,h,c] k[r,j,g(h),c]    j ∈ Ancestors(i)   (causal; tree mask for draft rows)
a[r,i,h,c]      = softmax_j(s) · v                                  softmax-merge combine over j
ag              = a * sigmoid(gate)
y_part          = Σ_{(h,c)∈u} ag Wo[l,(h,c),d] ;  y = Σ_u y_part     all-reduce #1
```
### MLP (every layer)
```
x'              = x + y                                              residual add
n2              = RMSNorm_d(x') (1+w2) ; qa2 = Quant128(n2)
gu[r,i,m2]      = Σ_d qa2 Wgu[l,d,m2]                               m2 = gate|up, split by m across ranks (no comm)
hm[r,i,m]       = silu(gate) * up ; qh = Quant128(hm)
z_part[u,r,i,d] = Σ_{m∈u} qh Wdown[l,m,d] ;  z = Σ_u z_part         all-reduce #2
x[l+1]          = x' + z
```
### Head, draft, sampling (per step)
```
hf      = RMSNorm(x[L])
logit[r,i,v] = Σ_d hf Wlm[d,v]   (bf16, vocab split across ranks; 1.27 GB/GPU)
tgt[r,i] = argmax_v logit        reduction over v: local argmax + all-gather of (val,id) pairs
accept   = longest prefix of drafts equal to tgt; bonus token                (VERIFY)
MTP draft s=1..2: e = fc([Embed(tok), hidden]) -> 1 full-attn decoder layer (own KV cache) -> draft lm_head (FP8, 151.7K vocab) -> argmax
x[r,1,t+1] = Embed(tgt)                                                      THE SERIAL EDGE
```

## 2. Indices carried by each tensor (Missing-Index Rule)
| Tensor | Carries | Lacks | Consequence |
|---|---|---|---|
| All W (in_proj, out, qkv, o, gate_up, down, lm_head, draft) | l, (n,d,m) | r, i, t | shared by every row of a step: weight read once per step at N = rows in the window (48 decode / 48+P mixed) |
| GDN state S[r,l,hv] | r, l, hv, (t via overwrite) | i except current | per-request, NOT shared: 1.57 MB/layer/request/GPU fp32, read + written every step -> N = 3 rows (1 + 2 drafts) per byte read |
| conv state | r, l | | small (3 rows x 10240 ch) |
| KV cache k,v[r,l,j,g] | r, l, j, g | h (6 q heads share one g) | GQA: N = 6 heads x 3 rows per KV byte |
| Activations x, y, partials | r, i, d | l is sequential | the AR payload: rows x 5120 x 2 B |
| rstd (norm scale) | r, i | d | a row scalar: the only part of RMSNorm that needs the whole d |
| tgt / drafts | r, i | d, v | low-cardinality choke point (one id) |

## 3. Dependency edges and their kind
1. Within a layer: n1 -> in_proj -> (GDN conv/recurrence | attention) -> out_proj -> **AR#1** -> add+norm -> gate_up -> silu -> down -> **AR#2** -> next layer. Strict serial chain per row; rows (r,i) are independent everywhere except: GDN recurrence (i depends on i-1 within a request), attention (i reads j<=i), conv (i reads i-3..i-1).
2. Across layers: x[l+1] = f(x[l]) serial; weights differ per l (no sharing across l).
3. Across steps: the serial sampling edge tgt[t] -> Embed -> step t+1 (broken by MTP speculation). Kept data across steps: S, conv state, KV.
4. Across requests: none (no shared prefixes in ShareGPT beyond the chat template, ~tens of tokens).
5. Cross-device edges (TP=2): exactly the two `Σ_u` combines per layer (128 per forward), plus the lm_head argmax pair gather, plus the MTP layer's 2 combines x 2 draft steps (+ its own head gather).

## 4. Reductions and their combine tags (where a split inserts a combine)
| Index reduced | Where | Combine | Nonlinearity right after? |
|---|---|---|---|
| d (in in_proj/qkv/gate_up) | GEMM K | + | yes for gate_up (silu) — TP splits n/m instead, so no comm |
| (hv,c) / (h,c) in out_proj/o_proj | GEMM K | + (AR#1) | RMSNorm (row scalar) |
| m in down_proj | GEMM K | + (AR#2) | RMSNorm (row scalar) |
| d in RMSNorm | rstd | Σ x² then rsqrt | it IS the nonlinearity, but it is only a per-row scalar: norm(x)·W = rstd(x) · ((x⊙(1+w)) W) |
| j in attention | softmax | online-softmax merge | |
| c in GDN q/k L2 norms, output gated norm | per head | per head (no cross-rank: heads split) | |
| v in argmax | lm_head | (max, id) merge | |
| t in GDN recurrence | across steps | ordered product (not commutative) | |

Observation for step 4: RMSNorm's cross-d dependency is a single scalar per row. So `n·W = rstd ⊙ (x⊙g)·W` with `x = y_part[0] + y_part[1] + residual`: by distributivity the next GEMM can be computed on the local partial and the peer partial separately and the row scalar applied after. That is the only algebraic way to start the next layer's GEMM before the all-reduce finishes (but it doubles that GEMM's activation side and needs W read once with both partials present, see 4-knobs).

## 5. Window-level view of one decode step (champion), per GPU
```
[embed] -> 64 x { norm+quant | in_proj GEMM | mixer (GDN kernel / attn) | out GEMM | AR#1 (+L2 prefetch gate_up) |
                  add+norm+quant | gate_up GEMM | silu+quant | down GEMM | AR#2 (+L2 prefetch next in_proj) }
       -> final norm -> lm_head GEMM (bf16) -> local argmax -> all-gather pairs -> verify
       -> MTP x2 { fc, 1 attn layer (2 ARs), draft lm_head FP8 -> argmax pair gather }
```
All windows are serial on one stream in decode (the row-split side stream is used only for steps >= 256 tokens).
