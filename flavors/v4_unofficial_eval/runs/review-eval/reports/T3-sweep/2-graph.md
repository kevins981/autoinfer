# T3 sweep, step 2: the graph

Indices: `r` request (16 live), `l` layer (64 target + 1 MTP), `i,j` positions, `t` step, `u` row inside the verify window
(u = 0 is the last accepted token, u = 1,2 the drafts; M = 3 rows/request, 48 rows/step), `d,d'` hidden (5120),
`m` MLP inner (17408), `h` attention Q head (24), `g` KV head (4), `hk` GDN key head (16), `hv` GDN value head (48),
`a,b` GDN head dims (128), `v` vocab (248,320), `p` TP rank (2), `s` draft step (1..2).

## 1. One target layer, per scalar index

Shared prologue (every layer):
```
xn[r,u,d]      = RMSNorm_d(x[r,u,·])[d] * w_norm[l,d]          (reduction over d; fused with residual add)
```
Full-attention layer (l mod 4 = 3):
```
qg[r,u,h,c]    = Σ_d xn W_qg[l,d,h,c]   c in 0..511 (q 256 + gate 256)      [FP8 W, no r/u/t index]
k,v[r,u,g,c]   = Σ_d xn W_k/W_v[l,d,g,c]
q,k            = RMSNorm_c(q), RMSNorm_c(k); RoPE on c<64 (position-dependent R[i])
KEEP k,v[r,l,j,g,c] (fp8 KV cache, paged, prefix-cacheable)
s[r,u,h,j]     = Σ_c q[r,u,h,c] k[r,j,g(h),c]     j in Ancestors(i(r,u)) (prefix + earlier rows u'<=u)
α              = softmax_j s ;  o[r,u,h,c] = Σ_j α v[r,j,g(h),c]
o'             = o * sigmoid(gate)                (output gate)
y[r,u,d]       = Σ_{h,c} o' W_o[l,h,c,d]           (TP: split over h -> all-reduce +)
```
GDN layer (l mod 4 != 3):
```
qkvz[r,u,·]    = Σ_d xn W_qkvz[l,d,·]  ; ba[r,u,·] = Σ_d xn W_ba[l,d,·]
q,k,v          = conv1d_k4 over positions (KEEP conv_state[r,l,3 last positions]) ; SiLU ; L2norm(q,k)
g[r,u,hv]      = -exp(A_log) * softplus(a + dt_bias) ; beta = sigmoid(b)
S[r,l,hv,a,b] (fp32, KEEP, in-place overwrite):
  S_u = exp(g_u) S_{u-1} + beta_u k_u^T (v_u - exp(g_u) S_{u-1} k_u)     (delta rule, serial in u)
o[r,u,hv,b]    = Σ_a q_u[a] S_u[a,b]
o'             = RMSNorm_b(o) * SiLU(z)            (gated norm, per head)
y[r,u,d]       = Σ_{hv,b} o' W_out[l,hv,b,d]       (TP: split over hv/hk -> all-reduce +)
```
MLP (every layer):
```
x1 = x + y (residual) ; xn2 = RMSNorm(x1)
h[r,u,m]  = SiLU(Σ_d xn2 W_g[l,d,m]) * Σ_d xn2 W_u[l,d,m]        (TP: split m, no comm)
y2[r,u,d] = Σ_m h W_d[l,m,d]                                     (TP: all-reduce +)
x[l+1]    = x1 + y2
```
Head and sampling:
```
logit[r,u,v] = Σ_d RMSNorm(x[L])[d] W_lm[v,d]     (bf16, 1.27 GB/GPU; TP: split v -> argmax-merge / gather)
target token  = argmax_v logit  (temp 0)  ; accept drafts while draft[u] == target[u-1]   (exact verify)
```
MTP drafter (per draft step s = 1,2; only for the next window):
```
e[r,s]    = Embed(token_{s-1})  ; hcat = [RMSNorm(e) ; RMSNorm(hidden_prev)] ; x0 = Σ hcat W_fc   (bf16 fc)
x_mtp     = one full-attention layer + MLP (FP8), own KV cache over MTP positions
draft_s   = argmax_v Σ_d RMSNorm(x_mtp) W_lm[v,d]     (shared target lm_head; local argmax per rank + 2-value merge)
```
THE SERIAL EDGES: `token[t] -> Embed -> layer 0 at t+1` (target) and `draft_s -> Embed -> MTP step s+1` (drafter: 2 serial
draft steps per window, each a full lm_head read).

## 2. Data nodes and their indices (what lacks which index = what can be shared)
| data node | indices | lacks | size per GPU | sharing consequence |
|---|---|---|---|---|
| W_qkvz, W_qg, W_k/v, W_o, W_out, W_g/u/d | l, d, out | **r, u, t** | 12.17 GB total | shared by all 48 rows of a step (N=48) and across draft/verify rows |
| W_lm (target) | v, d | r, u, t, l | 1.27 GB bf16 | read once per verify window (48 rows); **read again per draft step** (2x) by the drafter |
| W_mtp (1 layer) | d, ... | r, u, t | ~0.2 GB FP8 + 26 MB fc | read once per draft step (2x per window) |
| Embed | v, d | r, t | 1.27 GB (gather rows only) | gathers ~2 KB/row |
| KV cache k,v | r, l(16), j, g | — | ~16 KB/token/GPU fp8 | per request (prefix caching dedups identical prefixes: none here) |
| GDN state S | r, l(48), hv, a, b | **j, t** (fixed size) | 1.57 MB/req/layer, 1.21 GB for 16 req | **in-place overwrite**: rollback for spec decode -> the champion keeps one state slot per window row u (3 slots/req) and writes all 3 |
| conv_state | r, l, 3 positions | — | 15 KB/req/layer | as S |
| activations x, xn, h | r, u, d/m | l (reused buffers) | 48x17408x2 B = 1.7 MB max | tiny vs weights at decode |
| logits | r, u, v | — | 48 x 124,160 x 4 B = 24 MB | target: needed for argmax; drafter: avoided by local argmax |

## 3. Dependency edges that matter for cost
- Within a layer: norm -> GEMM -> (attention | GDN recurrence) -> GEMM -> **all-reduce** -> residual -> norm -> GEMM -> act -> GEMM -> **all-reduce**.
  2 all-reduces per layer x 64 layers = 128 dependent cross-GPU exchanges per target forward, +2 per MTP step.
- GDN rows u = 0,1,2 are serial in the recurrence (S_u depends on S_{u-1}) but cheap; the window's GEMMs merge all u.
- Verify -> accept count -> (a) which state slot the next step reads, (b) which drafts are re-embedded. Step t+1's target forward depends on
  step t's accept result (serial edge) and on step t's draft tokens (drafter runs after the target forward, before step t+1).
- CPU edges (not in the framework's template): scheduler -> input prep -> graph replay; per step the drafter rebuilds FlashInfer
  attention metadata eagerly (log); detokenize/stream on the API process (overlaps GPU).
- Prefill rows (~260/request) join a step as a chunk (mixed step, PIECEWISE graph, NCCL all-reduce since >96 rows).

## 4. TP split as the framework derives it (index rules)
TP=2 splits `m` (MLP: Rule 1 at W_g/u, Rule 2 all-reduce after W_d), `h/g` (attention: heads independent, all-reduce after W_o),
`hk/hv` (GDN: heads independent incl. state S, all-reduce after W_out), `v` (lm_head: argmax-merge combine, 2 numbers per row).
The residual stream x[r,u,d] is replicated on both ranks (that is why each all-reduce is a full [rows,5120] bf16 sum).
