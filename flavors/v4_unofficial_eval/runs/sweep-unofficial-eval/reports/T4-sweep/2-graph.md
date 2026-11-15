# T4 sweep, step 2: the graph

Indices: r request (16), l layer (64; type(l) in {GDN, ATT}), i/t position (t = current step,
rows of a verify window: t, t+1, t+2 with drafts), j past position, d hidden (5120), m MLP
(17408), h/g heads (ATT: 24 q / 4 kv; GDN: 48 v-heads / 16 k-heads), c (GDN state col, 128),
v vocab (248,320), s draft step (1..2). TP rank p in {0,1} splits h, g, m, v (column/row split).

## Per-layer equations (one row = one (r,i))
```
# shared prologue (every layer)
u[r,i,d]      = RMSNorm(x[r,l,i,d])                          (fused with residual add)
# ---- GDN layer (48) ----
qkvz[r,i,n]   = Σ_d u[r,i,d] Wqkvz[l,d,n]        n: q,k 16x128, v,z 48x128 (TP: heads split)
ba[r,i,h]     = Σ_d u W_ba[l,d,h]                  (beta, a -> decay g)
conv[r,i,n]   = Σ_{τ=0..3} w_conv[l,n,τ] qkv[r,i-τ,n]        KEEPS conv_state[r,l,3 rows]
S[r,l,h,t]    = g_t S[t-1] (I - β k kᵀ) + β k vᵀ    (delta rule; RECURRENT, in-place keep)
o[r,i,h,c]    = q[r,i] · S[r,l,h,i]
o'            = RMSNormGated(o, z)
y_attn[r,i,d] = Σ_{h,c} o'[r,i,h,c] Wout[l,(h,c),d]   ── reduced over h (TP split) ─► ALL-REDUCE #1
# ---- ATT layer (16) ----
qkv[r,i,·]    = Σ_d u Wqkv[l,d,·]   (q 24x256 + gate 24x256, k,v 4x256), q/k-norm, partial RoPE
k,v[r,l,i,g]  KEPT (FP8 paged KV)
a[r,i,h]      = Σ_j softmax_j(q[r,i,h]·k[r,j,g(h)]) v[r,j,g(h)]   j ≤ i (verify rows: causal among drafts)
a'            = a * sigmoid(gate)
y_attn        = Σ_h a' Wo[l,h,d]                    ── reduced over h ─► ALL-REDUCE #1
# ---- MLP (every layer) ----
x'            = x + y_attn ; u' = RMSNorm(x')
gu[r,i,m]     = Σ_d u' Wgu[l,d,m]   (m split by TP: no comm, Rule 1)
h[r,i,m]      = silu(g)*up
y_mlp[r,i,d]  = Σ_m h Wdown[l,m,d]                   ── reduced over m ─► ALL-REDUCE #2
x[l+1]        = x' + y_mlp
# ---- head ----
logits[r,i,v] = Σ_d RMSNorm(x[L]) Wlm[d,v]           (v split by TP; HEAD: local argmax + gather of (val,idx))
tok[r,i]      = argmax_v                             LOW-CARDINALITY CHOKE POINT
# ---- MTP drafter (s = 1,2), serial in s ----
e[r,t+s]      = fc([Embed(tok[t+s]); h_prev[r,t+s-1]])  -> 1 ATT layer + MLP (2 all-reduces) ->
d_tok[t+s+1]  = argmax over v<98304 of RMSNorm(.) Wlm_draft (fp8)
# ---- verify ----
accept while d_tok == tok (greedy, exact); bonus token = target argmax at first mismatch
```

## Data nodes, indices, and what lacks which index (sharing)
| data node | indices | lacks | consequence (Missing-Index rule) |
|---|---|---|---|
| W* (all body weights, lm_head) | l, (d, n/m/v) | r, i, t | every row of a step shares one HBM read: N = rows in window (48 decode; 48 + prompt chunk in mixed) |
| draft weights (MTP layer, draft head) | (d, ·) | r, i, t, **s** and l | the 2 draft steps reuse the same weights but are serial in s (step s+1 needs tok of s) |
| k,v (ATT) | r, l, j, g | h (6 q-heads share one g), i (shared by the 3 verify rows) | GQA sharing 6x; verify rows share one KV read |
| GDN state S | r, l, h | i, j (fixed-size) | read+write per step per request; in-place overwrite; HEAD keeps pending rows instead (replay) |
| conv_state | r, l, n, 3 rows | | tiny |
| activations x, u, y | r, i, l, d | | 48 x 5120 bf16 = 0.5 MB per tensor: the all-reduce payload |
| logits | r, i, v | | 48 x 124K x 2 B = 12 MB per GPU if materialized |
| sampled tok | r, i | v, d | cardinality 248K: speculation site (MTP) |

## Dependency edges that matter
1. Layer chain l -> l+1 (serial within a step), with 2 TP combine nodes (all-reduce, "+") per
   layer = 128 per target forward + 2 per draft step. These cross the PCIe/host link.
2. Within GDN: S[t-1] -> S[t] (recurrent, serial in t). Verify window rows t..t+2 are serial in
   the recurrence but parallel in everything else.
3. Serial edge step t -> t+1 through `tok` (the sampling choke point). Draft s -> s+1 serial.
4. Host-side edges: scheduler/input prep for step t+1 depends on accepted counts of step t
   (async scheduling in V2 runner partly breaks it).
5. Prefill chunk rows of a new request: parallel in i for linear layers, serial in i only inside
   GDN recurrence (chunked form: parallel within chunk by the dual form) and causal attention.

## Combine tags of split indices (for split/merge pricing)
- m in MLP down, h in out_proj/o_proj: "+" (all-reduce).
- v in lm_head: argmax-merge (value, index) — HEAD already uses this (tiny gather).
- j in attention: softmax-merge (split-KV in FlashInfer).
- t in GDN recurrence: NOT a simple reduction (product of (I - βkkᵀ) factors): can be chunked
  with the WY/dual form (exact up to fp), which is how prefill works.
