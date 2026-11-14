# T58 sweep, step 2: the graph

Phase 1 (no run history). Notation as in framework/01 §4. Per-rank sizes at TP=2 from 1-inputs.md.

## Indices

| Index | Meaning | Size in the steady state |
|---|---|---|
| r | request | 16 active |
| i, j | token position (query, key) | per step per request: 1 committed-but-unprocessed token + k=3 drafts = 4 query rows; j over the past (~260-490) |
| t | engine step (target verify step) | ~2,140 decode steps + mixed steps per trial |
| k | draft depth inside one step (MTP recursion) | 1..3 |
| l | layer | 64 (48 GDN `l mod 4 != 3`, 16 attention) |
| d | hidden | 5120 (split across ranks only where TP splits a reduction) |
| c | projection output channel | per rank: qkvz 8192, attn qkv 7168, gate_up 17408 |
| m | MLP inner | 17408 total, 8704 per rank (gate_up output 2x8704 per rank) |
| h, g | value head (GDN 48 / attn q 24), key head group (GDN 16 / attn KV 4) | halved per rank |
| e (dv), f (dk) | GDN state axes | 128 x 128 |
| τ | conv tap | 4 |
| v | vocab | 248,320 (124,160 per rank) |
| kb | K-block of 128 (W8A8 scale group) | K/128 |

## Data nodes

| Node | Indices | Missing (=> shareable over) | Size per rank | Lifetime |
|---|---|---|---|---|
| Body weights W[l,·,·] (fp8 + 128x128 scales) | l, c/d | **r, i, t, k** | 12.18 GB | static |
| Target lm_head W_V[v,d] (bf16; champion: 7-bit plane + host copy) | v, d | r, i, t, l | 1.27 GB (0.57 plane) | static |
| Embedding E[v,d] | v, d | r, i, t | gather only | static |
| MTP weights W_mtp (fc, attn layer) | ·,· | r, i, t, **k** | ~0.21 GB | static |
| Draft head W_V'[v',d] (first 98,304 ids, fp8 + PCA plane P,U) | v', d | r, i, t, **k** | 0.25 GB (+U) | static |
| Residual stream x[r,l,i,d] | r, l, i, d | — | 64 rows x 10 KB per layer (transient) | in-step |
| Projection outputs qkvz[r,l,i,c], gate_up, ... | r,l,i,c | — | transient | in-step |
| GDN state S[r,l,h,e,f] (fp32) | r, l, h | **i/j (fixed size)** | 1.57 MB per (r,l); 1.21 GB at r=16 | across steps, overwritten in place |
| GDN replay log (k̂, v, decay, β of last <= 8 tokens) | r, l, h, i | — | small (in slot S[1]) | across steps |
| Conv state (last 3 inputs) | r, l, c | — | 30 KB per (r,l) | across steps |
| KV cache k,v[r,l,j,g] (fp8) | r, l, j, g | — | 1 KB per (r,l,j) | append-only |
| AR payload (partial sums) | r, i, d | — | rows x 10 KB per AR, 129 ARs per verify step | in-step, crosses devices |
| Target logits / argmax | r, i, (v) | — | champion never materializes full logits (local argmax + certificate) | in-step |
| Draft tokens x̂[r,t+k] | r, k | — | ints | in-step |

## Compute nodes (one verify step; per layer l)

GDN layer (48x):
```
n[r,i,d]      = RMSNorm(x[r,l,i,·])                    + AR of previous layer fused in
qkvz[r,i,c]   = Σ_d n[r,i,d] W_qkvz[l,d,c]             reduce d (local, d not split)
ba[r,i,h]     = Σ_d n[r,i,d] W_ba[l,d,h]
u[r,i,c]      = Σ_τ wconv[l,c,τ] qkv[r,i-τ,c]          reads conv state for i-1..i-3 (KEEP)
β,decay[r,i,h]= f(b, a)                                elementwise
S[r,l,h,i]    = decay_i S[i-1] (I - β_i k_i k_iᵀ) + β_i v_i k_iᵀ     serial over i  (KEEP S)
o[r,i,h,e]    = Σ_f q[r,i,h,f] S[r,l,h,i][e,f]
g[r,i,h,e]    = RMSNormGated(o, z)
p[r,i,d]      = Σ_{h,e} g W_out[l,(h,e),d]             reduce over (h,e): SPLIT across ranks -> AR (+)
x[r,l+½,i,d]  = x + AR(p)
n2 = RMSNorm(x) ; gu = n2 W_gu ; a = silu(gate)*up ; p2 = Σ_m a W_down   (m split -> AR (+))
x[r,l+1,i,d]  = x + AR(p2)
```
Attention layer (16x): qkv projection, per-head q/k RMSNorm, partial RoPE, KV append
(KEEP k,v[j]), `s = q·k[j]`, softmax over j ≤ i (causal, incl. the 4 rows of this step),
`a = Σ_j α v[j]`, sigmoid gate, o_proj (split -> AR), then the same MLP.

After layer 64: final norm (AR fused) -> target head: `logit[r,i,v] = Σ_d h W_V` (v split ->
combine = max/argmax across ranks) -> greedy token per row.

Verify: compare target argmax at rows (t..t+3) with drafts x̂[t+1..t+3]; accepted length a ∈ 1..4.

Drafter (MTP), after the verify step (serial edge): draft step k=1 runs on all verified rows of
each request (up to 64 rows: fc(concat(norm(E[x[i+1]]), norm(h_L[i]))) -> attn layer -> norm ->
draft head -> argmax); steps k=2,3 run on 16 rows (one per request), each depending on the
previous step's token (serial edge through argmax, cardinality V').

## Dependency edges that matter

| Edge | Kind | Crosses device? | Note |
|---|---|---|---|
| layer l -> l+1 (x) | serial, high cardinality | no | 64-deep chain per step |
| GEMM partial -> AR -> next norm | reduce over a split index | **yes**: 129 per step (2 per layer + final), bf16 rows x 10 KB | PCIe through host, ~25 GB/s per direction |
| S[r,l,h,i-1] -> S[i] | recurrence over i (4 rows per step) | no | KEEP S across steps; in-place overwrite (02 §4.4) |
| target step t -> drafter -> step t+1 | serial edge through argmax (low cardinality) | combine across ranks | already speculated (MTP k=3) |
| draft step k -> k+1 | serial through argmax (low cardinality) | combine | could itself be speculated (tree) |
| request admission -> prefill (mixed step) | scheduling | — | prefill of ~260 tokens merged into a decode step (chunked prefill) |

## Which tensors lack which indices (what can be shared)

1. **Body weights lack r, i, t, k.** Every row in a window shares one read: at 64 rows per verify
   step the W8A8 GEMM is near the ridge (~82 rows), the Marlin bf16 path above its ridge (~41).
   Weights are re-read every step t (12.2 GB per step): the only ways to amortize are more rows
   per window (more requests r, more speculative rows i, merging mixed rows) or keeping part of W
   on chip across steps (L2 is 72 MB: < 1% of W, so no reuse across steps in general).
2. **S lacks i/j but carries r and l**: no sharing across requests; read (1.21 GB) every step and
   written when the replay log overflows. Its cost per step is ~10% of the weight bytes.
3. **KV carries r, j**: tiny here (16 layers, 2 KV heads/rank, fp8, ~500 tokens).
4. **Target lm_head lacks r, i, l, t** (read once per step for all 64 rows); the drafter's head
   and MTP layer also **lack k**: the three draft steps re-read them three times (Window across k
   is blocked by the serial argmax edge, unless k is speculated/treed).
5. **AR payload carries r, i** but not l in its size: 129 link transfers per step, each
   rows x 10 KB; link bytes are independent of weight bytes, so they are the "orthogonal resource"
   for Window overlap (Link || HBM/compute).
6. **No index of the model is missing from both the ARs' inputs and the GEMMs' weights within one
   layer** except r/i: so overlapping an AR with compute needs another row group (Split r or i into
   two groups and stagger them: what the champion does for mixed steps, not for decode).
