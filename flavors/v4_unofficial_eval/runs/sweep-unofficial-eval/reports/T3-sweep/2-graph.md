# T3 sweep, step 2: the graph

Indices: r request (16 live), l layer (64; type(l) = GDN for l mod 4 != 3, ATT otherwise), i query
position (per step: 1 verified row + 2 draft rows = 3 rows per request in decode, M = 48; ~260 rows
for a prefill), j past position, h/g heads (ATT: 24 q heads, 4 kv groups; GDN: 16 key heads, 48 value
heads, 3 value heads per key head), d = 5120, m = 17408, k/v head dims 128 (GDN) / 256 (ATT),
p = TP rank (2), n = vocab id (248,320), t = engine step, u = draft depth (0,1,2).

## Data nodes (per GPU at TP=2), with their indices and which ones they lack
| node | indices | missing (=> shareable across) | size |
|---|---|---|---|
| W_mlp[l] (gate_up, down) | l, d, m/p | **r, i, t** | 134 MB/layer/GPU |
| W_gdn_in[l], W_gdn_out[l] | l, d, h/p | r, i, t | 58 MB/layer/GPU |
| W_att_qkv[l], W_o[l] | l, d, h/p | r, i, t | 52 MB/layer/GPU |
| W_lm (target), W_emb | n/p, d | r, i, t, **l** | 1.27 GB/GPU each (bf16) |
| W_draft_lm (fp8 copy, n < 98,304) | n/p, d | r, i, t, l | 0.25 GB/GPU |
| W_mtp (1 layer) | d, m | r, i, t | ~0.2 GB/GPU |
| x[r,i,l,d] residual | r, i, l, d (replicated over p) | p | 10 KB/row |
| S[r,l,h,k,v] GDN state (fp32) | r, l, h/p | **i, j** (fixed size, in-place) | 1.57 MB/req/layer/GPU |
| conv[r,l,3,c] | r, l | | 30 KB/req/layer/GPU |
| pending rows (replay) k,v,decay,beta[r,l,i<=3] | r, l, i | | ~0.8 KB/row/head |
| K,V[r,l,j,g,256] fp8 (ATT layers) | r, l, j, g/p | h (GQA: 6 q heads share a g) | 2 KB/token/layer/GPU |
| logits[r,i,n] | r, i, n/p | | 0.5 MB/row/GPU (bf16) |
| draft tokens x̂[r,t+1..t+2] | r, u | | ints |

## Compute nodes per layer (one decode step, rows M)
```
ATT layer l:  n1 = RMSNorm(x)                      [r,i,d]           (fused add+norm)
              qkvg = n1 · W_att_qkv[l]  (Σ_d)      [r,i,h/p]         GEMM (column-parallel: no comm)
              q,k = RMSNorm_head + RoPE(64 dims)
              K,V[r,l,j=pos] <- k,v (fp8 append)                       KV write
              a = softmax_j(q·K[r,l,j≤i]) · V       Σ_j (softmax-merge) attention (per r, g)
              a *= sigmoid(gate)
              y = a · W_o[l]  (Σ_{h/p})             partial over p  -> ALL-REDUCE (+) #1
GDN layer l:  n1 = RMSNorm(x)
              qkvz,ba = n1 · W_gdn_in[l] (Σ_d)      column-parallel
              conv1d over (conv state, current rows) causal in i
              q,k L2-norm; g = decay(a, A_log, dt_bias); beta = sigmoid(b)
              for each row i in order: S <- S·exp(g_i) + beta_i k_i ⊗ (v_i - S^T k_i)   RECURRENCE in i
                                       o_i = S^T q_i
              o = RMSNorm_gated(o, z)
              y = o · W_gdn_out[l] (Σ_{h/p})         -> ALL-REDUCE (+) #1
both:         x += y; n2 = RMSNorm(x)
              h = silu(n2·W_gate) * (n2·W_up)        column-parallel over m/p
              y2 = h · W_down (Σ_{m/p})              -> ALL-REDUCE (+) #2
              x += y2
after L:      logits = norm(x) · W_lm  (n split over p) -> local argmax + tiny cross-rank max (HEAD), or gather
              accept/reject drafts vs argmax (exact)
MTP draft:    2 serial steps u=1,2: e = Emb(x̂) ⊕ hidden -> 1 layer (ATT+MLP, its own KV) -> W_draft_lm -> argmax
```

## Dependency edges that shape the schedule
1. **Layer chain** l -> l+1 (serial within a request; 64 layers x 2 all-reduces = 128 dependent
   cross-GPU exchanges per forward at TP=2).
2. **Row recurrence inside GDN** (i -> i+1 through S): only GDN has it; ATT and all GEMMs are
   row-independent.
3. **Serial sampling edge** step t -> t+1 (through the accepted token count and the next draft):
   the step's tail (lm_head, accept, draft x2) is on the critical path of the next step.
4. **Draft chain** u=1 -> u=2 (each draft step is a full MTP layer + draft lm_head).
5. **CPU edge**: the scheduler/metadata of step t+1 needs results of step t (async scheduling
   overlaps most of it).

## Missing-index observations (what can be shared)
- All weights lack r, i, t: every row of every request in a window shares one weight read
  (batching; spec rows ride the same read: M = 48 instead of 16 at nearly the same time).
- W_lm lacks l: read once per step for the target (all 48 rows) and once per draft step.
- S lacks i, j: fixed per request; its read+write per step (1.57 MB x 2 per layer per request per
  GPU) is the GDN analogue of the KV scan, and it does not shrink with batching (carries r).
- K/V lack h (GQA): 6 q heads per kv head; at decode q_len=3 the 3 rows of a request also share
  the same K/V read (spec verification batches over i).
- x is replicated over p (TP): every all-reduce produces a full copy on both GPUs.
- Nothing lacks p except weights' complementary halves: weights are split, not shared, across GPUs.
