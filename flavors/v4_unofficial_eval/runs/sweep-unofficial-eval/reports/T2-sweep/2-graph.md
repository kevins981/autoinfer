# T2 sweep, step 2: the graph

Indices: `r` request (16), `l` layer (64; type(l) ∈ {GDN, ATT}), `i` row in the step's window
(position; in decode 3 per request: the bonus token + 2 drafts), `j` past position, `t` step,
`d` hidden (5120), `m` MLP inner (17408), `h` head (ATT: 24 q / 4 kv `g`; GDN: 48 v / 16 k),
`v` vocab (248,320), `k_s` draft slot (1..2), `P` TP rank (2).

## Per-layer nodes (one row = one compute node; data nodes in brackets)
```
x0[r,i,d]          = Embed[tok[r,i], d]                                     (vocab-parallel: lookup + all-reduce)
for l:
  u[r,i,d]         = RMSNorm_d(x[r,l,i,·] (+ residual))                     reduction over d
  -- GDN layer --
  qkvz[r,i,c]      = Σ_d u[r,i,d] Wqkvz[l,d,c]       c ∈ 16384 (split by heads across P)
  ab[r,i,h]        = Σ_d u[r,i,d] Wba[l,d,h]         (bf16 weight, small)
  qkv'[r,i,c]      = Σ_{τ<4} conv[l,c,τ] qkv[r,i-τ,c]   [keep: conv_state[r,l,c,3]]
  S[r,l,h,i]       = g(a)·S[r,l,h,i-1]·(I - β k kᵀ) + β k vᵀ   [keep: S[r,l,h] 128x128 fp32]
  o[r,i,h,e]       = q[r,i,h]·S[r,l,h,i]
  y[r,i,d]         = Σ_{h,e} (RMSNorm(o)·swish(z))[r,i,h,e] Wout[l,(h,e),d]   reduced over h -> all-reduce (P)
  -- ATT layer --
  q,gate,k,v       = Σ_d u Wqkv[l,d,·]  ; q,k RMSNorm + partial RoPE(i)
  [keep: k,v[r,l,j,g,256] fp8]
  a[r,i,h,e]       = Σ_j softmax_j(q[r,i,h]·k[r,j,g(h)]) v[r,j,g(h),e] , j ≤ i (causal incl. this step's rows)
  y[r,i,d]         = Σ_{h,e} (a·σ(gate))[r,i,h,e] Wo[l,(h,e),d]           reduced over h -> all-reduce (P)
  -- MLP --
  u2               = RMSNorm_d(y + x)
  hgu[r,i,m]       = Σ_d u2 Wgu[l,d,m] ; h = silu(g)·u                        (m split across P, no comm)
  x[r,l+1,i,d]     = Σ_m h[r,i,m] Wdown[l,m,d]                               reduced over m -> all-reduce (P)
logits[r,i,v]      = Σ_d RMSNorm(x[r,L,i]) Wlm[d,v]                          (v split across P -> all-gather)
acc[r]             = Verify: #prefix of drafts equal to argmax_v logits[r,i,v]   (greedy, T=0)
-- MTP draft, k_s = 1..2 (serial) --
hd[r,d]            = MTPLayer(fc([Embed(tok); x[r,L,last accepted]]))         (1 ATT layer + MLP, own KV)
dlogits[r,v]       = Σ_d hd[r,d] Wlm[d,v]  -> draft tok = argmax_v           (same Wlm as target)
SERIAL EDGE:       tok[r, next step] = (accepted tokens, bonus, drafts)
```

## Missing-index table (what can be shared)
| Data node | Indices it has | Indices it lacks -> sharing it allows |
|---|---|---|
| W* (Marlin FP8 weights) | l, d, c/m | **r, i, t, k_s** -> all 48 rows of a step share one read; rows of different steps (t) or draft slots could share it if put in one window |
| Wlm (bf16 lm_head) | d, v | **r, i, t, k_s, l** -> target verify (48 rows) and the 2 draft steps all read the same 1.27 GB/GPU; today read 3x per step |
| Embed | v, d | gathered rows only |
| S (GDN state) | r, l, h | i (overwritten in place; MTP verify needs per-row states for rollback) |
| k,v cache | r, l, j, g | h (GQA: 6 q heads share a kv head) |
| Draft MTP weights | d,c (one layer) | k_s, r -> both draft steps share the read if merged |
| all-reduce payload | r, i, d | l is sequential -> 128 serial exchanges per forward |

## Dependency edges (serial chain inside a step)
`Embed -> [l: norm -> mixer GEMM -> (conv,GDN | attn) -> out GEMM -> AR -> norm -> gate_up -> down -> AR] x64
 -> lm_head -> all-gather -> verify (argmax, accept) -> MTP step 1 -> lm_head -> AG -> argmax -> MTP step 2 -> ... -> next step`.

- Inside a layer all edges are true data dependencies; across layers `l -> l+1` (W carries `l`, so no sharing).
- Across requests `r` there are no edges (except they share W reads and the step barrier).
- Across steps `t`: the serial edge goes through a **discrete token** (low cardinality: Speculate site)
  and through the kept state S / KV.
- The two all-reduces per layer sit on the critical path: the next RMSNorm reduces over all `d`.
- Draft step k_s=2 depends on k_s=1 (token edge) -> also serial.

## Reduction / combine tags (needed when splitting)
- Σ_d in every GEMM input (split d -> + combine), Σ_m in down (TP split -> all-reduce `+`),
  Σ_h in out/o-proj (TP split -> all-reduce `+`), softmax_j (split j -> (m,ℓ,o) merge),
  argmax_v (split v -> (max,idx) merge: exact, O(2·P) values instead of O(V)),
  RMSNorm Σ_d x² (split d -> scalar `+` per row).
- GDN recurrence over i: sequential in i (chunked form for prefill: re-bracket (QKᵀ)V vs Q(KᵀV)).

## Where each knob can act (sites listed for step 4)
1. Keep/Recompute: S (fp32 state: in-place vs per-row checkpoints for MTP rollback), KV, draft logits.
2. Rewrite: argmax over v (local argmax + merge), lm_head row dedup (target vs draft), GDN chunk form, RMSNorm+AR fusion.
3. Speculate: token edge (MTP draft count, tree, other drafters).
4. Split/Merge: TP vs PP vs hybrid; split v (vocab) vs split r for sampling (batch-sharded); split M of a mixed step (chunked prefill size); merge draft steps.
5. Place: which GPU holds lm_head / MTP; L2 vs HBM tier for the next GEMM's weights; host memory as link.
6. Window: overlap AR (link) with GEMM (HBM) by splitting a non-reduced index (out cols) or r; overlap draft with verify; prefetch weights into L2 during AR.
