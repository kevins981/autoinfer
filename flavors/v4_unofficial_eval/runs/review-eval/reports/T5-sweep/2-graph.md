# T5 sweep, step 2: the graph

Indices: r request (16 in flight), l layer (64; type(l) = GDN for l mod 4 != 3, ATT otherwise),
i query position (in a decode step: 3 rows per request, i in {t, t+1^, t+2^}: the last accepted token
and 2 drafts), j past position, t step, d/d' hidden (5120), m MLP inner (17408), h/g heads,
hv GDN value head (48), hk GDN key head (16), c conv tap (4), v vocab (248,320), k draft depth (1..2),
p TP rank (2).

## 1. Nodes per layer (scalar-index form; per rank p, sizes per rank)

```text
# residual + norm (row-local; no weights except a d-vector)
u[r,i,d]        = x[r,i,d] + res_in[r,i,d]                                  (fused add)
n[r,i,d]        = rmsnorm_d(u)[r,i,d] * (1+w_n[l,d])                        reduction over d (local, d replicated)
nq[r,i,d], s[r,i,d/128] = quant_dyn128(n)                                     W8A8 activation (group = 128 d)

# GDN layer (48 of them)
qkvz[r,i,n']    = Σ_d nq[r,i,d] Wqkvz[l,d,n'] * scales                       N=8192 per rank; W has no r,i,t
ba[r,i,hv]      = Σ_d n[r,i,d] Wba[l,d,hv]                                    N=48 (bf16 input)
cv[r,i,ch]      = silu( Σ_c conv_w[l,ch,c] · qkv[r,i-c,ch] )                  needs conv state = last 3 rows of qkv per r (KEEP)
S[r,hv,:,:](i)  = g(i)·S(i-1)·(I - β k kᵀ) + β v kᵀ                           gated delta rule; S fp32 [128x128] per (r,hv); serial in i
o_g[r,i,hv,e]   = Σ_e' q[r,i,hk(hv),e'] S[r,hv,e',e](i)
o2[r,i,hv,e]    = rmsnorm_e(o_g) * silu(z[r,i,hv,e])                           then quant_dyn128
y_l[r,i,d]      = Σ_{hv,e} o2 · Wout[l,(hv,e),d]                               K=3072 per rank: PARTIAL SUM over the TP-split (hv,e)
x'[r,i,d]       = AllReduce_p( y_l )                                           combine tag '+' (Link node)

# ATT layer (16 of them)
qkvg[r,i,·]     = Σ_d nq W_qkv[l,d,·]                                          N=7168 per rank (12 q heads x256 x2 (q, gate) + 2 kv heads x 2 x 256)
q,k = RoPE_partial(qnorm(q), knorm(k), pos(i))                                 RoPE on 64 of 256 dims
KV[r,l,j,g,·]   <- k,v (fp8, scale 1.0)                                        KEEP, append-only, carries r,j
a[r,i,h,·]      = Σ_j softmax_j( q·k[r,j] / sqrt(256) ) v[r,j]                  j ∈ Ancestors(i) (prefix + earlier draft rows of the same request)
ag              = a * sigmoid-type gate(gate[r,i,h,·]); quant_dyn128
y_l             = Σ_(h,·) ag W_o[l,(h,·),d]                                    K=3072, partial -> AllReduce

# MLP (all 64 layers), after a second residual add + norm + quant
gu[r,i,m']      = Σ_d nq W_gu[l,d,m']                                           N=17408 per rank (gate 8704 | up 8704)
hq              = quant_dyn128( silu(gate) * up )                               fused
y_m[r,i,d]      = Σ_m' hq W_down[l,m',d]                                        K=8704, partial -> AllReduce

# after layer 64
h_fin[r,i,d]    = rmsnorm(x + res)
logits[r,i,v_p] = Σ_d h_fin W_lm[d,v_p]                                         vocab split over p (124,160 per rank), bf16
accept[r]       = verify(argmax_v logits[r,i], drafts[r])                       batch-sharded: per-rank (max, argmax) then a tiny combine
x[r,t_next]     = token at first mismatch                                       THE SERIAL EDGE (+ accepted count, low cardinality)

# MTP drafter (2 serial steps k=1,2; 1 layer: attention + MLP, weights W_mtp[·] carry no l of the body)
e[r,k]          = fc( [embed(x̂[r,k-1]) ; h_fin or h_mtp[r,k-1]] )                needs the target's last hidden state
h_mtp[r,k]      = MTP_layer(e)                                                   own KV cache (attention layer), own AllReduces (2)
x̂[r,k]          = argmax_v( h_mtp W_lm_draft[d,v] )                             fp8 draft lm_head, local argmax then combine
```

## 2. Dependency edges and the serial chain of one decode step
`[sched/prepare (CPU)] -> embed -> 64 x {norm -> in_proj -> (GDN conv+recurrence | attention) -> out_proj -> AR -> norm -> gate_up -> act -> down -> AR}
-> final norm -> lm_head -> verify/sample -> [MTP k=1: fc -> layer (2 AR) -> draft head] -> [MTP k=2] -> next step`.
That is 128 body ARs + ~4 drafter ARs + sampling combines on the critical path, each a PCIe round
trip through host memory: every AR is a hard Window boundary (the next GEMM needs the reduced sum).

## 3. Missing-index table (what can be shared)
| Tensor | Has | Lacks | Consequence |
|---|---|---|---|
| W_qkvz, W_qkv, W_out, W_o, W_gu, W_down [l,·,·] | l | r, i, t, k | all 48 rows of a step share one read (N=48); a mixed step shares with its prefill rows. Different l never share (rules out cross-layer merging). |
| W_lm [d,v] | — | r, i, t, l | one read per step for 48 rows. Not shared with the drafter unless the drafter used the same head (it uses an FP8 copy). |
| W_mtp, W_lm_draft | — | r, i, t, **k** | the 2 serial draft steps re-read the same weights: k is missing from the weight, so draft steps could share one read only if they were in one window, which the serial x̂[k-1] -> x̂[k] edge forbids (Speculate site, see 4-knobs). |
| KV[r,l,j] | r, l, j | i (within a request), t | the 3 verify rows of one request share the KV read (N=3 per request); across requests nothing (no shared prefixes in this workload; prefix caching stays on, hit rate 0%). |
| S[r,hv] (GDN state, fp32) | r, l, hv | i, j | the 3 verify rows of a request read S once, but the recurrence is serial in i; draft-rejection needs the S of the accepted prefix: vLLM keeps per-row intermediate states (in-place-overwrite keep + per-position copies). |
| conv state [r,l,ch,3] | r, l | | tiny (3 rows x 10240 ch) |
| quant scales s[r,i,d/128] | r,i | | produced in the norm, consumed by the GEMM (already fused) |
| AR payload y_l [r,i,d] | r, i | — | 48 x 5120 bf16 = 480 KB per AR in decode; per-row, so it can be split along r or d (split-AR / pipelining sites) |

## 4. Index sizes that matter for the knobs
- d = 5120 (reduced in every GEMM and in the norm; the TP split puts a reduction across p after
  out_proj/o_proj/down: rule 2 -> AR).
- m = 17408 and heads split across p: rule 1, no comm until the following K-reduction.
- v = 248,320: split across p (vocab-parallel head); the argmax over v is a reduction with a tiny
  combine (value, index) since batch-sharded sampling.
- i per request per step = 3 (1 + 2 drafts): the Speculate knob sets this; the GEMM reuse N = 16 x 3 = 48.
- j: ~260 prompt + up to ~1,000+ output tokens; only the 16 ATT layers read KV, so KV bytes are
  small (see calibration).
- hv = 48 (24/rank): the GDN state S per request per layer is 24 x 64 KB = 1.57 MB/rank: GDN state
  traffic is the second-largest byte stream per decode step after weights.
