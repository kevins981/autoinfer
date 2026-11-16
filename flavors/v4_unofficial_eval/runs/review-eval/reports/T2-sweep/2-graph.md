# T2 sweep, step 2: the graph

Indices: r request (16), l layer (64: type GDN or ATT), i query row (per request: 1 + 2 drafts in a
decode step, or a prompt chunk in a mixed step), j past position, d/d' hidden (5120), m MLP (17408),
h/g heads (ATT: 24 q / 4 kv, dim 256; GDN: 16 k / 48 v heads, dim 128), v vocab (248,320),
p rank (TP=2 splits h, g, m, v). t = engine step; s = draft step (1..2).

## Compute and data nodes of one engine step (per GPU under TP=2)
Per layer l (GDN layer):
```
x̂[r,i,d]   = RMSNorm(x[r,i,d]) (+ residual add)                       reads x (r,i)
z[r,i,c]   = Σ_d x̂ W_in[l,d,c]   (in_proj: q,k,v,a,b,gate; c split by heads over p)   W_in: no r,i
conv[r,i]  = causal conv1d over the last 4 positions of z           keeps conv state C[r,l] (Keep, overwrite)
S[r,l,t]   = gated delta rule(S[r,l,t-1], k, v, a, b)  fp32          keeps S[r,l] (Keep, in-place overwrite)
o[r,i,·]   = q · S (per head) , gated norm                           reads S (r)
y_p[r,i,d] = Σ_{c∈p} o W_out[l,c,d]  -> AllReduce_p(+)               W_out: no r,i ;  link edge
```
ATT layer: qkv = x̂ W_qkv (TP over heads) -> q/k norm + partial RoPE + gate -> write k,v[r,l,j] (fp8,
Keep, append-only) -> FlashInfer attention over j ∈ Ancestors(i) (softmax over j) -> W_o ->
AllReduce. MLP (every layer): gate_up = x̂ W_gu (m split over p) -> SiLU·mul -> W_down -> AllReduce.

After layer 64:  logits_p[r,i,v∈p] = x̂ W_lm[v∈p,d] (bf16)  -> AllGather over v  -> sample/verify
(greedy: argmax_v; rejection: compare draft tokens with argmax) -> accepted tokens.
MTP drafter (s = 1,2): h = MTP_layer(embed(token), hidden) (1 ATT-type layer + MLP, FP8) ->
logits_p = h W_lm -> AllGather -> argmax -> next draft token.  (SERIAL EDGE between s=1 and s=2,
and between the drafter and the next target step.)

Mixed step: the same graph with one request having a prompt chunk of n_p rows (GDN uses the
chunked "dual form" kernels: chunk_scaled_dot_kkt, solve, chunk_fwd_h, chunk_fwd_o), plus 45
decode rows of the others. Rows are merged across r at every linear layer (Window merge).

## Edges that cost something
| edge | bytes per step (per GPU) | resource |
|---|---|---|
| weights -> every linear node | 12.2 GB (FP8) | DRAM, shared over all rows (no r, i, t index) |
| W_lm -> logits (target + 2 drafts) | 3 x 0.635 GB (bf16) | DRAM |
| S[r,l] read + write (48 layers) | 2 x 16 x 24 x 128 x 128 x 4 B x 48 = 2.4 GB | DRAM (index r: no sharing) |
| k,v[r,l,j] read (16 layers) | ~16 x 400 x 2 x 2 heads x 256 B x 16 = ~0.1 GB | DRAM (index r) |
| all-reduce after attn/GDN out and after MLP down | 128+ per target forward, each M x 5120 x 2 B (0.5 MB at M=48) | link (PCIe via host, SHM) |
| logits all-gather | target: 48 x 124k x 2 B = 12 MB; draft: 16 x 124k x 2 B = 4 MB, x2 | link |
| sampled token -> next step (serial) | tiny | CPU/GPU sync |

## Missing indices (what can be shared)
- All weights (W_in, W_qkv, W_o, W_gu, W_down, W_lm, MTP layer) lack r, i, t, s: shared by all rows
  in a window -> batching across requests, draft rows and prompt rows (Merge).
- W_lm lacks s: target lm_head and the two draft lm_heads read the same tensor, but the draft
  steps are serial (s=2 needs s=1's token), so they cannot share a window unless speculated.
- S[r,l] lacks i: the 3 verify rows of a request share one state read (the MTP kernel already does).
- k,v carry r: no sharing across requests (no shared prefixes in this workload; prefix caching on).
- logits[r,i,v]: the consumer at temperature 0 is argmax_v, a reduction over v whose combine op
  is (max, argmax): the AllGather of full logits is only needed because the consumer is written as
  "gather then reduce"; split-v reduction makes it a gather of 2 numbers per row.
- The all-reduce payload y[r,i,d] carries r,i: it can be split along r or i into pieces whose
  communication overlaps the next piece's compute (different resources: link vs DRAM/SM).

## Serial structure
Per step: [target forward: 64 layers, each GEMM -> AllReduce -> norm -> GEMM ...] -> lm_head ->
gather -> verify -> [draft s=1: MTP layer -> lm_head -> gather -> argmax] -> [draft s=2: same] ->
next step's input prep (CPU) -> next step. The per-layer AllReduce sits on the critical path
(each GPU needs the full sum before the next norm).
