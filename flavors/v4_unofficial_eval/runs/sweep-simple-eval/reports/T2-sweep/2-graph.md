# T2 sweep, step 2: the graph

Indices: r (request, 16), l (layer, 64: type GDN or ATT), i/t (position; per step each request has
q = 1 + accepted-draft-count... in BASE q = 3 rows: last token + 2 drafts), j (past positions),
d (5120), m (17408), h/g (24 q heads, 4 kv heads; GDN: 48 v-heads, 16 k-heads), dk=dv=128 (GDN),
e256 head_dim, v (vocab 248,320), s (draft depth 1..2), rank (TP 0/1).

## Data nodes
| node | indices | size (per GPU, TP=2) | kept across steps? |
|---|---|---|---|
| x (residual) | r,i,d | 10 KB/row | no |
| W_qkvz, W_ba, W_out (GDN) | l,(heads),d | 43 + 0.5 + 15.7 MB | static |
| W_qkv(+gate), W_o (ATT) | l,h,d | 36.7 + 15.7 MB | static |
| W_gate_up, W_down | l,d,m | 89.1 + 44.6 MB | static |
| W_lm (target, shared by MTP) | v,d | 1.27 GB bf16 | static |
| W_mtp layer + fc | d,m | ~0.25 GB | static |
| k,v cache (ATT) | r,l,j,g | 4 kv heads/2 x 256 x 2 x 1 B = 1 KB per token per layer | append-only keep |
| GDN state S | r,l,hv,dk,dv | 24 x 128 x 128 x 4 B = 1.57 MB per request per layer | in-place overwrite keep (BASE: written once per spec position = 3 slots) |
| conv state | r,l,ch,3 | ~ (qkv channels/2) x 3 x 2 B ~ 25 KB | overwrite keep |
| logits | r,i,v | 48 x 124,160 x 2 B = 11.9 MB (local shard) | no |
| draft tokens x̂ | r,s | ints | no (low cardinality) |

## Compute nodes per layer (one step, M rows = 48 in steady decode)
GDN layer l: norm -> in_proj_qkvz (GEMM, reduce over d) + in_proj_ba -> conv1d update (over t window
of 4) -> recurrence `S[t] = a_t S[t-1] (I - b_t k k^T) + b_t v k^T`, out = S q (per v-head, no comm:
heads split) -> gated rmsnorm -> out_proj (reduce over the split head channel: **all-reduce +**)
-> residual add + norm -> gate_up (no comm, split m) -> SiLU*mul -> down (reduce over split m:
**all-reduce +**).
ATT layer l: norm -> qkv(+gate) GEMM -> qk-norm + partial RoPE -> KV write -> attention over j
(softmax-merge over j; heads split, no comm) -> sigmoid gate -> o_proj (**all-reduce +**) -> MLP as above.
End: final norm -> lm_head (split v: no reduction, but argmax/softmax over v is a reduction over the
split index -> **all-gather of logits** in BASE) -> rejection/greedy accept -> sample.
MTP drafter (per draft step s): embed(x̂) (vocab-parallel: all-reduce) + fc([norm(h), norm(emb)])
(**all-gather**, bf16) -> 1 ATT layer (2 all-reduces) -> lm_head (shared) -> all-gather -> argmax.

## Dependency edges that matter
- l -> l+1 (serial), and within a layer the two all-reduces sit on the critical path (Rule 3 of
  `05`: rmsnorm follows each reduction).
- **Serial edge t -> t+1** through Sample (low cardinality: speculated by MTP).
- Draft chain s=1 -> s=2 (serial, each needs the full lm_head + argmax).
- Verify -> state commit: in BASE, S is written for all 3 positions before acceptance is known.
- Step -> step on the host: scheduling, input prep, and (for mixed prefill steps in BASE) eager
  per-kernel launches; both ranks must launch before each all-reduce completes (cross-rank barrier
  128+ times per step).

## Missing-index facts (what can be shared)
- All W lack r,i,t: every row in a window reuses them (N = rows in the step). Decode N=48 < ridge
  (~80 bf16-MMA, ~165 FP8): memory-bound; prefill-mixed steps N ~ 100-2000: compute-bound above ~80.
- W_lm lacks s and t: the 2 draft steps and the target verify each re-read the 1.27 GB lm_head
  (3 reads per step) because s is serial (s=2 needs argmax of s=1). Only the Speculate knob or a
  smaller drafter head changes this.
- k,v carry r,j: no sharing across requests (ShareGPT has no common long prefix beyond the chat
  template; prefix caching already dedups that).
- S carries r,l but not j: fixed size; carries t only via overwrite (02 §4.4: the rollback trap).
- Logits carry v split across ranks; argmax over v is an associative reduction with combine
  (max, argmin-index): **can be combined from per-rank (value, index) pairs** instead of gathering
  all logits (Rewrite: push the reduction below the gather).
- The all-reduce input carries d only; it lacks nothing that would allow skipping it; splitting
  along d (chunks) lets the GEMM producing it and the all-reduce pipeline (Window knob, Mechanism 2).
