# T8 sweep 2: the graph

Indices: r (request, 16), t (step), i (rows per step: 1 committed + k drafts, M = 16*(k+1) = 48), l (64), h/g heads, d (5120), m (17408), j (past positions).

Per layer l (all with r,i unless noted):
- x -> RMSNorm -> [GDN layers] qkvz = x W_qkvz[l] (no r,i: weight shared by all M rows), ba = x W_ba[l]; conv1d (keeps conv state c[r,l], 3 rows); S[r,l,h] (fp32 128x128, in-place overwrite keep, carries r but not i) updated row by row: S_i = decay_i S_{i-1} + k_i^T delta_i; o_i = q_i S_i; gated norm; out = o W_out[l] -> **all-reduce (+ over d split, TP)**.
- [attention layers] qkv = x W_qkv[l]; qk-norm, rope; KV append k,v[r,l,j] (fp8, append-only keep); attn over j<=i (softmax-merge reduce); o_proj -> **all-reduce**.
- MLP: gate_up = x W_gu[l] (m split over TP, Rule 1: no comm); SiLU*mul; down = h W_d[l] (m reduced -> **all-reduce**).
- After layer 64: final norm, lm_head W_lm (vocab split over TP) -> argmax over vocab (reduce; local argmax + small exchange) -> verify vs drafts (serial edge t->t+1).
- Draft (MTP): eh_proj(concat(embed(x_{i+1}), h_i)) -> 1 transformer layer (own weights, own KV) -> fp8 lm_head -> argmax; repeated k times serially (draft step s+1 depends on draft token s).

Missing-index facts that matter:
- All weights (W[l], W_lm, W_draft) lack r and i: reuse N = M = 48 per load -> every linear is DRAM-bound (ridge ~80-160 rows at fp32-acc FP8).
- S[r,l] lacks i: rows of the verify window share one state read; the k+1 intermediate states are only needed for the accepted prefix (DeltaLog already exploits this).
- Draft weights lack t,s: the k draft steps re-read the same draft weights (N=16 each step; serial dependency prevents merging them).
- The target lm_head lacks r,i: one 0.64 GB/GPU (bf16, half vocab) read per step.
- Communication edges: 2 all-reduces per layer = 128 per target step (+2 per draft step), each M x 5120 bf16 = 0.49 MB at M=48.
