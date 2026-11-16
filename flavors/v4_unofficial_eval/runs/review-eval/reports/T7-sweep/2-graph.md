# T7 sweep, step 2: graph (one decode step, per GPU at TP=2)
Indices: r (16 requests), i (rows per request in a step: 1 + k drafts = 3), l (64 layers), h/g (heads), d (5120), m (17408/2), v (vocab 248320), t (step).

Data nodes (index set; what it lacks -> what can be shared):
- W_qkvz/W_ba (GDN in-proj), W_qkv (attn), W_o, W_gate_up, W_down: [l, d, m]; lack r, i, t -> shared across all 48 rows of a step (Merge/Window) and across steps if speculated.
- W_lm (target lm_head) [d, v]: lacks r,i,t,l. W_mtp (draft layer + fp8 draft lm_head): lacks r,i; read once per draft step (k times per step).
- KV cache k,v[r, l_attn, j, g]: append-only. GDN state S[r, l_gdn, h, 128, 128] fp32: in-place overwrite (DeltaLog makes it checkpoint + log).
- conv state [r, l_gdn, 3, channels] (short conv window).
- activations x[r,i,l,d] (bf16), quantized copies (fp8 + group scales).

Compute nodes per layer l: norm(+quant) -> in-proj GEMM -> [GDN: conv1d -> recurrence (reads S, writes S/log)] or [attn: rope -> KV write -> paged attention]
-> out-proj GEMM -> all-reduce(+) [edge crosses devices: split over h] -> residual+norm(+quant) -> gate_up GEMM -> SiLU*mul(+quant)
-> down GEMM -> all-reduce(+) -> residual. After layer 64: final norm -> lm_head (split v; local argmax / sharded sampling) -> rejection (verify)
-> accepted tokens -> draft: MTP layer + draft lm_head, k serial steps (serial edge through argmax token) -> next step's rows.

Edges of note:
- serial edge t -> t+1 through Sample (low cardinality): the one the champion speculates (MTP, chain of k=2).
- inter-layer edge y[l] -> x[l+1]: high cardinality (not speculable exactly).
- AR edges: 2 per layer = 128 per target forward + per draft step; each a device crossing over host PCIe.
- draft chain: k serial MTP steps, each reading W_mtp (lacks r,i: shared across the 16 requests in a step).
Sharing visible from missing indices: weights lack i -> verifying more draft rows per step reuses the same weight reads (Gate 1 of Speculate);
W_mtp lacks the draft-step index -> but steps are serial (each needs the previous argmax), so no window merge exactly.
