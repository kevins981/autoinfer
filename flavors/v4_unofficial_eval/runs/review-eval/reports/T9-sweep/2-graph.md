# T9 sweep, 2: graph (compressed)
Indices: r (request, 16), t (position/verify row, k+1 = 3 rows), l (64 layers), h (heads), d (hidden), v (vocab).
Data nodes: W[l] (no r, no t: shared across the whole step window -> why speculation over t pays), lm_head W_v (no r,t),
KV[l_attn, r, t'] (append-only keep), S_gdn[l_gdn, r] (in-place overwrite keep; DeltaLog = keep inputs log instead of 3 states),
x[r,t] token ids (low cardinality: the only exact speculation choke point), drafts x^[r, t+1..t+k].
Compute per decode step: drafter (MTP layer + lm_head) k times sequential over t (edge x^[t+1] -> x^[t+2]);
target: per layer [norm -> qkvz GEMM -> AR? ...] each layer: in-proj GEMM, GDN recurrence or attention, o-proj GEMM -> AR, norm, gate_up GEMM, act, down GEMM -> AR (2 ARs per layer = 128/step, edge over the device split of d/heads);
lm_head (full vocab, target) -> rejection (argmax compare) -> accepted count a[r] -> commit S, KV.
Missing indices that allow sharing: W lacks r,t (batching + spec share weight reads); S lacks t (only last accepted state needed: DeltaLog);
the AR payload has r,t but the latency (~40 us) is per call, not per row: rows are nearly free on the link at M<=96.
