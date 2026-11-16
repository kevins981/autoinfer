# T4 sweep, step 2: the graph

Indices: r request (16), l layer (64 target + 1 MTP), i/t position (t = current step's rows: 3 per request
in decode = verified token + 2 drafts; ~261 rows for a new request's prefill), j past positions, d hidden
(5120), m MLP channel (17408), h/g heads (attention: 24 q / 4 kv; GDN: 16 k / 48 v heads), c conv channel,
v vocab (248,320), p draft step (1..2), and the device index (2 GPUs, TP).

## 1. Data nodes (per GPU at TP=2)
| node | indices | lacks | size | kept across steps? |
|---|---|---|---|---|
| W_qkvz[l], W_out[l] (GDN) | l, d, (head channels) | r, t, j | 41.9 + 15.7 MB FP8 | weight |
| W_qkv[l], W_o[l] (attn) | l, d, h | r, t, j | 36.7 + 15.7 MB | weight |
| W_gate_up[l], W_down[l] | l, d, m | r, t, j | 89.1 + 44.6 MB | weight |
| W_ba[l], conv_w[l], norms, A_log, dt_bias | l | r, t | small (bf16) | weight |
| W_lm (target lm_head) | v, d | r, t, l | 1.27 GB bf16 | weight |
| W_mtp (1 layer + fc), W_lm_draft (= the lm_head weights, bf16, separate module) | v, d | r, t, l, p | 0.19 GB + 1.27 GB | weight |
| x[r,l,t,d] residual stream | r,l,t,d | — | M x 5120 bf16 | no (per step) |
| k,v[r,l,j,g] (16 attn layers) | r,l,j,g | — | fp8; ~2 KB/token/layer/GPU | Keep (append-only) |
| S[r,l,hv] GDN state | r,l,hv | t, j | 24 x 128x128 fp32 = 1.57 MB/layer/req | Keep (overwrite) + 3 spec slots per req |
| conv_state[r,l,c,3] | r,l,c | | 5120 x 3 bf16 = 30 KB/layer/req | Keep (overwrite) + spec slots |
| logits[r,t,v] | r,t,v | | M x 124,160 per GPU | no |
| draft tokens x̂[r,t+1..t+2] | r,p | | ints | kept 1 step (proposals) |
| AR buffers (host pinned) | — | | 2 x 2 x 16 x 64 KB | transport |

Missing-index observations (what can be shared):
- All weights lack r and t: every row of every request in a step shares one weight read (M = 48 in decode).
  They also lack p: the MTP layer and draft lm_head are read once per draft step; the two draft steps read
  them twice (p is serial: step 2 needs step 1's token).
- W_lm_draft and W_lm are the same values (MTP loads the checkpoint lm_head) but different modules:
  a shared read across target-verify and draft would need them in the same window, which the serial
  edge (target logits -> sample -> draft) forbids, except draft step p=1 and the target lm_head do share
  the step... no: draft p=1 needs the target's hidden state AND the sampled token (serial). Not shareable.
- S[r,l] lacks t: one state per request per layer, but the spec-decode kernel materializes S after EACH of
  the 3 rows (3 slots written per request per layer) because the accept edge comes later (see §3).
- k,v carry r and j: no sharing across requests (no shared prefixes in this benchmark beyond the chat
  template's first tokens).

## 2. Compute nodes per layer (decode step; per GPU)
GDN layer l (48x):
1. AR_prev: all-reduce of the previous MLP output (host-mapped hmar), + L2 prefetch of W_qkvz head on a side stream.
2. NormQ: Gemma add-RMSNorm + dyn128 FP8 quant (writes residual).
3. G_qkvz = x · W_qkvz (W8A8 Triton, N=8192, K=5120); G_ba = x · W_ba (bf16, N=48).
4. Conv: causal conv1d update (reads/writes conv_state slots).
5. GDN: gated delta rule over the request's 3 rows: reads S[accepted-1 slot], writes S after each row
   (3 slots), output o[r,t,hv]; fused gated RMSNorm with z.
6. G_out = o · W_out (K=3072) -> partial sum over heads (split index hv reduced over: TP Rule 2 -> AR).
7. AR (hmar) + L2 prefetch of W_gate_up head; NormQ.
8. G_gate_up (N=17408) -> SiLU·mul + dyn128 quant -> G_down (K=8704) -> partial sum over m (-> AR at the next layer's step 1).
Attention layer (16x): same with G_qkv (N=7168), q/k norm + partial RoPE, KV write (fp8), FlashInfer
decode/prefill attention over j (3 query rows per request, causal among them), output gate, G_o (K=3072).

Per step (outside layers): embed; final norm; target lm_head (vocab-split, M=48) -> batch-sharded
sampling + rejection (exact greedy verification) -> accepted count per request; MTP draft p=1,2: fc(concat
(norm(h), norm(embed(x̂)))) -> MTP layer (full attention, its own KV) -> lm_head (vocab-split) -> local
argmax + 2-int cross-GPU reduce -> next draft token; scheduler / input prep on CPU; GDN metadata.

## 3. Dependency edges that matter
- Serial chain within a step: l -> l+1 (x), with 2 AR combine nodes per layer (TP split of hv/heads and
  of m; Rule 2). Each AR is a cross-device edge over PCIe through host memory: 128 per target forward + 2
  per MTP forward x 2 draft steps (+ lm_head local argmax reduces).
- The accept edge: target logits -> sample/verify -> num_accepted[r] -> (next step) which S slot is the
  source. The GDN node of step n cannot know which of its 3 rows will be accepted, so it writes S after every
  row (Keep with in-place overwrite + branch copies: 02 §4.4 "multi-branch / rollback trap").
- The draft edge: target hidden h[r,t] + accepted token -> draft p=1 -> token -> draft p=2 (serial over p).
- The step edge: drafts of step n are the 2 extra rows of step n+1 (Speculate on the x[t+1] edge).
- CPU edges: scheduler output -> input prep -> graph replay; sampled tokens -> CPU (detokenize, stop
  checks) -> next schedule. The draft attention metadata is rebuilt eagerly between draft steps
  (FlashInfer cannot update it in place).
- Prefill edges: a new request's ~261 rows run as a mixed step with the 15 decoding requests (chunked
  prefill, piecewise graphs up to 2048 tokens); its GDN prefill uses the chunked kernel (dual form) and
  writes S once.

## 4. Index-size asymmetries to remember
- M = 48 rows in decode << FP8 ridge (~80 rows at fp32-acc) -> weight-read bound; mixed steps M ~ 300 ->
  compute-bound GEMMs.
- S (16,384 fp32 per head) vs per-row inputs k,v (256 bf16 per head): 64x/128x asymmetry: a state write
  costs as much as caching ~128 rows of inputs (02 §4.3).
- Vocab 248K x 5120: the lm_head (1.27 GB/GPU) costs ~10% of the body per read; it is read once by the
  target and twice by the drafter per step (3.8 GB/step at bf16).
- AR payload per decode AR: 48 x 5120 x 2 B = 0.49 MB, latency-bound on PCIe (~40 us each, x ~132).
