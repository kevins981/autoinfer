# Status, Devil's Advocate Limitations, Open Questions, and Next Steps

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) and the dedicated knob files (`02`–`06`) first.

---

## 1. Coverage Table (The Retrofitting Check)

Every optimization we have tested is classified by what it took for the framework to enumerate it:
- **(a) Covered by existing primitives and moves:** Falls out of the enumeration tables without changing the framework.
- **(b) Needed a new *generic* primitive:** Exposed a structural gap that was fixed by adding one general-purpose primitive used across multiple optimizations.
- **(c) Needed a special-case rule:** A red flag that the abstraction is being patched per paper.

| Knob / File | (a) Optimizations Covered by Existing Moves | (b) Generic Primitives & Rules Used | (c) Special-Case Rules |
|---|---|---|---|
| **Knob 1: `Keep / Recompute`**<br>([02-knob-keep-recompute.md](file:///scratch/kevinsong/learning/framework/02-knob-keep-recompute.md)) | • KV cache (`{k,v}` vs `{q,k}` vs `{x}`)<br>• Sliding-window eviction<br>• Linear-attention fixed state `S[t-1]`<br>• ReplaySSM / DeltaLog (input caching vs state write-backs) | • Future-readers check (Rule 1)<br>• `t`-free formula check (Rule 2)<br>• HBM bytes written + capacity $C$ | None |
| **Knob 2: `Rewrite`**<br>([03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md)) | • MLA weight absorption (`(q W_ukᵀ) cᵀ`)<br>• DeepSeek Decoupled RoPE & Commuting/Latent RoPE<br>• Multi-tenant LoRA batching (Punica / S-LoRA)<br>• Linear Attention dual-form (`(QKᵀ)V` vs `Q(KᵀV)` at $M=d$)<br>• Cross-slot shared-base MoE factoring | • 4 algebraic laws (Associativity, Distributivity, Unrolling, Dedup)<br>• 3 hardware principles (Drop blocking index; Re-bracket size asymmetry; Commute past blocker) | None |
| **Knob 3: `Speculate`**<br>([04-knob-speculate.md](file:///scratch/kevinsong/learning/framework/04-knob-speculate.md)) | • Chain & Tree Speculative Decoding (Medusa, EAGLE-2, Sequoia)<br>• Expert prefetching (`sel[t]`)<br>• Certified sparse attention (Quest / RetrievalAttention)<br>• **NSA / MoBA KV-block prefetch (`sel_blk[t]`)**<br>• **Looped-depth parallel verify (`W_shared` missing `l`)**<br>• **Shared-expert dispatch shadow & Rollback-free SSM verify** | • Unified transformation `[Verify \|\| B(â)]`<br>• Gate 1 (Window Gate)<br>• Gate 2 (Cardinality Spectrum Gate: discrete choke-points vs continuous floats) | None |
| **Knobs 4 & 5: `Split / Merge` & `Place`**<br>([05-knob-split-and-place.md](file:///scratch/kevinsong/learning/framework/05-knob-split-and-place.md)) | • TP, DP, PP, EP, Context-Parallel decode<br>• Why EP beats TP at scale (64 GPUs)<br>• GQA KV head sharing<br>• Prefill/Decode & Attn/FFN disaggregation<br>• Peer-GPU / CPU KV offloading | • Combine tags (`+`, `softmax-merge`)<br>• Nonlinearity barrier (`ReLU`)<br>• Runtime lookup index (`sel`)<br>• Memory tiers | None |
| **Knob 6: `Window` & Combinations**<br>([06-knob-window-and-combinations.md](file:///scratch/kevinsong/learning/framework/06-knob-window-and-combinations.md)) | • Continuous & selective batching (Orca)<br>• Chunked prefill (Sarathi-Serve)<br>• Shared-prefix cascade attention (RadixAttention + Hydragen)<br>• **Dual-batch overlap (DualPipe / NanoFlow)**<br>• **Phase-shifted KV-scan under prefill MLP**<br>• **Draft-in-Verify all-to-all bubble**<br>• **Operator-specific MoE window size**<br>• **Stage-speculative Pipeline Parallelism** | • Window Dependency Rule (`A → B` $\implies W(A) < W(B)$)<br>• Shared-Read Rule (shared tensor counted once per window)<br>• Concurrent window cost `max(F, BW, Link_BW)` | None |

---

## 2. Devil's Advocate: Honest Limitations of Our Current Primitives

To avoid fooling ourselves, here are the places where our current primitives simplify reality or could break:

1. **The "Synchronous Barrier at Window Boundaries" Limitation (`Window` knob):**
   - *What our model assumes:* Execution is a sequence of discrete windows $W_1, W_2, \dots$, and `Total_Time = Σ_W Time(W)`.
   - *Where it breaks:* Suppose a single long network transfer `A` (`20 µs`) overlaps with a sequence of three dependent short compute ops `B1 → B2 → B3` (`5 µs` each). Because `B1 → B2 → B3` is a dependency chain, the Window Dependency Rule forces `B1`, `B2`, and `B3` into three separate windows (`W1`, `W2`, `W3`). To let `A` overlap all three, our representation has to artificially "split" `A` into `A1, A2, A3` across `W1, W2, W3`—even if `A` is a single indivisible transfer.
2. **The "Unlimited On-Chip Weight Sharing Within a Window" Loophole (`Window` knob):**
   - *What our model assumes:* If two nodes in the same window $W$ read weight tensor `W_MLP`, `W_MLP` is counted once in `HBM_bytes(W)`.
   - *Where it breaks:* In real hardware, two nodes only share a single HBM read of `W_MLP` if `W_MLP` stays resident in on-chip SRAM/registers while both run (i.e., they are fused into one kernel or one grouped GEMM).
   - *Simple guard:* Shared reads inside a window $W$ only count once for nodes of the **same operator** (or fused operator pair) whose combined active weight slice fits on chip.
3. **Resource Contention Inside a Window (`F + BW` Interference):**
   - *What our model assumes:* `Time(W) = max(FLOPs/F, HBM_bytes/BW, Link_bytes/Link_BW)`.
   - *Where it breaks:* On real GPUs, running a heavy HBM-streaming kernel (`KV_Scan`) at the exact same time as a heavy compute kernel (`MLP`) shares SMs and L2 cache bandwidth; you typically get ~80–85% of peak `(F, BW)` rather than 100% of both for free (which is why systems like NanoFlow explicitly partition SMs).
4. **Search-Space Blowup Without Periodic Templates (`Window` and `Place` knobs):**
   - *What our model assumes:* Any valid topological assignment of nodes to windows is in the search space.
   - *Where it breaks:* A 32-layer model with 2 request chunks has hundreds of nodes. Enumerating arbitrary cross-layer pairings (`layer 1 attn(R1) || layer 27 mlp(R2)`) is only tractable if the agent searches over **repeating window templates** (e.g., a 1-layer or 2-layer steady-state schedule repeated across $l = 1..L$).

---

## 3. Key Lessons Learned

1. **Enumerability (plausibility) comes before optimality.** The first test of the framework is not "does this win on an H100?"—it is "does this optimization exist as a distinct, reachable row when we systematically vary the knobs?"
2. **Always explain Compute-bound vs. HBM-bound via the reuse count $N$.** Counting how many tokens $N$ reuse each loaded byte from HBM ($N > 333$ vs. $N < 333$) makes every bottleneck table immediately transparent.
3. **Unifying Merge and Overlap into Execution Windows (`Window` knob) resolved a major gap.** Before Windows, keeping two request tiles `R1, R2` unmerged had three different physical meanings (serial, lockstep, or pipelined overlap) that collapsed into a single point in the knob space.
4. **The Shared-Read Rule inside Windows automatically explains why DualPipe needs large batches.** Putting `mlp(R1)` in Window 3 and `mlp(R2)` in Window 4 overlaps network transfers, but forces Window 3 and Window 4 to each load `W_MLP` from HBM—which only wins when each half-batch is near or above the ridge point ($N \ge 333$).
5. **The Two Gates of Speculation (Window Gate + Cardinality Gate) explain both why vanilla Transformers only speculate tokens/experts AND where new architectures open new speculation sites.**
6. **Never skip candidate families silently—always write the explicit pruning proof (`05` §4.1).** Jumping straight to "50/50 symmetric TP or PP" by habit hides why intra-layer alternating cuts lose (64 PCIe hops vs. 1 for the same serial `Attn → FFN` dependency) and misses the exact cases where **asymmetric splits ($\alpha^* \neq 0.5$)** win (e.g., balancing a `1.27 GB` `LM_Head` with a `33 / 31` layer split, or pinning `75.5 MB` recurrent states `S` in GPU VRAM while streaming only the `16` Full-Attention KV caches over PCIe across every 4-layer block in [examples/qwen3_8-27b-2x4090/handoff-implementation.md](file:///scratch/kevinsong/learning/framework/examples/qwen3_8-27b-2x4090/handoff-implementation.md)).
7. **Distinguish Append-Only Keep (`k[1..t]`) from In-Place Overwrite Keep (`S[t]`) (`02` §4.4).** Recurrent state `S[t]` is only fixed-size because it overwrites `S[t-1]`, which breaks speculative rollback and shared-prefix branching unless paired with input-window caching (`Row B: DeltaLog`) or lazy suffix materialization (`Row C: keep raw k,v up to the ~150-token crossover`).
8. **High-level intuition first.** Every derivation should start with a 2-sentence plain-English intuition (like the "bus vs. security line" picture for Phase-Shifted KV-Scan) before showing the index equations and tables.

---

## 4. Open Questions and Next Steps

1. **Continue sweeping popular and recent paper optimizations:**
   - **Ring Attention / Striped Context Parallel Prefill** (`Split` along `i, j` + `Place` + `Window` ring overlap).
   - **CPU KV Offloading + Layer Prefetching** (FlexGen / InfiniGen).
   - Recent architecture-specific papers (e.g., hybrid Mamba/Attention serving, Disaggregated MoE serving).
2. **Formalize periodic cross-layer Window templates ($\Delta$-shifted pipelines):**
   Write a compact notation for periodic window templates so intra-layer and cross-layer overlap are enumerated by varying a single template offset parameter.
3. **Include Head (`h, g`) and Vocabulary (`V`) indices in the standard template:**
   Needed so TP head-count limits and output-logit / sampling bottlenecks appear in the same tables.
