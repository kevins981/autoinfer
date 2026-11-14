# Knobs 4 & 5: Split / Merge and Place (Parallelism, Disaggregation, and Offloading)

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) first.

---

## 1. What the `Split / Merge` and `Place` Knobs Are

- **Knob 4 (`Split / Merge`):**
  - **`Split`:** Partition any index range (`r`, `i`, `j`, `m`, `d`, `e`, `h`) into pieces (subsets).
    - You can split **within one device** (e.g., splitting a 4096-token prompt along `i` into chunks, or splitting `j` into shared prefix + private suffix).
    - Or you can split **and `Place` across multiple devices** (which is how all forms of parallelism—TP, DP, EP, Context Parallelism—are generated!).
  - **`Merge`:** The inverse operation—fuse nodes that differ only in an index (`r`, `i`, `t`) that is missing from a shared input (when placed in the same Execution Window—see `06`).
- **Knob 5 (`Place`):** Assign each compute node (or split piece) to a **device**, and each kept data node to a **(device, memory tier)** (Local HBM, Peer GPU HBM over `Link_BW`, or CPU DRAM over PCIe).
  - Any dependency edge whose producer and consumer sit on different devices or tiers automatically inserts a **transfer node** (`Link_bytes / Link_BW`).

---

## 2. The Four Mechanical Rules When Splitting an Index Across Devices

When you split an index across GPUs, you do **not** need to memorize Megatron-LM or DeepSpeed formulas—the communication pattern follows mechanically from four rules on the index:

| Property of the Index Being Split | What Happens on the Graph | Example |
|---|---|---|
| **Rule 1: Index is NOT reduced over** | **Zero communication!** Each GPU computes its slice independently. | Splitting `r` (Data Parallel) or splitting `m` in the first MLP layer `h[r,i,m] = ReLU(Σ_d a[r,i,d] W_1[d,m])`. |
| **Rule 2: Index IS reduced over (`Σ` or `softmax`)** | Inserts a **combine node** across GPUs using the reduction's combine tag (`+` $\rightarrow$ AllReduce / ReduceScatter; `softmax_j` $\rightarrow$ online-softmax merge `(m, ℓ, o)`). | Splitting `m` in the second MLP layer `o[r,i,d] = Σ_m h[r,i,m] W_2[m,d]` inserts an AllReduce (`+`) after `W_2`. |
| **Rule 3: A nonlinearity (`ReLU`) immediately follows a split reduction** | Because `ReLU(A + B) ≠ ReLU(A) + ReLU(B)`, the combine node is forced to run **before** the nonlinearity! | Splitting `d` in `h = ReLU(Σ_d a[d] W_1[d,m])` forces an AllReduce *before* `ReLU`. |
| **Rule 4: Index is a runtime Lookup Index (`sel[r,i,s]`)** | Tokens must travel to whichever GPU holds `W[sel[r,i,s]]` and return after $\implies$ **2 data-dependent All-to-All transfers** (`disp` and `comb`) + load imbalance! | Splitting expert index `e` across GPUs (Expert Parallelism). |

---

## 3. Worked Derivation 1: Enumerating All Parallelism Modes on an MoE Layer

Suppose we have 2 GPUs and want to distribute the MoE layer (`h[r,i,s,m] = ReLU(Σ_d a W_1[sel,d,m])`, `o[r,i,s,d] = Σ_m h W_2[sel,m,d]`). We enumerate splitting each index:

| Index Split Across 2 GPUs | Expert Weights per GPU | Communication Inserted (by Rules 1–4) | Load Balance | Familiar Name |
|---|---|---|---|---|
| **Request `r`** | Full copy (`2×` memory) | None within the layer (Rule 1) | Depends on request lengths | **Data Parallel (DP)** |
| **Inner dim `m`** | Half of every expert | Zero comm at `W_1` (Rule 1); **1 AllReduce (`+`)** after `W_2` (Rule 2) | Perfect | **Tensor Parallel (TP)** |
| **Hidden dim `d`** | Half of every expert | **AllReduce before `ReLU`** at `W_1` (Rule 3) | Perfect | Valid, rarely optimal |
| **Slot `s`** | Full copy | Combine over slots | Fine | Dominated by `r` |
| **Lookup `e = sel[r,i,s]`** | **Half the experts (4 of 8)** | **2 All-to-Alls** (`disp` before `W_1`, `comb` after `W_2`; volume $\approx K \times$ TP) (Rule 4) | **Data-dependent** (router skew) | **Expert Parallel (EP)** |
| **Past pos `j` (in `attn`)** | — | **Online-softmax merge `(m, ℓ, o)`** across GPUs (Rule 2) | Perfect | **Context / Sequence Parallel Decode** |

### Why EP Beats TP at Scale (64 GPUs) Even Though TP Wins on 2 GPUs
On 2 GPUs, TP has less communication than EP and perfect balance. Why do large MoE systems (like DeepSeek-V3) use **DP Attention + EP Experts** on 64 GPUs? Two index-size facts:
1. **Index too small to split (`#KV heads < #GPUs`):** If a model has 8 KV heads (`g = 1..8`), Attention TP cannot split `g` across 64 GPUs without replicating the KV cache $8\times$. Splitting Attention along `r` (DP) avoids KV replication.
2. **Reshard cost between Attention and Experts:** Once Attention is split by `r` (tokens already distributed across GPUs), transitioning to Experts split by `e` (EP) costs **one All-to-All**, whereas transitioning to Experts split by `m` (TP) would require an All-Gather of all tokens onto every GPU first.

---

## 4. Worked Derivation 2: `Place` at the Subgraph Level $\rightarrow$ Disaggregation & Pipeline Parallelism

Instead of splitting an index symmetrically across identical GPUs, we can **Place** different subgraphs (or request groups) onto different device pools:

| What Is Placed on Pool 0 vs. Pool 1 | Crossing Edge (`Link_bytes`) | When It Wins | Familiar Name |
|---|---|---|---|
| **Layers `1..16` on GPU 0; Layers `17..32` on GPU 1** | Activation `x[l=17] = y[l=16]` (`8 KB/token`) | When weights exceed 1 GPU's HBM capacity $C$ and links are slow; requires multiple requests in flight (or Stage-Speculative PP—see `06`) to avoid idle bubbles. | **Pipeline Parallelism (PP)** |
| **Attention + Router on Pool 0; MoE Experts on Pool 1** | `a, sel, w` (`disp` and `comb`) | When Attention (`KV-bandwidth-bound`) and Experts (`Weight-bandwidth-bound`) want **different batch sizes or hardware ratios** (and micro-batches pipeline across the two pools). | **Attention–FFN Disaggregation (MegaScale-Infer)** |
| **Many-ready-row requests (`A`, $N > 333$) on Pool 0; 1-ready-row requests (`B`, $N < 333$) on Pool 1** | Kept `k,v[1..n_A]` transferred **once** per request after `A`'s first pass | When separate pools avoid chunked-prefill interference and `KV_bytes / Link_BW` is small relative to generation lifetime. | **Prefill / Decode Disaggregation (DistServe / Splitwise)** |

### 4.1 The Systematic Weight-Placement Sweep: Proving When Symmetric TP/PP Wins—and When Asymmetric Splits Beat 50/50!

**Never jump straight to "symmetric 50/50 TP or PP" by habit.** When distributing a model across 2 identical GPUs (`GPU 0`, `GPU 1`), a blind agent must sweep all ways to place or split the weight nodes (`W_attn[l]`, `W_ffn[l]`, `W_LM`) and split ratios $(\alpha, 1-\alpha)$, and explicitly prove why dominated groups are pruned:

1. **Why Intra-Layer `W_attn` (GPU 0) vs. `W_ffn` (GPU 1) Is Strictly Dominated by Layer-Cut `PP=2` on Homogeneous GPUs:**
   - Within every layer `l`, `Attn[l] → FFN[l] → Attn[l+1]` is a **strictly serial dependency chain**.
   - Placing `W_attn[l]` on GPU 0 and `W_ffn[l]` on GPU 1 serializes single-batch execution (`GPU 1` cannot start `FFN[l]` until `GPU 0` finishes `Attn[l]`), which is the *exact same* sequential bottleneck as placing Layers `1..L/2` on GPU 0 and `L/2+1..L` on GPU 1 (`PP=2`), **except** alternating inside every layer crosses the inter-GPU link **$2L$ times** instead of **1 time**!
   - *Explicit Pruning Rule:* Along any serial chain of weight blocks on homogeneous GPUs, a single cut (`PP`) strictly dominates alternating cuts unless the two block types have fundamentally different batching/hardware requirements (as in MoE Attention–FFN Disaggregation above).

2. **Why Placing `W_Q` on GPU 0 and `W_K, W_V` on GPU 1 Is Dominated by Splitting Head Index `h, g` (`TP`):**
   - Inside Attention, `q = x W_Q`, `k = x W_K`, `v = x W_V` are independent (`W_Q ⊥ W_K ⊥ W_V`), so placing whole matrices on different GPUs is just a special case of splitting the output channel index—except in GQA (`40 Q heads` vs `8 KV heads`), `W_Q` is $5\times$ larger than `W_K`, leaving GPU 1 idle 80% of the time! Splitting each head group evenly across GPUs (`TP`) achieves the same parallelism with better balance.

3. **The Symmetric vs. Asymmetric Balance Equation ($\alpha = 0.5$ vs. $\alpha \neq 0.5$):**
   - Suppose we split weights (via TP channel fraction $\alpha$ or PP layer fraction $\alpha$) across two identical GPUs that also perform other work $T_{\text{other}, i}$ and hold other state $M_{\text{state}, i}$:
     $$\text{Time} = \max\Big(\alpha \cdot T_{\text{weights}} + T_{\text{other, 0}},\;\; (1-\alpha) \cdot T_{\text{weights}} + T_{\text{other, 1}}\Big)$$
     $$\text{Memory}_0 = \alpha \cdot M_{\text{weights}} + M_{\text{state, 0}} \le C, \qquad \text{Memory}_1 = (1-\alpha) \cdot M_{\text{weights}} + M_{\text{state, 1}} \le C$$
   - **When Symmetric ($\alpha = 0.5$) Is Provably Optimal:** When $T_{\text{other, 0}} = T_{\text{other, 1}}$ and $M_{\text{state, 0}} = M_{\text{state, 1}}$, $\max(\alpha, 1-\alpha)$ is uniquely minimized at $\alpha = 0.5$.
   - **When Asymmetric ($\alpha \neq 0.5$) Beats 50/50:** Whenever $T_{\text{other, 0}} \neq T_{\text{other, 1}}$ or $M_{\text{state, 0}} \neq M_{\text{state, 1}}$, the optimal split sets the two branches equal:
     $$\alpha^* = \frac{1}{2} + \frac{T_{\text{other, 1}} - T_{\text{other, 0}}}{2 \cdot T_{\text{weights}}}$$
     *Concrete examples where $\alpha^* \neq 0.5$ wins:*
     - **Large `LM_Head` on Stage 1 in `PP=2`:** In `Qwen3.8-27B`, `LM_Head` (`248K vocab × 5120 = 1.27 GB`) sits on GPU 1 and equals **3 full Transformer layers** (`412 MB/layer`). Thus **33 layers on GPU 0 + (31 layers + `LM_Head`) on GPU 1** beats a naive `32 / 32` layer split!
     - **Co-locating a Speculative Draft Model or Asymmetric KV/State Pool on One GPU:** If GPU 0 hosts a `1.5 GB` draft model (or extra pinned recurrent state `S`), shifting $\Delta \alpha$ of target layers/channels or KV cache to GPU 1 equalizes both memory and window time.

---

## 5. Worked Derivation 3: `Place` on Data Nodes $\rightarrow$ Peer-GPU / CPU KV Offloading

Data nodes (`W`, `k,v[r,j]`) also have a `Place` choice across memory tiers: `{Local HBM, Peer GPU HBM over Link_BW, CPU DRAM over PCIe}`.

**Snapshot:** 16 decode requests (`1000` tokens each = `256 MB` total KV cache), 1 GPU running weights (`368 MB / 3 TB/s = 122 µs` per layer).

| Data Node | Placed On | Transfer / Read Time | Can It Overlap in the Same Window (`06`)? |
|---|---|---|---|
| Model weights `W` | Peer HBM (`600 GB/s`) | `368 MB / 600 GB/s = 613 µs` | **No** (everything waits on weights $\implies 5\times$ slower) |
| **Subset of `k,v` (`73 MB` $\approx 4.5$ requests)** | **Peer HBM (`600 GB/s`)** | **`73 MB / 600 GB/s = 122 µs`** | **YES!** Streams over `Link_BW` (`122 µs`) **in the same Window** while local HBM streams `W` (`122 µs`)! |

**Condition:** Offloading kept KV data to a slower tier (Peer HBM or CPU PCIe) and streaming it in the same Window as weight loads is **free on latency** whenever:
$$\frac{\text{Offloaded\_KV\_Bytes}}{\text{Link\_BW}} \le \frac{\text{Local\_Weight\_Bytes}}{\text{HBM\_BW}}$$
and wins whenever local HBM capacity $C$ would otherwise cap the batch size!
