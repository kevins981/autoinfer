# Knob 6: Execution Windows (`Window`) and Multi-Knob Combinations

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) first.

---

## 1. What the `Window` Knob Is (Unifying Batching and Multi-Resource Overlap)

Earlier drafts treated **Batching / Merging** (fusing tokens that read the same weights) and **Scheduling / Engine Overlap** (running network all-to-all at the same time as compute) as two completely different mechanisms.
The **`Window`** knob unifies both into one mechanical primitive:

> **Definition of an Execution Window (`Window` knob):**
> On each device, assign every compute and transfer node to a discrete sequential **Execution Window** $W_1, W_2, W_3, \dots$ subject to two rules:
> 1. **Window Dependency Rule:** If `A → B` is a dependency path in the graph, `A` must be placed in a strictly earlier window than `B` (`window(A) < window(B)`). Only mutually independent nodes (`A ⊥ B`) may be placed in the **same window** (`{A || B}`). *(This prevents collapsing a dependent chain like an entire forward pass into a single window!)*
> 2. **Window Cost & Shared-Read Rule:**
>    - **Shared inputs counted once per window:** If multiple nodes placed in the *same* window $W$ read the same tensor (e.g., weight matrix `W_MLP` or shared prefix `k_shared`), that tensor is counted **once** in `HBM_bytes(W)`. If they are placed in *different* windows, each window loads the tensor from HBM separately.
>    - **Concurrent resource consumption across the window:** All nodes in $W$ execute concurrently across the device's resources `(F, BW, Link_BW)`:
>      $$\text{Time}(W) = \max\!\left(\frac{\sum_{u \in W} \text{FLOPs}(u)}{F},\; \frac{\sum_{u \in W} \text{HBM\_bytes}(u)}{BW},\; \frac{\sum_{u \in W} \text{Link\_bytes}(u)}{Link\_BW}\right), \qquad \text{Total\_Time} = \sum_W \text{Time}(W)$$

Look at the two ways placing independent nodes `A` and `B` into the **same Window** beats running them in separate windows (`Time(W[A || B]) < Time(W[A]) + Time(W[B])`):
- **Mechanism 1 (Shared HBM Read — "Batching / Merging"):** `A` and `B` read the **same weight/KV tensor** from HBM, so combining them into one Window cuts HBM bytes in half and raises reuse count $N$ toward `333`!
- **Mechanism 2 (Orthogonal Resource Overlap — "Pipelining / Co-Scheduling"):** `A` and `B` bottleneck on **different hardware resources** (`F` vs `BW` vs `Link_BW`), so running them in the same Window takes $\max(T_A, T_B)$ instead of $T_A + T_B$!

---

## 2. Worked Derivations Under Mechanism 1 (Shared Reads in a Window $\rightarrow$ Batching, Chunked Prefill, Cascade Attention)

### 2.1 Continuous Batching, Selective Batching, and Chunked Prefill
**Snapshot:** Requests `B1..B16` each have `k,v[1..1000]` kept and **1 ready position** (`i = 1001`). Request `A` has nothing kept and **4096 ready positions** (`i = 1..4096`). Goal: `B` latency $\le 2\times$ solo.

| Tokens Placed in Window $W$ (at Linear Layers) | Reuse Count $N$ | Bottleneck ($N$ vs 333) | Window Time (vs Solo `B`) | Verdict |
|---|---|---|---|---|
| 16 separate windows (one per `B`) | $N = 1$ | HBM-bound ($1 \ll 333$) | 16 loads total | Wastes $16\times$ HBM bandwidth |
| **All `B1..B16` in 1 Window** (at linear layers `W`; KV-scan separate since `k,v` carry `r`) | **$N = 16$** | **HBM-bound ($16 < 333$)** | **1 load ($1\times$ latency)** | **Continuous + Selective Batching (Orca)!** |
| All `B1..B16` + all 4096 rows of `A` in 1 Window | $N = 4112$ | Compute-bound ($4112 \gg 333$) | $12.3\times$ | **Violates latency goal ($\le 2\times$)!** |
| **All `B1..B16` + a Chunk of $n = 317$ rows of `A` (`Split` along `i` + `Window`)** | **$N = 16 + 317 = 333$** | **At Ridge Point ($N = 333$)** | **1 load ($1\times$ latency; `A` finishes in 13 windows)** | **Chunked Prefill (Sarathi-Serve)!** Rides the `333 - 16 = 317` free compute rows per weight load! |

### 2.2 Shared-Prefix Cascade Attention (`Rewrite` + `Split along j` + `Window`)
**High-level intuition:** Suppose `B1..B16` share a **2,000-token system prompt** (`j = 1..2000`) plus **500 private tokens** (`j = 2001..2500`).
1. **Dedup `Rewrite`:** Collapses `k,v[r, 1..2000]` into `k_shared[1..2000], v_shared[1..2000]` (**no `r` index!**).
2. **`Split` along `j` + `Window` across `r`:** Split `j = 1..2500` into Piece 1 (`1..2000`, `r`-free) and Piece 2 (`2001..2500`, carries `r`), connected by `j`'s combine tag **`softmax-merge (m, ℓ, o)`**.
3. Place all 16 requests' Piece 1 (`q[1..16] × k_shared[1..2000]`) in the **same Window**: `k_shared` (`32 MB`) is read from HBM **once** at reuse **$N = 16$** instead of 16 times at $N = 1$ (**Hydragen / FlashInfer Cascade Attention**)!

---

## 3. Worked Derivation Under Mechanism 2: Dual-Batch Overlap (DualPipe / NanoFlow)

**Setup (from `05`):** 2 GPUs running **DP Attention + EP Experts** on 100 requests per GPU.
For any batch `R`, each layer is a 4-step dependency chain alternating between `HBM/Compute` and `Link`:
```text
attn(R) [HBM/Compute]  ──►  disp(R) [Link]  ──►  mlp(R) [HBM/Compute]  ──►  comb(R) [Link]
```
If all 100 requests stay in one group `R`, the Window Dependency Rule (`A → B` $\implies W(A) < W(B)$) forces the 4 steps into 4 separate windows—leaving `Link` 100% idle during `attn`/`mlp`, and `HBM/Compute` 100% idle during `disp`/`comb`!

Now apply **`Split` along `r` within each GPU** to split `R` into two independent half-batches `R1` (50 requests) and `R2` (50 requests), and enumerate the valid **Window assignments**:

| Window Schedule for `R1` and `R2` | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 | Valid (`A→B` $\implies W(A)<W(B)$)? | Verdict |
|---|---|---|---|---|---|---|---|
| **1. Aligned (Lockstep)** | `attn(R1)`, `attn(R2)` | `disp(R1)`, `disp(R2)` | `mlp(R1)`, `mlp(R2)` | `comb(R1)`, `comb(R2)` | — | ✓ | Shares weights in W1 & W3, but **0% overlap** between `Link` and `HBM/Compute`! |
| **2. Serial** | `attn(R1)` | `disp(R1)` | `mlp(R1)` | `comb(R1)` | `attn(R2)`... | ✓ | 0% overlap *and* loads weights twice! |
| **3. Staggered by 1 Window** | `attn(R1)` | **`disp(R1)` $\|$ `attn(R2)`** | **`mlp(R1)` $\|$ `disp(R2)`** | **`comb(R1)` $\|$ `mlp(R2)`** | `comb(R2)` | **✓** | **Dual-Batch Overlap (DualPipe / NanoFlow)!** In W2, W3, W4, one half-batch uses `Link_BW` while the other uses `F / BW`! |

### Why the Shared-Read Rule Automatically Explains When DualPipe Wins vs. Loses
In Variation 3 (Staggered), `mlp(R1)` is in **Window 3** and `mlp(R2)` is in **Window 4**.
By the **Shared-Read Rule**, because they are in *different* windows, **Window 3 and Window 4 each load the expert weights `W` from HBM separately!**
- **Wins at large batch ($N_{\text{per expert}} \ge 333$):** Both `mlp(R1)` and `mlp(R2)` are **Compute-bound**, so loading `W` twice is hidden behind compute FLOPs, and overlapping `disp`/`comb` saves the entire network transfer time!
- **Loses at small decode batch ($N \ll 333$):** Loading `W` twice from HBM (`2 × T_weights`) costs *more* HBM time than the network time saved!

---

## 4. Four Novel Multi-Knob Optimizations Discovered via the `Window` Knob

### 4.1 Novel Candidate 1: "Phase-Shifted KV-Scan" (Overlapping Long-Context Decode with Compute-Bound Prefill on 1 GPU)

**High-level intuition in plain words (the "Bus vs. Security Line" picture):**
Suppose on **1 GPU** we have:
- **`Long`:** 1 decode token with a massive **100,000-token KV cache** (`1,600 MB` in HBM).
- **`Prefill`:** Prompt chunks `Prefill_1`, `Prefill_2` of **500 tokens each**.

Think of the shared weight matrices (`W_QKV` and `W_MLP`) as **buses** that all tokens ride together, and think of `Long`'s 100k KV-cache scan as a slow **security line** that only `Long` has to walk through:
- **Check bottlenecks via Reuse Count $N$:**
  - `QKV_proj(Prefill_1 + Long)`: **$N = 501 > 333 \implies$ Compute-bound!**
  - `KV_Scan(Long)`: **Only $N = 1$ token (`q_Long`) reads `Long`'s 100k KV cache ($1 \ll 333$) $\implies$ 333× HBM-bandwidth-bound!** (Reads zero model weights, ~0 FLOPs.)
  - `MLP(Prefill_1)`: **$N = 500 > 333 \implies$ Compute-bound!** (Compute units are 100% busy; HBM bandwidth has spare room!)
- **In baseline chunked prefill:** Everyone gets off the `W_QKV` bus and **waits** for `Long` to finish its slow 100k KV security line before boarding the `W_MLP` bus together.
- **What the `Window` knob discovers:** After riding `W_QKV` together in Window 1, `Prefill_1` immediately runs its compute-bound `MLP(Prefill_1)` in **Window 2**, while **in that exact same Window 2**, `Long` streams its 100k KV cache from HBM! Then in **Window 3**, `Long` (whose KV scan is now finished!) simply hops onto **`Prefill_2`'s `W_MLP` bus**!

| Window | Nodes Placed in Window | Bottleneck Pairing | Shared Weights Read |
|---|---|---|---|
| **Window 1** | `QKV_proj(Prefill_1 + Long)` | Merged ($N = 501$) | `W_QKV` read **once** |
| **Window 2** | **`[Attn(Prefill_1) → MLP(Prefill_1)]` $\|$ `KV_Scan(Long)`** | **Compute (`F`) $\|$ HBM (`BW`)!** | `W_MLP` read for `Prefill_1`; `KV_Scan` reads 0 weights |
| **Window 3** | `QKV_proj(Prefill_2)` → `Attn(Prefill_2)` → **`MLP(Prefill_2 + Long)`** | Merged ($N = 501$) | `W_MLP` read **once** for `Prefill_2 + Long`! |

---

### 4.2 Novel Candidate 2: "Draft-in-Verify-Bubble" (Asymmetric Speculative Pipelining Under EP)

**High-level intuition:**
When running **Speculative Decoding** ($k+1 = 5$ tokens verified per request) on an **Expert-Parallel (EP)** model, the target verification pass has two network all-to-all steps (`v_disp`, `v_comb`) per layer that are **$5\times$ larger** than normal decode—leaving **HBM and Compute 100% idle** during those network windows. Meanwhile, the **Drafter** (e.g., a 1-layer EAGLE head) is a serial chain of HBM-bound steps with **zero network traffic**.
- **Move (`Split` requests into `R1, R2` + `Speculate` + `Window`):** While Cohort `R2` runs its target `Verify` pass, place Cohort `R1`'s serial `Draft` steps **inside `R2`'s `v_disp` and `v_comb` network all-to-all windows** (`HBM/Compute || Link`)!

---

### 4.3 Novel Candidate 3: Decoupled Operator Window Size for MoE (`Split / Merge` + `Window`)

**High-level intuition:**
In an MoE model with $E = 8$ experts and top-$K = 2$ routing, each expert only sees **$K / E = 1/4$** of the tokens in a window.
- Dense attention projections (`W_QKV`) reach the ridge point at **$N = 333$ tokens** per window.
- At 333 tokens, each MoE expert only sees $333 \times (2/8) \approx 83$ tokens ($N = 83 < 333 \implies$ still **$4\times$ HBM-bandwidth-bound**!). Experts want $333 \times (8/2) \approx 1332$ tokens to reach the ridge!
- **Move:** Run **4 Attention windows of 333 tokens** (`W1..W4`, respecting causal order), buffer their outputs `a` in HBM, and merge all $4 \times 333 = 1332$ tokens into **1 MoE Window (`W5`)**—cutting expert weight loads by $E/K = 4\times$!

---

### 4.4 Novel Candidate 4: "Stage-Speculative Pipeline Parallelism" (`Place` + `Speculate` + `Window`)

**High-level intuition:**
When a 32-layer model is split across **4 GPUs using Pipeline Parallelism (`Place`)**, single-request decode leaves **3 out of 4 GPUs sitting 100% idle in every window** because GPU 0 (`L1–8`) cannot start token `t+1` until GPU 3 (`L25–32`) finishes token `t` four windows later.
- **Move:** Apply **`Speculate`** at GPU 0's stage boundary (`y[L8, t] → Guess → x̂[t+1]`), and use GPU 0's **3 idle bubble windows** (`W2, W3, W4`) to immediately pipeline speculative tokens $\hat{x}[t+1], \hat{x}[t+2], \hat{x}[t+3]$ right behind `t` at **zero extra hardware cost**!

| Window | GPU 0 (`L1–8`) | GPU 1 (`L9–16`) | GPU 2 (`L17–24`) | GPU 3 (`L25–32`) |
|---|---|---|---|---|
| **W1** | Runs `t` $\rightarrow$ guesses $\hat{x}[t+1]$ | Idle | Idle | Idle |
| **W2** | **Runs $\hat{x}[t+1]$** $\rightarrow$ guesses $\hat{x}[t+2]$ | **Runs `t`** | Idle | Idle |
| **W3** | **Runs $\hat{x}[t+2]$** $\rightarrow$ guesses $\hat{x}[t+3]$ | **Runs $\hat{x}[t+1]$** | **Runs `t`** | Idle |
| **W4** | **Runs $\hat{x}[t+3]$** | **Runs $\hat{x}[t+2]$** | **Runs $\hat{x}[t+1]$** | **Runs `t` $\rightarrow$ true $x[t+1]$!** |
| **W5** | Verify / continue from $x[t+1]$ | **Runs $\hat{x}[t+3]$** | **Runs $\hat{x}[t+2]$** | **Runs $\hat{x}[t+1]$ $\rightarrow$ verifies!** |
