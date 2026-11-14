# Knob 3: Speculate (Breaking Dependencies via Guess, Verify, and Recover)

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) first.

---

## 1. What the `Speculate` Knob Is (The Unified Transformation)

Earlier drafts described speculation loosely as *"delete an edge `A → B`."* That hid how verification works and why only certain edges in a computation graph can be speculated.

In our framework, **Speculation is a single unified transformation** on any dependency edge `A(x) ──(a)──► B(a)`:

```text
BASELINE (Sequential — requires 2 Windows because B depends on A):
  Window 1:  x ──► [ A(x) ] ──► a_true
  Window 2:  a_true ──► [ B(a_true) ] ──► y

SPECULATED (Unified Transformation):
  Window 1:  x ──► [ Guess(x) ] ──► â
  Window 2:  [ Verify(x, â) ]   ||   [ B(â) ]       <-- Independent! Placed in the SAME Window!
  Window 3 (only if Verify == False, rate 1 - p):  [ Recover ]
```

The expected execution time is:
$$\text{Time}_{\text{spec}} = \text{Time}(\text{Guess}) + \text{Time}\Big(\text{Window}\big[\text{Verify}(x, \hat{a}) \;\|\; B(\hat{a})\big]\Big) + (1 - p)\cdot \text{Time}(\text{Recover})$$

---

## 2. The Two Gates Every Speculation Candidate Must Pass

For $\text{Time}_{\text{spec}} < \text{Time}(A) + \text{Time}(B)$, a candidate edge `A ──(a)──► B` must pass **Two Gates**:

### Gate 1: The Window Gate (`Time(Window[Verify || B(â)]) < Time(A) + Time(B)`)
- **In Exact Speculation:** To verify what a neural network node `A(x)` outputs with 100% mathematical certainty, `Verify(x, â)` must run `A(x)` to obtain `a_true` and check `â == a_true`. Because you still run `A(x)` inside `Verify`, breaking the dependency `A → B` **only** saves time if placing `A(x)` and `B(â)` in the **same Window** is faster than running them sequentially:
  - Either `A(x)` and `B(â)` **share weights** in that Window (by the Missing-Index Rule!),
  - Or `A(x)` and `B(â)` use **orthogonal hardware resources** (`F` vs `BW` vs `Link_BW`)!
- **In Approximate Speculation:** `Verify(x, â)` is a cheap confidence or bound check (`Conf(x, â) > τ`) that skips running `A(x)` altogether when the check passes, trading a bounded accuracy difference $\epsilon$ for skipping `A(x)`.

### Gate 2: The Cardinality Spectrum Gate ($p > 0$)
Everything on a computer is discrete, but tensors sit on a **cardinality spectrum** (number of possible values):
- **High-cardinality end (`4,096` FP16 floats = $(2^{16})^{4096}$ possible values):** The probability of guessing a continuous activation vector (`q[t]`, `a[t]`, `y[l, t]`) *bit-for-bit* (`â == a_true`) is **$p \approx 0$**! High-cardinality edges can only be speculated in **Approximate** mode (`||â - a_true|| < ε` or `Conf > τ`).
- **Low-cardinality choke-points ($2$ to $10^5$ discrete values):** A sampled token ID (`x[t+1]`), an MoE router selection (`sel[t]`), a sparse-attention KV block ID (`sel_blk[t]`), or a binary halting gate (`halt[k]`) has small cardinality and concentrated probability mass, so **$p = P(\hat{a} == a_{\text{true}}) \approx 0.70\text{–}0.95$**!

---

## 3. Thought Experiment 1: Testing Every Site in a Vanilla Transformer (Proving Why Only Tokens & Experts Win!)

Suppose our blind agent has never heard of speculative decoding and tests the Unified Speculation Transformation at **every site `(1)–(5)`** of a standard Transformer at decode ($N = 1$, HBM-bound):

| Candidate Site `a` (`A ──(a)──► B`) | Gate 1: Does `Window[ A(x) \|\| B(â) ]` Beat `Time(A) + Time(B)`? | Gate 2: Cardinality of `a` (Can Exact `â == a_true` Have $p > 0$?) | Exact Verdict | What If We Allow *Approximate* `Verify` (`Conf > τ` or `\|\|â - a\|\| < ε`)? |
|---|---|---|---|---|
| **1. Query `q[t]`**<br>(`QKV_proj ──► KV_Scan`) | **NO!** At batch 1, `QKV_proj` reads `W_QKV` from HBM and `KV_Scan` reads `KV_cache` from the *same* HBM. Different tensors, same bottleneck $\implies \text{Time}(W[A \| B]) = T_A + T_B$! | **Huge** (`4096` floats $\implies p = 0$). | **Fails both gates!** | Still fails (`Verify` without reading `W_Q` cannot certify accuracy). |
| **2. Attention Output `a[t]`**<br>(`KV_Scan ──► MLP`) | **NO at batch 1** (`KV_Scan` and `MLP` both bottleneck on HBM and read different tensors). | **Huge** (`4096` floats $\implies p = 0$). | **Fails both gates!** | **Approximate Work-Skipping (`Skip 95% of KV blocks` when block-max bound $< \epsilon$):** Wins for long context (**Quest / RetrievalAttention**)! |
| **3. Router Choice `sel[t]`**<br>(`Attn+Router ──► Load_Expert`) | **YES!** `A` (`Attn`) uses HBM/Compute, while `B` (`Prefetch_Expert(sel_guess)` from CPU/Peer GPU) uses **PCIe/Link**! Orthogonal resources $\implies \max(T_A, T_B) < T_A + T_B$! | **Tiny!** (`top-2 of 8` discrete expert IDs $\implies p \approx 0.80$). | **WINS! (Expert Prefetching)** | Skipping tail experts when top-1 confidence $> 0.9$ also wins! |
| **4. Layer Output `y[l, t]`**<br>(`Layer l ──► Layer l+1` on 1 GPU) | **NO on standard Transformers!** Because `W[l, d, m]` **carries index `l`**, `Layer 1` and `Layer 2` read *different* weights (`W[1]` vs `W[2]`) $\implies$ **zero weight sharing** ($N = 1$ on both)! | **Huge** (`4096` floats $\implies p = 0$). | **Fails both gates on standard models!** *(See §5 for Looped models!)* | Skipping layers `l+1..L` when confidence $> \tau$ = **Approximate Early Exit**. |
| **5. Next Token `x[t+1]`**<br>(`Step t ──► Step t+1`) | **YES!** `A` is `Target(x[t])` and `B` is `Target(x̂[t+1])`. Because `W[l, d, m]` has **NO `t` index**, running `Step t` and `Step t+1` in the same Window **shares 100% of model weights** ($N = 2$ instead of $N = 1$)! | **Small!** (`1 discrete token ID` out of vocab $V \implies p \approx 0.70\text{–}0.85$). | **BIG WINNER! (Speculative Decoding)** | — |

### The Deep Insight from Row 4 vs. Row 5
Why does everyone speculate across *time steps `t`* (`x[t+1]`) instead of across *layers `l`* (`y[l, t]`) on 1 GPU?
1. **The Missing-Index Law:** Model weights `W[l, d, m]` carry `l` (so `Layer 1` and `Layer 2` cannot share weights in a Window), whereas `W[l, d, m]` does **not** carry `t` (so `Step t` and `Step t+1` share 100% of weights in a Window!).
2. **The Low-Cardinality Choke-Point Law:** Between `Step t` and `Step t+1`, the entire 4,096-float hidden state collapses through `Sample(...)` into **a single discrete integer `x[t+1]`** ($p \approx 0.80$), whereas between `Layer l` and `Layer l+1` the edge is a 4,096-float vector ($p = 0$)!

---

## 4. Worked Derivation: Chain vs. Tree Speculation (Medusa / EAGLE-2 / Sequoia)

**High-level intuition:**
At batch size 1, verifying a single chain of $k + 1 = 5$ speculative tokens in one window only reaches **$N = 5 \ll 333$**—leaving `333 - 5 = 328` tokens of free compute slack in the verification window!
Instead of making a single chain 64 tokens long (where $p^{64} \approx 0$), `Guess` can output **$B_d$ branching guesses at depth $d$**—forming a **tree of $M = 32$ candidate tokens** verified together in the **same Window** ($N = 32 \ll 333$, still 1 HBM weight load!).

| Speculation Variation | Candidate Nodes $M$ in Verification Window | Reuse Count $N$ on Weights & Past KV Cache | Window Time (vs Solo Decode, for $M \le 333$) | Acceptance Governing Term |
|---|---|---|---|---|
| **1. No speculation** | $M = 1$ | $N = 1$ | $1\times$ | 1 token |
| **2. Chain ($1$ guess per step)** | $M = k + 1$ (e.g., 5) | $N = 5$ | $1\times$ ($5 \ll 333$) | Bound by top-1 accuracy $p_1^d$ |
| **3. Tree ($B_d$ guesses at depth $d$)** | **$M = 1 + \sum_{d=1}^k \prod_{m=1}^d B_m$** (e.g., 32) | **$N = 32$** (shares `W` *and* past `k,v[1..t]`!) | **$1\times$ ($32 \ll 333$, free on HBM!)** | **Top-$B_d$ cumulative accuracy** $\gg p_1$ (**Medusa / EAGLE-2 / Sequoia**)! |
| **4. Oversized tree ($M = 1000$)** | $M = 1000$ | $N = 1000 > 333$ | $3\times$ (Compute-bound!) | Violates latency goal |

Because causal attention sums over `j ∈ Ancestors(i)`, every tree node `i` shares the exact same historical prefix `{1..t}` ($N = M = 32$ batched KV read!) and applies a `[32 × 32]` tree mask on the new nodes. And as batch size $B_{\text{batch}}$ grows, the free slack $B_{\text{batch}} \times M \le 333$ shrinks, automatically reducing optimal tree size $M^*$ (**Sequoia**).

---

## 5. Thought Experiment 2: Plugging In Modern (2024–2026) Architectures Reveals New Speculation Targets!

Why did the vanilla Transformer in §3 only have two exact winners (`x[t+1]` and `sel[t]`)? Because those were the only two low-cardinality choke-points in a 2020 Transformer!
When we plug modern architectures into the exact same Two-Gate table, four new speculation optimizations immediately emerge:

| Modern Architecture Template | New Bottleneck Edge `A ──(a)──► B` | Gate 1 (Window Sharing / Overlap) & Gate 2 (Cardinality) | New Speculation Optimization Discovered |
|---|---|---|---|
| **1. Block-Sparse / Native Sparse Attention (DeepSeek NSA, MoBA)**<br>`sel_blk[t] = topK_b(q[t] · K_sum[b])` | `topK_b ──(sel_blk[t])──► Load_KV_Blocks(sel_blk[t])` | • **Gate 2:** `sel_blk[t]` is a **discrete lookup index** (top-16 block IDs out of 1,000 $\implies p \approx 0.85$ from token $t-1$ or layer $l-1$)!<br>• **Gate 1:** Overlaps prefetching KV blocks from CPU/peer-HBM (`Link_BW`) with `QKV_proj` (`HBM/Compute`)! | **Exact Speculative KV-Block Prefetch + Delta Recovery:** Predict `sel_blk` before `W_Q`, prefetch the 16 KV blocks, verify against true `topK_b`, and fetch only the 1–2 missed blocks on a mismatch! |
| **2. Looped / Recurrent-Depth Reasoning Models (Huginn, Universal Transformer)**<br>`z[k+1] = Block(z[k], W_shared)` for loop step `k = 1..R` | `Block(z[k]) ──(z[k+1])──► Block(z[k+1])` | • **Gate 1:** Unlike normal layers, **`W_shared[d, m]` has NO loop index `k`!** So running loop steps `k = 1, 2, 3, 4` in the **same Window** loads `W_shared` **ONCE ($N = 4$)**!<br>• **Gate 2:** `Verify` checks fixed-point convergence $\|\hat{z}[k] - \text{Block}(\hat{z}[k-1])\| < \epsilon$ (or discrete `halt[k] ∈ {0, 1}`). | **Depth-Parallel Speculation Within a Single Token:** Guess the trajectory $\hat{z}[1..4]$ and verify 4 recurrent depth steps in 1 weight load! |
| **3. Shared + Routed Fine-Grained MoE (DeepSeek-V3)**<br>`y = Shared_Exp(a) + Σ_s Routed_Exp(a, sel[s])` | `Router(a) ──(sel)──► AllToAll_Disp(a, sel)` | • **Gate 2:** `sel` is a **discrete expert ID** ($p \approx 0.80$ when guessed from `x[t]` before `Attn`).<br>• **Gate 1:** While `Shared_Exp(a)` runs on **HBM/Compute**, `AllToAll_Disp(a_guess, sel_guess)` runs on **Link** in the same Window! | **Speculative Dispatch Under Shared-Expert Shadow:** Hide EP all-to-all dispatch inside the deterministic Shared Expert's compute window! |
| **4. Hybrid SSM + Attention with Input Caching (ReplaySSM + Spec Decode)**<br>`S[t] = A_t S[t-1] + k[t]ᵀ v[t]` | `Verify(x̂[t+1..t+k]) ──► Update(S)` | In standard SSMs, verifying $k=5$ speculative tokens overwrites state `S`, requiring expensive state rollback or $5\times$ state clones when tokens $3..5$ are rejected! | **Rollback-Free Speculative SSM Verification:** Keep only `S[t]` (never write `S[t+1..t+5]` during the verify window!); compute all $k+1$ verify outputs from `{S[t], k[t+1..t+5], v[t+1..t+5]}` in SRAM, and only advance `S` along the accepted prefix! |
