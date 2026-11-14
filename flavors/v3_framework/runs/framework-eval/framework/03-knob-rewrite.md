# Knob 2: Rewrite (Deterministic Algebraic Transformations)

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) first.

---

## 1. What the `Rewrite` Knob Is

A **`Rewrite`** is a deterministic ($p = 1$) mathematical identity over real numbers $\mathbb{R}$ (ignoring floating-point rounding order) that changes the equations in the computation graph while preserving the exact mathematical output.

In our scalar index notation, every `Rewrite` is one of **four elementary rules**:
1. **Associativity (Swap reduction order):** `(A · B) · C ↔ A · (B · C)` — changes which two tensors multiply first, and therefore changes the shape and indices of the intermediate tensor.
2. **Distributivity (Factor out / expand across a sum):** `A · B + A · C ↔ A · (B + C)` — pushes a matrix multiplication outside (or inside) a sum over requests `r`, expert slots `s`, or feature partitions `d`.
3. **Recurrence Unrolling:** Substitutes `S[t-1]` into `S[t]` across $L$ steps before materializing `S[t]` to HBM.
4. **Subgraph Deduplication:** Collapses two compute/data nodes that have identical inputs and formulas into a single shared node.

---

## 2. The Three Hardware Principles for When a `Rewrite` Wins

Changing an equation algebraically only speeds up hardware execution if it does at least one of three things:

| Principle | What to Look For in the Equations | Why It Speeds Up Hardware |
|---|---|---|
| **Principle 1: Drop a Blocking Index to Unlock Sharing**<br>*(Use Distributivity or Dedup to remove `r`, `s`, or `j` from a big tensor)* | A large weight matrix `W` is trapped inside a per-request (`r`) or per-slot (`s`) sum, giving it reuse $N = 1$. | Factoring `W` outside the sum drops the blocking index from the weight multiplication, raising reuse from $N = 1$ to $N = B$ (or cutting matmuls by $K\times$)! |
| **Principle 2: Re-Bracket Across an Index Size Asymmetry**<br>*(Use Associativity `(AB)C ↔ A(BC)` when one dimension is much smaller than another)* | In a chain of two matmuls `[M × D] × [D × J]`, one bracketing forms an `[M × J]` intermediate while the other forms a `[D × D]` intermediate. | Moves the heavy projection off the large index ($J = 1000$ past tokens) and onto the small index ($M = 1$ current token), or flips at the crossover $M = D$! |
| **Principle 3: Isolate or Commute Past a Position-Dependent Blocker**<br>*(When a `t,j`-dependent matrix sits between two tensors you want to associate)* | In `q[t] · R[t-j] · W_ukᵀ · c[j]ᵀ`, the RoPE rotation `R[t-j]` depends on both `t` and `j`, blocking `q[t]` and `W_ukᵀ` from associating. | Splitting `d` via Distributivity (`d_nope ⊕ d_rope`) or commuting `R[j]` past `W_uk` lets `W_ukᵀ` touch `q[t]` and absorb! |

---

## 3. Worked Derivations Under Principle 1 (Drop a Blocking Index)

### 3.1 Multi-Tenant LoRA Batching (Punica / S-LoRA)
**High-level intuition:** Suppose 16 requests (`r = 1..16`) in a batch each use a *different* rank-16 LoRA adapter `A[r, d, r_lora] × B[r, r_lora, m]` on top of a shared `4096 × 4096` base weight `W_base[d, m]`.

| Bracketing / Distribution of `y[r]` | Indices on the Big `[4096 × 4096]` Matrix | Reuse Count $N$ on the Big Matrix | Verdict |
|---|---|---|---|
| **Option 1 (Merge LoRA into weights):**<br>`y[r] = x[r] · (W_base + A[r] B[r]) = x[r] · W_eff[r]` | `W_eff[r, d, m]` **carries index `r`!** (16 different `32 MB` matrices) | **$N = 1$** (reads `16 × 32 MB = 512 MB` from HBM!) | Terrible for multi-tenant serving |
| **Option 2 (Distributivity + Associativity):**<br>`y[r] = (x[r] · W_base) + (x[r] · A[r]) · B[r]` | `W_base[d, m]` **has NO `r` index!** | **$N = 16$** on `W_base` (reads `32 MB` **once**!), plus tiny rank-16 `A[r], B[r]` | **Punica / S-LoRA!** ($16\times$ less HBM weight traffic) |

### 3.2 Cross-Slot Factoring in Shared-Base / LoRA-MoE (Candidate Novel)
**High-level intuition:** In a fine-grained MoE where each token activates $K = 8$ experts (`s = 1..8`) and each expert's down-projection is a shared base matrix plus a low-rank expert delta (`W_2[e] = W_base + U[e] V[e]`):
- **Naive evaluation:** Multiplies each of the $K = 8$ slot activations `h[s]` by `W_2[sel[s]]` separately ($8\times$ work).
- **Apply Distributivity across `Σ_s`:**
  $$\sum_{s=1}^{K} w[s]\, h[s]\, \big(W_{\text{base}} + U[\text{sel}[s]] V[\text{sel}[s]]\big) = \underbrace{\left(\sum_{s=1}^{K} w[s]\, h[s]\right)}_{\text{Sum } K=8 \text{ slots FIRST!}} W_{\text{base}} + \sum_{s=1}^{K} w[s]\big(h[s] U[\text{sel}[s]]\big) V[\text{sel}[s]]$$
- **Result:** Pushing `Σ_s` inside `W_base` collapses $K = 8$ matmuls into **1 matmul**, and drops the lookup index `sel[s]` from `W_base` so `W_base` achieves full batch reuse $N = B$!

---

## 4. Worked Derivations Under Principle 2 (Re-Bracket Across Size Asymmetry)

### 4.1 MLA Weight Absorption (`1 Query Token ≪ J Past Tokens`)
**High-level intuition:** In Multi-head Latent Attention (MLA), past keys are generated from a compressed 512-dim latent `c[j]` via `k[j] = c[j] · W_uk`.
If we keep `{c[j]}` in the KV cache (`1 MB` instead of `16 MB` for 1000 tokens), look at the two ways to bracket the attention score `s[t, j]` at decode step `t`:

| Bracketing of `s[t, j]` | Which Index `W_uk` Multiplies Over | FLOPs at Decode Step `t` ($J = 1000$ past tokens) | Time (HBM + Compute) |
|---|---|---|---|
| **Left Bracketing (Recompute `k[j]`):**<br>`s[t, j] = q[t] · (c[j] · W_uk)ᵀ` | Multiplies `W_uk` across **$J = 1000$ past tokens `j`**! | `8.4 GFLOP` (`8.4 µs` compute!) | `0.3 µs (read c) + 8.4 µs = 8.7 µs` (slower than standard `{k,v}` cache at `5.3 µs`!) |
| **Right Bracketing (Weight Absorption):**<br>`s[t, j] = (q[t] · W_ukᵀ) · c[j]ᵀ` | Multiplies `W_ukᵀ` across **only the 1 current query token `t`**! | `0.008 GFLOP` (`0.008 µs`!) | **`0.36 µs` ($15\times$ faster than `{k,v}` cache and $16\times$ less memory!)** |

### 4.2 Linear Attention / SSM Dual-Form (`(QKᵀ)V` vs. `Q(KᵀV)` at Crossover $M = d$)
**High-level intuition:** When processing a chunk of $M$ tokens in a Linear Attention / Gated DeltaNet / Mamba-2 layer (with state dimension $d = 128$), associativity gives two exact ways to compute $a = (Q K^T) V$:

| Bracketing of `Q Kᵀ V` for $M$ Tokens | Intermediate Tensor Shape | Total FLOPs | Winning Regime |
|---|---|---|---|
| **Left Bracketing: `(Q Kᵀ) · V`** *(Attention Form)* | `[M × M]` | $4 M^2 d$ | **Wins when $M < d$ ($M < 128$)** — e.g., during **Speculative Verification** of $M = 16\text{ or }32$ tokens! ($8\times$ fewer FLOPs than state form, and avoids materializing the `[128 × 128]` state increment for unverified tokens!) |
| **Right Bracketing: `Q · (Kᵀ V)`** *(State Form)* | `[d × d]` (`128 × 128`) | $4 M d^2$ | **Wins when $M > d$ ($M > 128$)** — e.g., during **Chunked Prefill** of $M = 512$ prompt tokens! |

---

## 5. Worked Derivation Under Principle 3 (Isolate or Commute Past a RoPE Blocker)

**High-level intuition:** Why couldn't we just apply MLA weight absorption (§4.1) directly to a standard Transformer with **RoPE (Rotary Position Embedding)**?
Look at the equation when RoPE rotation matrices `R[t]` and `R[j]` are included:
$$s[t, j] = \big(q[t]\, R[t]\big) \cdot \big(c[j]\, W_{uk}\, R[j]\big)^T = q[t] \cdot \underbrace{R[t - j]}_{\text{Depends on } t \text{ AND } j!} \cdot W_{uk}^T \cdot c[j]^T$$
Because `R[t-j]` sits **between** `q[t]` and `W_ukᵀ` and depends on `j`, it blocks `(q[t] · W_ukᵀ)` from associating!

Enumerating algebraic ways around `R[t-j]` yields both DeepSeek's solution and a clean architectural alternative:

| Rewrite Option Around `R[t-j]` | Algebraic Move | What Is Kept in Cache | Result |
|---|---|---|---|
| **Option 1: Split `d = d_nope ⊕ d_rope` via Distributivity** | Set $R = I$ on `d_nope` (so `W_uk_nope` absorbs into `q_nope[t]`), and keep a tiny separate `k_rope[j]` (`64`-dim) for the RoPE part. | `c[j]` (512-dim) + `k_rope[j]` (64-dim) | **DeepSeek-V2/V3 Decoupled RoPE!** |
| **Option 2: Commute `R[j]` to the Left of `W_uk` (`c[j] R[j] W_uk`)** | Either apply RoPE in the latent space (`c[j] R_c[j] W_uk`) or constrain `W_uk` on RoPE dims to $2\times 2$ complex-linear blocks $\begin{pmatrix}a & -b\\ b & a\end{pmatrix}$ so `W_uk R[j] = R[j] W_uk`! | **Only `c[j]` (512-dim)** (`k_rope` eliminated!) | **`W_uk` is now adjacent to `q[t] R[t]` and absorbs 100%!** |
