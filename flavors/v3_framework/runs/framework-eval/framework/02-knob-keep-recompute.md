# Knob 1: Keep / Recompute (Caching & State Management Across Steps)

Read [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) first.

---

## 1. What the `Keep / Recompute` Knob Is

When a request generates token `t` (say, token `1001`) after already having tokens `j = 1..1000`, the computation graph at step `1001` references tensors from past positions `j ≤ 1000`.
For **every data node produced at an earlier step**, the `Keep / Recompute` knob makes a binary choice:
- **Keep (Cache):** Store the tensor in memory (HBM or another memory tier) when it is first computed, and read it back on future steps `t`.
- **Recompute (Drop):** Do not store it across steps; if a future step needs it, recompute it from whatever upstream inputs were kept.

---

## 2. The Two Rules for `Keep / Recompute`

Before scoring any candidate keep set with hardware numbers, every candidate data node must pass two simple checks:

### Rule 1 (Usefulness — Future Readers Check)
> **Does any node at future step `t` actually read this tensor?**
> Trace backward from the output `y[t]` along dependency edges within the layer. If a past tensor at `j < t` is not in the backward slice of `y[t]`, keeping it saves zero work.

*Concrete Example (`Why we don't cache Q`):*
Look at the attention equations at step `t = 1001`:
```text
s[1001, j] = Σ_d q[1001, d] · k[j, d]      (for j = 1..1001)
a[1001, d] = Σ_j α[1001, j] · v[j, d]
```
Trace backward from `a[1001]`: it reads `q[1001]` (the current step's query), `k[1..1001]`, and `v[1..1001]`.
Notice that past query vectors **`q[1], q[2], ..., q[1000]` are never read at step 1001**! So keeping `q[j]` is useless.

### Rule 2 (Validity — Formula Must Not Depend on Future Steps `t`)
> **Can this tensor actually be computed at step `j ≤ 1000` without knowing token `t = 1001`?**
> Look at the indices inside the formula that produces the tensor. If the formula contains the future query index `t` (`1001`), the tensor did not exist at step `1000` and **cannot** be kept from step `1000`.

*Concrete Example (`Why we CAN cache k[5], but CANNOT cache attention score s[t, 5]`):*
- **Key vector `k[5] = x[5] · W_K`:** The formula only contains past position `j = 5`. At step `1000`, token 5 is already known, and `k[5]` never changes when we move to step `1001` or `1002`. **Valid to keep!**
- **Attention score `s[1001, 5] = q[1001] · k[5]`:** At step `1000`, we computed `s[1000, 5] = q[1000] · k[5]` (how much token 1000 attended to token 5). At step `1001`, we need `q[1001] · k[5]` (how much token 1001 attends to token 5). Because `q[1001]` does not exist at step `1000`, `s[1001, j]` and `α[1001, j]` **cannot be cached across steps**!

---

## 3. Worked Derivation 1: Re-Deriving the KV Cache (Without Knowing It Exists)

**Snapshot:** Request `B1` at token `t = 1001`, nothing kept yet.
 In one layer, the data nodes indexed by past position `j ≤ 1000` are:
`{ x[j], q[j], k[j], v[j], s[1001,j], α[1001,j], a[j], h[j], o[j], y[j] }`.

### Enumeration Table (1000 Past Positions, 1 Layer, `d = 4096`, FP16)

| Candidate Keep Set | Passes Rule 1 (Has Readers)? | Passes Rule 2 (Free of `t=1001`)? | HBM Read Cost | Recompute Cost (FLOPs) | Total Time | HBM Capacity Used | Verdict |
|---|---|---|---|---|---|---|---|
| `∅` (keep nothing) | — | ✓ | ~0 | Recompute all 1000 rows (`x → k,v` + upstream layers) | $\ge 70\text{ }\mu\text{s}$ | 0 | Baseline |
| `{q[j]}` (or `{a[j], h[j]}`) | **✗ (0 readers at step 1001)** | ✓ | — | Saves 0 work | $\ge 70\text{ }\mu\text{s}$ | 8 MB | Dead (Rule 1) |
| `{s[1001,j]}` or `{α[1001,j]}` | ✓ | **✗ (Contains `q[1001]`)** | — | — | — | — | **Illegal (Rule 2)** |
| `{x[j]}` | ✓ (feeds `k,v`) | ✓ | 8 MB (`2.7 µs`) | Recompute `k,v` for 1000 rows: `67 GFLOP` (`67 µs`) | `70 µs` | 8 MB | Valid ($2\times$ less memory, $13\times$ slower) |
| `{k[j]}` (or `{v[j]}`) | ✓ | ✓ | 8 MB (`2.7 µs`) | Recompute `v` for 1000 rows: `33.5 GFLOP` (`33.5 µs`) | `36 µs` | 8 MB | Valid |
| **`{k[j], v[j]}`** | **✓** | **✓** | **16 MB (`5.3 µs`)** | **0** | **`5.3 µs`** | **16 MB** | **Winner (KV Cache!)** |
| `{x[j], k[j], v[j]}` | ✓ | ✓ | 24 MB (`8 µs`) | 0 | `8 µs` | 24 MB | Dominated by `{k,v}` |

**Conditional Verdict:**
Keep `{k[j], v[j]}` wins whenever reading 2 vectors (`16 KB/token ÷ BW`) is faster than recomputing 2 projections (`67 MFLOP/token ÷ F`), provided `16 MB/request/layer` fits in HBM capacity `C`. If capacity `C` is exhausted, `{x[j]}` (combined with `Rewrite` into a compressed latent `c[j]`—see [03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md)) becomes the winning row!

---

## 4. Worked Derivation 2: Same Table, Different Model Templates $\rightarrow$ 3 More Optimizations

### 4.1 Sliding-Window Attention Eviction
- **Template change:** Causal bound is `t - W < j ≤ t` instead of `1 ≤ j ≤ t`.
- **Result from Rule 1:** For any position `j ≤ t - W`, `k[j]` and `v[j]` are outside the index bound of all future steps $\ge t$. They have **zero future readers** $\implies$ dropped automatically! No special "ring buffer" rule is needed.

### 4.2 Linear Attention / Recurrent Fixed-Size State
- **Template change:** Remove `softmax` so attention is linear:
  $$a[t] = \sum_{j \le t} \big(q[t] \cdot k[j]\big)\, v[j] = q[t] \cdot \underbrace{\left(\sum_{j \le t} k[j]^T v[j]\right)}_{S[t]}$$
- **Result from Rule 2:** In softmax attention, `q[t]` was trapped *inside* `softmax(q[t]·k[j])`, so the sum over `j` depended on `t` and could not be kept. In linear attention, `q[t]` factors *outside* the sum over `j`!
- That creates a brand-new `t`-free data node $S[t-1] = \sum_{j \le t-1} k[j]^T v[j]$ of fixed size `[d_k × d_v]` (`128 × 128` numbers, independent of context length!). In the Keep/Recompute table, keeping `{S[t-1]}` dominates keeping `{k[1..t-1], v[1..t-1]}` as soon as $t > d_k$.

### 4.3 ReplaySSM / DeltaLog (Trading State Write-Backs Against Input Caching)
- **Why bytes *written* to HBM matter:** In a recurrent SSM / Gated DeltaNet layer, updating the state $S[t] = A_t S[t-1] + k[t]^T v[t]$ at every decode step reads `S[t-1]` (`128 × 128 = 16,384` numbers) **and writes `S[t]` (`16,384` numbers) back to HBM**, whereas the step's inputs `k[t], v[t]` are only `256` numbers ($64\times$ smaller!).
- By unrolling the recurrence over $L = 8$ steps ([03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md)), a new candidate row appears in the Keep/Recompute table:
  - **Row A (Eager State Keep — In-Place Overwrite):** Keep `{S[t]}`. Every step reads `16,384` + writes `16,384` = `32,768` numbers of HBM traffic.
  - **Row B (ReplaySSM / DeltaLog — Checkpoint + Input Ring Buffer):** Keep `{S[t - L], k[t-L+1..t], v[t-L+1..t]}`. Each step reads `S[t-L]` (`16,384`) + recent inputs (`256 × L`), writes only `256` numbers (`k[t], v[t]`), and flushes `S` to HBM only once every $L = 8$ steps—cutting HBM traffic nearly in half!
  - **Row C (Token-History Keep — No State `S` Materialized):** Keep `{k[1..t], v[1..t]}` directly (`256` numbers/token) and use the Left-Bracketed Attention form `(Q Kᵀ) V` ([03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md)). Because a single `[128 × 128]` matrix `S` (`16,384` numbers read + `16,384` written, often required to stay in 16-bit BF16/FP16 for recurrent numerical stability even when `k, v` are FP8) equals **64–150+ tokens** of raw `k, v` vectors, **Row C actually beats Row A for short prefixes or short branched suffixes** ($t < d_k$)!

### 4.4 Crucial Distinction: Append-Only Keep (`k[1..t]`) vs. In-Place Overwrite Keep (`S[t]`)
Notice a structural difference when applying **Rule 1 (Future Readers)** to a recurrent state `S[t]`:
- **Append-Only Keep (`k[1..t], v[1..t]`):** Past entries `k[1..t-1]` are never modified when `k[t]` is written. If **Speculative Decoding** rejects draft tokens `t+2..t+4`, or if **16 parallel requests branch** from a shared prefix at step `t`, `k[1..t]` is still intact in memory at zero extra cost.
- **In-Place Overwrite Keep (`S[t]`):** `S[t]` only achieves $O(1)$ fixed memory capacity because step `t` **destructively overwrites `S[t-1]`** (assuming step `t+1` only reads `S[t]`).
  - **The Multi-Branch / Rollback Trap:** As soon as you introduce **Speculative Decoding** (where rejecting token `t+2` creates a future reader for `S[t+1]`!) or **Shared-Prefix Branching** (where $B$ requests all read `S_prefix` and diverge), overwriting `S` in-place either destroys needed state or forces $B\times$ expensive full-state clones.
  - **How the table solves both:** Pairing `S[t - L]` with **Row B (ReplaySSM)** makes speculative rollback free (just drop the rejected `k, v` vectors without touching `S[t - L]`), and pairing a single shared `S_prefix` with **Row C (private `k, v` suffix)** avoids cloning `S_prefix` across branches until the private suffix exceeds the crossover length!
