# Overview, Representation, and the Six Knobs

**Status:** Work in progress, checkpoint of 2026-11-13.

### File Map of This Repository
- [CLAUDE.md](file:///scratch/kevinsong/learning/framework/CLAUDE.md) — Project stage, deliverable ("enumerability/plausibility before optimality"), and communication style.
- [01-overview-and-representation.md](file:///scratch/kevinsong/learning/framework/01-overview-and-representation.md) *(this file)* — Motivation, the 3 inputs (Model, Hardware, Workload), the Cost Model, and a 1-table overview of the 6 Knobs.
- [02-knob-keep-recompute.md](file:///scratch/kevinsong/learning/framework/02-knob-keep-recompute.md) — **Knob 1 (`Keep / Recompute`):** Rules, principles, and derivations (KV cache, sliding window, linear attention state, ReplaySSM / DeltaLog).
- [03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md) — **Knob 2 (`Rewrite`):** The 4 algebraic identities, 3 hardware principles, and derivations (MLA weight absorption, RoPE vs. absorption, Punica/S-LoRA, Linear Attention dual-form, Shared-Base MoE factoring).
- [04-knob-speculate.md](file:///scratch/kevinsong/learning/framework/04-knob-speculate.md) — **Knob 3 (`Speculate`):** The unified transformation, the Two Gates (Window Gate + Cardinality Gate), why vanilla Transformers only speculate tokens & experts, Chain vs. Tree speculation, and modern-architecture targets (NSA/MoBA, Looped depth, Shared-expert shadow, Rollback-free SSM).
- [05-knob-split-and-place.md](file:///scratch/kevinsong/learning/framework/05-knob-split-and-place.md) — **Knobs 4 & 5 (`Split / Merge` & `Place`):** Index-splitting rules, TP / DP / EP / PP / Context-Parallel decode, Why EP beats TP at scale, Disaggregation, and KV offloading.
- [06-knob-window-and-combinations.md](file:///scratch/kevinsong/learning/framework/06-knob-window-and-combinations.md) — **Knob 6 (`Window`) & Multi-Knob Combinations:** Unifying batching (Merge) and multi-resource overlap, continuous/selective batching, chunked prefill, Shared-Prefix Cascade Attention, Dual-Batch Overlap (DualPipe/NanoFlow), and novel multi-knob schedules.
- [07-status-and-open-questions.md](file:///scratch/kevinsong/learning/framework/07-status-and-open-questions.md) — Coverage table, Devil's Advocate limitations, lessons learned, and open questions.
- [examples/qwen3_8-27b-2x4090/handoff-implementation.md](file:///scratch/kevinsong/learning/framework/examples/qwen3_8-27b-2x4090/handoff-implementation.md) — End-to-end case study & implementation handoff for `Qwen3.5/3.8-27B` (FP8) on 2× RTX 4090 + PCIe + DDR4.
- [examples/deepseek-v4_1-flash-nvl72/01-systematic-derivation.md](file:///scratch/kevinsong/learning/framework/examples/deepseek-v4_1-flash-nvl72/01-systematic-derivation.md) & [handoff-implementation.md](file:///scratch/kevinsong/learning/framework/examples/deepseek-v4_1-flash-nvl72/handoff-implementation.md) — Systematic framework derivation & implementation handoff for `DeepSeek-V4.1-Flash` on `NVIDIA Vera Rubin NVL72` (multi-user agentic coding).

---

## 1. Motivation

Today, LLM inference optimization advances mostly by "sparks of genius": someone notices that decode is memory-bound and proposes speculative decoding; someone else notices expert weights are read redundantly and proposes expert parallelism; someone notices network all-to-alls leave compute idle and proposes dual-batch overlap (DualPipe, NanoFlow); someone notices recurrent states are expensive to write back and proposes caching inputs instead (ReplaySSM, DeltaLog). Each new architecture (MLA, hybrid SSM/attention, fine-grained MoE, looped-depth models) restarts the process from scratch.

We want to replace that with a **systematic enumeration procedure**: given
1. **a model definition** (the computation graph and its indices),
2. **a hardware platform** (compute rate, memory bandwidth/capacity, link bandwidth), and
3. **a workload snapshot** (active requests, known tokens, and kept state),

an agent should be able to **enumerate the space of valid ways to serve the model** and compare them with back-of-the-napkin cost estimates—so that known optimizations appear naturally as rows in an enumeration table, and novel optimizations can be discovered the exact same way.

---

## 2. Core Design Principles

1. **Enumerability (plausibility) first, napkin optimality second, fine accuracy last.**
   - First ask: *Does our representation allow an agent to systematically generate this variation as a point in the design space?*
   - Second ask: *Does coarse napkin math (FLOPs, HBM bytes, link bytes, capacity) explain the condition under which this variation wins or loses?*
   - Leave kernel-level and hardware-microarchitecture constants to lower layers unless a missing detail is **structural** (an entire dimension or resource is unrepresented).
2. **Build through examples, but play devil's advocate against retrofitting.**
   - Every optimization we test is classified as **(a)** covered by existing moves, **(b)** enabled by a new *generic* primitive, or **(c)** requiring a special-case rule (a red flag).
   - For every primitive, we explicitly ask: *Where is it too strict (excluding valid hardware schedules) or too loose (allowing impossible free lunches)?*
3. **Derive, don't assume (The Systematic Enumeration & Explicit Pruning Rule).**
   - The test for any derivation: *Could an agent that has never heard of this optimization reach it by listing the candidate sites on the graph, checking validity, and comparing rows in a table?*
   - **Never silently jump to familiar candidates** (e.g., jumping straight to symmetric 50/50 TP or PP). Always sweep the full set of candidate nodes, indices, or split ratios first. If a group of cases can be safely skipped, **explicitly state the dominance proof for why that group loses**—and check what conditions break that dominance proof (which is often where asymmetric or hybrid optimizations hide!).
4. **No prefill/decode primitives.**
   - "Prefill" and "decode" are *derived labels*, not built-in modes. Without a KV cache, decode step 1000 and prefill of a 1000-token prompt are the exact same computation graph; the difference is purely the result of an earlier `Keep / Recompute` decision.
5. **Output is conditional, not absolute.**
   - The framework reports *"Design point $X$ beats baseline when `<inequality in model/hardware/workload parameters>`"*, never *"Always do $X$"*.

---

## 3. Scope and the Four-Layer Model Stack

Something belongs in this framework (the **Structure** layer) if its effect can be written as a choice on the computation/data graph (rewrite, split/merge, place, keep/recompute, window schedule) and scored from average workload statistics using `(FLOPs, HBM_bytes, Link_bytes, Capacity)`.

| Layer | Question It Answers | Provides to Others | Consumes from Others |
|---|---|---|---|
| **Structure** *(this framework)* | Which ways of computing, caching, splitting, grouping, placing, and overlapping are in the design space, and when do they win? | Per-request costs, design-point trade-offs, required accuracy thresholds ("wins if $p \ge 0.75$") | Efficiency factors, empirical acceptance/routing stats |
| **Policy / Scheduling** | Who to admit, evict, preempt, or route dynamically over time | Concurrency distributions, queue delays | Cost curves from Structure |
| **Implementation** | How to build kernels, allocators, page tables, SM warps | Achieved hardware efficiency | Operator shapes and co-scheduled window targets |
| **ML** | Drafter training, quantization, sparse patterns, router correlation | Acceptance rate $p$, accuracy loss | Target `(cost, p)` requirements from Structure |

---

## 4. Input 1: The Model Template (Scalar Index Notation)

Block diagrams ("Attention → MoE") hide which dimensions are shared, which are reduced over, and which depend on the current token position. We write every equation **per scalar index**, with every reduction written explicitly.

| Index | Meaning | Behavior |
|---|---|---|
| `r` | Request | Independent across requests (unless prefix-deduped) |
| `l` | Layer | Sequential within one request (`l → l+1`) |
| `i`, `j` | Token positions (`j ∈ Ancestors(i)`, which is `j ≤ i` on a chain) | `i` is the query position; `j` ranges over causal ancestors (past + current context) |
| `t` | Current generation step (a specific value of `i`) | Connected to `t+1` by the serial sampling edge |
| `d`, `d'`, `m` | Hidden / feature dimensions | Use distinct names per tensor axis |
| `h`, `g` | Query head, KV head group | `g` is shared across multiple `h` in GQA |
| `e`, `s` | Expert (`1..E`), top-$K$ slot (`1..K`) | `sel[r,i,s] ∈ 1..E` is a data-dependent **lookup index** |

### Running Model Template (1 Layer Shown, Attention + Top-2 MoE)
```text
q[r,i,d]     = Σ_d' x[r,i,d'] W_Q[d',d]              (and k, v with W_K, W_V)
s[r,i,j]     = Σ_d  q[r,i,d] k[r,j,d]                for j ∈ Ancestors(i) (causal bound)
α[r,i,j]     = softmax_j s[r,i,j]                    reduction over j; combine op = softmax-merge
a[r,i,d]     = Σ_j  α[r,i,j] v[r,j,d]
g[r,i,e]     = Σ_d  a[r,i,d] W_g[d,e]                router logits
sel[r,i,s]   = top2_e g[r,i,e]                       LOOKUP INDEX (value computed from data)
w[r,i,s]     = softmax_s g[r,i,sel[r,i,s]]
h[r,i,s,m]   = ReLU( Σ_d a[r,i,d] W_1[sel[r,i,s],d,m] )
o[r,i,s,d]   = Σ_m  h[r,i,s,m] W_2[sel[r,i,s],m,d]
y[r,i,d]     = Σ_s  w[r,i,s] o[r,i,s,d]
x[r,l+1,i,·] = y[r,l,i,·]                            layer transition edge
x[r,1,i+1,·] = Embed(Sample(y[r,L,i,·]))             THE SERIAL EDGE (step t → t+1)
```

### How Indices Immediately Reveal Sharing (The "Missing-Index Rule")
- **Dense weights `W_Q[l, d', d]`, `W_MLP[l, d, m]` have NO `r`, `i`, or `t` index.** Every token `(r, i, t)` processed together at layer `l` reuses the exact same weight tensor loaded from HBM. Conversely, because `W` *does* carry `l`, Layer 1 and Layer 2 do *not* share weights (unless the model is weight-tied / looped across depth).
- **KV cache tensors `k[r, l, j, d]`, `v[r, l, j, d]` HAVE `r` and `j`.** Request `r=1` never reads request `r=2`'s KV cache (unless deduplicated on a shared prompt prefix).
- **Expert weights `W_1[l, e, d, m]` carry `e = sel[r, i, s]`.** Two tokens `(r, i)` only share an expert weight load if their runtime router lookup hits the same expert `e`.

---

## 5. Input 2: The Hardware Model and the "333-Row Rule"

We model each device as a coarse vector of three resource rates plus one capacity limit:
- **Compute rate `F`** (running example: `1000 TFLOP/s` in FP16)
- **HBM bandwidth `BW`** (running example: `3 TB/s` — counts both bytes read and bytes written)
- **Inter-device link bandwidth `Link_BW`** (running example: `600 GB/s`)
- **HBM capacity `C`** (bounds total bytes kept resident across requests)
- **Memory tiers:** Local HBM, Peer GPU HBM (over `Link_BW`), CPU DRAM (over PCIe).

### The Balance Point ("Ridge Point") and Reuse Count $N$
To know whether a matrix multiplication $[N \times K] \times [K \times M]$ is **compute-bound** or **memory-bandwidth-bound**, count how many tokens **$N$** reuse each weight element ($K \times M$ matrix, $b_W$ bytes per weight element, e.g., $b_W = 2$ for FP16 or $b_W = 1$ for FP8) loaded from HBM:
- **Total FLOPs:** $2 \cdot N \cdot K \cdot M$ (1 multiply + 1 add per element)
- **Weight Bytes Read:** $b_W \cdot K \cdot M$ bytes (dominates token reads/writes when $K, M \gg N$)
- **Arithmetic Intensity (FLOPs per byte):**
  $$\frac{2 \cdot N \cdot K \cdot M\text{ FLOPs}}{b_W \cdot K \cdot M\text{ Bytes}} = \frac{2N}{b_W}\text{ FLOPs/byte} \quad (\text{equals } N\text{ FLOPs/byte in FP16 where } b_W = 2)$$
- Equating arithmetic intensity $\frac{2N}{b_W}$ to the hardware ridge point $\frac{F(\text{dtype})}{BW}$ gives the universal threshold $N_{\text{ridge}} = \frac{b_W \cdot F(\text{dtype})}{2 \cdot BW}$. On our running FP16 example (`F = 1000 TFLOP/s`, `BW = 3 TB/s`, $b_W = 2$):
  $$N_{\text{ridge}} = \frac{2 \cdot 1000\text{ TFLOP/s}}{2 \cdot 3\text{ TB/s}} \approx 333\text{ tokens per weight load}$$
  *(Note: On an RTX 4090 with FP8 weights where $b_W = 1$, $F_{\text{FP8}} = 660\text{ TFLOP/s}$, and $BW = 1\text{ TB/s}$, $N_{\text{ridge}} = \frac{1 \cdot 660}{2 \cdot 1} = 330$ tokens—nearly the exact same threshold!)*

This gives a universal 1-line check for any operator:
- **$N > 333$ tokens reusing the loaded data $\implies$ Compute-bound (`FLOPs / F > Bytes / BW`).** Compute units are 100% busy; HBM bandwidth has spare capacity.
- **$N < 333$ tokens reusing the loaded data $\implies$ HBM-bandwidth-bound (`Bytes / BW > FLOPs / F`).** HBM bandwidth is 100% busy; compute units sit idle.
  - *Example:* In single-token decode attention, a request's historical KV cache `k[r, 1..t, d]` is only multiplied by **$N = 1$** query token `q[r, t, d]`. Since $N = 1 \ll 333$, scanning a long KV cache in decode is **333× memory-bandwidth-bound**!

---

## 6. Input 3: The Workload Snapshot

A computation graph is instantiated from **(Model Template + Workload Snapshot)**.
A snapshot is a concrete list of active requests, each described by:
1. **Known tokens so far:** `x[r, 1..n_r]` (prompt tokens plus any tokens already generated).
2. **Data nodes currently kept in memory** from earlier steps: e.g., nothing (`∅`), or `k,v[r, 1..n_r-1]`, or MLA latents `c[r, 1..n_r-1]`, or an SSM state `S[r]`.
3. **Workload Goal:** e.g., maximize throughput subject to per-token latency $\le 2\times$ solo decode latency.

---

## 7. Summary of the Six Knobs and the Window Cost Model

| Knob | Dedicated File | What Site It Acts On | What It Does |
|---|---|---|---|
| **1. Keep / Recompute** | [02-knob-keep-recompute.md](file:///scratch/kevinsong/learning/framework/02-knob-keep-recompute.md) | Data node across steps `t` | Save a `t`-free tensor in memory across steps vs. drop it and recompute from upstream inputs. |
| **2. Rewrite** | [03-knob-rewrite.md](file:///scratch/kevinsong/learning/framework/03-knob-rewrite.md) | Algebraic expression | Apply Associativity `(AB)C = A(BC)`, Distributivity `A(B+C) = AB+AC`, Recurrence Unrolling, or Subgraph Dedup ($p = 1$, exact over $\mathbb{R}$). |
| **3. Speculate** | [04-knob-speculate.md](file:///scratch/kevinsong/learning/framework/04-knob-speculate.md) | Dependency edge `A ──(a)──► B` | Insert `Guess(x) → â`, run `[Verify(x, â) \|\| B(â)]` in the same Window, and run `Recover` on failure (rate $1-p$). |
| **4. Split / Merge** | [05-knob-split-and-place.md](file:///scratch/kevinsong/learning/framework/05-knob-split-and-place.md) | Index (`r, i, j, m, d, e`) | **Split** an index range into pieces (inserting a combine node if the index is reduced over), or **Merge** nodes that differ only in an index missing from a shared input. |
| **5. Place** | [05-knob-split-and-place.md](file:///scratch/kevinsong/learning/framework/05-knob-split-and-place.md) | Compute / Data node | Assign to a device and memory tier (inserting a transfer node on crossing edges). |
| **6. Window** | [06-knob-window-and-combinations.md](file:///scratch/kevinsong/learning/framework/06-knob-window-and-combinations.md) | Independent nodes on a device | Assign nodes to discrete sequential **Execution Windows** $W_1, W_2, \dots$ (`A → B` $\implies W(A) < W(B)$; shared reads in the same window counted once). |

### The Unified Window Cost Formula
Inside each Execution Window $W$ on a device, all nodes are mutually independent (`A ⊥ B`), shared tensors read by multiple nodes in $W$ are counted **once**, and resources `(F, BW, Link_BW)` run concurrently:
$$\text{Time}(W) = \max\!\left(\frac{\sum_{u \in W} \text{FLOPs}(u)}{F},\; \frac{\sum_{u \in W} \text{HBM\_bytes}(u)}{BW},\; \frac{\sum_{u \in W} \text{Link\_bytes}(u)}{Link\_BW}\right), \qquad \text{Total\_Time} = \sum_W \text{Time}(W)$$
subject to resident bytes $\le C$.
