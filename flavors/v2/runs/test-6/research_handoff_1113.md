# Implementation Handoff: Candidate Optimizations for Qwen3.5/3.8-27B (FP8) on 2× RTX 4090 + CPU DDR4

## 1. Target System & Baseline Facts

### Model Architecture (`Qwen3.5/3.8-27B`, Text-Only Backbone, FP8 Weights & KV)
- **Layers ($L = 64$):** 3:1 Hybrid stack repeating `Linear → Linear → Linear → Full` 16 times:
  - **48 Linear Attention layers (`Gated DeltaNet`):** $H_k = 16$ query/key heads, $H_v = 48$ value heads (3 value heads per key head), head dim $d_k = d_v = 128$, local conv width $k_{\text{conv}} = 4$, output gate.
  - **16 Full Attention layers (`GQA`):** $H_q = 40$ query heads, $G_{kv} = 8$ KV heads (5 query heads per KV head), head dim $d_{\text{full}} = 256$, `attn_output_gate = true`.
- **FFN (all 64 layers):** SwiGLU with hidden size $D = 5120$, intermediate size $M = 17408$.
- **Vocabulary (`LM_Head`):** $V = 248,320 \times 5120$ ($\approx 1.27\text{ GB}$ in FP8).
- **Footprints:**
  - **Total Model Weights (FP8, 1 B/param):** **~27.0 GB** ($\approx 412\text{ MB/layer}$: `145 MB` attn + `267 MB` FFN, plus `1.27 GB` LM head). Does *not* fit on 1× 4090 (24 GB); fits across 2× 4090 (48 GB total, leaving **~21 GB free VRAM** for states and activations).
  - **Per-Request Full-Attention KV Cache (16 layers, FP8):** $16 \times 2 \times 8 \times 256 \times 1\text{ B} = \mathbf{64\text{ KB/token}}$ (`0.0625 MB/token`, e.g., `2.0 GB` at 32K context, `8.0 GB` at 128K context).
  - **Per-Request Linear-Attention Recurrent State `S` (48 layers, 16-bit BF16/FP16 required for numerical stability):** $48 \times 48 \times 128 \times 128 \times 2\text{ B} = \mathbf{75.5\text{ MB fixed per request}}$ (independent of context length), plus tiny `conv_state` (`~1.4 MB`).

### Hardware Platform
- **GPUs:** 2× NVIDIA RTX 4090 (`24 GB` GDDR6X each = `48 GB` total; `1,008 GB/s` HBM bandwidth per GPU; `660 TFLOP/s` FP8 / `330 TFLOP/s` FP16 Tensor Core compute per GPU).
  - **GPU Balance Point (Ridge Point):** $\sim 330$ tokens per FP8 weight load ($N < 330$ is memory-bandwidth-bound; $N > 330$ is compute-bound).
- **Interconnect:** PCIe Gen 4 x16 (**~25 GB/s** practical unidirectional per GPU, **~50 GB/s** aggregate across both GPUs; no NVLink; per-collective P2P latency $\alpha \approx 10\text{–}15\text{ }\mu\text{s}$).
- **Host CPU & Memory:** Infinite DDR4 capacity, **~50–80 GB/s** aggregate host memory bandwidth.

---

## 2. Candidate Optimizations to Implement & Benchmark

Each optimization below is independent and targets a specific bottleneck regime. They can be implemented standalone or composed.

---

### Optimization 1: Multi-Step Input Replay ("DeltaLog Keep") for the 48 Gated DeltaNet States
**Target Bottleneck:** Decode HBM bandwidth at short-to-medium context ($L_{\text{ctx}} \le 8\text{K}$) and state rollback overhead during speculative decoding.

#### What It Does (High-Level)
In standard recurrent decoding, every token step $t$ reads the 16-bit state matrix `S[t-1]` (`75.5 MB/req` across 48 layers) from GPU HBM, updates it, and **writes the new `S[t]` (`75.5 MB/req`) back to GPU HBM**—costing **`151 MB/req` of HBM traffic per token** (which is *more* HBM traffic than all 16 Full-Attention KV caches combined whenever $L_{\text{ctx}} < 2,400$ tokens!).

Instead of writing `S[t]` back to HBM on every decode step:
1. Keep a frozen checkpoint `S[t - W]` (`75.5 MB`) in HBM and append only the compact post-conv vectors `{k[i], v[i], α[i], β[i]}` (`0.49 MB/token` across 48 layers) to a tiny ring buffer of length $W$ (e.g., $W = 8\text{ to }16$ steps).
2. At decode step $t$, load `S[t - W]` (`75.5 MB`) + the last $\le W$ input vectors (`~4 MB`) into on-chip SRAM, replay the $\le W$ rank-1 delta updates inside registers to compute `o_lin[t]`, and **skip writing `S[t]` back to HBM**.
3. Flush/advance the HBM checkpoint `S[t - W] → S[t]` only once every $W$ steps.

#### Why We Expect It to Work
- On RTX 4090, single-token decode is $330\times$ memory-bandwidth-bound (Tensor Cores are >99% idle). Replaying $W = 8$ rank-1 outer products in SRAM uses idle compute while eliminating $7/8$ of the `75.5 MB` HBM state writebacks, cutting Linear-Attention HBM traffic by **~43%** (`151 MB/step → ~86 MB/step` per request).
- **Free Bonus for Speculative Decoding:** Because `S[t - W]` is never overwritten in-place during the window, rejecting speculative draft tokens requires **zero state backup copies and zero rollback cost**—simply discard the rejected `k, v` entries in the ring buffer.

#### Key Assumptions & Implementation Notes
- Assumes fused custom kernel (Triton/CUDA) for Gated DeltaNet decode that accepts `(S_checkpoint, k_ring[1..w], v_ring[1..w], alpha_ring[1..w], beta_ring[1..w])` and only writes `S_out` when `flush=True`.
- Sweep $W \in \{4, 8, 16\}$ to verify SRAM register pressure and latency crossover.

---

### Optimization 2: Dual-Form Unrolled Tree Speculation for Gated DeltaNet
**Target Bottleneck:** Single-request / low-batch decode latency ($B = 1\text{–}4$), where GPU weight reads (`13.5 ms` per pass) are heavily memory-bound ($N = 1 \ll 330$).

#### What It Does (High-Level)
Standard tree-based speculative decoding (e.g., EAGLE-2 / Medusa / Sequoia verifying a tree of $M = 32$ candidate tokens in one forward pass) breaks down on Gated DeltaNet if implemented recurrently: branching the state `S` (`75.5 MB`) across a 32-node tree requires cloning/reading hundreds of megabytes of `S` matrices at $N=1$ reuse and rolling them back on rejection.

Instead, during the **Verification Pass of $M$ tree tokens** ($M \approx 16\text{–}64 < d_k = 128$), switch the 48 Gated DeltaNet layers from the recurrent state form to a **Hybrid Dual Form**:
1. **Cross-History Term (Batched State Read):** Multiply all $M$ tree queries `Q_tree` (`[M × 128]`) against the single frozen root state `S[t]` (`[128 × 128]`) in **one batched matmul** (`N = M = 32` reuse on `S[t]`, reading `75.5 MB` from HBM only **once**!).
2. **Intra-Tree Term (Left-Bracketed `[M × M]` Attention Form in SRAM):** Do **not** materialize any intermediate `[128 × 128]` states `S[t+1..t+M]` for the tree nodes. Because $M = 32 < d_k = 128$, compute the intra-tree delta interactions using the `[M × M]` tree-masked parallel chunk form directly in SRAM (`8×` fewer FLOPs than state updates and zero HBM state writes).
3. **Post-Verify State Advance:** After token verification selects the winning path of length $a \le 6$, advance `S[t] → S[t+a]` along **only the accepted path** (or append the $a$ accepted `k,v` vectors to Optimization 1's ring buffer).

#### Why We Expect It to Work
- Verifying $M = 32$ tokens in one pass reuses the `27 GB` FP8 model weights 32 times ($N = 32 \ll 330$, so weight load time stays flat at `~13.5 ms`).
- Using the dual form keeps Gated DeltaNet state HBM traffic at **1× `75.5 MB` read + 0 rejected writes** regardless of tree branching factor.

#### Key Assumptions & Implementation Notes
- Requires deriving the exact `[M × M]` tree-masked intra-chunk transition matrix for Gated DeltaNet's householder-like update $S_i = \alpha_i S_{i-1} (I - \beta_i k_i k_i^T) + \beta_i k_i v_i^T$ (the WY representation used in FlashLinearAttention chunked prefill, masked by the `[M × M]` ancestor matrix `j ∈ Ancestors(i)`).
- Pack the 3 value heads per key head ($H_v = 48, H_k = 16 \implies 3:1$) into a contiguous `[128 × 384]` slice per key head so each `q[h_k]`, `k[h_k]` is loaded once across its 3 value heads.

---

### Optimization 3: 3:1 Strided PCIe KV Offloading (Pin `S` in VRAM, Stream Full-Attention `KV` from CPU DDR4 Across 4-Layer Blocks)
**Target Bottleneck:** Long-context ($L_{\text{ctx}} = 32\text{K–}256\text{K}$) or medium-batch workloads where total state exceeds the `21 GB` of free GPU VRAM and spills to CPU DDR4.

#### What It Does (High-Level)
When offloading state to CPU DDR4 over PCIe Gen 4 (`~25 GB/s` per GPU = `~50 GB/s` aggregate across both GPUs):
1. **Asymmetric State Placement by Layer Type:** Always pin the 48 Linear Attention states `S` (`75.5 MB/req`, fixed size) in GPU VRAM (21 GB holds up to ~200 requests' worth of `S`), and offload **only the 16 Full-Attention KV caches (`64 KB/token`)** to CPU DDR4.
2. **4-Layer Strided PCIe Prefetch Window:** Because only 1 out of every 4 layers is a Full-Attention layer (`l = 4, 8, 12, ..., 64`), do not fetch Layer $4k$'s KV cache on-demand at Layer $4k$. Instead, stream Layer $4k$'s KV cache from CPU DDR4 into a double-buffered GPU staging buffer **concurrently while the GPUs execute Layers $4k-3, 4k-2, 4k-1$ (the 3 preceding Linear Attention layers) and Layer $4k-4$'s FFN**.

#### Why We Expect It to Work
- Executing one 4-layer block (`3 Linear + 1 Full = 1.65 GB` of FP8 weights) takes at least **`0.82 ms` per 4-layer block** (`13.1 ms` across all 16 blocks) just to read weights from GPU HBM.
- During that `13.1 ms`, both GPUs' PCIe Gen 4 links (`~50 GB/s` total, backed by `50–80 GB/s` DDR4) can stream up to **`655 MB` of Full-Attention KV cache per decode step completely hidden behind GPU weight reads**.
- Because Qwen3.8-27B only has 16 Full-Attention layers (`64 KB/token` total), `655 MB` corresponds to **~10,500 tokens of offloaded KV context per step at zero latency penalty** ($4\times$ more hidden context than a standard 64-layer full-attention model!).

#### Variations to Test (`Data-Only Stream` vs. `CPU-Side Full-Attention Compute`)
- **Variation 3A (Single-Batch / Low-Batch, $B = 1\text{–}2$):** Stream `KV` over PCIe into GPU double-buffer during the preceding 3 Linear layers; run `Full_Attn` on GPU. (Works even for $B=1$ because historical `KV[1..t-1]` does not depend on Layers $4k-3..4k-1$).
- **Variation 3B (Multi-Batch, $B \ge 4$):** Split the active batch into two micro-batches `R1` and `R2`. Keep `KV` in CPU DDR4 and **execute `Full_Attn` on the Host CPU (using AVX-512 + `60 GB/s` DDR4)** for micro-batch `R2` while the GPUs execute the 3 Linear layers + FFNs for micro-batch `R1`. Only send the tiny query `q` (`20 KB/req`) and output `o_full` (`20 KB/req`) over PCIe—reducing PCIe traffic by **$200\times$** and utilizing CPU DDR4 bandwidth (`60 GB/s`) as a third parallel memory bus alongside the two GPUs!

---

### Optimization 4: 3-Layer Head-Start Speculative KV-Page Prefetching (for 64K–256K Ultra-Long Context in DDR4)
**Target Bottleneck:** Ultra-long context ($L_{\text{ctx}} = 64\text{K–}256\text{K}$, where a single request's Full-Attention KV cache is `4–16 GB`), exceeding the `655 MB/step` free PCIe prefetch budget of Optimization 3A.

#### What It Does (High-Level)
Combine block-sparse KV page selection (e.g., Quest / page min-max bounding boxes) with Qwen3.8-27B's 3:1 hybrid layer spacing:
1. Keep compact per-page key summaries (e.g., min/max vectors for each 16-token KV page—only `~1/8`th the size of full KV) pinned in GPU VRAM for the 16 Full-Attention layers, while full KV pages live in CPU DDR4.
2. **Predict 3 Layers Early (`l = 4k + 1`):** Instead of waiting until Layer $4k+4$ to compute `q[l=4k+4]` and discover which DDR4 KV pages are needed (which would stall the GPU waiting on PCIe), use the hidden state exiting Layer $4k+1$ (`x[l=4k+1] · W_Q[l=4k+4]`, or Layer $4k$'s query) to score the page summaries and **guess the top-$K$ KV pages** (`e.g., top 4K–8K tokens`) **before Layers $4k+2$ and $4k+3$ run**.
3. **Overlap Prefetch with Layers $4k+2$ and $4k+3$:** Stream those guessed KV pages over PCIe while the GPU executes the remaining Linear layers.
4. **Verify / Optional Delta Fetch at Layer $4k+4$:** When true `q[l=4k+4]` is computed, score the GPU-resident page summaries again. Either run approximate top-$K$ attention directly, or fetch only the small delta of missed high-scoring pages (`1 - p ≈ 5–10%`) from DDR4.

#### Why We Expect It to Work
- Residual stream representations change smoothly across consecutive layers (`cosine similarity > 0.85` across a 3-layer span), making early top-$K$ page recall high ($p \approx 0.90+$).
- Turns an impossible `4–16 GB` PCIe stall (`80–320 ms/token`) into a `< 650 MB` PCIe transfer completely hidden inside the `0.82 ms` Linear-layer execution windows.

---

### Optimization 5: Shared-Prefix Lazy State Materialization (Keep 1 Shared `S_prefix` + Private `k,v` Suffix Across Branched Requests)
**Target Bottleneck:** Multi-branch / Best-of-$N$ / Agent fan-out workloads where $B$ parallel requests ($B = 8\text{–}32$) share a system prompt or prefix of length $L_{\text{prefix}}$ and generate private suffixes of length $L_{\text{suffix}}$.

#### What It Does (High-Level)
When $B = 16$ requests branch from a shared prefix, standard engines clone the `75.5 MB` Gated DeltaNet state `S_prefix` 16 times (`1.21 GB` VRAM, `2.42 GB/step` HBM read+write traffic).
However, a single token's raw `{k, v, α, β}` in the 48 Linear layers is only **`0.49 MB`**, meaning `S` (`75.5 MB` read + `75.5 MB` write = `151 MB`) is larger than the raw `{k, v}` history for any suffix shorter than **`~150–300` tokens**!

Instead of cloning `S_prefix` on fork:
1. Keep **one shared, read-only `S_prefix` (`75.5 MB`)** in HBM across all $B$ branched requests.
2. For each request's private generated suffix tokens ($L_{\text{suffix}} < 150$), do **not** materialize a private `[128 × 128]` state `S[r]`. Instead, store the private suffix tokens as an append-only `{k, v, α, β}` buffer (`0.49 MB/token`, zero state writeback!).
3. During decode, compute all $B$ requests' historical contribution from `S_prefix` via **one batched matmul** (`Q_decayed[B × 128] × S_prefix[128 × 128]`, reading `75.5 MB` **once** at reuse $N = B$), and add the intra-suffix contribution by scanning each request's short `{k, v}` buffer.
4. Only materialize a private `S[r]` if a branch's private suffix grows past the crossover threshold ($L_{\text{suffix}}^* \approx 150$ tokens).

#### Why We Expect It to Work
- Cuts Linear-Attention state memory and HBM bandwidth by **$5\times\text{–}15\times$** during the first 150 generated tokens of branched requests or beam/tree search.

#### Key Assumptions & Implementation Notes
- Requires tracking the cumulative householder/decay transition from the end of the prefix so `q[r, t]` can be projected against the frozen `S_prefix` (using the associative chunk decomposition of Gated DeltaNet).

---

### Optimization 6: Minimum Chunk-Size Enforcement & Asymmetric `TP=2` vs. `PP=2` Switching
**Target Bottleneck:** High-throughput serving with Chunked Prefill + Batched Decode ($B \ge 16$).

#### What It Does (High-Level)
Two structural rules specific to this model and hardware:
1. **Never Micro-Chunk Prefills Below $M_{\text{chunk}} \ge 512$ Tokens on Gated DeltaNet:**
   - In standard Transformers, splitting a 4,096-token prefill into 32 chunks of 128 tokens across 32 windows is free. In Qwen3.8-27B, every separate window that processes a chunk of request `A` must read and write `A`'s `75.5 MB` state `S[A]` (`151 MB/window`). Slicing prefill into 32 windows wastes $32 \times 151\text{ MB} = \mathbf{4.83\text{ GB}}$ of HBM traffic. Keep prefill chunk sizes $M_{\text{chunk}} \ge 512$ (or skip updating `S` across decode-only windows).
2. **Switch from `TP=2` to Asymmetric `PP=2` (`33 / 31` Layer Split) as Concurrency $B$ Grows:**
   - **At low batch ($B = 1\text{–}4$):** Use **Symmetric `TP=2`** (split heads `h` and FFN `m` 50/50 across GPU 0 and GPU 1) so both GPUs' `1,008 GB/s` memory buses stream weights simultaneously (`~13.5 ms` weight read + `~1.5 ms` PCIe AllReduce latency across 128 collectives). Use custom P2P PCIe two-shot AllReduce to minimize the 128-message PCIe latency overhead.
   - **At larger batch ($B \ge 16$) or multi-microbatch prefill:** `TP=2` runs 128 PCIe AllReduces of size $10\text{ KB} \times B$ per pass. Switch to **Asymmetric `PP=2`**: place **Layers 1..33 (`13.6 GB`) on GPU 0**, and **Layers 34..64 (`12.8 GB`) + `LM_Head` (`1.27 GB`) = `14.1 GB` on GPU 1**. This balances the `1.27 GB` vocabulary head against two extra Transformer layers on GPU 0 and reduces PCIe communication from **128 collectives to 1 point-to-point activation transfer** per micro-batch.

---

### Optimization 7: Top-$K$ Vocabulary Slicing (`FR-Spec`) on the `1.27 GB` `LM_Head` + Output-Gate Head Skipping
**Target Bottleneck:** Drafter overhead in speculative decoding and per-head state/KV bandwidth.

#### What It Does (High-Level)
1. **Drafter Vocabulary Slicing (`LM_Head` is `1.27 GB`!):**
   - Because `vocab_size = 248,320`, `LM_Head` (`248,320 × 5120 = 1.27 GB` in FP8) is as large as **3 full Transformer layers**!
   - During speculative drafting (`Guess`), restrict the drafter's `LM_Head` projection to a cached subset of the **top-16,384 most frequent token IDs** (`84 MB` instead of `1.27 GB`—**15× smaller**, saving `~1.2 ms` per draft step). The target model's verification pass (`Verify`) still checks the full `248,320` vocabulary in a single batched window ($N = M$), preserving 100% lossless target distribution.
2. **Approximate Output-Gate Head Skipping (`attn_output_gate = true`):**
   - In both the 16 Full-Attention layers and 48 Linear-Attention layers, each head's output is multiplied elementwise by a gate `g[r, i, h, d] = σ(x[r, i] · W_gate[h, d])` that depends **only** on the current input `x[r, i]`.
   - Compute `g[r, i, h, :]` first; if $\|\text{g}[r, i, h, :]\|_\infty < \epsilon$ for head `h`, skip reading `S[r, h]` (`128 × 128` matrix) or scanning `KV[r, g(h)]` for that head. (Empirically profile head gate sparsity on target calibration prompts first to verify what fraction of heads fall below $\epsilon$).

---

## 3. Recommended Implementation & Profiling Order

1. **First (Quickest High-Confidence Wins on 2× 4090):**
   - Stand up the **TP=2 FP8 baseline** (`13.5 GB` weights/GPU, 16-bit `S`, FP8 `KV`) and profile per-layer time breakdown between Weights, `S` read/write, `KV` read, `LM_Head`, and PCIe AllReduce latency.
   - Implement **Optimization 1 (DeltaLog Keep, $W=8$)** and **Optimization 7A (Top-16K Vocab Slicing for Drafter + Last-Token-Only Prefill `LM_Head`)**.
2. **Second (Latency / Speculative Decoding Multipliers):**
   - Implement **Optimization 2 (Dual-Form Unrolled Tree Speculation for Gated DeltaNet)** on top of Optimization 1's rollback-free ring buffer.
3. **Third (Long-Context & High-Concurrency Scaling into DDR4):**
   - Implement **Optimization 3A (3:1 Strided PCIe KV Double-Buffering)** across the 4-layer blocks for $L_{\text{ctx}} \ge 32\text{K}$, followed by **Optimization 4 (3-Layer Early Speculative KV-Page Prefetch)** for $128\text{K–}256\text{K}$ and **Optimization 5 (Shared-Prefix Lazy State Materialization)** for parallel branch workloads.
