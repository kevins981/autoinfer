# Round 3 ideas: bs-r3-adjacent_field (databases / storage-systems view)

I approached this the way a database person approaches a slow system: before tuning operators, check
admission control and the buffer pool. That is where the main finding came from.

## Finding F1 (measured): the k=3 TTFT blow-up (T1-008, KB11) is an admission-control stall, not slow prefill
Sources: `workspaces/leader-1.logitsflags/out/C3/server.log` (k=3), `.../C2b/server.log` (k=2, same job),
`bank/T1-009/official/1/trial*/server.log`.

- **k=3: every one of the 17 scheduler log lines reads `Running: 14 reqs, Waiting: 2 reqs, GPU KV cache usage:
  95.3%`** (one reads 13/2/88.5%). `Running: 16` never appears. The benchmark is closed-loop at c=16, so two
  requests sit in the queue for the whole run. That is the TTFT p50 792 ms. Throughput also falls because the
  effective concurrency is 14, not 16.
- **k=2 (C2b and the official T1-009 run): `Running: 16, Waiting: 0, usage 83.8%`.** A single request shows 5.2%.
- **Why (inferred from the code and the numbers):** this is a hybrid model, so GDN states live in the same block
  pool as the attention KV. The page size is unified: the attention block size is forced to 1600 tokens to match
  the mamba page (`interface.py:1115` log line), and the log shows 4 KV-cache groups. Each mamba group holds
  `1 + num_speculative_blocks` state blocks per request (`vllm/v1/kv_cache_interface.py:1138-1156`;
  `MambaSpec.num_speculative_blocks` = k). For a short request:
  - at k=2: 3 mamba groups × 3 + 1 attention block = **10 blocks**;
  - at k=3: 3 × 4 + 1 = **13 blocks**.

  The measured usage fits this exactly: 5.24% vs 6.81% per request, a ratio of 1.30 = 13/10. That puts the pool at
  ≈191 blocks (5 GiB, `--kv-cache-memory 5368709120` in the harness). 16 requests at k=3 need ≥208 blocks, so
  only 14 fit.
- Consistent with T1-008's own numbers: TPOT p50 went down (19.3 vs 20.5 ms, because of a smaller batch and more
  tokens per step) while TTFT went up 4x. A slower prefill could not produce that combination; queueing does.
- So **3 draft tokens were never actually tested at c=16.** The acceptance gain was real (measured 1 + .772 +
  .548 + .396 = 2.716 vs 2.34 for k=2, `C3/bench400/bench.log` per-position).

---

## Idea 1 (top bet): k=3 with the block pool sized so 16 requests fit
- **direction:** Re-run T1-009 + `num_speculative_tokens: 3` with a KV budget large enough for 16 k=3 requests
  plus some growth: `--kv-cache-memory` ≈ 5.75–6.0 GiB (~215–225 blocks). If the memory isn't there, free it
  elsewhere rather than give up:
  - (a) trim the CUDA-graph capture list. 41 sizes up to 2048 are captured for 1.34 GiB/rank; decode needs only
    multiples up to 64, and the prefill sizes could be thinned;
  - (b) see Idea 3, which removes the speculative mamba blocks altogether.

  Check that the server log shows `Running: 16, Waiting: 0`, and rerun the 128K request.
- **why:** F1, all measured. Memory headroom, measured from the C3 log: free at start 22.89 GiB − model 14.99 −
  KV 5.0 − graphs 1.34 = **1.56 GiB** before activation peaks (memory profiling is skipped when
  kv_cache_memory is set). So +0.75 GiB is plausible but not certain. The 128K request needs ~82 attention + 12
  mamba blocks, which fits either way.
- **bottleneck:** the decode steps (66% of wall time). k=3 trades a ~9% longer step for 16% more tokens per step.
- **upside (napkin):** T1-008 measured the per-request decode rate at k=3 as +6% (TPOT p50 19.3 vs 20.5 ms),
  with 14 running. With 16 running the step gets a little longer, because verification at M=64 instead of 56 is
  memory-bound (assumed ~+1–2%). So **per-sequence rate +4–5%, at full concurrency 16 → +3 to +5% tok/s**, and
  TTFT goes back to about k=2 levels. Confidence: medium-high that admission is the cause (the log is
  unambiguous), medium on the size of the gain.
- **novelty:** k=3 was tried (T1-008) but diagnosed wrongly as "slower prefill/mixed steps". This is new: a
  zero-code fix to a measured cause. Validity: a memory/KV-budget change, allowed if the 128K request still works.
  Spec decode stays exact.
- **cheapest test (~15 GPU-min, one job):**
  1. T1-009 and T1-009+k=3+`--kv-cache-memory 6000000000`, back to back, bench400 each; grep `Running:` in the
     server log.
  2. Then `long_request 128000` on the k=3 server.

  An even cheaper pre-check: start the k=3 server with the larger KV budget and confirm it starts and logs
  ≥215 blocks (≈ `GPU KV cache size` ≥ 290K tokens). That costs ~3 GPU-min.
- **follow-up if it pays:** k=4 needs 16 blocks per request (256 blocks ≈ 6.7 GiB), so it is probably out of
  memory unless Idea 3 lands. Position-3 acceptance is still 39.6%, so k=4 might add ~+0.3 tokens per step.

## Idea 2 (bold, speculative): lossless compression of the all-reduce payload (a storage/network trick)
- **direction:** The host-staged decode AR is PCIe-bandwidth-bound: 491 KB per AR at M=48 ≈ 25 µs per
  direction, with write and read phases only partly overlapping (KB2, KB5). Storage and network systems almost
  never ship floating-point data raw. bf16 activations have a low-entropy exponent byte (a narrow dynamic range
  within a row). Encode each partial losslessly on the GPU before pushing it to host memory, e.g.:
  - per group of 64 values: base exponent + a 4-bit exponent delta + the raw sign and 7 mantissa bits
    = 12 bits per value;
  - groups whose exponent range exceeds 16 (outlier channels, zeros) go raw into an overflow area.

  The peer decodes bit-exactly and then adds. **Bitwise lossless**, so the bits on the wire decode to exactly
  the baseline's values. This is not quantized communication; still worth a one-line confirmation from the
  manager.
- **why:** decode ARs measured at 128 × ~44 µs = 5.6 ms of a 38.5 ms step (KB9), at the PCIe floor per KB5.
  Fewer bytes is the only lever left there. SM-side encode/decode of 491 KB is ~1–2 µs of DRAM-speed work
  (assumed), against ~25 µs per direction of PCIe.
- **upside (napkin):** if 25% fewer bytes (12/16 bits; assumed, depends on the data), transport goes from ~40 to
  ~30 µs per AR, saving ~1.2 ms/step (3% of a decode step). Prefill-step ARs gain too, on their transport part.
  **~+2 to +3% tok/s.** Confidence: low-medium. Everything hinges on the measured compressibility of the real
  partials (outlier channels could break groups) and on the encode/decode staying in the same kernel.
- **novelty:** high. Nothing like it in tree.md or ideas.md. All comm work so far changed transport, never bytes.
- **cheapest test:** mostly GPU-free.
  1. One ~2 GPU-min job dumps a few hundred real o_proj/down_proj partials (pre-AR) at c=16 decode.
  2. A CPU script computes the fraction of 64-value groups whose exponent range fits 4 bits and the resulting
     bytes/value.

  Go if the payload is ≤ 0.8× the raw size; drop otherwise.

## Idea 3: GDN checkpoint + log replay (R2-B), now with a second payoff: it frees the block pool for more drafts
- **direction:** R2-B as proposed in round 2: write one fp32 state per step plus small per-token log records, and
  replay the accepted tokens at the next step. Bitwise exact. The new angle: place the log records in a small
  dedicated buffer, not in the speculative state slots, so the mamba groups need `num_speculative_blocks = 0`.
  Per request, that takes the blocks from 10 to 4 at k=2 and from 13 to 4 at k=3 or k=4.
- **why:** F1 shows the mamba speculative slots take 6 of 10 blocks per request at k=2 (≈60% of the pool use at
  c=16). The traffic side is measured in round 2 (KB9: 75 MB of state writes per GDN layer per step, a spill into
  the next GEMM of +19 µs).
- **upside:** round 2's +3.5–5% from traffic, plus it makes k=3/k=4 fit without growing the KV budget (Idea 1's
  memory risk disappears). Confidence: medium. The effort is medium-high: kernel work, plus an audit of the
  align-mode/prefix-cache consumers (`vllm/v1/worker/mamba_utils.py:1557-1610`).
- **novelty:** a re-proposal (R2-B), with a new measured motivation (F1).
- **cheapest test:** round 2's ceiling microbenchmark (state_indices columns 1..k set to the null slot: time
  saved per GDN layer), ~1 GPU-min.

## Not proposed
- Overlapping the decode AR with the GEMM (epilogue writes straight to host memory): Marlin's fp32 reduce writes C
  only at the end of the k-split, so there is little overlap to win (a reading of the code, not measured).
- Prefill batching/admission games: prefill steps are compute-bound (M≈430 ≫ the ~165 FLOP/B ridge), so grouping
  prompts saves nothing. fp16-accumulate tensor cores (2x peak on the 4090) change the math: forbidden.
- Duplicating some of rank 0's weights on rank 1 to rebalance the GPU 0 clock skew in prefill: costs ~0.7 GB,
  which Idea 1 needs, and wins ≤3%.

## My bets
1. **Idea 1.** The cause is measured and unambiguous in the logs. It is a flag-level change with a ~15 GPU-min
   test, and it turns a "failed" experiment into a likely +3–5%. Do it first. The only risk is memory headroom,
   which has two fallbacks.
2. **Idea 3,** if a team has the engineering capacity. It stacks with Idea 1 and removes its memory risk.
3. **Idea 2** as the bold one. Its first test is almost free (one dump, then CPU analysis), so it can be killed
   cheaply.

## Follow-up (manager's answers)
- Idea 2 validity: bitwise-lossless compression of the all-reduce payload counts as baseline-precision
  communication; no human ruling needed. Conditions: lossless for **all** inputs, with a raw fallback for
  incompressible groups (the design above has one: the overflow area), and nothing tuned to the benchmark's
  data distribution beyond generic compressibility. The group size and delta width should be fixed generically,
  not fit to dumped ShareGPT activations.
- Idea 1 validity: raising --kv-cache-memory is allowed. It is a memory change, so it needs the 128K-request test
  and an OOM check under bench load.
