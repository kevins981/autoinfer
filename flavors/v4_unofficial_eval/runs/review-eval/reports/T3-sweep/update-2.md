# T3 map update 2 (closing)

## Tested since update-1
| candidate | result | evidence |
|---|---|---|
| per-target prefetch size (gate_up / qkvz / qkv / MTP M=16) | uniform 16 MiB within ~1 us of each target's best; no gain to tune | worker job (hmar_pf2.log) |
| fp8 drafter lm_head on the current champion (KV 4.4 GiB to fit) | **win: T3-003 = champion, 826.1 (3 starts 821.2/821.9/835.1, +1.2%)**; head microbench 1484 -> 783 us at M=16 (~3% napkin, delivered 1.2%) | jobs 63, 65, 67; review T3-4 |
| prefetch under the drafter-head AR | bf16 head: -134 us in microbench; fp8 head: +89 us (worse); quick bench +0.15% -> dropped | T3-005, job 64 |
| exact greedy target fast path (NaN -> exact fallback, LoRA/bonus guards) | within noise on T3-003: 827.6 (817.0/838.3); per-step host sync eats the ~1% | T3-004, job 69; review T3-3 turn 2 |
| TRITON_ATTN (in-place draft attention metadata; no eager rebuild) | flag-only: +4.1% quick (job 71) but **invalid**: FP8 query + FP8 P with an FP8 KV cache | T3-006, review T3-5 |
| TRITON_ATTN with bf16 query (one-line default in triton_attn.py) | **853.6 (854.1/853.2), +3.3% vs T3-003, GSM8K 93.6%: promoted, champion** | T3-007, job 75; review T3-5 turn 2 |

## Verdicts changed
- The fp8 draft head: earlier teams saw +6.7% (BASE) and neutral (T2-003 with KV 4.25 + preemptions); on T3-001 at KV 4.4 it is +1.2%
  with 0 preemptions. Memory headroom was the real blocker (graph capture OOM at 5 GiB).
- GPU idle per decode step (~2 ms, union-based) is partly the drafter's eager FlashInfer metadata rebuild: a CPU-side cost the
  framework's (F, BW, Link) window model has no column for.

## Framework notes
- Missing tier: L2 as a Place target in link windows (T3-001).
- Missing resource: host CPU / launch path and host syncs (T3-004's sync, FlashInfer's eager rebuild): a fourth window resource.
- Missing dimension: memory headroom as a binding constraint on draft-side copies (T3-003).
- Backend choice is not "the same math" by default: a backend swap can silently change precision (T3-006); the contract check must read
  the kernel's dtype path, not only the flag.

## Current ranking (for the next team)
1. Profile T3-007 to see what idle remains per decode step and per mixed step (TRACE=graph); compare FlashInfer vs Triton target-side attention time.
2. Remove other per-step CPU gaps (union idle ~2 ms/decode step; mixed steps ~7-22 ms idle).
3. GDN DeltaLog (3 state writes per request per layer confirmed; real-server savings depend on L2 write-back): needs a dram-bytes check.
4. hmar epilogue fusion (residual + norm + quant), ~0.8%.
