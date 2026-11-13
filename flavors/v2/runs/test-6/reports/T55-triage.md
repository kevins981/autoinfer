# T55 triage of research_handoff_1113.md (leader-55, 0 GPU-min)

Done independently, then checked against T54's triage (reports/T54-triage.md), which agrees with this one row by row
and has the config.json fact check (24 Q / 4 KV heads, mamba_ssm_dtype float32). Split agreed with T54: T55 took the
GDN-state ideas (1, 5), T54 took speculation, partition and vocab (2, 6, 7). Sparring: sparring/ T55-1 (summary below).

## Memo inputs that are wrong for this system
| Memo claim | This system | Evidence |
|---|---|---|
| State write 75.5 MB/req every step (151 MB traffic/token) | The champion already writes no state per decode step: replay kernel + lazy checkpoint every 8 accepted tokens + two-slot rollback | KB15 (T1-012), KB48 (T1-035), KB111 (T37-002) |
| GDN state 16-bit | fp32 (checkpoint config); 25 MB per layer per GPU at 16 requests; can't drop (GDN math fixed) | KB9, task_definition, T54 fact check |
| Ridge ~330 rows (660 TF FP8) | exact paths are bf16 MMA (~165 TF) or e4m3 with fp32 promotion (~340 TF measured roof): ridge ~87 / ~180 rows; measured decode GEMMs step up above 64 rows | KB118, KB120, KB131 |
| PCIe all-reduce ~1.5 ms/pass, 10-15 us latency P2P | no P2P on this box; host-staged AR ~50 us each, ~141 per decode step = 7.0 ms (28% of decode) | KB109, KB130, leader-53 anatomy |
| B=1-4 latency regime, 32K-256K contexts, fan-out | c=16 closed loop, ~260 in / ~228 out tokens, independent requests | TASK.md |

## One row per idea
| # | Idea | Verdict | Evidence | What would have to be true to pay here |
|---|---|---|---|---|
| 1 | DeltaLog Keep (checkpoint + ring of inputs, replay, flush every W) | **Already tested in this run**: it is the champion's GDN path | replay (KB15, +2.6%), lazy checkpoint every 8 (KB48, +1.0%), two-slot rollback-free state (KB111). What's left is the state READ: replay 38.5-40 us/layer in situ vs ~25 us DRAM floor, 23.4 us fully L2-warm (T22); prefetch in the AR window did not reproduce the warm time (T22), and the window's spare 11-19 MB is fungible with weight prefetch (T12 write-up: <=1%; T52-001). Ceiling ~0.7 ms of 25 ms decode, ~1.5-2% e2e. | The dense per-step read of fp32 S would have to be avoidable. Sparring T55-1 agreed it is required: q_t is dense and only exists at that step, the 4 verify rows already share one read, requests don't share state, 1.2 GB/GPU of state can't stay in a 72 MB L2. |
| 2 | Dual-form unrolled tree speculation | **Wrong workload** (T54's slice) | extra verify rows cost ~0.07 ms AR + ~0.07 ms GEMM each above 64 rows (KB118/120/131) vs ~0.61 ms per accepted token (KB111); offline exact tree acceptance (agents/leader-38.w1/tree/full_cpu_analysis.txt): chain k=3 2.73, dynamic N=3 2.78, N=7 3.73. My model: N=7 at 128 rows ~37 ms/step -> 0.100 vs 0.106 tokens/ms for the chain, i.e. a loss. Sparring: the AR bytes per extra row are required under TP (each node is its own residual row; bf16 comm). | Rows above 64 would need to cost <~0.04 ms each (they cost ~0.14). |
| 3A/3B | Strided PCIe KV offload / CPU attention | **Wrong workload** | KV ~16 MB/request, fits; attention 0.31 ms/step; host link already carries the ARs. | Long contexts that overflow VRAM. |
| 4 | Early speculative KV-page prefetch (Quest) | **Wrong workload, and approximate** | contexts <~1K; top-K attention is inexact (task_definition). | Long context plus an exact variant. |
| 5 | Shared-prefix lazy state (one S_prefix + private suffix logs) | **Wrong workload; key assumption wrong** | no fan-out in the benchmark; prefix-state caching already +0.5% (KB86). Also the crossover "S > raw suffix log below ~150-300 tokens": at fp32/TP=2, S is 1.5 MB/layer/request per GPU vs ~8 KB per token of k,v,(a,b), so ~190 tokens; benchmark contexts (prompt ~260 + output) are mostly past it, and a parallel-form recompute over the log costs O(T^2). | Branched requests with short private suffixes. |
| 6.1 | Prefill chunk >= 512 for GDN | **Already true here** | prompts ~260 tokens, budget 2048: not chunked; mixed-step GDN (chunked core + conv/state) is ~2.3 ms of a ~41.5 ms small mixed step (KB91), mixed steps ~35% of wall, so the whole GDN share of mixed steps is <~2% e2e (sparring T55-1's open question, answered from KB91). | Long prompts split into small chunks. |
| 6.2 | TP -> asymmetric PP=2 | **Already tested** (T54's slice) | KB112 (decode 0.91-0.92 of TP with trees), KB123 (calibrated closed-loop sim). Sparring note: PP end-to-end is a calibrated sim, not a server run. | PP decode >= 0.93 of TP. |
| 7A | Top-16K drafter vocab | **Already tested, superseded** | KB71 (32K list -7.3%), KB84 (exact PCA screen, 115 us per draft step). | — |
| 7B | Output-gate head skipping | **Not allowed (approximate)**; and the state update must run anyway, so the S read can't be skipped | task_definition | An exact zero-gate rate (implausible). |

## Decision
No memo idea in T55's slice (1, 5) has a path to +5%. Their mechanisms are already in the champion (idea 1) or target
traffic this benchmark doesn't have (idea 5). The residue is the GDN replay latency gap (~0.7 ms/step, <=2% e2e),
already costed by T12/T22. T55 writes no proposal and spends no GPU time; the 30 GPU-min seed is unused.
