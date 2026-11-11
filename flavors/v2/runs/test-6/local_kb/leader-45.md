# leader-45 (T45, research_radical) working notes

## Inspiration cards (first set + redraw)
Cards considered: Holding, Rework, Fixed choices, Boundaries, Unused structure; Per-item overhead, Repetition,
Predictability, Trade-offs, Idleness; HW (HBM 1008 GB/s, 72 MiB L2, four one-way PCIe lanes, green contexts).
Takeaway: "Idleness" and "Unused structure" only point to the drafter phase (near-full, <=2%) and the CPU (AMX ~7 TF,
dead). "Trade-offs" points to the one parameter that gates every tokens-per-step idea: the in-situ cost per extra row.

## Facts I found (cite)
- MTP input convention: post-norm is correct (T45-001).
- Memory clock P2 cap 10251 vs 10501 MHz under CUDA: DRAM roof ~984 GB/s (T45-002).
- Closed-loop sim + sensitivities + tree row-cost sweep + comm min-cut: T45-002, agents/leader-45/sim/.
## Sparring (T45-1, 4 turns)
- Partner flagged the additive-floor assumption (sum of kernel floors = step). Supported for same-row chains and SM
  transport (KB108 chain), untested for tree-sized M (>64 rows) in TP.
- Trailing prompt waves under decode GEMMs: infeasible (CE chain ~0.3-0.5 ms per ~1 MB message spans 3-5 GEMMs).
## Killed at 0 GPU-min
CPU prefill, MTP pre-norm, alt TP partitions (min-cut), async prefill lane, prompt-as-draft rows <64/step,
drafter-phase co-scheduling, prefill coalescing, lossy shadow rows (approx-AR self-drafting: +10% tokens for +11% step),
context AR codec (~+1-2%), PP decode + prefill-in-bubble (wash vs champion, consistent with T38).
## Stage 1 (T45-003, job 408)
Chain M=48/64/80/96 (us/layer): 356.8/368.6/404.9/411.6; gate_up 94.9/104.1/131.0/131.5 = bf16-MMA compute roof from
~M=75. Kill rule hit (86 us/row/pass vs 40). Sub-ridge slack only with an e4m3 gate_up (untested at M>64).
## Stage 1b (T45-004, jobs 411/413)
bs_gemm gate_up flat ~120 us M=80-128 (Marlin to 165). All-bs chain M=64 344.2 vs champion-like 368.6 us/layer:
champion's gate_up at M=64 sits on Marlin's step (104 vs bs 82.6). Exploit item, ~+1.5-2.5% e2e est., untested in server.
## T45-005 (gate_up -> bs_gemm W8A8 at 48<M<=96)
ABBA +3.83% all B > all A (job 415); FP8 GSM8K 94.8 vs bf16 ref 93.2 (job 416); official #1 1173.4 tok/s (+2.7% vs
1142.2 cross-job, below the 1199.3 +5% bar), p99 TTFT 831 / TPOT 25.9, all gates pass. Balance 20.7.
