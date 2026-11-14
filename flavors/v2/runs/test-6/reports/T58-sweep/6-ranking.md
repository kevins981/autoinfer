# T58 sweep, step 6: the promising candidates, ranked

Pricing uses the GPU-side whole-trial table (3-calibration §4b): trial ~76 s; decode-like work ~57 s (1 ms per decode step ≈ +3.1%);
mixed steps' excess: ~4.4 s fixed + ~16 s of prefill rows; 184 "small" mixed steps (P ≤ 191, total rows 97-255, hostar path) per trial.
I only rank what survived 5-history.md: untested, or tested on code different enough that the result may not hold.

| Rank | Candidate | Napkin gain (e2e) | Wins when | Riskiest assumption | Cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | **M1: AR-window L2 prefetch for 97-255-row hostar ARs.** Raise `VLLM_HOSTAR_L2PF_MSG_MAX` from 1 MiB to cover 255 rows (2.6 MB raw) | Small mixed steps lose t31b and the prefetch (KB131: GEMM +9.5 ms). Microbench hot-vs-cold bs_gemm at M=128: qkvz 72->54, gate_up 158->142, down 89->79, out 37->34, so ≤ ~3 ms/step if fully realized; x0.5-0.75 in-situ transfer (KB93) = 1.5-2.3 ms x 184 steps = **0.3-0.4 s, +0.4-0.6%** | The 88-128 us mid-size AR window lands enough prefetched lines (hint mode lands ~40% under load, KB97). The bs_gemm at M=128 is L2-sensitive (it is partly compute-limited) | That the prefetch blocks don't slow the mid-size AR itself (KB78/97: prefetch stretches the AR ~2 us + more at larger ranges) | One env var, no build. (a) A GPU-side mixed-step span comparison in one nsys job; or directly (b) an ABBA (`scripts/abba.sh`, A = champion, B = env set); stop-loss after the first pair |
| 2 | **M2: mid-M (97-255 rows) FP8 GEMM** keeping decode-like efficiency at M=128-192 (e.g. t31-style register-direct with 2 m16 tiles per warp, or bs_gemm with an L2-warm tile walk) | GEMM excess in small mixed steps 9.5 ms (KB131); halving it gives **~0.9 s, +1.2%**. With M1 the target shrinks | In-chain GEMM band at M=128 drops from ~0.07 to ~0.035 ms/row | That a kernel exists: KB131's ncu shows math-pipe throttle at ~74% of the e4m3 roof and L2 55-70% at M=128 | Microbench of a t31 variant at M=128 vs bs_gemm, cold and L2-warm (1-2 GPU-min), before any server work |
| 3 | **W7': lower the wave threshold by capturing the GDN/attention cores per wave size** (KB133 open item; uses memory freed elsewhere) | KB133 est. ~+1% if the threshold falls to ~384 | Host time per wave step falls below its GPU time | Graph memory at 0.45 GiB free; the 128K gate | Host-probe of a wave step at 384-768 rows with captured cores (T52's T29_HOSTPROBE), CPU-side mostly |
| 4 | **S2: load-adaptive draft depth in the drain** (k up to 5-6 when ≤ ~8 requests run) | Drain ~2-3% of slot time (KB69); +20-25% tokens there for +2-5% step time: **+0.4-0.8%** | Step cost is flat in rows at low batch (19.0-20.5 ms at 1-9 requests; my steplog) | KB85's plumbing penalty of ragged/adaptive K (-8.5% before). Also needs kMaxLogTokens and kLogCap lifted in the GDN replay kernel (CUDA build) and drafter graphs at those sizes | Offline: count drain steps x per-position acceptance at k=6 from a logged run (KB114's "k=6 run logging" test, ~3 GPU-min) |
| 5 | K2a: lazy checkpoint cap 8 -> 16 | writes -0.28 GB/step ≈ -0.25 ms vs replay compute +~0.2 ms (more pending tokens replayed each step; the kernel is latency/compute-bound, KB51/54): **~0** | the replay-compute growth is smaller than estimated | kernel compute per replayed token | microbench of the replay kernel at cap 8 vs 16 (needs a build) |
| — | k=4 uniform at full batch | **loses on my pricing**: +10.2% tokens (KB127 oracle) vs GEMM step 64->80 (+42 us/layer, KB120) + AR rows + one drafter level ≈ +4.6 ms (+18%) | — | — | — |

Ranks 1-4 are all exploit-sized (≤ ~1% each), and that is the honest picture. The framework sweep found no untested candidate above ~1-2% on this champion.
Everything structural it lists was tested in this run:
- the AR transport;
- parallelism;
- speculation depth/trees;
- weight coding;
- prefill chunking;
- f16 accumulation.
The run reached most of the closed directions by intuition, and the framework re-derives them as rows (5-history.md).

## Plan

1. M1 first: zero code, one env var. If the screen is positive, file the record with a full ABBA, then an official eval.
2. M2 microbench only if M1 shows the small-mixed-step GEMMs respond to L2 residency (the same physics).
3. Compare notes with T56/T57 now (my ranking is written). T56 offered M2 ("mid-M FP8 GEMM") and a lower wave threshold; avoid duplicating.

## Addendum after sparring (T58-1, sparring/)

My summary line "no untested lever above ~1-2%" is not supported as a general claim:
- **The drafter's headroom is unresolved.** With k=3 the ceiling is 4.0 vs 2.68 tokens per step (+49%).
- **What the history bounds is narrow:** fine-tuning the stock 1-layer MTP head with affordable self-generated data (KB60).
- **Untested:** a drafter fed richer features, such as intermediate target hidden states (EAGLE-3-like), or with more capacity.
  - Napkin: conditional acceptance 0.74 -> 0.82 gives ~3.04 tokens/step (+13%), minus ~+1 ms/step of drafter cost: **~+8-9%**.
  - Riskiest: that richer features raise greedy agreement at all.
- **Cheapest test** (no new generation pass): T43's dump (agents/leader-43.w1/dd/full: residuals at L16/32/48, ids, drafter tokens and confidences for ~32K verify rows), combined with T3's offline MTP pipeline (agents/leader-3.w1).
  - Train two equal small adapters on the frozen head, with and without h_32/h_48.
  - Split held-out data by sequence.
  - Measure agreement with the target's greedy token.
- This is a research-sized item (training), outside this team's budget; ranked separately from ranks 1-4.

## Addendum 2: outcomes and hand-off (after T58-001, T58-002 and sparring T58-1 turns 7-8)

- **Rank 1 (M1)** was built as T58-001 and is the champion: 1229.6 official, +0.72% same-job (6-start ABBA), +0.33% cross-job.
- **Rank 2 (M2)** was microbenched in T58-002, job 512.
  - The hybrid that uses t31b for out/down at 97-144 rows saves 17-22 µs per GDN layer, about +0.2-0.3% end to end.
  - It was not pursued: it would need an FP8 GSM8K A/B, which costs more than it would gain.
- **Rank 3 (W7': lower wave threshold with per-wave-size core capture).** I withdraw my earlier "≤0.5% cap". Handed on, as follows:
  - **Supported:** about 76 steps per trial carry 384-767 prompt tokens, at ~115 ms GPU each, ~8.7 s per trial in total.
  - **To be tested:** whether two waves beat joined execution at 384-767 rows. My scenario assumes 8-12% less GPU time; that number is not measured.
  - **Unresolved feasibility:** whether whole-wave graph capture for the needed sizes fits in ~0.45 GiB of free memory.
  - **Conditional payoff:** ~+0.9-1.4% end to end, but only if host overhead is zero and both points above hold. It is an optimistic scenario, not a ceiling.
  - **First discriminator:** compare two waves against joined execution in GPU time at 384-767 rows. `T30_WAVE_BENCH` with `T30_WAVE_BENCH_BINS` in vllm/t30_wave.py already times eligible steps without the host. Then measure the extra memory capture needs at one representative size.
  - It is a multi-day build with memory risk, more than T58's budget.
- **Rank 4 (S2, drain depth):** +0.4-0.8% is an assumed estimate.
  - KB85's -8.5% plumbing loss doesn't settle the sparse, drain-only case, but it is the main risk.
  - Before any build, account for the payoff from a trace: drain steps x per-position acceptance at k=6.
  - Not attempted here.
- **The reason for closing is about resources, not exhaustion.** The remaining opportunities need multi-day builds with unresolved feasibility. It is not true that "everything left is <0.8%".
- **Largest open lever:** the drafter fed richer features (Addendum 1), which is research-sized.
