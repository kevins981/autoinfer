## Run start (human, 2026-11-14)
The task definition, assumption ledger and focus were agreed with the human before the run started
(task_definition.md, assumption_ledger.md, focus.md): initialize() skips the definition. The run
starts from the stock baseline (bank/BASE); map.md starts empty.

## 2026-11-14 manager: run opened
- Definition skipped (agreed before the run). Pool 1000 GPU-min.
- Opened T1 (leader-1) and T2 (leader-2), sweep teams, slot A, each seed 20 + start 80 = 100 GPU-min. Pool free: 800.

## 2026-11-14 manager: rule change from the human (relayed via another session)
- Map disagreements: keep both views in map.md (attributed, with reasoning and evidence), mark the entry
  contested with what would settle it; the manager does not settle them. Tell open leaders the first
  time a contested entry is recorded. (No contested entries yet.)

## 2026-11-14 manager: T1 sweep report + map proposal 1
- reports/T1-sweep.md received (files 1-6). proposals/T1-map-1.md accepted as the initial map.md skeleton
  (evidence: sweep files, job 1 microbench (peer-access error confirmed in its log), BASE profile).

## 2026-11-14 manager: infra issue, vLLM socket path too long
- T1 job 5: ZMQError "ipc path ... longer than 107 characters"; TMPDIR=$CACHE_ROOT/tmp/v3_framework-framework-eval + uuid = 113 chars.
- official_eval -> harness/eval.sh -> workspace env.sh -> task env.sh sets the same TMPDIR; nothing sets VLLM_RPC_BASE_PATH, so official evals likely fail the same way.
- Announced workaround for unofficial jobs (VLLM_RPC_BASE_PATH=$CACHE_ROOT/tmp, 85-char path). Leaders to hold official evals until the human decides. Raised with the human.

## 2026-11-14 manager: T2 sweep report + map proposal 1
- reports/T2-sweep.md received. proposals/T2-map-1.md accepted (A-E): mixed-step rank-skew caveat, new knob sites, 7 table rows,
  ReplaySSM row refined (T1's entry; a refinement backed by kernel-code evidence, not a disagreement, so not contested), method notes.
- T1/T2 agreed a work split between themselves (noted, not imposed by the manager).

## 2026-11-14 manager: socket-path issue fixed (human)
- Human added `export VLLM_RPC_BASE_PATH=$CACHE_ROOT/tmp` to tasks/ and run task/ env.sh (verified, line 27 in both). Official evals unblocked; announced.

## 2026-11-14 (human) shared map dropped
map.md is removed (the design copy is kept outside the run). Each team keeps its own map: its sweep
folder reports/<team>-sweep/, plus update-<n>.md at each resync and at closing. Teams read other
teams' maps only after their own phase 1. No map proposals; the manager no longer edits or judges
maps. funding_policy.md updated. T1-map-1/T2-map-1 stay in proposals/ as those teams' notes.

## 2026-11-14 manager: shared map dropped (human)
- map.md removed; manager.md, leader_sweep.md, common.md, funding_policy.md updated by the human. Each team keeps its own map
  (reports/<team>-sweep/ + update-<n>.md at each resync and at closing). No map proposals; the manager no longer edits or judges maps.
  Forwarded verbatim to leader-1 and leader-2. Earlier map decisions above (T1-map-1, T2-map-1) are superseded.

## 2026-11-14 manager: new champion T1-004
- Official #1 707.6 tok/s (709.6/705.6) vs BASE 635.9/635.9: both trials +11%, far outside the noise. Gates pass (GSM8K 94.4%, latency, 128K).
- Validity: draft fp8 head (ruling 1), local argmax exact for greedy drafts; HOSTMEM AR sums rank0+rank1 in the same order on both
  ranks in bf16 (bitwise equal to NCCL per T1); generic size threshold (1.31 MB env default), enabled only for TP=2 without P2P. Nothing fitted to data.
- Note: the AR extension JIT-compiles via cpp_extension.load at server start (costs GPU time per fresh workspace, not a validity issue).

## 2026-11-14 manager: job 22 exit-75 refusal (17:26:09) — assessed as a teardown race, not outside use
- Refusal named pids 1072115/1072116 (same user, ~15 GB each). Job 21 (T2 w8a8 server, started 17:25:28) was
  terminated just before (output "Terminated"; no ledger entry, so uncharged); its vLLM TP workers were most likely still
  exiting when job 22 checked. Gone minutes later; nvidia-smi now shows only job 23's workers. Run GPU work not halted;
  reported to the human.

## 2026-11-14 manager: new champion T2-002 (credit T1 + T2)
- Official #1 757.6 (756.3/758.9) vs T1-004 709.6/705.6: both trials ~+7%, clear. Gates pass. GSM8K 93.2% vs 94.4% (3/250 questions):
  consistent with fp-reordering flips (graphs vs eager mixed steps), far above the 89.6% gate; accepted.
- Validity: flags only. Batch-sharded sampling: each request sampled from full-vocab logits on one rank (exact). Capture sizes:
  BASE's list + generic 128..512, same kernels, padding discarded; not keyed to the benchmark. Nothing fitted to data.

## 2026-11-14 manager: new champion T1-009; T1 resync 1 -> extension 80
- T1-009 official 786.4 (789.7/783.2) vs T2-002 756.3/758.9: clear. Gates pass. Flag only (Triton attention: fp reordering).
  GSM8K drift: 94.4 (BASE) -> 93.2 (T2-002) -> 92.8 (T1-009), all well above the 89.6 gate; flagged to the human.
- T1 resync 1: win since last grant -> extension 80 GPU-min (reports/T1-1.md). Pool free after: 720.
- 2026-11-14: T2-004 official 788.8 not promoted (+0.3% over T1-009, trials overlap); W8A8 valid per contract.

## 2026-11-14 manager: new champion T2-006; ruling on FP8 A/B carry-over; T2 resync 1 -> extension 60
- T2-006 official 819.9 (821.6/818.3) vs T1-009 789.7/783.2: clear. Gates pass. FP8-activation A/B from T2-004 (job 32) carried over
  (only a flag added that doesn't touch FP8 paths); ruling added to task_definition.md, to be confirmed with the human.
- T2 resync 1: win since last grant -> extension 60 (reports/T2-1.md). Pool free after: 660.
- 2026-11-14: human confirmed the FP8-activation A/B carry-over ruling (task_definition.md); T2-006 stands.

## 2026-11-14 manager: usage-limit stop (~19:20-19:30 UTC) and resume
- All agents went idle mid-task when the plan's usage limit ran out; it reset. Resumed leader-1 and leader-2 (told to resume their workers). No GPU jobs were running or queued. Balances: T1 75.2, T2 79.9; pool free 660.

## 2026-11-14 manager: new champion T1-014
- Official 842.7 (841.6/843.8) vs T2-006 821.6/818.3: both trials ahead. Gates pass (GSM8K 93.6%).
- Validity: T1-012 greedy local argmax exact (first max, lowest global id on ties, as torch.argmax; vocab padding excluded;
  generic fallback whenever a request needs logits). T1-013 row split: same per-row math, ARs on row halves (same sums), fp reordering
  at most; generic token-count threshold.
- 2026-11-14: T1 recommends closing itself (T1-015 official -0.7%; remaining map items high-effort). No sparring on record -> offered one before deciding.

## 2026-11-14 manager: T1 closed, T3 opened
- T1 recommended closing; sparring offered and done (sparring/T1-1.md), two closing tests run (k=1 screen negative; Marlin gap is
  environmental). Close per policy (hard limit on k; remaining items high-effort, variant aimed at cause tried). 16.2 GPU-min returned.
- Opened T3 (leader-3, slot A), seed 20 + start 80. Pool free after: 576.2. Champion given in its task: T1-014 (T2-008 official pending).

## 2026-11-14 manager: new champion T2-008
- Official 880.1 (880.4/879.7) vs T1-014 841.6/843.8: clear. Gates pass (GSM8K 94.4%). Same serve flags as T1-014; code adds
  GDN DeltaLog, an exact rewrite (same fp32 ops and order, bitwise-tested), no change to FP8/comm/sampling paths.

## 2026-11-14 manager: new champion T3-003
- Official 897.4 (895.7/899.2) vs T2-008 880.4/879.7: both trials ahead; same-job ABBA +2.41% all B > all A (job 85). Small but consistent.
- Valid: prefetch only (L2 hints/loads by extra AR CTAs, no writes), outputs bitwise identical by construction; same flags as T2-008;
  generic env defaults (40 MB, <=128 rows) keyed to hardware, not the benchmark. Gates pass (GSM8K 93.6%).
- 2026-11-14: ruling: fused RMSNorm+FP8 quant counts as fp reordering if quant stage bitwise exact; same-job GSM8K-250 on both arms instead of a new bf16-ref A/B (task_definition.md). To report to the human.
- 2026-11-14: human confirmed the fused norm + FP8 quant ruling.

## 2026-11-14 manager: T3 resync 1 -> extension 60; T3-005 promotion pending
- T3-003 win since the start grant -> extension 60 (reports/T3-1.md). Pool free after: 516.2.
- T3-005 official 909.7 (907.4/912.1) vs T3-003 (895.7/899.2): cross-job +1.4%, same-job ABBA mixed (+0.85%, job 101). Not clear
  enough: asked for a repeated same-job A/B before promotion.

## 2026-11-14 manager: new champion T2-009; T2 resync 2 -> extension 60
- T2-009 official 906.9 (904.8/909.0) vs T3-003 895.7/899.2; same-job ABBA +1.29% all B > all A (job 102; narrow on the 2nd pair). Fused-op
  ruling applied: quant bitwise, same-job GSM8K-250 93.6 vs 93.2. Gates pass. Promoted.
- T3-005 (909.7, builds on T3-003, pending a same-job A/B) must now beat T2-009 to be promoted; told leader-3.
- T2 resync 2: wins since last grant -> extension 60 (reports/T2-2.md). Pool free after: 456.2.
- 2026-11-14: T2 asked how a reduced-vocab draft subset may be chosen. Answered from existing ruling 2 + no-fitting ruling: (a) tokenizer/model-only, (b) generic public corpus (not chosen for resembling ShareGPT chat), (c) draft outputs on generic prompts: all allowed, source documented in the record; never benchmark data or GSM8K. No new ruling needed.

## 2026-11-15 manager: new champion T3-006
- Official 919.0 (922.8/915.2) vs T2-009 904.8/909.0; same-job 6-start ABBA +1.86% all B > all A (job 110). Prefetch only (exact), same flags. Gates pass.
- T3-005's repeated A/B vs T3-003 also all B > all A, +1.86% (job 108); T3-005 superseded by T3-006, not promoted separately.

## 2026-11-15 manager: new champion T2-011
- Official 931.8 (930.7/932.8) vs T3-006 922.8/915.2: both trials ahead; same-job ABBA on T2-009 +3.14% all B > all A (job 116). Gates pass.
- Valid under ruling 2 option (a): subset = tokenizer's base-vocab boundary [0,151643) + special tokens, no benchmark data; coverage check on
  vLLM docs/code only. K picked among structural candidates (98304 vs 151643) with a smoke bench; the chosen K is the structural, more
  inclusive one, so not fitted. Generality note for the human: ids >= 151643 (the vocab extension) can't be drafted; workloads that use them
  heavily would see lower acceptance (speed only; outputs stay exact).

## 2026-11-15 manager: new champion T3-009; T3 resync 2 -> extension 50
- T3-009 official 951.5 (955.0/947.9) vs T2-011 930.7/932.8; same-job ABBA +1.40% all B > all A (job 123). Exact (same per-row math; group-128
  per-token quant is row-local). Gates pass. Also notes a silent regression since T2-009 (row-split disabled), now fixed.
- T3 resync 2: wins since last grant -> extension 50 (reports/T3-2.md). Pool free after: 406.2.
- 2026-11-15: T2-012 official 953.1 not promoted (+0.17% over T3-009, trials overlap; no same-job A/B on T3-009). Promotion would need one showing it consistently.

## 2026-11-15 manager: T2 closed, T4 opened
- T2 recommended closing. Spar T2-1 already discussed closing; it named k=3 and mixed-GEMM efficiency as unchecked, and T2 checked both
  (k=3 -11.7% screen, job 133; mixed GEMMs 80-90% of ceiling). Remaining ideas <=1%. Closed; 17.2 GPU-min returned.
- Opened T4 (leader-4, slot A), seed 20 + start 80. Pool free after: 323.3. Champion in its task: T3-009.

## 2026-11-15 manager: T3 resync 3 -> last chance 35
- No win since resync 2 (T3-010..013 null/negative). Measured lead: T3-015 profile (job 142) attributes most of k=3's step excess to fixable
  implementation (GEMM tile config at M=56/64, DeltaLog 2-row cap). Last chance 35 GPU-min (asked 50). Pool free after: 288.3.

## 2026-11-15 manager: run PAUSED by the human
- Told leader-3 and leader-4 (and through them their workers): no new GPU jobs, let the running job finish and record it, cancel waiting jobs, go idle.
- No grants, closures or openings while paused. Waiting for the human to resume.
- Cancelled waiting job 145 (T4 worker); job 146 (T3 worker) was already cancelled by its team. Job 143 (T3) left running to finish.

## 2026-11-15 manager: run ENDING (human): finalize()
- Asked leader-3 and leader-4 for write-ups and last map updates (no GPU jobs). Then: close T3/T4, stop all agents, reports/final.md. No new teams.
- T3 closed (run end); 54.3 GPU-min returned.
- T4 closed (run end); 95.5 returned. All agents stopped. reports/final.md written. Run finalized: champion T3-009 951.5 tok/s (+49.6%), 561.9 of 1000 GPU-min used.
- T4 closed (run end); 95.5 returned. All agents stopped. reports/final.md written. Run finalized: champion T3-009 951.5 tok/s (+49.6%), 561.9 of 1000 GPU-min used.
