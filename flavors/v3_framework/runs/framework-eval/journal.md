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
