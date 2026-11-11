team:         T14
builds_on:    T14-003
performance:  "unofficial same-job ABBA, 300 prompts c=16: A 927.5 (928.9, 926.1), B 935.7 (938.7, 932.7): +0.9%, every B above every A"
summary:
  hypothesis: as T14-003; the earlier ABBA lost its 4th start to the job timeout.
  change:     same change.diff (ws leader-14.m1, policy default on). A = ws leader-14.champ (= T9-005, own build), B = leader-14.m1 with AI_POLICY unset. T9-005 serve_args. Job 120, 10.4 GPU-min, order A B B A, fresh start each, 300 prompts (not 400, to fit all 4 starts), no gsm8k; stray-process check logged before each start; each arm's _C printed from its own workspace (job120.log).
    Why T14-003's first start took 309 s: the first start in that job had to compile the host/dma-staged-AR and lmh_codec JIT extensions for a new workspace (~40 s each phase, in server.log 07:31:33->07:32:56) and read the 29 GB weights from cold page cache (model loading 59.7 s vs 10 s warm). The JIT caches (.cache/torch_extensions, keyed by source hash) were warm for both arms afterwards, and the weights were page-cached (cat of the safetensors took 9 s) before this job; starts took 150-160 s.
  results:    Observed (tok/s / p99 TTFT ms / p99 TPOT ms / median TTFT ms / MTP acceptance):
    1A  928.9 /  871 / 34.5 / 162 / 2.338
    2B  938.7 / 1115 / 39.2 / 175 / 2.332
    3B  932.7 / 1152 / 38.3 / 178 / 2.332
    4A  926.1 / 1121 / 38.7 / 165 / 2.333
    A mean 927.5 (spread 2.8), B mean 935.7 (spread 6.0), B-A +8.2 tok/s = +0.9%. min B 932.7 > max A 928.9 (gap 3.8).
    Combined with T14-003 (400 prompts, different job): B above A there too (+0.2%, +0.5%, one A).
  conclusion: Interpretation: consistent small gain from the policy, ~+0.9% in this job (T14-003: +0.2..0.5% vs one A), same sign in both jobs and every B above every A here; 2 starts per arm so this is suggestive, not proof. The size is larger than the in-situ estimate (-0.6 s Marlin cp / trial of ~100+ s -> ~+0.5%), within start-to-start noise. p99 TTFT/TPOT show no consistent change (noisy). Not comparable to official 400-prompt numbers (300 prompts here).
validity:     as T14-003 (default-on policy, bitwise at M<=48, <=1 ulp elsewhere, min_workspace_size check only loosened, no memory change).

official #1 (2026-11-09T07:59): FAILED at apply_diff (official/1/prepare.log)

official #2 (2026-11-09T08:12): 956.6 tok/s at c=16 (mean of 2 trials: 955.8 / 957.5, CV 0.1%) (+50.4% vs BASE), GSM8K 94.0%, MTP acceptance 2.33, TTFT p50 173 ms, TPOT p50 15.1 ms; gates: GSM8K pass, latency pass, 128K request pass (official/2/results.json)
