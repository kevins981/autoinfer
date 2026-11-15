team:         T2
builds_on:    T1-006 + T2-005 (T2-005's humming change applied to T1-006, which itself builds on T2-004)
performance:  unofficial: 738.8 / 747.8 tok/s (mean 743.3) vs champion T2-005 715.7 / 720.2 (+3.5%) and T1-006 676.6 / 680.7 (+9.5%)
summary:
  hypothesis: T2-005's W8A8 (humming) and T1-006's target-side distributed argmax act on different parts of the step (GEMMs vs logits gather) and stack.
  change:     T1-006's code + T2-005's 3 humming changes (vllm/envs.py default activation-quant config, vllm/utils/humming.py NVRTC lib path, humming.py is_layer_skipped whole-component fix); serve args = T1-006's + --quantization humming. Built by leader-2.w1 (workspace t2-hm3).
  results:    Observed: unofficial_eval --ws t2-hm3 --gsm8k --long: 738.8 / 747.8 tok/s, GSM8K-250 93.2%, 128K pass, p99 TTFT 940 ms / TPOT 37.1 ms, MTP acceptance 2.302.
  conclusion: Interpretation: the two changes stack roughly additively (T2-005 vs T2-004 +8.4%; T1-006 vs T2-004 +2.5%; here vs T2-004 +12.3%).
validity:     leader-2 review: union of two reviewed changes (T2-005's validity review applies to the humming part; T1-006's to the argmax part, reviewed and officially confirmed by T1/manager). The diff touches only those files; GSM8K-250 93.2% passes; 128K pass; latencies well inside the gates.
unofficial #1 (copied from unofficial/ws/t2-hm3/1): 738.8 / 747.8 tok/s (200 prompts); GSM8K 93.2%, 128K pass, p99 TTFT 940 ms / TPOT 37.1 ms (BASE official 1341 / 46.1), MTP acceptance 2.302

official #1 (2026-11-15T17:14): 761.6 tok/s at c=16 (mean of 2 trials: 774.9 / 748.3, CV 2.5%) (+19.8% vs BASE), GSM8K 93.6%, MTP acceptance 2.3, TTFT p50 177 ms, TPOT p50 19.3 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
