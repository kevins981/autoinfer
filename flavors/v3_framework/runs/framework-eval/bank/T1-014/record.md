team:         T1
builds_on:    T1-012 + T1-013
performance:  "unofficial: +2.94% vs T2-006 (ABBA 200 prompts, all B > all A: B 796.5/810.2 vs A 779.0/781.8); official #1: 842.7 tok/s (+2.8% vs T2-006 819.9, +32.5% vs BASE), all gates pass"
summary:
  hypothesis: T1-012 (greedy local argmax, decode-side sampling comm, ~+1.2%) and T1-013 (row-split AR overlap in mixed steps,
              +1.14%) touch disjoint code and disjoint step parts -> ~+2.3% together.
  change:     T2-006 + T1-012's files (model_runner.py, sample/greedy_local.py) + T1-013's files (qwen_gdn_linear_attn.py,
              qwen3_next.py, qwen3_5/nvidia/model.py, rowsplit.py). Disjoint file sets; copied, imports checked.
  results:    Observed (agents/leader-1/jobs/bundle): acceptance B 2.342/2.353 vs A 2.343/2.350; TTFT/TPOT p99 B 977/930 / 35.3/37.8 vs A 830/965 / 38.6/37.5 ms; gsm8k 50 on B 90.0%; both paths logged active.
  conclusion: Interpretation: official +2.8% matches the unofficial +2.94% and the sum of the parts (+1.18% + 1.14%).
validity:     union of T1-012 and T1-013 reviews (exact up to fp reordering; generic keys: request sampling params, token count).

official #1 (2026-11-14T20:29): 842.7 tok/s at c=16 (mean of 2 trials: 841.6 / 843.8, CV 0.2%) (+32.5% vs BASE), GSM8K 93.6%, MTP acceptance 2.333, TTFT p50 156 ms, TPOT p50 17.6 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
