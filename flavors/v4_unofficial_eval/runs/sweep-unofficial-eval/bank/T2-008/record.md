team:         T2
builds_on:    T1-005
performance:  unofficial negative (1 start): -7.34% vs HEAD T1-005; latency gate failed (p99 TTFT 1930 ms)
summary:
  hypothesis: with the draft step now cheap (FP8 draft head + local argmax, T1-001) and faster ARs (T1-005), a 3rd MTP draft token pays: acceptance ~2.35 -> ~2.7 tokens/step (+15%, if p3 ≈ 0.35-0.4) for a step cost increase of ~+8-10% (one more draft step ~1 ms, 64 instead of 48 verify rows: AR payload +33%, GDN state writes 4 instead of 3 per layer, still one Marlin pass and below the ridge). Napkin +2..5% (reports/T2-sweep/4-knobs.md SP1).
  change:     serve args: HEAD's with num_speculative_tokens 3 (exact speculative decoding, same MTP head).
  results:    Observed: unofficial #1 negative (1 start): 675.1 vs ref 731.3/725.9 (-7.34%); MTP acceptance 2.724 (vs ~2.35 at k=2: +16%, as napkin predicted, p3 ≈ 0.37); p99 TTFT 1930 ms (gate: <= 1609), TPOT p99 37.3 ms. Graph capture 1.05 GiB, prefill graphs to 512, decode graphs to 16 requests.
  conclusion: Interpretation: acceptance rose as predicted but the step cost rose ~25% (napkin +8-10%): the per-step cost of the 4th verify row and 3rd draft step is larger than priced (likely GDN 4-state writes + drafter steps + mixed steps carrying 64 decode rows); TTFT p99 regression also rules it out as is. Not pursued further; would need KR1 (cheaper per-row state) to revisit. Screen negative (1 start).
validity:     Exact spec decode (verification against the target unchanged); draft count is open per the task. Memory: more verify rows, larger decode graphs and GDN per-row state slots (4 per request) -> --long.

unofficial #1 (2026-11-15T06:51): negative (1 start): -7.34% vs HEAD T1-005 (B 675.1 vs reference mean 728.6 over 2 starts, 0 run in this job; 200 prompts); GSM8K/128K not run (stopped after 1 start), p99 TTFT 1930 ms / TPOT 37.3 ms, MTP acceptance 2.724; gates failed: latency p99_ttft_ms 1929 > 120% of BASE 1341 (unofficial/1/unofficial.json)
