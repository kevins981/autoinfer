team:         T5
builds_on:    T1-042 (+ T3's env-gated CUDA-event timers, workspace t3-distill.t)
performance:  unofficial: k=3 + prefill-graph cap 1024 + KV 5.75 GiB = -1.0% vs champion (same job); cap 1024 alone -4.0%
summary:
  hypothesis: On the built champion stack, plain MTP k=3 (M=64 at 16 req) beats k=2 once 16 requests fit:
              estimated step +8% vs E +17.6% => +5-9% decode throughput (leader-5 napkin; KB20 null was on T1-015).
  change:     no serving-code change (timers inert unless AUTOINFER_TIMERS). Flags only for the probe:
              B = num_speculative_tokens 3, prefill_cudagraph_capture_token_sizes truncated at 1024 and
              max_prefill_cudagraph_capture_tokens 1024, --kv-cache-memory 6174015488 (5.75 GiB);
              A = champion flags; C = champion flags with the same cap-1024 grid.
  results:    Observed (job 28, 11.1 GPU-min, worker leader-5.w1; 150 ShareGPT prompts c=16 each, fresh server per run,
              order B1 A C B2; files agents/leader-5.w1/job1/):
              | run | tok/s | TTFT mean/p99 | TPOT mean/p99 | accept len |
              | B1 k=3 cap1024 KV5.75 | 865.4 | 275/1179 | 17.47/43.9 | 2.707 |
              | A champion            | 873.6 | 266/1206 | 16.48/37.7 | ~2.33 |
              | C k=2 cap1024         | 838.9 | 269/1294 | 16.40/38.6 | ~2.33 |
              | B2 k=3 cap1024 KV5.75 | 865.2 | 273/1328 | 17.23/42.5 | 2.707 |
              B/A 0.991/0.990; B/C 1.032/1.031; C/A 0.960. KV pool B 298,627 tok; 0 OOM; B holds 16 req (one sample 14+1 waiting,
              A also had one 15+1). 128K request on B2: pass (59.3 s), health 200 after.
              Median step at 16 req (ms): k=2 fwd 22.35 / sample 1.12 / propose 1.68 = 25.15; k=3 fwd 26.32 / 1.14 / 2.53 = 29.99 (+19.2%).
              k=3 fwd by M: 52 (13 req) 25.76, 56 25.8, 60 26.29, 64 26.32; k=2: 36 21.58, 42 21.76, 45 22.29, 48 22.35.
              Estimate vs measured: step +8% est vs +19.2% measured; E +17.6% est vs +16.2% measured.
  conclusion: Interpretation: plain k=3 does not pay as is (k=3 vs k=2 on the same grid is +3%, but the grid cap that makes room
              costs ~4%; single runs, noise at 150 prompts unquantified though B1/B2 agree within 0.03%). The cost is NOT per-row:
              k=3 at M=52 with 13 requests costs +3.4 ms over k=2 at M=48 with 16 requests => a step change at M>48 (the
              "cliff" T3 saw unbuilt is real on the built engine). Suspect Marlin thread_m_blocks 3->4 (T1-016's tile override
              covers only 2-3 m-blocks) or 4-token GDN/attention paths; being localized in T5-002. The cap-1024 grid's -4% says the
              memory for k=3 should come from GDN slots (T3 route B), not from the prefill graphs.
validity:     Probe only (no code change apart from inert timers). k=3 is exact spec decode; the graph cap and KV size are memory
              flags, covered by the 128K pass and no OOM. Not proposed for official eval.
