team:         T1
builds_on:    T2-003 + T1-001
performance:  +0.75% vs T2-003 (707.0; not the champion any more, T2-004 743.9): 712.3 tok/s, starts 705.2 / 719.4
              (eval/1, GPU job 20). Within noise: start 1 is below T2-003's mean.
summary:
  hypothesis: T1-001's fp8 draft lm_head (+6.7% on BASE) stacks on T2-003 (prefill graphs, batch-sharded sampling,
              MTP local argmax).
  change:     flags only: T2-003's serve args + "draft_lm_head_quantization":"fp8" in the same speculative-config
              + --kv-cache-memory 4563402752 (4.25 GiB instead of 5 GiB), needed because fp8 head + prefill graphs
              OOM with 5 GiB: job 11 (graphs 2048) OOMs during spec-prefill graph capture (all-reduce output alloc);
              job 16 (graphs 1024) starts and serves, then OOMs in GDN chunk prefill. Neither had T2-003's sampling flags.
  results:    Observed: 712.3 tok/s (705.2, 719.4) vs T2-003 708.9/705.1; acceptance 2.337 vs 2.349; TTFT p99 1215 ms,
              TPOT p99 42.7 ms (gate_latency pass); 128K request pass at 4.25 GiB KV (~228K-token capacity);
              vllm:num_preemptions_total 0 -> 1 in each start (metrics_after.txt; T2-003: 0). Server logs show no preemption warning.
  conclusion: Interpretation (corrected after review T1-2): adding the fp8 draft head with the reduced KV budget to T2-003
              produced no resolved throughput gain (observed +0.75%, CV 1.4%, start 1 below T2-003's mean). The
              interaction mechanism is not established: T1-001's fp8 head still produces bf16 logits and uses the full
              gather path, so its BASE gain is not simply "the gather"; a matched-KV ablation with head/comm timings
              would be needed. The KV trim costs one preemption per start. Not pursued further on T2-004.
validity:     ruling 1 (fp8 draft head), verification exact. KV budget change tested with the 128K request (pass, capacity 227,284 tokens); 1 preemption per start.
              Reviewed: reviews/T1-2 (codex); corrections adopted.
