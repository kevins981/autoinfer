# T9 write-up (leader-9; opened 07:08 UTC, run ends 08:15; no worker started)
## Result
- T9-001: reduced-vocab fp8 drafter head (ids [0,131039) by tokenizer id order + 33 special tokens, rows re-sharded evenly over TP),
  on T5-001: 908.0 tok/s (912.7/902.5/908.8; jobs 126, 127), +1.41%, acceptance 2.335 vs 2.343, 128K pass. Submitted as champion candidate.
## What I learned
1. Re-test old negatives when the cost balance moved: T1-002 lost on BASE (K=32K, acceptance -4.6%). On the fp8-head champion the
   draft head was still 6% of the decode step, and a larger K (2^17) keeps acceptance nearly unchanged (-0.3%).
2. A vocab subset under TP must be re-sharded evenly: a prefix of the global vocab leaves rank 0's shard unchanged (no gain on the
   critical path), and replication at K=131K would read more bytes than today's sharded head.
3. Choose such parameters before measuring (one K), or the choice is fitted to the benchmark.
4. History search before GPU work saved a wasted run: k=3 on the champion was already measured (T7-001, neutral).
## Left for others
- Mixed-step host launch overhead in the GDN prefill path (update-1.md; T8-002's profile): the largest open gap.
- Free the full fp8 draft head copy when the reduced head serves all greedy drafts (memory only, ~0.64 GB/GPU) -> KV room for k=3.
- Map files: T9-sweep.md (1-6 compressed, unreviewed: review backend unavailable).
Status: T9-001 promoted to champion (manager announcement).
