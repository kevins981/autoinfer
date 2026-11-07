# IDEAS (run test-4): scored but untried. An archive, not a queue.
| round | id | idea | by | N | U | why not selected |
|---|---|---|---|---|---|---|
| 1 | R1-C | Marlin tile configs / split-K / fuse GEMMs sharing input at M<=64 (brainstorm/r1/systematic_engineer.md #4) | systematic_engineer | 3 | 2 | AF measured L2-hot Marlin near its floor (43-48 us vs 53-60 in-server); lower confidence than A/B |
| 1 | R1-D | L2 prefetch of next layer's weights during AR (adjacent_field.md #4) | adjacent_field | 4 | 1 | measured ceiling ~+3% |
| 1 | R1-E | NCCL env tuning (protocol/channels) | both (negative) | 1 | 1 | measured no gain at decode sizes; prefill-size check given to T1 |
