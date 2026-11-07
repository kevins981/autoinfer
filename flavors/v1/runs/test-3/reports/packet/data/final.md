# Final report (run test-3, stopped after round 3 at the human checkpoint)
- Champion: **T1-009, 730.9 tok/s official (+15.8% vs BASE 631.4)**; gates and 128K pass.
- Best measured: **T1-012, 749.7 (+18.7% vs BASE, +2.6% vs T1-009)**: valid, exact, gates pass, but below the 3% bar. Recommended for promotion by the human's decision; preemption/resume untested.
- Path: host-staged AR (T1-002) → + host all-gather (T1-004) → + prefill graphs (T1-005, r1 champion) → logits diet (T1-006/007/009, r2 champion) → GDN state replay (T1-012, r3).
- What was learned: shared_kb.md KB1-KB17 (incl. GPU 0's locked clock, KB8; the KV/memory ceiling, KB16).
- Open: R3-B v2 (smaller GDN spec state → k=3 at c=16), lossless AR compression, L2 prefetch, preemption test for T1-012. See ideas.md.
- Handoff packet: reports/packet/ (open index.html; also reports/packet.zip). Rebuild with `python3 reports/packet_src/build_packet.py` from the run folder.
