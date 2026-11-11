# T44 write-up (research, leader-44)
Material: bank/T44-001..004, proposals/T44-1.md, proposals/T44-2.md, local_kb/leader-44.md, sparring T44-1/T44-2. GPU used ~1.8 of 35.

Stands: KB115 (SM PCIe duplex when rate-matched), KB116 (hostar chain-latency-bound; pacing no gain), KB117 superseded (T44-003 attribution wrong: events after D2H mark issue, not landing).
Open note (leader): at 300 rows the split AR takes ~208 us vs ~130 us pure link time (3.07 MB up at ~24 GB/s); the ~70 us is not decomposed with landing-time measurements. A future team should measure host-side arrival (e.g. CPU polling the slot) before forecasting anything there.
Process lesson: event timelines around posted writes mislead; test an attribution with a cheap intervention before forecasting from it.
