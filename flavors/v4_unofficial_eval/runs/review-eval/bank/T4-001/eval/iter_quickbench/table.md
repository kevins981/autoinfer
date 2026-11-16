# Iteration logs (--enable-logging-iteration-details), 200 prompts c=16, no nsys. ON=max 512, OFF=0 (job rp-iter).

**What elapsed_ms measures (core.py `capture_iteration_details`, ~line 631-675):** monotonic time inside `with capture_iteration_details(...)` which wraps only `_wait_for_output(future)` (blocking wait for the worker's model output) in `step()` / `step_with_batch_queue()`. It excludes scheduling, the execute_model launch, `update_from_output` and output processing. With async scheduling / batch queue (size 2) the core has already scheduled and launched step N+1 before blocking on step N, so elapsed = remaining time of step N when the core starts waiting; it is not the step period, and it can be shortened by core-side CPU work happening before the wait. Summed over iterations vs bench duration below tells how much of the wall time the core spends blocked.

Totals: ON 1535 iterations, sum elapsed 60.0 s; OFF 1518 iterations, 58.9 s. Bench wall: ON 58.7 s (784.2 tok/s), OFF 57.4 s (795.6 tok/s); this includes the server-side warmup/idle.

| kind | ON n | ON median ms | ON mean | ON total s | OFF n | OFF median | OFF mean | OFF total s | mean ON-OFF ms |
|--|--|--|--|--|--|--|--|--|--|
| decode D=16 (gen tok 48) | 772 | 31.6 | 31.5 | 24.35 | 764 | 31.0 | 31.1 | 23.74 | +0.5 |
| decode D<16 | 575 | 29.5 | 27.0 | 15.52 | 572 | 28.7 | 26.5 | 15.14 | +0.5 |
| mixed p<128 | 80 | 55.9 | 58.4 | 4.67 | 76 | 65.5 | 66.7 | 5.07 | -8.3 |
| mixed 128-256 | 26 | 80.1 | 83.0 | 2.16 | 24 | 77.4 | 80.1 | 1.92 | +2.9 |
| mixed 256-512 | 41 | 130.4 | 130.9 | 5.37 | 41 | 119.4 | 120.0 | 4.92 | +11.0 |
| mixed >=512 | 38 | 209.8 | 206.3 | 7.84 | 38 | 194.3 | 207.7 | 7.89 | -1.5 |
| prefill-only (no decode) | 3 | 37.1 | 37.0 | 0.11 | 3 | 52.5 | 58.8 | 0.18 | -21.8 |