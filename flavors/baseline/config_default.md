# CONFIG (baseline defaults; copied into each run as config.md, where the human may edit it)
The run stops at whichever limit comes first (checked before each experiment or gpujob starts, so
the last job may run past it); set a limit to `none` to drop it. `experiment` and `gpujob` read
these lines (keep their format).

- max gpu hours: 4        (GPU time of all jobs: experiments + gpujobs)
- max experiments: none
- max hours: none         (wall clock; includes pauses)
- model: claude-opus-5-5 / medium   (read by new_run; the same as v1's manager)
