# A very large second-level cache

**What it is.** All 128 SMs of a GPU share one second-level cache of 72 MiB. That is large next to many of the arrays a program reads repeatedly.

**What it makes possible.** Data that fits in the cache and is read again soon does not have to come from main memory again. Whether it is still there depends on what else passed through the cache in between, which the order of work decides.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** Ada Tuning Guide; CUDA Programming Guide: L2 Cache Control.
