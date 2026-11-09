# Telling the cache what to keep

**What it is.** A program can influence the second-level cache instead of leaving it to the default policy. Up to 49.5 MiB can be set aside for data marked to persist. A window of memory (up to 128 MiB) is marked per stream or per graph node, with a ratio that says what fraction of accesses in it get the persisting treatment; accesses can also be marked as streaming (use once, evict first). At the instruction level, a load or store can carry an eviction hint (evict first, evict last, do not cache), and there are instructions to prefetch into the cache, to change the priority of a range, and to discard a range without writing it back.

**What it makes possible.** The program knows which data it will need again and which it will not; the default policy does not. These controls let it pass that knowledge on.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes). The hint instructions were assembled for this chip; not run. The set-aside cannot be changed while the multi-process service is in use.

**Where to read more.** CUDA Programming Guide: L2 Cache Control; PTX ISA reference: `prefetch`, `createpolicy`, `applypriority`, `discard`.
