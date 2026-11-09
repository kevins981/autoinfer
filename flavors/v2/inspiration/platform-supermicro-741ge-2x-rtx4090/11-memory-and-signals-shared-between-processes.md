# Memory and signals shared between processes

**What it is.** GPU memory allocated in one process can be opened in another, which then reads and writes the same bytes without a copy. Events can be shared the same way, so one process can wait for work another process started. Memory pools give allocation and release that are ordered with the work on a stream, and a pool can be created so that its memory is exportable.

**What it makes possible.** Work on one GPU can be split across processes without paying for copies between them.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Interprocess Communication; Stream-Ordered Memory Allocator.
