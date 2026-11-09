# How the host learns that the GPU is done

**What it is.** When a host thread waits for GPU work, it can spin on a core, yield its core between checks, or sleep until an interrupt arrives. These differ in how fast the thread reacts and how much processor time it burns. Besides waiting, the host can poll an event without blocking, or attach a function that runs when a stream reaches a given point. The device also reports stream operations that write a value to memory or wait until memory holds a value.

**What it makes possible.** The gap between the GPU finishing and the host acting on it is set by the waiting method, and it is paid at every hand-over.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes). The waiting modes are from the documentation.

**Where to read more.** CUDA Programming Guide: Asynchronous Execution; driver API: context scheduling flags, stream memory operations.
