# Host memory as common ground

**What it is.** Host memory can be pinned (locked in place) and registered with the driver, after which either GPU can copy to and from it directly. The same pinned buffer can serve both GPUs. Memory that already exists, including memory shared between processes, can be registered after the fact, and can be registered read-only. Pinned host memory can also be mapped into a GPU's address space, so a kernel reads or writes it directly, each access crossing the link.

**What it makes possible.** The one place both GPUs can reach is host memory. How that memory is laid out, who writes it and who reads it is open to design.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Unified and System Memory (page-locked host memory).
