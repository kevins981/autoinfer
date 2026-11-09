# Control over the GPU's address space

**What it is.** A program can manage GPU memory the way an operating system manages virtual memory: reserve a range of addresses without backing it, create physical memory separately, and map, unmap or remap physical memory into the range. Physical memory can be exported to another process as a file descriptor. Sparse arrays, whose backing is supplied later piece by piece, are supported.

**What it makes possible.** A buffer can grow without being copied, memory can change hands without moving, and address layout can be chosen for convenience rather than dictated by the order of allocations.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Virtual Memory Management.
