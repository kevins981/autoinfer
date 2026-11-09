# The processor layout

**What it is.** Each GPU has 128 identical processor blocks (SMs). One SM holds up to 1,536 threads (48 groups of 32, called warps) from up to 24 thread blocks. It has 65,536 registers to divide among them (at most 255 per thread) and 100 KB of fast scratch memory (shared memory): 48 KB per thread block by default, up to 99 KB on request. The scratch memory, the first-level cache and the texture cache share one 128 KB pool, and the split can be chosen per kernel.

**What it makes possible.** Registers, scratch memory and thread count trade against each other inside one SM: a kernel that uses fewer registers per thread fits more threads, and one that asks for more scratch memory leaves less first-level cache. A kernel does not have to occupy all 128 SMs.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Compute Capabilities; Ada Tuning Guide.
