# Memory that lives on both sides

**What it is.** Unified (managed) memory is one address range valid on the host and on the GPU. Pages move to whichever side touches them, and both sides can hold pages at the same time. A program can give hints: where a range should preferably live, that it is mostly read, or that it should be moved now ahead of use. A range may be larger than the GPU's free memory; pages are evicted to the host as needed. Ordinary pageable host memory cannot be read by the GPU directly on this system.

**What it makes possible.** Data need not have one fixed home, and the total can exceed what the GPU holds, at the price of page moves when the wrong side touches a page.

**How this was checked.** Queried from the device on this machine (CUDA driver API device attributes).

**Where to read more.** CUDA Programming Guide: Unified Memory.
