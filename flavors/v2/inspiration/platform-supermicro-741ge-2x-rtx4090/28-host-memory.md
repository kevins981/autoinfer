# Host memory

**What it is.** 125 GB of DDR5 memory across the two halves, shared with other users. The processors support 8 memory channels each at up to 4,000 MT/s; how many are populated was not checked. Host memory is slower to read than GPU memory (how much slower was not measured), but there is five times as much of it as one GPU has. A memory-backed file system (`/dev/shm`, 63 GB) lets processes share memory by name.

**What it makes possible.** The host has room to hold things the GPUs cannot, at the cost of the link to reach them.

**How this was checked.** Sizes read from `free` and `df`. Memory speed was not measured.

**Where to read more.** `lscpu`, `dmidecode` output where readable without root.
