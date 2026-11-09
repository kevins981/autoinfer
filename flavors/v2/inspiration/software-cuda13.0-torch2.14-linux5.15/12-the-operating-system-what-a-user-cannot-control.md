# The operating system: what a user cannot control

**What it is.** Some conditions are fixed for an ordinary user. The processor frequency policy raises a core's speed only after load appears on it. The system may move a process's memory pages between the two halves of the host on its own. No cores are reserved: any thread can be interrupted or moved to make room for another, and real-time priority is not available. Other people use the machine. The driver version (580) caps the CUDA version at 13.0. The kernel is Linux 5.15.

**What it makes possible.** These are the conditions any design has to work under. A design that depends on a quiet, pinned, fully warmed-up host will not always get one.

**How this was checked.** Read from `/sys` and `/proc` on this machine.

**Where to read more.** Linux kernel documentation: CPU frequency scaling, NUMA balancing, scheduler.
