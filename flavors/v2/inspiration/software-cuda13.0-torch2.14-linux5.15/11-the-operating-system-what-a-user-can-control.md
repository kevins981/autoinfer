# The operating system: what a user can control

**What it is.** Without administrator rights a user can still: choose which cores a thread or process may run on; choose that a thread's memory is allocated on the half of the host where it first touches it; request large pages for a memory range (the system grants them on request, not by default); lock up to about 15.7 GB of memory so it is never paged out; lower a process's priority; create shared memory. A tool to set core affinity is installed; the usual tool for memory placement is not, though the underlying system calls are available.

**What it makes possible.** Where threads run and where memory sits is left to the system's defaults unless the program says otherwise.

**How this was checked.** Read from `ulimit`, `/sys/kernel/mm/transparent_hugepage/` and the installed tools.

**Where to read more.** `man` pages: `sched_setaffinity`, `mbind`, `madvise`, `mlock`, `taskset`.
