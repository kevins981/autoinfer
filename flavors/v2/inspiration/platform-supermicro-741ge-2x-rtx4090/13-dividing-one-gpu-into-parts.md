# Dividing one GPU into parts

**What it is.** Green contexts give a program several lightweight contexts on one GPU, each owning a fixed subset of the SMs. On this chip family a part is at least 4 SMs, in multiples of 2. Work in one part does not take SMs from another. The documentation notes that disjoint parts are not guaranteed to run at the same time.

**What it makes possible.** Different kinds of work can each get a known share of the chip instead of contending for all of it.

**How this was checked.** From the documentation; not tried on this machine.

**Where to read more.** CUDA Programming Guide: Green Contexts.
