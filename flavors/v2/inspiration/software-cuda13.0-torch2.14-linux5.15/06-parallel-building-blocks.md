# Parallel building blocks

**What it is.** A library of general parallel algorithms for the GPU is installed, in C++ and with a Python interface: sort, scan (running totals), reduce, select the items that meet a condition, partition, remove duplicates, histogram, run-length encode, search, merge.

**What it makes possible.** Problems that look inherently sequential, such as compacting a list or grouping equal items, have well-known parallel forms that are already written.

**How this was checked.** The package is installed (CCCL 13.3); not exercised here.

**Where to read more.** CCCL documentation (Thrust, CUB, `cuda.cccl`).
