# The framework underneath

**What it is.** The framework keeps its own pool of GPU memory and hands out pieces without asking the driver each time; the pool's behaviour can be tuned and inspected. New operations can be written in C++ or CUDA and loaded as extensions, built ahead of time or when the program starts. The framework exposes streams, events, pinned host memory and non-blocking copies. Each operation passes through a dispatch layer on the host before any GPU work is issued.

**What it makes possible.** The framework's defaults are one arrangement among several it supports, and anything it lacks can be added beside it.

**How this was checked.** From the installed framework (torch 2.14); not exercised here.

**Where to read more.** PyTorch documentation: CUDA semantics, memory management, C++ extensions.
