# Several processes on one GPU

**What it is.** By default, processes that use the same GPU take turns: the GPU switches between them in time slices, and only one runs at any instant. The multi-process service (MPS) funnels several processes into one shared context so their kernels can run at the same time, and can cap each process's share of the GPU. An ordinary user can start the service for their own processes; its control program is installed. It is not running now, and other people use this machine.

**What it makes possible.** Whether work split across processes overlaps or alternates on the GPU is a choice, not a fixed property.

**How this was checked.** The device reports the default sharing mode and that MPS is off. Starting MPS was not tried.

**Where to read more.** NVIDIA Multi-Process Service documentation.
