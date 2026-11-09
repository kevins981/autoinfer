# Ready-made math libraries

**What it is.** NVIDIA's tuned libraries are installed: dense linear algebra (including a lightweight matrix-multiply interface with many formats, and batched forms), sparse linear algebra, matrix multiply with one operand pruned in the fixed pattern the matrix hardware supports, solvers, Fourier transforms, random numbers, the standard deep-learning operations, image and JPEG routines, and reading files directly into GPU memory.

**What it makes possible.** Some of these implement an operation in a way nobody would write by hand, and some expose hardware modes that are otherwise awkward to reach.

**How this was checked.** The packages are installed; which code paths each takes on this chip was not checked.

**Where to read more.** cuBLAS, cuSPARSE, cuSPARSELt, cuSOLVER, cuFFT, cuRAND, cuDNN, NPP, nvJPEG and cuFile documentation.
