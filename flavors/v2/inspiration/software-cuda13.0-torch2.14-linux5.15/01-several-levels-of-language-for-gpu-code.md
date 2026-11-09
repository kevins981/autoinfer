# Several levels of language for GPU code

**What it is.** GPU code can be written at different levels, and several are installed here. At thread level: CUDA C++ (compiler 13.0) and Numba (Python compiled to the same thing). At block level: Triton, where one program describes a whole thread block and the compiler arranges the threads. At tile level: NVIDIA's CuTe DSL (the building blocks of its matrix-kernel library, usable from Python) and cuTile (kernels written as operations on whole tiles).

**What it makes possible.** The level sets what the author controls and what the compiler decides. The same kernel can be cheap to write at one level and only reach full speed at another, and pieces written at different levels can live in one program.

**How this was checked.** All import in this environment. cuTile's compiler was not run on this chip.

**Where to read more.** CUDA Programming Guide: Writing SIMT Kernels, Writing Tile Kernels; the Triton, Numba, CuTe DSL and cuTile documentation.
