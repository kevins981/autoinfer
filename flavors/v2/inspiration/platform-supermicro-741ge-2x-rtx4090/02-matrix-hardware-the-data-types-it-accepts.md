# Matrix hardware: the data types it accepts

**What it is.** The matrix-multiply units (tensor cores) work on small fixed tiles, one warp at a time. On this chip they accept: 16-bit float, bfloat16, a 19-bit float (TF32), 64-bit float, 8-bit and 4-bit integers, single bits (with XOR or AND followed by a bit count), and two 8-bit float formats (E4M3 and E5M2, which can be mixed in one product). The 8-bit floats accumulate in 32-bit float. Tile depth grows as the type shrinks: 16 for the 16-bit floats, 32 for 8-bit types, 64 for 4-bit, 256 for single bits. There is a sparse form in which one operand has half its values removed in a fixed pattern. Single instructions convert pairs of 32-bit floats to and from the 8-bit float formats.

**What it makes possible.** The same matrix hardware does several times more multiplies per instruction on narrower types, and narrower operands are also fewer bytes to fetch.

**How this was checked.** Each instruction was assembled for this chip (`ptxas -arch=sm_89`); not run.

**Where to read more.** PTX ISA reference: warp-level matrix instructions (`mma`, `mma.sp`), `cvt`.
