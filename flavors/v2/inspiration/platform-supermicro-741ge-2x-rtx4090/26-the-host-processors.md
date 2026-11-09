# The host processors

**What it is.** Two Intel Xeon Silver 4410Y processors: 12 cores each (24 cores, 48 hardware threads), 2.0 GHz base and up to 3.9 GHz. Each core has 2 MiB of second-level cache; each processor has 30 MiB of third-level cache. They have 512-bit vector instructions, including 16-bit float and bfloat16 arithmetic and 8-bit integer dot products, and a matrix extension (AMX) that multiplies tiles of bfloat16 or 8-bit integers.

**What it makes possible.** The host is slow next to a GPU but far from negligible, it is mostly idle while the GPUs work, and it sits on the path between them.

**How this was checked.** Read from `lscpu`.

**Where to read more.** Intel's processor specification page; Intel intrinsics guide.
