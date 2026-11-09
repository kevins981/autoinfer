# Graphics hardware a compute program can reach

**What it is.** The chip carries hardware built for graphics that compute code can also use. Texture units fetch from a 1-, 2- or 3-dimensional array and blend the neighbouring entries in hardware in a single fetch (the blend weights have limited precision), with built-in handling of out-of-range coordinates; arrays can be up to 131,072 wide in one dimension. The ray-tracing units search a hierarchy of bounding boxes in hardware; they are reached through a separate library.

**What it makes possible.** A table lookup with interpolation, or a search over spatial data, can be done by dedicated hardware that is otherwise idle in a compute program.

**How this was checked.** Texture limits queried from the device. The ray-tracing path was not checked on this machine.

**Where to read more.** CUDA Programming Guide: texture and surface memory; the OptiX documentation.
