# Sharing without copying

**What it is.** Several mechanisms let two parties use the same bytes. Host processes can share memory by name (63 GB available in the memory-backed file system). GPU memory can be shared between processes. Libraries in one process can hand each other GPU arrays without a copy through a standard exchange format (DLPack) or a common array interface. Message passing between processes is available with zero-copy for large messages.

**What it makes possible.** A boundary between processes or between libraries does not have to cost a copy.

**How this was checked.** The mechanisms are present in the installed software; not exercised here.

**Where to read more.** DLPack; PyTorch `torch.multiprocessing`; CUDA Programming Guide: Interprocess Communication; ZeroMQ guide.
