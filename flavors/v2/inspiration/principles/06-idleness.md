# Idleness

**The principle.** A system is a set of resources, each able to do work at some rate. A job is a set of pieces of work, where some pieces need the results of others. Two things set the minimum time for the job:
- **The longest chain of true dependencies:** pieces that really need each other's results must run one after another.
- **Total work divided by total capacity:** every resource working all the time can't go faster than this.

Any time beyond those two limits is waiting that nothing requires. It usually comes from the order in which the work was arranged, not from what the work needs.

**The waste.** A resource has free capacity, and a piece of work has all its inputs ready, but the work isn't running there.

**General directions.**
- **Separate true dependencies from arranged order:** find which waits are needed and which are only habit.
- **Run ready work on free capacity:** at the same time as other work, or overlapping with it.
- **Shorten the longest chain:** restructure the work so pieces depend on less, or on results that arrive earlier.
- **Move work toward free capacity:** a piece can run on a different resource, even one that does it more slowly, if that resource would otherwise wait.

**Illustrations** (examples only; the best use of this principle may look like none of them). A processor starts the next instruction before the current one finishes. A kitchen prepares side dishes while the main dish is in the oven. A program continues with other work while waiting for a disk read.
