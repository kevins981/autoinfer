# Generality

**The principle.** Work that can handle many cases has to find out which case it is in, and has to be built so that every case stays possible. The more that is known in advance about the inputs, the less there is to decide and the less work remains. Every fact that is fixed in this situation (a size, a type, a range, a condition that always holds) is a decision that could be made once, ahead of time, instead of every time.

**The waste.** Checks, indirection and cautious choices that exist for cases which never occur here.

**General directions.**
- **Fix what is known:** build a version for the case that actually occurs.
- **Remove decisions with one answer:** a check that always comes out the same isn't a check.
- **Use a narrower method:** one that only works under the conditions that hold here can be cheaper.
- **Keep the general version as a fallback:** the special version only needs to cover what it was built for.

**Illustrations** (examples only; the best use of this principle may look like none of them). A key fits one lock; a locksmith's toolkit opens any, slowly. A compiler replaces a calculation on constants with its result. A production line built for one product outruns a general workshop.
