# Unused structure

**The principle.** The cost of work depends on how much the method knows about regularity in its inputs. Data or mathematics with regularity (many zeros, symmetry, sorted order, repeated patterns, few independent directions, operations that may be regrouped) contains less than its size suggests. A method that knows the regularity can do proportionally less.

**The waste.** Work in proportion to the raw size of the problem, when its real content is smaller.

**General directions.**
- **Look for regularity:** measure it; don't assume it is absent.
- **Skip what contributes nothing:** parts known to be zero, unchanged or irrelevant.
- **Do one of a kind:** when parts are symmetric or repeated, do one and reuse it.
- **Use the laws of the mathematics:** regroup, factor or reorder to reach a cheaper equivalent form.
- **Choose a method whose cost follows the content:** not the size.

**Illustrations** (examples only; the best use of this principle may look like none of them). Sorted order lets you find a name in a phone book in a few steps. Multiplying by a table that is mostly zeros needs only the non-zero entries. Gauss summed 1 to 100 by pairing the ends.
