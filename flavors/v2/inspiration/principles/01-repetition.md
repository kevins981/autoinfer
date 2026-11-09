# Repetition

**The principle.** Any piece of work is a function: it takes inputs and produces an output. The output contains nothing that wasn't already determined by the inputs. So work is only necessary when the inputs are new. If the inputs were seen before, the output already exists somewhere. If part of the inputs is new, only the part of the output that depends on it needs to be made.

This holds for computing a value, and equally for moving or storing one: sending something to a place that already has it is the same waste.

**The waste.** The work done, minus the work the new information required.

**General directions.**
- **Remember:** keep an output so the work isn't done again. This trades memory for work.
- **Share:** produce an output once and let everyone who needs it use the same copy.
- **Update by difference:** when the inputs change a little, compute only what the change affects.
- **Do it ahead:** if the inputs are known long before the output is needed, the work can be done once, early.

**Illustrations** (examples only; the best use of this principle may look like none of them). Incremental compilation rebuilds only what a change touched. Rsync sends only the parts of a file the other side lacks. A spreadsheet recomputes only the cells that depend on the edited one.
