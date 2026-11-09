# Inspiration cards

Short cards that give leaders a different angle to think from. The `inspire` tool (to be written)
prints a few of them; the manager attaches a draw when a team opens, a leader can draw at any
time, and the manager may offer a draw at a last chance.

- Spec: [../design/inspiration.md](../design/inspiration.md).
- `principles/`: general principles of where systems lose time. One card per file. Nothing in
  them is specific to inference, to a task, or to an earlier run.

## Card format (principles)
- **The principle:** the way of thinking, stated generally.
- **The waste:** what is lost, in one sentence.
- **General directions:** what follows from the principle. Directions, not techniques.
- **Illustrations:** a few examples from other fields, marked as examples only.

The 21 principles draw on the kinds of waste in lean manufacturing, on Park et al.,
"Principles and Methodologies for Serial Performance Optimization" (OSDI 2025), and on TRIZ
(the trade-offs card).

## Platform cards
Facts and capabilities of one machine and one software stack. The folder name says which.
- `platform-supermicro-741ge-2x-rtx4090/`: hardware (GPU 01-19, communication 20-25, host 26-28).
- `software-cuda13.0-torch2.14-linux5.15/`: software (13 cards).

Format: **What it is**, **What it makes possible** (general, never a proposed use), **How this
was checked**, **Where to read more**. No measurements of the engine and nothing from earlier
runs. A different machine needs its own folders.
