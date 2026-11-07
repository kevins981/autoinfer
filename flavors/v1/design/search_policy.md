# Search policy (v1 default)

How the manager decides what to work on next, given everything so far. The manager reads this at
every `brainstorm()`. It is an isolated variable: a run can override it, and the human can edit it
mid-run. The run keeps a copy of the text used in each round.

1. Every leader whose work this round produced a record that beats the champion the round started
   with (officially, consistently, gates passing) continues next round with the single next step
   it proposed in its report (keep pushing what works; it knows the line best). This holds for
   each such leader, not only the one with the best result. These slots are taken before the
   brainstorm ideas are selected. If they take every slot, skip the brainstorm this round.
2. Fill the remaining slots from the brainstorm ideas by upside and novelty: most by upside, at
   least one slot for a high-novelty idea (protected exploration; don't reject it on napkin math
   alone).
3. If nothing improved this round, select purely by upside and novelty.
