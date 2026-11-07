# Search policy (v1 default)

How the manager decides what to work on next, given everything so far. The manager reads this at
every `brainstorm()`. It is an isolated variable: a run can override it, and the human can edit it
mid-run. The run keeps a copy of the text used in each round.

1. If a leader's work this round produced a new champion, that leader's team continues next round
   with the single next step it proposed in its report (keep pushing what works; it knows the line
   best). That slot is taken before the brainstorm ideas are selected.
2. Fill the remaining slots from the brainstorm ideas by upside and novelty: most by upside, at
   least one slot for a high-novelty idea (protected exploration; don't reject it on napkin math
   alone).
3. If nothing improved this round, select purely by upside and novelty.
