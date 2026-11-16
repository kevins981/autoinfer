#!/bin/bash
R=/scratch/kevinsong/learning/autoinfer/flavors/v4_unofficial_eval/runs/review-eval
eval "set -- $(tr '\n' ' ' < $R/bank/T5-001/serve_args.txt)"
TRACE=graph exec bash $R/task/scripts/profile_nsys.sh $R/workspaces/t8-champ $R/agents/leader-8/prof_graph 32 "$@"
