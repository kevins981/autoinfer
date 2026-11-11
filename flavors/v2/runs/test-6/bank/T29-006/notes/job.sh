#!/bin/bash
R=$AUTOINFER_RUN; source $R/workspaces/t29-v2/env.sh; cd $R/agents/leader-29.w1/fp
timeout 240 python probe.py 2>&1 | grep -v "^W1109\|deprecated"
