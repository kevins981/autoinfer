source $AUTOINFER_RUN/workspaces/leader-1.t2001/env.sh
cd $AUTOINFER_RUN/agents/leader-1/dbo
for P in Simple LL; do echo "== NCCL_PROTO=$P"; NCCL_PROTO=$P torchrun --nproc-per-node 2 dbo_bench.py 2>&1 | grep -v Warning | tail -20; done
