#!/bin/bash
source $AUTOINFER_RUN/task/env.sh
cd $(dirname $0)
run(){ echo "== $*"; env "$@" timeout 50 torchrun --nproc_per_node 2 --master_port 29555 arbench.py 48 2>&1 | grep -E "RESULT|Error|error" | head -5; }
run A=1
run NCCL_PROTO=LL
run NCCL_PROTO=LL128
run NCCL_PROTO=Simple
run NCCL_ALGO=Ring
run NCCL_ALGO=Tree
run NCCL_PROTO=LL NCCL_NTHREADS=64
run NCCL_PROTO=LL NCCL_NTHREADS=512
run NCCL_PROTO=LL NCCL_BUFFSIZE=4194304
run NCCL_P2P_DISABLE=1
run NCCL_SHM_USE_CUDA_MEMCPY=1
run NCCL_MAX_NCHANNELS=1
run NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
NCCL_DEBUG=INFO timeout 50 env torchrun --nproc_per_node 2 --master_port 29555 arbench.py 48 2>&1 | grep -E "Using|via |Channel 00|NET|SHM|P2P" | head -20
