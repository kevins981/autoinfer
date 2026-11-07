#!/bin/bash
source /scratch/kevinsong/learning/autoinfer/flavors/v1/runs/test-3/task/env.sh
D=$(dirname $0)
cd $D
run(){ CFG=$1 env "${@:2}" timeout 120 torchrun --nproc_per_node 2 --master_port 29611 ar_bench.py 2>&1 | grep -E "CFG|copy|Error|error" ; }
run default
run proto_simple NCCL_PROTO=Simple
run proto_ll128 NCCL_PROTO=LL128
run ch8 NCCL_MIN_NCHANNELS=8
run ch16 NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
run simple_ch8 NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=8
run shm_memcpy NCCL_SHM_USE_CUDA_MEMCPY=1
run shm_memcpy_simple NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_PROTO=Simple
NCCL_DEBUG=INFO CFG=dbg timeout 60 torchrun --nproc_per_node 2 --master_port 29612 ar_bench.py 2>&1 | grep -E "via|Channel 0|NCCL INFO.*(SHM|P2P|transport)" | head -8
