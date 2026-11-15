# Simple-protocol NCCL tuning sweep; bench.py copied from leader-2.w1 (T2), credit theirs.
source $AUTOINFER_RUN/task/env.sh
cd $(dirname $0); rm -f nccl.jsonl
run(){ label=$1; shift; env "$@" BPORT=$((29600+RANDOM%300)) timeout 60 python bench.py nccl "$label" nccl.jsonl 2>&1 | grep -iE "error|fail" | head -3; }
run S NCCL_PROTO=Simple
for n in 1 4 8 16; do run "S_nch$n" NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=$n NCCL_MAX_NCHANNELS=$n; done
for t in 128 256 512; do run "S_thr$t" NCCL_PROTO=Simple NCCL_NTHREADS=$t; done
for b in 1048576 524288 8388608; do run "S_buf$b" NCCL_PROTO=Simple NCCL_BUFFSIZE=$b; done
run S_nch8_buf1M NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_BUFFSIZE=1048576
run S_nch4_thr256 NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_NTHREADS=256
run S_again NCCL_PROTO=Simple
cat nccl.jsonl
