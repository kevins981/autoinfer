#!/bin/bash
cd $(dirname $0)
nvidia-smi --query-gpu=index,clocks.sm,power.draw,power.limit,temperature.gpu --format=csv -lms 200 > smi1.csv &
SMI=$!
CUDA_VISIBLE_DEVICES=0 ./copipe
CUDA_VISIBLE_DEVICES=1 ./copipe
kill $SMI
sort -t, -k2 -n smi1.csv | tail -3; awk -F, 'NR>1{print $1, $2, $3}' smi1.csv | sort | uniq -c | sort -rn | head -20
