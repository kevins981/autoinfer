#!/bin/bash
cd $(dirname $0)
taskset -c 0-11,24-35 ./duplex2
