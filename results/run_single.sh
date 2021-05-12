#!/bin/bash

DATA_PATH="/mnt/local/alex/data/tpc-h/"

echo "Query $1"
echo "===================================="

../build/bin/r3d3 --optimize-sass --debug-time --backend=r3d3 --backend-scheduler=list --data-load-tpch --data-path-tpch="$DATA_PATH" "../tests/tpch/q${1}.hir" | tee "q${1}_r3d3.log"
