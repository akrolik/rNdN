#!/bin/bash

make -C ../build

DATA_PATH="/mnt/local/alex/data/tpc-h/"

for i in {1..22}
do
	echo "Query $i"
	echo "===================================="

	../build/bin/r3d3 --debug-time --backend=r3d3 --data-load-tpch --data-path-tpch="$DATA_PATH" "../tests/tpch/q${i}.hir" | tee "q${i}_r3d3.log"

	echo
done
