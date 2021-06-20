#!/bin/bash

DATA_PATH="/mnt/local/alex/data/tpc-h/"

make -C ../build

for i in {1..22}
do
	echo "Query $i"
	echo "===================================="

	../build/bin/r3d3 --debug-time --backend=ptxas --data-load-tpch="$DATA_PATH" "../tests/tpch/q${i}.hir" | tee "q${i}_ptxas.log"

	echo 
done
