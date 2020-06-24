#!/bin/bash

make -C ../build

for i in $(seq 1 22)
do
	echo "Query $i"
	echo "===================================="

	../build/bin/r3d3 --load-tpch --print-time "../tests/tpch/q${i}.hir" | tee "q${i}.log"
	# for j in $(seq 1 10)
	# do
	# 	../build/bin/r3d3 --load-tpch --print-time "../tests/tpch/q${i}.hir" | tee "q${i}_${j}.log"
	# done
	echo
done
