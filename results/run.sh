#!/bin/bash

make -j -C ../build

for i in {1..22}
do
	echo "Query $i"
	echo "===================================="

	./run_single.sh $i "r3d3" ""
done
