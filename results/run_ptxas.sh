#!/bin/bash

make -j -C ../build

for i in {1..22}
do
	echo "Query $i"
	echo "===================================="

	./run_single.sh $i "ptxas" "--backend=ptxas"
done
