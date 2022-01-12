#!/bin/bash

make -j -C ../build

for i in {1..22}
do
	./run_single.sh $i "ptxas" "--backend=ptxas"
done
