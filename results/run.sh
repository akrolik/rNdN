#!/bin/bash

make -j -C ../build

for i in {1..22}
do
	./run_single.sh $i "r3d3" ""
done
