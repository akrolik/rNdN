#!/bin/bash

make -C ../build

for i in {1..22}
do
	./run_single.sh $i "--backend=ptxas"
done
