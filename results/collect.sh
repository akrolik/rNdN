#!/bin/bash

FILE=$1

function collect {
	echo -n "$1: "
	grep "$1:" $FILE | sed -e "s/[^:]*: //" | cut -d ' ' -f 1 | awk '{s+=$1} END {print s}'
}

function collect_1 {
	echo -n "$1: "
	grep "$1:" $FILE | awk 'END {print}' | sed -e "s/[^:]*: //" | cut -d ' ' -f 1
}


# Compilation

echo "Compile"
echo "--------------------"

collect_1 "Frontend"
collect_1 "Outliner"
collect_1 "Codegen"
collect_1 "CUDA assembly"

echo -n "[TOTAL] "
collect_1 "Compilation"

# Data

echo
echo "Data Cache"
echo "--------------------"

collect "Transfer 'tpch_.*' to GPU"

echo
echo "Data Intermediate"
echo "--------------------"

collect "Transfer to GPU"
collect "Transfer to CPU"
collect "Resize buffers"
collect "Initialize buffer"

# collect "CUDA transfer (.*) ->"
# collect "CUDA transfer (.*) <-"

# echo

# collect "CUDA allocation (.*)"
# collect "CUDA copy (.*)"
# collect "CUDA clear (.*)"
# collect "CUDA free (.*)"

echo
echo "Data Output"
echo "--------------------"

collect_1 "Output collection"

# Execution

echo
echo "Execution"
echo "--------------------"

collect "Builtin function '.*'"
collect "GPU function '.*'"
collect "Library function '.*'"

echo -en "\t"; collect "Library function 'order_lib'"
echo -en "\t"; collect "Library function 'group_lib'"
echo -en "\t"; collect "Library function 'loop_join_lib'"
echo -en "\t"; collect "Library function 'hash_join_lib'"

collect "Kernel '.*' execution"
collect "Runtime analysis"

# collect "CPU allocation (.*)"
# collect "CPU free (.*)"

echo -n "[TOTAL] "
collect_1 "Execution"
