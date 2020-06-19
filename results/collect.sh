#!/bin/bash

FILE=$1

function collect {
	echo -n "$1: "
	grep "$1:" $FILE | sed -e "s/[^:]*: //" | cut -d ' ' -f 1 | awk '{s+=$1} END {print s}'
}

function collect_1 {
	echo -n "$1: "
	grep "$1:" $FILE | sed -e "s/[^:]*: //" | cut -d ' ' -f 1
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
echo "Data"
echo "--------------------"

collect "CUDA transfer (.*) ->"
collect "CUDA transfer (.*) <-"

echo

collect "CUDA allocation (.*)"
collect "CUDA copy (.*)"
collect "CUDA clear (.*)"
collect "CUDA free (.*)"

# Execution

echo
echo "Execution"
echo "--------------------"

collect "Kernel '.*' execution"

collect "Library function 'order_lib'"
collect "Library function 'group_lib'"
collect "Library function 'loop_join_lib'"
collect "Library function 'hash_join_lib'"
collect "Kernel 'hash_create_.*' execution"
collect "Kernel 'join_.*' execution"

collect "Builtin function '.*'"
collect "Create dictionary"
collect "Runtime analysis"

collect "CPU allocation (.*)"
collect "CPU free (.*)"

echo -n "[TOTAL] "
collect_1 "Execution"
