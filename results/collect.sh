#!/bin/bash

FILE=$1

function collect {
	echo -n "$1: "
	grep "$1:" $FILE | sed -E -e "s/.*: (.*) us($| \[.*)/\1/" | awk '{s+=$1} END {print s}'
}

function collect_1 {
	echo -n "$1: "
	grep "$1:" $FILE | awk 'END {print}' | sed -E -e "s/.*: (.*) us($| \[.*)/\1/"
}


# Compilation

echo "Compile"
echo "--------------------"

collect_1 "Syntax"
collect_1 "Outliner"
collect_1 "Frontend compiler"
	echo -en "\t"; collect "Dependency access analysis '.*'"
	echo -en "\t"; collect "Dependency analysis '.*'"
	echo -en "\t"; collect "Dependency subgraph analysis '.*'"
	echo -en "\t"; collect "Data object analysis '.*'"
	echo -en "\t"; collect "Shape analysis '.*'"
	echo -en "\t"; collect "Geometry analysis '.*'"
	echo -en "\t"; collect "Compatibility analysis '.*'"
	echo -en "\t"; collect "Outline builder '.*'"
collect "Frontend optimizer '.*'"

collect "Control-flow graph '.*'"
collect "Register allocation '.*'"
	echo -en "\t"; collect "Live variables '.*'"
	echo -en "\t"; collect "Live intervals '.*'"
	echo -en "\t"; collect "Linear scan allocator '.*'"

collect "Structurizer '.*'"
	echo -en "\t"; collect "Dominators '.*'"
	echo -en "\t"; collect "Post-dominators '.*'"
	echo -en "\t"; collect "Structurize '.*'"

collect "SASS codegen '.*'"
collect "Backend optimizer '.*'"
collect "Scheduler '.*'"
	echo -en "\t"; collect "Block dependency analysis '.*'"
	echo -en "\t"; collect "List scheduler '.*'"
	echo -en "\t"; collect "Linear scheduler '.*'"

collect_1 "Binary generator"
	echo -en "\t"; collect "SASS assembler"
	echo -en "\t"; collect "ELF generator"

collect_1 "CUDA assembler"

echo -n "[TOTAL] "
collect_1 "Compilation"

# Data

echo
echo "Data Cache"
echo "--------------------"

collect "Transfer 'tpch.*' to GPU"
collect "String pad cache"

# collect "CUDA transfer 'tpch.*' (.*) ->"
# collect "CUDA allocation 'tpch.*' (.*)"

echo
echo "Data Intermediate"
echo "--------------------"

collect "Transfer to GPU"
collect "Transfer to CPU"
collect "Resize buffers"
collect "Initialize buffer"

# echo

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

# collect "CUDA transfer 'output.*' (.*) <-"

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
echo -en "\t"; collect "Library function 'like_lib'"
echo -en "\t"; collect "Library function 'like_cache_lib'"

collect "Kernel '.*' execution"
	echo -en "\t"; collect "CUDA kernel 'like' execution"

collect "Runtime analysis"
echo -en "\t"; collect "Runtime analysis data"

# collect "CPU allocation (.*)"
# collect "CPU free (.*)"

echo -n "[TOTAL] "
collect_1 "Execution"
