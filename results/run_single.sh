#!/bin/bash

DATA_PATH="/mnt/local/alex/data/tpc-h/"
OUTPUT_PATH="q${1}_${2}.log"
OPTIONS="$3"

../build/bin/rNdN $OPTIONS --debug-time --debug-options --data-load-tpch="$DATA_PATH" "../tests/tpch/q${1}.hir" | tee "$OUTPUT_PATH"

RESULT=$(cat "$OUTPUT_PATH" | grep -v "\[" | grep -v "^Debug" | grep -v "^Loading" | grep -v "^Execut" | grep -v "^Init" | grep -v "^Parsing"| grep -v "^\$")
EXPECTED=$(cat "expected/q${1}.log")

# Query 1 has double precision ouput that varies between runs, remove the decimal

if [[ "$1" == "1" ]]; then
	RESULT=$(echo "$RESULT" | sed 's/\([0-9]\{10,\}\)\.[0-9]*/\1.000000/g')
	EXPECTED=$(echo "$EXPECTED" | sed 's/\([0-9]\{10,\}\)\.[0-9]*/\1.000000/g')
fi

if [[ "$RESULT" == "$EXPECTED" ]]; then
	echo "[BENCHMARK] Success" | tee -a "$OUTPUT_PATH"
else
	echo "[BENCHMARK] Failure" | tee -a "$OUTPUT_PATH"
fi
