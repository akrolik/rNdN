#!/bin/bash

DATA_PATH="/mnt/local/alex/data/tpc-h/"
OUTPUT_PATH="q${1}_r3d3.log"
OPTIONS="$2"

echo "Query $1 (Options=$OPTIONS)"
echo "===================================="

../build/bin/r3d3 $OPTIONS --optimize-sass --debug-time --data-load-tpch="$DATA_PATH" "../tests/tpch/q${1}.hir" | tee "$OUTPUT_PATH"

RESULT=$(cat "$OUTPUT_PATH" | grep -v "\[" | grep -v "^Loading" | grep -v "^Execut" | grep -v "^Init" | grep -v "^Parsing"| grep -v "^\$")
EXPECTED=$(cat "expected/q${1}.log")

if [[ "$RESULT" == "$EXPECTED" ]]; then
	echo "[BENCHMARK] Success" | tee -a "$OUTPUT_PATH"
else
	echo "[BENCHMARK] Failure" | tee -a "$OUTPUT_PATH"
fi
