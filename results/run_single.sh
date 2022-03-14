#!/bin/bash

QUERY=${1}
SF=${2}
NAME=${3}
OPTIONS=${4}

DATA_PATH="/mnt/local/alex/data/tpc-h/sf${SF}"
OUTPUT_PATH="q${QUERY}_sf${SF}_${NAME}.log"
QUERY_PATH="../tests/tpch/q${QUERY}.hir"

if [[ "$QUERY" == "11" && "$SF" != "1" ]]; then
	QUERY_PATH="../tests/tpch/q${QUERY}_${SF}.hir"
fi

../build/bin/rNdN $OPTIONS --debug-time --data-page-size=10737418240 --data-page-count=1 --data-scale-tpch=${SF} --data-load-tpch="$DATA_PATH" "$QUERY_PATH" | tee "$OUTPUT_PATH"

RESULT=$(cat "$OUTPUT_PATH" | grep -v "\[" | grep -v "^Debug" | grep -v "^Loading" | grep -v "^Execut" | grep -v "^Init" | grep -v "^Parsing"| grep -v "^\$")
EXPECTED=$(cat "expected/sf${SF}/q${QUERY}.log")

# Query 1 has double precision ouput that varies between runs, remove the decimal

if [[ "$QUERY" == "1" ]]; then
	RESULT=$(echo "$RESULT" | sed 's/\([0-9]\{10,\}\)\.[0-9]*/\1.000000/g')
	EXPECTED=$(echo "$EXPECTED" | sed 's/\([0-9]\{10,\}\)\.[0-9]*/\1.000000/g')
fi

if [[ "$RESULT" == "$EXPECTED" ]]; then
	echo "[BENCHMARK] Success" | tee -a "$OUTPUT_PATH"
else
	echo "[BENCHMARK] Failure" | tee -a "$OUTPUT_PATH"
fi
