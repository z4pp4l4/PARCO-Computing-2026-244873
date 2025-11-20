#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 MATRIX_NAME SCHED_TYPE CHUNK_SIZE NUM_THREADS"
    echo "Example: $0 Trefethen_2000 static 10 8"
    exit 1
fi

#Taking input arguments
MATRIX=$1
SCHED=$2
CHUNK=$3
THREADS=$4

if [ "$SCHED"== "static" ]; then
    REPL="schedule(static,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED" =="dynamic" ]; then
    REPL="schedule(dynamic,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED"=="guided" ]; then
    REPL="schedule(guided,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED"=="auto" ]; then
    REPL="schedule(auto) num_threads($THREADS)"
else
    echo "Invalid schedule type: $SCHED"
    echo "Choose: static | dynamic | guided | auto "
    exit 1
fi

echo "Using OpenMP: $REPL"
#code generation phase
sed "s/SCHEDULE_PLACEHOLDER/$REPL/g" spmv_template.c > spmv_generated.c

#compilation phase
gcc -fopenmp spmv_generated.c -o spmv_exec

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

./spmv_exec "$MATRIX"
