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

if [ "$SCHED" == "static" ]; then
    REPL="schedule(static,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED" == "dynamic" ]; then
    REPL="schedule(dynamic,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED" == "guided" ]; then
    REPL="schedule(guided,$CHUNK) num_threads($THREADS)"
elif [ "$SCHED" == "auto" ]; then
    REPL="schedule(auto) num_threads($THREADS)"
else
    echo "Invalid schedule type: $SCHED"
    echo "Choose: static | dynamic | guided | auto "
    exit 1
fi

echo "Using OpenMP: $REPL"
if [ ! -f Deliverable_1/scripts/Final_parallel_code.c ]; then
    echo " Error: Final_parallel_code.c not found!"
    exit 1
fi

#code generation phase
sed "s/SCHEDULE_PLACEHOLDER/$REPL/g" Deliverable_1/scripts/Final_parallel_code.c > scripts/Final_parallel_code_generated.c

#compilation phase
gcc -fopenmp Deliverable_1/scripts/Final_parallel_code_generated.c -o Deliverable_1/scripts/SpmV_final_executable.exe -lm -std=gnu99

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

.Deliverable_1/scripts/SpmV_final_executable.exe "$MATRIX" "$SCHED" "$CHUNK" "$THREADS"

