#!/bin/bash

THREADS_TO_TEST=(4 8 12 16 32 64)
CHUNK_SIZES=(5 10 20 50 100 500)
SCHED_TYPES=("static" "dynamic" "guided" "auto")

if [ $# -ne 1 ]; then
    echo "Usage: $0 MATRIX_NAME"
    echo "Example: $0 Trefethen_2000"
    exit 1
fi

MATRIX=$1 
EXECUTABLE="./SpmV_final_executable.exe"
C_CODE="Final_parallel_code.c"

echo "### SPMV BENCHMARK: Matrix $MATRIX ###"
echo "Threads: ${THREADS_TO_TEST[@]} | Chunk Sizes: ${CHUNK_SIZES[@]} | Schedules: ${SCHED_TYPES[@]}"
echo "--------------------------------------------------------------------------------------------------"
echo "Compiling $C_CODE..."
gcc -fopenmp "$C_CODE" -o "$EXECUTABLE" -lm -std=gnu99
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful. Running benchmarks..."
echo ""
# Print the header for the results table
printf "%-12s | %-12s | %-10s | %-14s | %-14s | %-10s\n" "Schedule" "Chunk Size" "Threads" "Seq. Time (ms)" "Par. Time (ms)" "Speedup"
printf -- "--------------------------------------------------------------------------------------------\n"

# iterate all scheduling types
for SCHED in "${SCHED_TYPES[@]}"; do
    # iterate through all thread counts
    for THREADS in "${THREADS_TO_TEST[@]}"; do
        # Determine which chunk sizes to use for the current schedule
        if [ "$SCHED" == "auto" ]; then
            # 'auto' schedule does not use a chunk size parameter, so we use a dummy value (e.g., 0)
            CHUNKS_USED=(0)
        else
            CHUNKS_USED=("${CHUNK_SIZES[@]}")
        fi
        # Loop through relevant chunk sizes
        for CHUNK in "${CHUNKS_USED[@]}"; do
            # Use '0' as a placeholder for the chunk size in the output for 'auto' schedule
            CHUNK_LABEL=$CHUNK
            if [ "$SCHED" == "auto" ]; then
                CHUNK_LABEL="N/A"
            fi
            # Run the executable and capture output
            OUTPUT=$("$EXECUTABLE" "$MATRIX" "$SCHED" "$CHUNK" "$THREADS" 2>/dev/null)
            
            if [ $? -eq 0 ] && [ ! -z "$OUTPUT" ]; then
                read -r SEQ_TIME PAR_TIME SPEEDUP <<< "$OUTPUT" #reading output from C exec
                
                # Validate that we got numeric results
                if [ ! -z "$SEQ_TIME" ] && [ ! -z "$PAR_TIME" ] && [ ! -z "$SPEEDUP" ]; then
                    printf "%-12s | %-12s | %-10d | %-14.6f | %-14.6f | %-10.2f\n" "$SCHED" "$CHUNK_LABEL" "$THREADS" "$SEQ_TIME" "$PAR_TIME" "$SPEEDUP"
                else
                    echo "Error: Invalid output format for $SCHED/$CHUNK/$THREADS"
                fi
            else
                echo "Error running benchmark for $SCHED/$CHUNK/$THREADS"
            fi
        done
    done
done

printf -- "--------------------------------------------------------------------------------------------\n"
echo "Benchmark completed."
