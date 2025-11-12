#!/bin/bash
# ============================================================================
# Compile and Run All Sparse Matrix-Vector Multiplication Benchmarks
# Across Multiple Thread Counts (4, 8, 12, 16, 32, 64)
# ============================================================================

set -e  # Stop on error

COMPILER="gcc"
COMPILER_FLAGS="-fopenmp -std=c99 -lm"
OUTPUT_BASE_DIR="../results"

# ============================================================================
# List of implementations (each C file = one technique)
# ============================================================================
declare -a SOURCES=(
    "CSR_matrix_vector_multiplication.c:SEQUENTIAL"
    "CSR_matrix_vector_multiplication_parallel.c:PARALLEL"
    "CSR_matrix_vector_mult_parallel_static.c:STATIC"
    "CSR_matrix_vector_mult_parallel_dynamic.c:DYNAMIC"
    "CSR_matrix_vector_mult_parallel_guided.c:GUIDED"
    "CSR_matrix_vector_mult_parallel_auto.c:AUTO"
    "CSR_matrix_vector_mult_parallel_runtime.c:RUNTIME"
)

# ============================================================================
# Define all thread counts to test
# ============================================================================
THREAD_LIST=(4 8 12 16 32 64)

# ============================================================================
# Compilation phase
# ============================================================================
echo "============================================================================"
echo "COMPILATING THE C PROGRAMS"
echo "============================================================================"

TOTAL=${#SOURCES[@]}
COUNT=0

for entry in "${SOURCES[@]}"; do
    COUNT=$((COUNT + 1))
    SOURCE_FILE="${entry%:*}"
    LABEL="${entry#*:}"
    EXECUTABLE="CSR_matrix_vector_multipl_cluster_${LABEL}.exe"

    echo "[${COUNT}/${TOTAL}] Compiling ${SOURCE_FILE} → ${EXECUTABLE}"
    if $COMPILER $COMPILER_FLAGS -o "$EXECUTABLE" "$SOURCE_FILE"; then
        echo "${LABEL} compiled successfully"
    else
        echo "Failed to compile ${SOURCE_FILE}"
        exit 1
    fi
done

echo ""
echo "All implementations compiled successfully!"
echo ""
echo "============================================================================"
echo "EXECUTING THE PROGRAMS"
echo "============================================================================"

mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type"
COUNT=0
for entry in "${SOURCES[@]}"; do
    COUNT=$((COUNT + 1))
    LABEL="${entry#*:}"
    EXECUTABLE="CSR_matrix_vector_multipl_cluster_${LABEL}.exe"
    for THREADS in "${THREAD_LIST[@]}"; do
        export OMP_NUM_THREADS=$THREADS
        echo "------------------------------------------------------------"
        echo "[${COUNT}/${TOTAL}] Running ${LABEL} with ${THREADS} threads..."
        echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
        OUT_DIR="$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}/${THREADS}_threads"
        mkdir -p "$OUT_DIR"
        if ./"$EXECUTABLE"; then
            echo "✓ Completed ${LABEL} with ${THREADS} threads"
        else
            echo "✗ Execution failed for ${LABEL} (${THREADS} threads)"
        fi
        # Move any result files (if they exist) to the correct folder
        find "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}" -maxdepth 1 -type f -name "RESULTS_*" -exec mv {} "$OUT_DIR/" \; 2>/dev/null || true
        echo ""
    done
done

echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo "All runs completed for thread counts: ${THREAD_LIST[*]}"
echo "Results organized under:"
echo "  $OUTPUT_BASE_DIR/CLUSTER/scheduling_type/<version>/<threads>_threads/"
echo ""
echo "Done!"
echo "============================================================================"
