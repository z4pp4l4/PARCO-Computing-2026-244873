#!/bin/bash
# ============================================================================
# Compile and Run All Sparse Matrix-Vector Multiplication Benchmarks
# SEQUENTIAL: Run ONCE (thread count irrelevant)
# PARALLEL: Run across Multiple Thread Counts (4, 8, 12, 16, 32, 64)
# ============================================================================

set -e  # Stop on error
COMPILER="gcc"
COMPILER_FLAGS="-fopenmp -std=gnu99 -lm"
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

# List of PARALLEL implementations (everything except SEQUENTIAL)
declare -a PARALLEL_SOURCES=(
    "CSR_matrix_vector_multiplication_parallel.c:PARALLEL"
    "CSR_matrix_vector_mult_parallel_static.c:STATIC"
    "CSR_matrix_vector_mult_parallel_dynamic.c:DYNAMIC"
    "CSR_matrix_vector_mult_parallel_guided.c:GUIDED"
    "CSR_matrix_vector_mult_parallel_auto.c:AUTO"
    "CSR_matrix_vector_mult_parallel_runtime.c:RUNTIME"
)

# ============================================================================
# Define all thread counts to test (for PARALLEL versions only)
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
        echo "        ✓ ${LABEL} compiled successfully"
    else
        echo "        ✗ Failed to compile ${SOURCE_FILE}"
        exit 1
    fi
done

echo ""
echo "All implementations compiled successfully!"
echo ""
# ============================================================================
# EXECUTION PHASE
# ============================================================================
echo "============================================================================"
echo "EXECUTING THE PROGRAMS"
echo "============================================================================"
echo ""

mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type"
# =====================================================================
# SEQUENTIAL: Run ONCE (no thread parameter)
# =====================================================================
echo "====== SEQUENTIAL (Run ONCE - thread count irrelevant) ======"
echo ""
LABEL="SEQUENTIAL"
EXECUTABLE="CSR_matrix_vector_multipl_cluster_${LABEL}.exe"

mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}"

echo "Running ${LABEL}..."
if ./"$EXECUTABLE"; then
    echo "✓ Completed ${LABEL}"
else
    echo "✗ Execution failed for ${LABEL}"
fi
echo ""
echo "====== PARALLEL VERSIONS (Run with multiple thread counts) ======"
echo ""
# Calculate total runs for progress display
TOTAL_PARALLEL_RUNS=$((${#PARALLEL_SOURCES[@]} * ${#THREAD_LIST[@]}))
CURRENT_RUN=0

for entry in "${PARALLEL_SOURCES[@]}"; do
    LABEL="${entry#*:}"
    EXECUTABLE="CSR_matrix_vector_multipl_cluster_${LABEL}.exe"
    
    mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}"
    
    for THREADS in "${THREAD_LIST[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo "------------------------------------------------------------"
        echo "[$CURRENT_RUN/$TOTAL_PARALLEL_RUNS] Running ${LABEL} with ${THREADS} threads..."
        echo "------------------------------------------------------------"
    
        OUT_DIR="$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}/${THREADS}_threads"
        mkdir -p "$OUT_DIR"    
        if ./"$EXECUTABLE" $THREADS; then
            echo "✓ Completed ${LABEL} with ${THREADS} threads"
        else
            echo "✗ Execution failed for ${LABEL} (${THREADS} threads)"
        fi
        # Move any result files (if they exist) to the correct folder
        find "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}" -maxdepth 1 -type f -name "RESULTS_*" -exec mv {} "$OUT_DIR/" \; 2>/dev/null || true
        
        echo ""
    done
done

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo ""
echo "SEQUENTIAL:"
echo "  ✓ Run 1 time (thread count doesn't affect single-threaded code)"
echo "  ✓ Results in: $OUTPUT_BASE_DIR/CLUSTER/scheduling_type/sequential/"
echo ""
echo "PARALLEL VERSIONS:"
echo "  ✓ Run ${#PARALLEL_SOURCES[@]} implementations × ${#THREAD_LIST[@]} thread counts = $TOTAL_PARALLEL_RUNS total runs"
echo "  ✓ Thread counts tested: ${THREAD_LIST[*]}"
echo "  ✓ Results organized in: $OUTPUT_BASE_DIR/CLUSTER/scheduling_type/<version>/<threads>_threads/"
echo ""

echo "============================================================================"
echo "Done!"
echo "============================================================================"