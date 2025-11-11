#!/bin/bash
# ============================================================================
# Compile and Run All Sparse Matrix-Vector Multiplication Benchmarks
# Using all CSR_matrix_vector_mult_* implementations
# ============================================================================

set -e  # Stop on error

COMPILER="gcc"
COMPILER_FLAGS="-fopenmp"
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

mkdir -p "$OUTPUT_BASE_DIR/LOCAL"
mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type"

COUNT=0
for entry in "${SOURCES[@]}"; do
    COUNT=$((COUNT + 1))
    LABEL="${entry#*:}"
    EXECUTABLE="CSR_matrix_vector_multipl_cluster_${LABEL}.exe"

    echo "[${COUNT}/${TOTAL}] Running ${LABEL}..."
    mkdir -p "$OUTPUT_BASE_DIR/LOCAL/${LABEL,,}"
    mkdir -p "$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LABEL,,}"

    if ./"$EXECUTABLE"; then
        echo "Completed ${LABEL}"
    else
        echo "Execution failed for ${LABEL}"
    fi
    echo ""
done
echo "============================================================================"
echo "ORGANIZING RESULTS"
echo "============================================================================"

for entry in "${SOURCES[@]}"; do
    LABEL="${entry#*:}"
    LOWER="${LABEL,,}"
    SRC_FILE="$OUTPUT_BASE_DIR/LOCAL/${LOWER}/RESULTS_*.txt"
    DEST_DIR="$OUTPUT_BASE_DIR/CLUSTER/scheduling_type/${LOWER}"

    if ls $SRC_FILE 1> /dev/null 2>&1; then
        cp $SRC_FILE "$DEST_DIR/" 2>/dev/null || true
        echo "   ✓ Organized results for ${LABEL}"
    fi
done

echo ""
echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo "Results stored in:"
echo "  $OUTPUT_BASE_DIR/LOCAL/{sequential,parallel,static,dynamic,guided,auto,runtime}/"
echo "  $OUTPUT_BASE_DIR/CLUSTER/scheduling_type/{same}/"
echo ""
echo "Executables available:"
for entry in "${SOURCES[@]}"; do
    LABEL="${entry#*:}"
    echo "  ./CSR_matrix_vector_multipl_cluster_${LABEL}.exe"
done

echo ""
echo "Done!"
echo "============================================================================"
