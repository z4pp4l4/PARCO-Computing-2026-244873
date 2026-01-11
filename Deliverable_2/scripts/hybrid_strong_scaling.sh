#!/bin/bash
# Strong Scaling Script (Hybrid MPI + OpenMP)

EXEC=./MPI_implementation.out
SRC=./MPI_implementation.c

PROC_NUM=(1 2 4 8 16 32 64 128)

# Selected matrices (same philosophy as MPI-only case)
MATRIX_DIR=../src
MATRICES=(
  Journals.mtx
  torso3.mtx
  Ga41As41H72.mtx
  nemeth19.mtx
)

RESULTS_DIR=../results/strong_scaling_hybrid
mkdir -p "${RESULTS_DIR}"

# OpenMP affinity
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Compilation
echo "Compiling code..."
mpicc "${SRC}" -o "${EXEC}" -std=c99 -fopenmp || exit 1
echo "Compilation OK"
echo

for MATRIX in "${MATRICES[@]}"; do
    MATRIX_PATH="${MATRIX_DIR}/${MATRIX}"

    echo "----------------------------------------"
    echo "Matrix: ${MATRIX}"
    echo "----------------------------------------"

    OUT_FILE="${RESULTS_DIR}/${MATRIX%.mtx}.log"

    {
        echo "Strong scaling results (Hybrid MPI+OMP) for ${MATRIX}"
        echo "Date: $(date)"
        echo "========================================"
        echo
    } > "${OUT_FILE}"

    # BASELINE: 1 MPI rank, max threads
    export OMP_NUM_THREADS=64
    echo "[Baseline] np=1, OMP_NUM_THREADS=64"
    mpirun -np 1 "${EXEC}" "${MATRIX_PATH}" | tee tmp_baseline.log >> "${OUT_FILE}"

    BASELINE_T1=$(grep -E "^\s*1\s*\|" tmp_baseline.log | \
                  awk -F'|' '{gsub(/ /,"",$2); print $2}')

    if [[ -z "${BASELINE_T1}" ]]; then
        echo "[ERROR] Failed to extract BASELINE_T1"
        exit 1
    fi

    echo "BASELINE_T1=${BASELINE_T1}" >> "${OUT_FILE}"
    echo >> "${OUT_FILE}"

    # Scaling runs
    for NP in "${PROC_NUM[@]}"; do
        [ "${NP}" -eq 1 ] && continue

        OMP_THREADS=$((128 / NP))
        [ "${OMP_THREADS}" -gt 64 ] && OMP_THREADS=64
        [ "${OMP_THREADS}" -lt 1 ] && OMP_THREADS=1

        export OMP_NUM_THREADS=${OMP_THREADS}

        echo "Running np=${NP}, OMP_NUM_THREADS=${OMP_THREADS}"
        echo "----------------------------------------" >> "${OUT_FILE}"
        echo "np=${NP}, OMP_NUM_THREADS=${OMP_THREADS}" >> "${OUT_FILE}"
        echo "----------------------------------------" >> "${OUT_FILE}"

        BASELINE_T1="${BASELINE_T1}" \
        mpirun -np "${NP}" "${EXEC}" "${MATRIX_PATH}" >> "${OUT_FILE}" 2>&1

        echo >> "${OUT_FILE}"
    done
done

rm -f tmp_baseline.log
echo "Strong scaling (Hybrid) completed."
