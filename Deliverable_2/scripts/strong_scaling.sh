#!/bin/bash
# Strong Scaling Script (selected SuiteSparse matrices)

EXEC=./MPI_impl_2D_partitioning.out
SRC=./MPI_impl_2D_partitioning.c

PROC_NUM=(1 2 4 8 16 32 64 128)

# Matrices are stored in ../src/
MATRIX_DIR=../src
MATRICES=(
  nemeth19.mtx
  Trefethen_20000.mtx
  webbase-1M.mtx
  Flan_1565.mtx
  #torso3.mtx
  #Ga41As41H72.mtx
  
)

RESULTS_DIR=../results/strong_scaling
mkdir -p "${RESULTS_DIR}"

# Compilation
echo "Compiling code..."
mpicc "${SRC}" -o "${EXEC}" -std=c99 -fopenmp || exit 1
echo "Compilation OK"
echo

for MATRIX in "${MATRICES[@]}"; do
    MATRIX_PATH="${MATRIX_DIR}/${MATRIX}"

    echo "--------------------------------"
    echo "Matrix: ${MATRIX}"
    echo "--------------------------------"

    OUT_FILE="${RESULTS_DIR}/${MATRIX%.mtx}.log"

    {
        echo "Strong scaling results for ${MATRIX}"
        echo "Date: $(date)"
        echo "========================================"
        echo
    } > "${OUT_FILE}"

    # BASELINE RUN (np = 1)
    echo "Running baseline np=1"
    mpirun -np 1 "${EXEC}" "${MATRIX_PATH}" | tee tmp_baseline.log >> "${OUT_FILE}"

    BASELINE_T1=$(grep -E "^\s*1\s*\|" tmp_baseline.log | \
                  awk -F'|' '{gsub(/ /,"",$2); print $2}')

    if [[ -z "${BASELINE_T1}" ]]; then
        echo "[ERROR] Failed to extract BASELINE_T1"
        exit 1
    fi

    echo "BASELINE_T1=${BASELINE_T1}" >> "${OUT_FILE}"
    echo >> "${OUT_FILE}"

    # Other process counts
    for NP in "${PROC_NUM[@]}"; do
        [ "${NP}" -eq 1 ] && continue

        echo "Running ${MATRIX} with np=${NP}"
        echo "----------------------------------------" >> "${OUT_FILE}"
        echo "np = ${NP}" >> "${OUT_FILE}"
        echo "----------------------------------------" >> "${OUT_FILE}"

        BASELINE_T1="${BASELINE_T1}" \
        mpirun -np "${NP}" "${EXEC}" "${MATRIX_PATH}" >> "${OUT_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[ERROR] np=${NP}" >> "${OUT_FILE}"
        else
            echo "[OK] np=${NP}"
        fi
        echo >> "${OUT_FILE}"
    done
done

rm -f tmp_baseline.log
echo "Strong scaling runs completed"
echo "Results directory: ${RESULTS_DIR}"
