#!/bin/bash
# Strong Scaling Script (Hybrid MPI + OpenMP)

EXEC=./MPI_impl_2D_partitioning.out
SRC=MPI_impl_2D_partitioning.c

PROC_NUM=(1 2 4 8 16 32 64 128)

MATRICES=(
  bcsstk05.mtx
  bcsstm05.mtx
  breasttissue_10NN.mtx
  dataset20mfeatpixel_10NN.mtx
  Journals.mtx
  nemeth05.mtx
  nemeth19.mtx
  tols2000.mtx
  Trefethen_2000.mtx
  ww_36_pmec_36.mtx
)

RESULTS_DIR=results_strong_scaling_hybrid_2D
mkdir -p ${RESULTS_DIR}

# OpenMP affinity (IMPORTANT)
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Compilation
echo "Compiling code..."
mpicc ${SRC} -o ${EXEC} -std=c99 -fopenmp || exit 1
echo "Compilation OK"
echo

for MATRIX in "${MATRICES[@]}"; do
    echo "----------------------------------------"
    echo "Matrix: ${MATRIX}"
    echo "----------------------------------------"

    OUT_FILE=${RESULTS_DIR}/${MATRIX%.mtx}.log
    echo "Strong scaling results for ${MATRIX}" > "${OUT_FILE}"
    echo "Date: $(date)" >> "${OUT_FILE}"
    echo "----------------------------------------" >> "${OUT_FILE}"
    echo >> "${OUT_FILE}"

    #BASELINE 
    export OMP_NUM_THREADS=64
    echo "[Baseline] np=1, OMP_NUM_THREADS=64"
    mpirun -np 1 ${EXEC} ${MATRIX} | tee tmp_baseline.log >> "${OUT_FILE}"

    BASELINE_T1=$(grep -E "^\s*1\s*\|" tmp_baseline.log | awk -F'|' '{gsub(/ /,"",$2); print $2}')

    if [[ -z "${BASELINE_T1}" ]]; then
        echo "[ERROR] Failed to extract BASELINE_T1"
        exit 1
    fi

    echo "BASELINE_T1=${BASELINE_T1}" >> "${OUT_FILE}"
    echo >> "${OUT_FILE}"

    # SCALING RUNS
    for NP in "${PROC_NUM[@]}"; do
        if [ "${NP}" -eq 1 ]; then
            continue
        fi

        OMP_THREADS=$((128 / NP))
        if [ "${OMP_THREADS}" -gt 64 ]; then
            OMP_THREADS=64
        fi
        if [ "${OMP_THREADS}" -lt 1 ]; then
            OMP_THREADS=1
        fi

        export OMP_NUM_THREADS=${OMP_THREADS}

        echo "Running np=${NP}, OMP_NUM_THREADS=${OMP_THREADS}"
        echo "----------------------------------------" >> "${OUT_FILE}"
        echo "np=${NP}, OMP_NUM_THREADS=${OMP_THREADS}" >> "${OUT_FILE}"
        echo "----------------------------------------" >> "${OUT_FILE}"

        BASELINE_T1=${BASELINE_T1} \
        mpirun -np ${NP} ${EXEC} ${MATRIX} >> "${OUT_FILE}" 2>&1

        echo >> "${OUT_FILE}"
    done
done

rm -f tmp_baseline.log
echo "Strong scaling completed."
