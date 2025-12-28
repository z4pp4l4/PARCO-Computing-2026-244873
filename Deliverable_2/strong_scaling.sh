
#!/bin/bash
# Strong Scaling Script

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

RESULTS_DIR=results_strong_scaling
mkdir -p ${RESULTS_DIR}

# Compilation
echo "Compiling code..."
mpicc ${SRC} -o ${EXEC} -std=c99 -fopenmp
if [ $? -ne 0 ]; then
    echo "Compilation error"
    exit 1
fi
echo "Compilation OK"
echo

#Runs
for MATRIX in "${MATRICES[@]}"; do
    echo " -------------------------------"
    echo "Matrix: ${MATRIX}"
    echo " -------------------------------"

    OUT_FILE=${RESULTS_DIR}/${MATRIX%.mtx}.log

    # Reset output file for this matrix
    echo "Strong scaling results for ${MATRIX}" > "${OUT_FILE}"
    echo "Date: $(date)" >> "${OUT_FILE}"
    echo "========================================" >> "${OUT_FILE}"
    echo >> "${OUT_FILE}"

    for NP in "${PROC_NUM[@]}"; do
        echo "Running ${MATRIX} with np=${NP}"
        echo "----------------------------------------" >> "${OUT_FILE}"
        echo "np = ${NP}" >> "${OUT_FILE}"
        echo "----------------------------------------" >> "${OUT_FILE}"
        mpirun -np ${NP} ${EXEC} ${MATRIX} >> "${OUT_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[ERROR] np=${NP}" >> "${OUT_FILE}"
            echo "Error with np=${NP}"
        else
            echo "[OK] np=${NP}"
        fi

        echo >> "${OUT_FILE}"
    done
done

echo "Strong scaling runs completed"
echo "Results directory: ${RESULTS_DIR}"