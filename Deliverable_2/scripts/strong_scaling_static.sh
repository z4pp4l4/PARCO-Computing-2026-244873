#!/bin/bash
# Strong Scaling Script (SuiteSparse matrices)

EXEC1=./MPI_impl_2D_partitioning.out
SRC1=./MPI_impl_2D_partitioning.c

EXEC2=./MPI_implementation.out
SRC1=./MPI_implementation.c

PROCS=(1 2 4 8 16 32 64 128)

MATRIX_DIR=../src
MATRICES=(
  nemeth19.mtx
  Trefethen_20000.mtx
  torso3.mtx
  Ga41As41H72.mtx
)

echo "Compiling code..."
mpicc -O3 "${SRC1}" -o "${EXEC1}" -std=c99 -fopenmp || exit 1
mpicc -O3 "${SRC2}" -o "${EXEC2}" -std=c99 -fopenmp || exit 1
echo "Compilation OK"
echo

for MATRIX in "${MATRICES[@]}"; do
    MTX="${MATRIX_DIR}/${MATRIX}"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        continue
    fi

    echo "*********************************************"
    echo "STRONG SCALING 2D pragma omp schedule(static): ${MATRIX}"
    echo "*********************************************"
    echo

    # BASELINE P=1
    echo "---------------------------------------------"
    echo "P=1 | Matrix=${MATRIX} (BASELINE)"
    echo "---------------------------------------------"

    BASELINE_T1=$(mpirun -np 1 "${EXEC1}" "${MTX}" \
        | grep "Total time:" \
        | awk '{print $3}')

    if [ -z "${BASELINE_T1}" ]; then
        echo "[ERROR] Failed to extract BASELINE_T1"
        exit 1
    fi

    echo ">> BASELINE_T1 = ${BASELINE_T1} s"
    export BASELINE_T1
    echo

    # -------------------------
    # STRONG SCALING
    # -------------------------
    for P in "${PROCS[@]}"; do
        if [ "${P}" -eq 1 ]; then
            continue
        fi

        echo "---------------------------------------------"
        echo "P=${P} | Matrix=${MATRIX}"
        echo "---------------------------------------------"
        mpirun -np "${P}" "${EXEC1}" "${MTX}"
        echo
    done

    unset BASELINE_T1
    echo
done

for MATRIX in "${MATRICES[@]}"; do
    MTX="${MATRIX_DIR}/${MATRIX}"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        continue
    fi
    echo "*********************************************"
    echo "STRONG SCALING 1D schedule(static): ${MATRIX}"
    echo "*********************************************"
    echo
    # BASELINE P=1
    echo "---------------------------------------------"
    echo "P=1 | Matrix=${MATRIX} (BASELINE)"
    echo "---------------------------------------------"

    BASELINE_T1=$(mpirun -np 1 "${EXEC2}" "${MTX}" \
        | grep "Total time:" \
        | awk '{print $3}')

    if [ -z "${BASELINE_T1}" ]; then
        echo "[ERROR] Failed to extract BASELINE_T1"
        exit 1
    fi

    echo ">> BASELINE_T1 = ${BASELINE_T1} s"
    export BASELINE_T1
    echo
    for P in "${PROCS[@]}"; do
        if [ "${P}" -eq 1 ]; then
            continue
        fi

        echo "---------------------------------------------"
        echo "P=${P} | Matrix=${MATRIX}"
        echo "---------------------------------------------"
        mpirun -np "${P}" "${EXEC2}" "${MTX}"
        echo
    done
    unset BASELINE_T1
    echo
done

echo "Strong scaling completed!"
