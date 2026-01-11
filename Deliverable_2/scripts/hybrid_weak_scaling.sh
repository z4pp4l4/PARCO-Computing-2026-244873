#!/bin/bash
# Weak Scaling Script (Hybrid MPI + OpenMP)

SRC=./MPI_impl_2D_partitioning.c
EXEC=./MPI_impl_2D_partitioning.out

# Better weak-scaling parameters
ROWS_PER_PROC=20000
NNZ_PER_PROC=200000

PROCS=(1 2 4 8 16 32 64 128)

# Synthetic matrices stored here
MATRIX_DIR=../src/matrices
mkdir -p "${MATRIX_DIR}"

# OpenMP affinity
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "------------------------------"
echo " Weak scaling (Hybrid MPI+OMP)"
echo "------------------------------"

# Compilation
echo "Compiling code..."
mpicc "${SRC}" -o "${EXEC}" -std=c99 -fopenmp || exit 1
echo "Compilation OK"
echo

for P in "${PROCS[@]}"; do
    N=$((ROWS_PER_PROC * P))
    NNZ=$((NNZ_PER_PROC * P))
    MTX="${MATRIX_DIR}/random_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "Generating matrix for P=${P}"
        ./random_mtx_generator.sh "${N}" "${NNZ}" "${MTX}"
    fi

    # Keep total cores roughly constant (≈128)
    OMP_THREADS=$((128 / P))
    [ "${OMP_THREADS}" -gt 64 ] && OMP_THREADS=64
    [ "${OMP_THREADS}" -lt 1 ] && OMP_THREADS=1

    export OMP_NUM_THREADS=${OMP_THREADS}

    echo
    echo "--------------------------------"
    echo "P = ${P} | Matrix = ${N} x ${N} | NNZ = ${NNZ}"
    echo "OMP_NUM_THREADS = ${OMP_THREADS}"
    echo "--------------------------------"

    mpirun -np "${P}" "${EXEC}" "${MTX}"
done

echo "Weak scaling (Hybrid) completed."
