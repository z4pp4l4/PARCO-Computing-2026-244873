#!/bin/bash
# Weak Scaling Script (synthetic matrices)

SRC=./MPI_impl_2D_partitioning.c
EXEC=./MPI_impl_2D_partitioning.out

ROWS_PER_PROC=20000
NNZ_PER_PROC=200000

PROCS=(1 2 4 8 16 32 64 128)

# Store synthetic matrices under src/matrices
MATRIX_DIR=../src/matrices
mkdir -p "${MATRIX_DIR}"

echo "------------------------------"
echo " Weak scaling (synthetic MTX)"
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

    echo
    echo "--------------------------------"
    echo "P = ${P} | Matrix = ${N} x ${N} | NNZ = ${NNZ}"
    echo "--------------------------------"

    mpirun -np "${P}" "${EXEC}" "${MTX}"
done
