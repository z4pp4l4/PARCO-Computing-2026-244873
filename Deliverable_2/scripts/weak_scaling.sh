#!/bin/bash
# Weak scaling benchmark (file-based, clean)

SRC=./MPI_impl_2D_partitioning.c
EXEC=./MPI_impl_2D_partitioning.out

PROCS=(1 2 4 8 16 32 64 128)
MATRIX_DIR=../src/matrices

# Compile once
echo "Compiling SpMV code..."
mpicc -O3 -std=c99 -fopenmp "${SRC}" -o "${EXEC}" || exit 1
echo "Compilation OK"
echo

for P in "${PROCS[@]}"; do
    MTX="${MATRIX_DIR}/weak_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        exit 1
    fi

    echo "---------------------------------------------"
    echo "P=${P} | Matrix=$(basename ${MTX})"
    echo "---------------------------------------------"

    mpirun -np "${P}" "${EXEC}" "${MTX}"
done
