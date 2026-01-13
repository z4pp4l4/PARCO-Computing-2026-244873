#!/bin/bash
# Weak scaling benchmark (file-based, clean)

SRC1=./MPI_impl_2D_partitioning.c
EXEC1=./MPI_impl_2D_partitioning.out

SRC2=./MPI_implementation.c
EXEC2=./MPI_implementation.out

PROCS=(1 2 4 8 16 32 64 128)
MATRIX_DIR=../src/matrices

# Compile once
echo "Compiling SpMV code..."
mpicc -O3 -std=c99 -fopenmp "${SRC1}" -o "${EXEC1}" || exit 1
mpicc -O3 -std=c99 -fopenmp "${SRC2}" -o "${EXEC2}" || exit 1
echo "Compilation OK"
echo
echo "---------------------------------------------"
echo "WEAK SCALING 2D hybrid MPI+OpenMP #pragma omp parallel schedule(static)"
echo "---------------------------------------------"
for P in "${PROCS[@]}"; do
    MTX="${MATRIX_DIR}/weak_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        exit 1
    fi

    echo "---------------------------------------------"
    echo "P=${P} | Matrix=$(basename ${MTX})"
    echo "---------------------------------------------"

    mpirun -np "${P}" "${EXEC1}" "${MTX}"
done
echo "---------------------------------------------"
echo "WEAK SCALING 1D hybrid MPI+OpenMP #pragma omp parallel schedule(static)"
echo "---------------------------------------------"
for P in "${PROCS[@]}"; do
    MTX="${MATRIX_DIR}/weak_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        exit 1
    fi

    echo "---------------------------------------------"
    echo "P=${P} | Matrix=$(basename ${MTX})"
    echo "---------------------------------------------"

    mpirun -np "${P}" "${EXEC2}" "${MTX}"
done