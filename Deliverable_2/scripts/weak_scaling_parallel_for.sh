#!/bin/bash
# Weak scaling benchmark - Hybrid MPI + OpenMP (fixed core budget)

SRC1=./MPI_impl_2D_partitioning.c
EXEC1=./MPI_impl_2D_partitioning.out

SRC2=./MPI_implementation.c
EXEC2=./MPI_implementation.out

MATRIX_DIR=../src/matrices
TOTAL_CORES=128

# (MPI ranks, OMP threads)
CONFIGS=(
  "1 128"
  "2 64"
  "4 32"
  "8 16"
  "16 8"
  "32 4"
  "64 2"
  "128 1"
)

echo "Compiling SpMV code..."
mpicc -O3 -std=c99 -fopenmp "${SRC1}" -o "${EXEC1}" || exit 1
mpicc -O3 -std=c99 -fopenmp "${SRC2}" -o "${EXEC2}" || exit 1
echo "Compilation OK"
echo

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ==========================
# 2D WEAK SCALING
# ==========================
echo "---------------------------------------------"
echo "WEAK SCALING 2D Hybrid MPI + OpenMP"
echo "---------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    P=$1
    T=$2

    MTX="${MATRIX_DIR}/weak_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        exit 1
    fi

    export OMP_NUM_THREADS=$T

    echo "---------------------------------------------"
    echo "MPI ranks = ${P}, OMP threads = ${T}  (P×T = $((P*T)))"
    echo "Matrix = $(basename ${MTX})"
    echo "---------------------------------------------"

    mpirun -np "${P}" "${EXEC1}" "${MTX}"
done

# ==========================
# 1D WEAK SCALING
# ==========================
echo "---------------------------------------------"
echo "WEAK SCALING 1D Hybrid MPI + OpenMP"
echo "---------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    P=$1
    T=$2

    MTX="${MATRIX_DIR}/weak_ps${P}.mtx"

    if [ ! -f "${MTX}" ]; then
        echo "[ERROR] Missing matrix ${MTX}"
        exit 1
    fi

    export OMP_NUM_THREADS=$T

    echo "---------------------------------------------"
    echo "MPI ranks = ${P}, OMP threads = ${T}  (P×T = $((P*T)))"
    echo "Matrix = $(basename ${MTX})"
    echo "---------------------------------------------"

    mpirun -np "${P}" "${EXEC2}" "${MTX}"
done
