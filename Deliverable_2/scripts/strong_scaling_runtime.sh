#!/bin/bash
# Strong Scaling - Hybrid MPI + OpenMP (schedule(runtime), no oversubscription)

SRC_2D=./MPI_impl_2D_partitioning.c
EXEC_2D=./MPI_impl_2D_partitioning.out

SRC_1D=./MPI_implementation.c
EXEC_1D=./MPI_implementation.out

MATRIX_DIR=../src
MATRICES=(
  nemeth19.mtx
  Trefethen_20000.mtx
  torso3.mtx
  Ga41As41H72.mtx
)

# Total cores available on the node
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

echo "Compiling code..."
mpicc -O3 -std=c99 -fopenmp "${SRC_2D}" -o "${EXEC_2D}" || exit 1
mpicc -O3 -std=c99 -fopenmp "${SRC_1D}" -o "${EXEC_1D}" || exit 1
echo "Compilation OK"
echo

# OpenMP runtime configuration
export OMP_PLACES=cores
export OMP_PROC_BIND=close
#export OMP_SCHEDULE=runtime

#####################################
# STRONG SCALING 2D
#####################################
for MATRIX in "${MATRICES[@]}"; do
    MTX="${MATRIX_DIR}/${MATRIX}"
    [ ! -f "${MTX}" ] && continue

    echo "**************************************************"
    echo "STRONG SCALING 2D | schedule(runtime) | ${MATRIX}"
    echo "**************************************************"

    # Baseline: P=1, T=128
    export OMP_NUM_THREADS=128
    BASELINE_T1=$(mpirun -np 1 "${EXEC_2D}" "${MTX}" \
        | grep "Total time:" | awk '{print $3}')
    export BASELINE_T1

    echo "Baseline T1 = ${BASELINE_T1}s"
    echo

    for cfg in "${CONFIGS[@]}"; do
        set -- $cfg
        P=$1
        T=$2

        export OMP_NUM_THREADS=$T

        echo "---------------------------------------------"
        echo "MPI ranks = ${P}, OMP threads = ${T}"
        echo "---------------------------------------------"

        mpirun -np "${P}" "${EXEC_2D}" "${MTX}"
        echo
    done

    unset BASELINE_T1
done

#####################################
# STRONG SCALING 1D
#####################################
for MATRIX in "${MATRICES[@]}"; do
    MTX="${MATRIX_DIR}/${MATRIX}"
    [ ! -f "${MTX}" ] && continue

    echo "*************************************************"
    echo "STRONG SCALING 1D | schedule(runtime) | ${MATRIX}"
    echo "*************************************************"

    # Baseline: P=1, T=128
    export OMP_NUM_THREADS=128
    BASELINE_T1=$(mpirun -np 1 "${EXEC_1D}" "${MTX}" \
        | grep "Total time:" | awk '{print $3}')
    export BASELINE_T1

    echo "Baseline T1 = ${BASELINE_T1}s"
    echo

    for cfg in "${CONFIGS[@]}"; do
        set -- $cfg
        P=$1
        T=$2

        export OMP_NUM_THREADS=$T

        echo "---------------------------------------------"
        echo "MPI ranks = ${P}, OMP threads = ${T}"
        echo "---------------------------------------------"

        mpirun -np "${P}" "${EXEC_1D}" "${MTX}"
        echo
    done

    unset BASELINE_T1
done

echo "Strong scaling completed!"
