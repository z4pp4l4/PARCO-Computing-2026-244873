#!/bin/bash
SRC=MPI_impl_2D_partitioning.c
EXEC=./MPI_impl_2D_partitioning.out
ROWS_PER_PROC=1000
NNZ_PER_PROC=10000

PROCS=(1 2 4 8 16 32 64 128)

mkdir -p matrices
echo "------------------------------"
echo " Weak scaling (synthetic MTX)"
echo "------------------------------"

# Compilation
echo "Compiling code..."
mpicc ${SRC} -o ${EXEC} -std=c99 -fopenmp || exit 1
if [ $? -ne 0 ]; then
    echo "Compilation error"
    exit 1
fi
echo "Compilation OK"
echo

for P in "${PROCS[@]}"; do
  N=$((ROWS_PER_PROC * P))
  NNZ=$((NNZ_PER_PROC * P))
  MTX="matrices/random_ps${P}.mtx"

  if [ ! -f "$MTX" ]; then
    echo "Generating matrix for P=$P"
    ./random_mtx_generator.sh "$N" "$NNZ" "$MTX"
  fi

  echo
  echo "--------------------------------"
  echo "P = $P | Matrix = ${N} x ${N} | NNZ = ${NNZ}"
  echo "--------------------------------"
  mpirun -np $P $EXEC "$MTX"
done
