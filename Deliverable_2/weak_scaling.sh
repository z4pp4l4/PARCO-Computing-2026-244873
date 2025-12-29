#!/bin/bash

EXEC=./spmv_mpi_2d_parallelIO
ROWS_PER_PROC=2000
NNZ_PER_PROC=100

PROCS=(1 2 4 8 16 32 64 128)

mkdir -p matrices
echo "------------------------------"
echo " Weak scaling (synthetic MTX)"
echo "------------------------------"

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
