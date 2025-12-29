#!/bin/bash
# Usage: ./gen_random_mtx.sh N NNZ output.mtx

N=$1
NNZ=$2
OUT=$3

if [ -z "$N" ] || [ -z "$NNZ" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 N NNZ output.mtx"
  exit 1
fi

echo "Generating random Matrix Market file:"
echo "  Size: ${N} x ${N}"
echo "  NNZ : ${NNZ}"
echo "  File: ${OUT}"

{
  echo "%%MatrixMarket matrix coordinate real general"
  echo "% Random matrix generated for weak scaling"
  echo "${N} ${N} ${NNZ}"

  for ((k=0; k<NNZ; k++)); do
    i=$((RANDOM % N + 1))
    j=$((RANDOM % N + 1))
    # random value in (0,1)
    val=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()}')
    echo "$i $j $val"
  done
} > "$OUT"
