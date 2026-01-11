#!/bin/bash
# Usage: ./random_mtx_generator.sh N NNZ output.mtx

N=$1
NNZ=$2
OUT=$3

if [ -z "$N" ] || [ -z "$NNZ" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 N NNZ output.mtx"
  exit 1
fi

awk -v N="$N" -v NNZ="$NNZ" '
BEGIN {
  srand(12345);   # fixed seed → reproducibility
  print "%%MatrixMarket matrix coordinate real general";
  print "% Random matrix for weak scaling";
  print N, N, NNZ;

  for (k=0; k<NNZ; k++) {
    i = int(rand()*N) + 1;
    j = int(rand()*N) + 1;
    v = rand();
    print i, j, v;
  }
}
' > "$OUT"
