#!/bin/bash
# Generate weak-scaling matrices (run ONCE)
SOURCE="generate_weak_mtx.c"
GEN="./generate_weak_mtx.out"
#compile generator
gcc -O3 "${SOURCE}" -o "${GEN}" -lm -std=c99

ROWS_PER_PROC=20000
NNZ_PER_ROW=10

PROCS=(1 2 4 8 16 32 64 128)

OUTDIR=../src/matrices
mkdir -p "${OUTDIR}"

for P in "${PROCS[@]}"; do
    N=$((ROWS_PER_PROC * P))
    OUT="${OUTDIR}/weak_ps${P}.mtx"

    if [ -f "${OUT}" ]; then
        echo "[SKIP] ${OUT}"
        continue
    fi

    echo "[GEN] P=${P}  N=${N}"
    ${GEN} "${N}" "${NNZ_PER_ROW}" "${OUT}"
done
