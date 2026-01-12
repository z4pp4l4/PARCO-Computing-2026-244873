#!/bin/bash
# Download selected matrices from SuiteSparse / TAMU
# Run this script from Deliverable_2/scripts/

set -e

DEST_DIR=../src
mkdir -p "${DEST_DIR}"

echo "----------------------------------------"
echo "Downloading SuiteSparse matrices"
echo "Destination: ${DEST_DIR}"
echo "----------------------------------------"

cd "${DEST_DIR}"

download_matrix () {
    GROUP=$1
    NAME=$2

    TAR="${NAME}.tar.gz"
    URL="https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${TAR}"

    if [ -f "${NAME}.mtx" ]; then
        echo "[SKIP] ${NAME}.mtx already exists"
        return
    fi

    echo "[DOWNLOAD] ${GROUP}/${NAME}"
    wget "${URL}"
    tar -xzf "${TAR}"
    mv "${NAME}/${NAME}.mtx" .
    rm -rf "${NAME}" "${TAR}"
}


download_matrix PARSEC Ga41As41H72
download_matrix Norris torso3
download_matrix Nemeth nemeth19
download_matrix JGD_Trefethen Trefethen_20000
download_matrix Janna Flan_1565
download_matrix Williams webbase-1M
echo "----------------------------------------"
echo "All matrices downloaded successfully!"
echo "----------------------------------------"
