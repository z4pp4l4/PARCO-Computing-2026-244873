#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Weak-scaling MatrixMarket generator
 *
 * Usage:
 *   ./generate_weak_mtx N NNZ_PER_ROW output.mtx
 *
 * Generates:
 *   - N x N matrix
 *   - exactly NNZ_PER_ROW nonzeros per row
 *   - banded sparsity pattern (row i connects to i..i+k)
 */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s N NNZ_PER_ROW output.mtx\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int nnz_per_row = atoi(argv[2]);
    const char *out = argv[3];

    if (N <= 0 || nnz_per_row <= 0) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    long long NNZ = (long long)N * nnz_per_row;

    FILE *f = fopen(out, "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    /* MatrixMarket header */
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%% Weak scaling synthetic matrix\n");
    fprintf(f, "%d %d %lld\n", N, N, NNZ);

    srand(12345);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < nnz_per_row; k++) {
            int j = i + k;
            if (j >= N)
                j = j % N;

            double v = 1.0;   // deterministic value
            fprintf(f, "%d %d %.6f\n", i + 1, j + 1, v);
        }
    }

    fclose(f);

    printf("Generated %s (%d x %d, NNZ=%lld)\n",
           out, N, N, NNZ);

    return 0;
}
