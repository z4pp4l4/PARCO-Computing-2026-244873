#include <stdio.h>
#include <stdlib.h>
#include <time.h>


static int cmp_int(const void *a, const void *b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s N NNZ_PER_ROW output.mtx\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int nnz_per_row = atoi(argv[2]);
    const char *out = argv[3];

    if (N <= 0 || nnz_per_row <= 0 || nnz_per_row > N) {
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
    fprintf(f, "%% Random synthetic matrix for weak scaling\n");
    fprintf(f, "%d %d %lld\n", N, N, NNZ);

    srand(12345);  // fixed seed → reproducibility

    int *cols = malloc(nnz_per_row * sizeof(int));
    if (!cols) {
        perror("malloc");
        return 1;
    }

    for (int i = 0; i < N; i++) {

        /* Generate unique random columns */
        int count = 0;
        while (count < nnz_per_row) {
            int c = rand() % N;
            int duplicate = 0;
            for (int k = 0; k < count; k++) {
                if (cols[k] == c) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate)
                cols[count++] = c;
        }

        /* Sort for nicer locality */
        qsort(cols, nnz_per_row, sizeof(int), cmp_int);

        for (int k = 0; k < nnz_per_row; k++) {
            double v = (double)rand() / RAND_MAX;
            fprintf(f, "%d %d %.6f\n",
                    i + 1, cols[k] + 1, v);
        }
    }

    free(cols);
    fclose(f);

    printf("Generated %s (%d x %d, NNZ=%lld)\n",
           out, N, N, NNZ);

    return 0;
}
