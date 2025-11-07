

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//#include "bcsstk05_csr.h"
//#include "west0067_csr.h"
#include "adder_dcop_32_csr.h"
//CONTINUARE ---------------------

/*
Tes results with different CSR matrices headers.


...


*/
void mat_vec_mult(const int *Arow, const int *Acol, const double *Aval, const double *x, double *y, int nrows) {
    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;
        for (int j = Arow[i]; j < Arow[i + 1]; j++) {
            sum += Aval[j] * x[Acol[j]];
        }
        y[i] = sum;
    }
}

long get_time_in_nanosec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main() {
    srand(time(NULL));

    printf("========  CSR Matrix-Vector Multiplication  =======\n");
    printf("Matrix size: %d x %d, non-zero values = %d\n", nrows, ncols, nnz);

    // Allocate vectors
    double *x = malloc(ncols * sizeof(double));
    double *result = calloc(nrows, sizeof(double));

    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    // Perform SpMV
    long start = get_time_in_nanosec();
    mat_vec_mult(Arow, Acol, Aval, x, result, nrows);
    long end = get_time_in_nanosec();

    long elapsed_ns = end -start;
    double elapsed_ms = elapsed_ns / 1e6;

    // Print summary
    printf("Computation finished in %.6f milliseconds.\n", elapsed_ms);
    printf("Sample results:\n");
    for (int i = 0; i < (nrows < 10 ? nrows : 10); i++)
        printf("result[%d]= %.6e\n", i, result[i]);

    // Free memory
    free(x);
    free(result);
    return 0;
}
