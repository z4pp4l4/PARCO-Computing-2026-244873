#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "bcsstk05_csr.h"
#define MATRIX_NAME "bcsstk05"
//#include "bcsstm05_csr.h"
//#define MATRIX_NAME "bcsstm05"
//#include "CAG_mat72_csr.h"
//#define MATRIX_NAME "CAG_mat72"
//#include "dataset20mfeatpixel_10NN_csr.h"
//#define MATRIX_NAME "dataset20mfeatpixel_10NN"
//#include "nemeth05_csr.h"
//#define MATRIX_NAME "nemeth05"
//#include "nemeth19_csr.h"
//#define MATRIX_NAME "nemeth19"
//#include "tols2000_csr.h"
//#define MATRIX_NAME "tols2000"
//#include "Trefethen_2000_csr.h"
//#define MATRIX_NAME "Trefethen_2000"

#define RUNS 15

void mat_vec_mult(const int *Arow,const int *Acol, const double *Aval,const double *x, double *y, int nrows) {
    #pragma omp parallel for

    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;
        for (int j = Arow[i]; j < Arow[i + 1]; j++)
            sum += Aval[j] * x[Acol[j]];
        y[i] = sum;
    }
}

long get_time_in_nanosec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

//FOR UNBIASED MEASUREMENTS: flush CPU cache
void flush_cache() {
    const size_t size = 512 * 1024 * 1024; // 512 MB (way largr then normal cache sizes)
    char *buffer = malloc(size);
    if (!buffer) {
        fprintf(stderr, "Cache flush: malloc failed\n");
        return;
    }
    for (size_t i = 0; i < size; i += 64){
        buffer[i] = (char)(i & 0xFF);
    }
    volatile char sink = 0;
    for (size_t pass = 0; pass < 3; pass++)
        for (size_t i = 0; i < size; i += 64)
            sink ^= buffer[i];

    (void)sink;
    free(buffer);
#ifdef __linux__
    system("sync"); // optional on Linux
#endif
}

// Simple 90th percentile
double percentile90(double *array, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (array[j] < array[i]) {
                double temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
    int idx = (int)(0.9 * n); //take 90th percintile
    if (idx >= n){
        idx = n - 1;
    } 
    return array[idx];
}

int main() {
    srand(time(NULL));
    printf("MATRIX-VECTOR MULTIPLICATION (CSR FORMAT) --> PARALLEL VERSION\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d,non_zero_val = %d\n", nrows, ncols,non_zero_val);
    printf("--- Cache flushed before each run for unbiased measurements ---\n\n");
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    double t[RUNS];

    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    printf("Timing runs:\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100); // allow short pause
        long start = get_time_in_nanosec();
        mat_vec_mult(Arow, Acol, Aval, x, y, nrows);
        long end = get_time_in_nanosec();
        t[r] = (end - start) / 1e6;
        printf("Run %2d: %.6f ms\n", r + 1, t[r]);
    }

    double sum = 0;
    for (int i = 0; i < RUNS; i++) sum += t[i];
    double avg = sum / RUNS;
    double p90 = percentile90(t, RUNS);

    char filename[256];
    snprintf(filename, sizeof(filename), "../results/LOCAL/RESULTS_%s_PARALLEL.txt", MATRIX_NAME);

    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t[i]);
        fprintf(f, "\nAverage: %.6f ms\n90th percentile: %.6f ms\n", avg, p90);
        fclose(f);
        printf("\nResults saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}
