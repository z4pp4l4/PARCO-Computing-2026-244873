#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "Trefethen_2000_csr.h"
#define MATRIX_NAME "Trefethen_2000"
#define RUNS 15

// ====================== Sequential version (NO parallelization) ======================
void mat_vec_mult_sequential(const int *Arow, const int *Acol, const double *Aval,
                             const double *x, double *y, int nrows) {
    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;
        for (int j = Arow[i]; j < Arow[i + 1]; j++)
            sum += Aval[j] * x[Acol[j]];
        y[i] = sum;
    }
}

// ====================== Parallel version (WITH #pragma) ======================
void mat_vec_mult_parallel(const int *Arow, const int *Acol, const double *Aval,
                           const double *x, double *y, int nrows) {
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

// FOR UNBIASED MEASUREMENTS: flush CPU cache
void flush_cache() {
    const size_t size = 512 * 1024 * 1024;
    char *buffer = malloc(size);
    if (!buffer) {
        fprintf(stderr, "Cache flush: malloc failed\n");
        return;
    }
    for (size_t i = 0; i < size; i += 64) {
        buffer[i] = (char)(i & 0xFF);
    }
    volatile char sink = 0;
    for (size_t pass = 0; pass < 3; pass++)
        for (size_t i = 0; i < size; i += 64)
            sink ^= buffer[i];
    (void)sink;
    free(buffer);
#ifdef __linux__
    system("sync");
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
    int idx = (int)(0.9 * n);
    if (idx >= n) idx = n - 1;
    return array[idx];
}

int main() {
    srand(time(NULL));
    printf("================================================================================\n");
    printf("MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("SEQUENTIAL vs PARALLEL COMPARISON\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    //printf("Threads: %d\n", omp_get_max_threads());
    printf("--- Cache flushed before each run for unbiased measurements ---\n");
    printf("================================================================================\n\n");

    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    double t_seq[RUNS], t_par[RUNS];

    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    // ====================== Test SEQUENTIAL ======================
    printf("SEQUENTIAL VERSION (no #pragma):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);

        long start = get_time_in_nanosec();
        mat_vec_mult_sequential(Arow, Acol, Aval, x, y, nrows);
        long end = get_time_in_nanosec();

        t_seq[r] = (end - start) / 1e6;
        printf("  Run %2d: %.6f ms\n", r + 1, t_seq[r]);
    }

    // Reset y array
    memset(y, 0, nrows * sizeof(double));

    printf("\n");

    // ====================== Test PARALLEL ======================
    printf("PARALLEL VERSION (with #pragma omp parallel for):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);

        long start = get_time_in_nanosec();
        mat_vec_mult_parallel(Arow, Acol, Aval, x, y, nrows);
        long end = get_time_in_nanosec();

        t_par[r] = (end - start) / 1e6;
        printf("  Run %2d: %.6f ms\n", r + 1, t_par[r]);
    }

    // ====================== Calculate Statistics ======================
    double avg_seq = 0, avg_par = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_par += t_par[i];
    }
    avg_seq /= RUNS;
    avg_par /= RUNS;

    double p90_seq = percentile90(t_seq, RUNS);
    double p90_par = percentile90(t_par, RUNS);

    double speedup = avg_seq / avg_par;

    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Version                      | Average (ms) | 90th Percentile (ms) | Speedup\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("SEQUENTIAL (baseline)        | %.6f       | %.6f               | 1.00x\n",
           avg_seq, p90_seq);
    printf("PARALLEL                     | %.6f       | %.6f               | %.2fx\n",
           avg_par, p90_par, speedup);
    printf("================================================================================\n\n");

    // ====================== Save Results ======================
    char filename[256];
    snprintf(filename, sizeof(filename), "../results/CLUSTER/scheduling_type/parallel/RESULTS_%s_SEQ_vs_PAR.txt", MATRIX_NAME);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
        //fprintf(f, "Threads: %d\n", omp_get_max_threads());
        fprintf(f, "Number of runs: %d\n\n", RUNS);

        fprintf(f, "SEQUENTIAL VERSION (no #pragma):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f ms\n", avg_seq);
        fprintf(f, "90th percentile: %.6f ms\n\n", p90_seq);

        fprintf(f, "PARALLEL VERSION (with #pragma omp parallel for):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_par[i]);
        fprintf(f, "Average: %.6f ms\n", avg_par);
        fprintf(f, "90th percentile: %.6f ms\n\n", p90_par);

        fprintf(f, "SUMMARY:\n");
        fprintf(f, "Sequential Average:  %.6f ms\n", avg_seq);
        fprintf(f, "Parallel Average:    %.6f ms\n", avg_par);
        fprintf(f, "Speedup:             %.2fx\n", speedup);

        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}