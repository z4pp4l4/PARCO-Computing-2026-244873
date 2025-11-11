#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>

#include "Trefethen_2000_csr.h"
#define MATRIX_NAME "Trefethen_2000"
#define RUNS 15

// ============================================================================
// Utility: nanosecond timer
// ============================================================================
long get_time_in_nanosec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// ============================================================================
// Utility: cache flush (to avoid cache bias)
// ============================================================================
void flush_cache() {
    const size_t size = 512 * 1024 * 1024; // 512 MB
    char *buffer = malloc(size);
    if (!buffer) return;

    for (size_t i = 0; i < size; i += 64)
        buffer[i] = (char)(i & 0xFF);

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

// ============================================================================
// Utility: 90th Percentile
// ============================================================================
double percentile90(double *array, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (array[j] < array[i]) {
                double tmp = array[i];
                array[i] = array[j];
                array[j] = tmp;
            }
    int idx = (int)(0.9 * n);
    if (idx >= n) idx = n - 1;
    return array[idx];
}

// ============================================================================
// SEQUENTIAL SpMV (no OpenMP)
// ============================================================================
void test_sequential(const int *Arow, const int *Acol, const double *Aval,
                     const double *x, double *y, int nrows, double *times) {
    printf("\nSEQUENTIAL (no parallelization):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);

        long start = get_time_in_nanosec();

        for (int i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (int j = Arow[i]; j < Arow[i + 1]; j++)
                sum += Aval[j] * x[Acol[j]];
            y[i] = sum;
        }

        long end = get_time_in_nanosec();
        times[r] = (end - start) / 1e6;
        printf("  Run %2d: %.6f ms\n", r + 1, times[r]);
    }
}

// ============================================================================
// RUNTIME SpMV with fixed schedule(guided,10)
// ============================================================================
void test_runtime_guided10(const int *Arow, const int *Acol, const double *Aval,
                           const double *x, double *y, int nrows, double *times) {
    printf("\nschedule(guided,10):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);

        long start = get_time_in_nanosec();

        #pragma omp parallel for schedule(guided,10)
        for (int i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (int j = Arow[i]; j < Arow[i + 1]; j++)
                sum += Aval[j] * x[Acol[j]];
            y[i] = sum;
        }

        long end = get_time_in_nanosec();
        times[r] = (end - start) / 1e6;
        printf("  Run %2d: %.6f ms\n", r + 1, times[r]);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    srand(time(NULL));
    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("SEQUENTIAL vs schedule(guided,10) RUNTIME\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, nnz = %d\n", nrows, ncols, non_zero_val);
    printf("================================================================================\n\n");

    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    double t_seq[RUNS], t_runtime[RUNS];

    // --- Run Sequential
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);

    // --- Run schedule(guided,10)
    test_runtime_guided10(Arow, Acol, Aval, x, y, nrows, t_runtime);

    // --- Compute averages and percentiles
    double avg_seq = 0, avg_rt = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_rt += t_runtime[i];
    }
    avg_seq /= RUNS;
    avg_rt /= RUNS;

    double p90_seq = percentile90(t_seq, RUNS);
    double p90_rt = percentile90(t_runtime, RUNS);

    // --- Summary
    printf("\n================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Mode                         | Avg (ms)  | 90th Perc (ms) | Speedup\n");
    printf("-----------------------------------------------------------------\n");
    printf("SEQUENTIAL                   | %.6f | %.6f | 1.00x\n", avg_seq, p90_seq);
    printf("schedule(guided,10)          | %.6f | %.6f | %.2fx\n",
           avg_rt, p90_rt, avg_seq / avg_rt);
    printf("================================================================================\n\n");

    // --- Save results
    char filename[256];
    snprintf(filename, sizeof(filename),
             "../results/CLUSTER/scheduling_type/runtime/RESULTS_%s_RUNTIME_guided_chunk10.txt",
             MATRIX_NAME);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule: guided, chunk=10 (runtime test)\n\n");
        fprintf(f, "SEQUENTIAL times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f | 90th: %.6f\n\n", avg_seq, p90_seq);
        fprintf(f, "RUNTIME guided (chunk=10) times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_runtime[i]);
        fprintf(f, "Average: %.6f | 90th: %.6f\n\n", avg_rt, p90_rt);
        fprintf(f, "Speedup: %.2fx\n", avg_seq / avg_rt);
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}
