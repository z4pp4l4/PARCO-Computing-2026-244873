#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

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
    const size_t size = 512 * 1024 * 1024; // 512MB
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
// Utility: compute 90th percentile
// ============================================================================
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

// ============================================================================
// SEQUENTIAL (no parallelization)
// ============================================================================
void test_sequential(const int *Arow, const int *Acol, const double *Aval,
                     const double *x, double *y, int nrows, double *times) {
    printf("SEQUENTIAL (no parallelization):\n");
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
// GUIDED schedule with chunk = 10
// ============================================================================
void test_guided_chunk10(const int *Arow, const int *Acol, const double *Aval,
                         const double *x, double *y, int nrows, double *times) {
    printf("schedule(guided,10):\n");
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
    printf("SEQUENTIAL vs schedule(guided,10)\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    printf("================================================================================\n\n");

    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    double t_seq[RUNS], t_guided[RUNS];

    // --- Run SEQUENTIAL
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    printf("\n");

    // --- Run GUIDED (chunk=10)
    test_guided_chunk10(Arow, Acol, Aval, x, y, nrows, t_guided);
    printf("\n");

    // --- Compute averages & 90th percentile
    double avg_seq = 0, avg_guided = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_guided += t_guided[i];
    }
    avg_seq /= RUNS;
    avg_guided /= RUNS;

    double p90_seq = percentile90(t_seq, RUNS);
    double p90_guided = percentile90(t_guided, RUNS);

    // --- Summary
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Mode                        | Avg (ms)  | 90th Perc (ms) | Speedup\n");
    printf("-----------------------------------------------------------------\n");
    printf("SEQUENTIAL                  | %.6f | %.6f | 1.00x\n", avg_seq, p90_seq);
    printf("schedule(guided,10)         | %.6f | %.6f | %.2fx\n",
           avg_guided, p90_guided, avg_seq / avg_guided);
    printf("================================================================================\n\n");

    // --- Save results to file
    char filename[256];
    snprintf(filename, sizeof(filename),
             "../results/CLUSTER/scheduling_type/guided/RESULTS_%s_GUIDED_chunk10.txt",
             MATRIX_NAME);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule: guided, chunk=10\n\n");
        fprintf(f, "SEQUENTIAL times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f | 90th: %.6f\n\n", avg_seq, p90_seq);
        fprintf(f, "GUIDED (chunk=10) times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_guided[i]);
        fprintf(f, "Average: %.6f | 90th: %.6f\n\n", avg_guided, p90_guided);
        fprintf(f, "Speedup: %.2fx\n", avg_seq / avg_guided);
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}