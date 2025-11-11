#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "Trefethen_2000_csr.h"
#define MATRIX_NAME "Trefethen_2000"
#define RUNS 15

long get_time_in_nanosec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void flush_cache() {
    const size_t size = 512 * 1024 * 1024;
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

// ====================== Sequential version (NO parallelization) ======================
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

// Auto schedule - default
void test_auto(const int *Arow, const int *Acol, const double *Aval,
               const double *x, double *y, int nrows, double *times) {
    printf("schedule(auto) - default:\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(auto)
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

int main() {
    srand(time(NULL));
    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("SEQUENTIAL vs AUTO SCHEDULING COMPARISON\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    //printf("Threads: %d\n", omp_get_max_threads());
    printf("Schedule Type: AUTO (compiler/runtime decides)\n");
    printf("================================================================================\n\n");
    
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;
    
    double t_seq[RUNS], t_auto[RUNS];
    
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    printf("\n");
    test_auto(Arow, Acol, Aval, x, y, nrows, t_auto);
    printf("\n");
    
    // Calculate averages and percentiles
    double avg_seq = 0, avg_auto = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_auto += t_auto[i];
    }
    avg_seq /= RUNS;
    avg_auto /= RUNS;
    
    double p90_seq = percentile90(t_seq, RUNS);
    double p90_auto = percentile90(t_auto, RUNS);
    
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Schedule Type                    | Average (ms) | 90th Perc (ms) | Speedup\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("SEQUENTIAL (baseline)            | %.6f       | %.6f         | 1.00x\n",
           avg_seq, p90_seq);
    printf("schedule(auto)                   | %.6f       | %.6f         | %.2fx\n",
           avg_auto, p90_auto, avg_seq / avg_auto);
    printf("================================================================================\n\n");
    
    // Save results to file
    char filename[256];
    snprintf(filename, sizeof(filename), "../results/LOCAL/scheudling_type/auto/RESULTS_%s_AUTO_with_SEQ.txt", MATRIX_NAME);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule Type: AUTO (compiler/runtime decides) vs SEQUENTIAL\n");
        fprintf(f, "Matrix size: %d x %d, non_zero_val: %d\n", nrows, ncols, non_zero_val);
        //fprintf(f, "Threads: %d\n", omp_get_max_threads());
        fprintf(f, "Number of runs: %d\n\n", RUNS);
        
        fprintf(f, "SEQUENTIAL (no parallelization - baseline):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_seq, p90_seq);
        
        fprintf(f, "schedule(auto) - default:\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_auto[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_auto, p90_auto);
        
        fprintf(f, "SUMMARY:\n");
        fprintf(f, "SEQUENTIAL (baseline):  %.6f ms\n", avg_seq);
        fprintf(f, "schedule(auto):         %.6f ms (%.2fx speedup)\n", avg_auto, avg_seq / avg_auto);
        
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }
    
    free(x);
    free(y);
    return 0;
}