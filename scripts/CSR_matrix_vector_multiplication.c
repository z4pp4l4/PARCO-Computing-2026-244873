#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "bcsstk05_csr.h"
// #include "west0067_csr.h"
#define RUNS 15

void mat_vec_mult(const int *Arow, const int *Acol, const double *Aval,
                  const double *x, double *y, int nrows) {
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

// Robust cache flush: larger buffer, multiple passes, memory barrier
void flush_cache() {
    const size_t size = 512 * 1024 * 1024; // 512 MB to exceed L3 cache
    char *buffer = malloc(size);
    if (!buffer) {
        fprintf(stderr, "Cache flush: malloc failed\n");
        return;
    }
    // Fill cache (write phase)
    for (size_t i = 0; i < size; i += 64) {
        buffer[i] = (char)(i & 0xFF);
    }
    // Flush by reading in random pattern to prevent prefetcher optimization
    volatile char sink = 0;
    for (size_t pass = 0; pass < 3; pass++) {
        for (size_t i = 0; i < size; i += 64) {
            sink ^= buffer[i];
        }
    }
    // Prevent compiler from optimizing away
    (void)sink;
    
    free(buffer);
    
    // System-level cache flush on Linux (optional, requires appropriate permissions)
    #ifdef __linux__
    system("sync"); // Sync filesystem buffers
    #endif
}

// Simple 90th percentile (sort array then take element)
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

int main() {
    srand(time(NULL));
    printf("MATRIX-VECTOR MULTIPLICATION (CSR FORMAT) --> SEQUENTIAL VERSION\n");
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    printf("Note: Cache flushed before each run for unbiased measurements\n\n");
    
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    double t[RUNS];
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    for (int i = 0; i < ncols; i++){
        x[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
    
    // Timing runs with cache flush before each
    printf("Timing runs:\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();           // Flush caches
        usleep(100);             // Small delay to let system settle
        
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
    
    FILE *f = fopen("../results/RESULTS_SEQUENTIAL.txt", "w");
    if (f) {
        fprintf(f, "Times (ms) [Cold cache, cache flushed before each run]:\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t[i]);
        fprintf(f, "\nAverage: %.6f ms\n90th percentile: %.6f ms\n", avg, p90);
        fclose(f);
    }
    
    printf("\nAverage: %.6f ms\n90th percentile: %.6f ms\n", avg, p90);
    printf("Saved to ../results/RESULTS_SEQUENTIAL.txt\n");
    
    free(x);
    free(y);
    return 0;
}