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

// Static schedule - default
void test_static_default(const int *Arow, const int *Acol, const double *Aval,
                         const double *x, double *y, int nrows, double *times) {
    printf("schedule(static) - default:\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(static)
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

// Static schedule - chunk size 1
void test_static_1(const int *Arow, const int *Acol, const double *Aval,
                   const double *x, double *y, int nrows, double *times) {
    printf("schedule(static, 1):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(static, 1)
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

// Static schedule - chunk size 10
void test_static_10(const int *Arow, const int *Acol, const double *Aval,
                    const double *x, double *y, int nrows, double *times) {
    printf("schedule(static, 10):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(static, 10)
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

// Static schedule - chunk size 50
void test_static_50(const int *Arow, const int *Acol, const double *Aval,
                    const double *x, double *y, int nrows, double *times) {
    printf("schedule(static, 50):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(static, 50)
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

// Static schedule - chunk size 100
void test_static_100(const int *Arow, const int *Acol, const double *Aval,
                     const double *x, double *y, int nrows, double *times) {
    printf("schedule(static, 100):\n");
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(static, 100)
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
    printf("SEQUENTIAL vs STATIC SCHEDULING COMPARISON\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    //printf("Threads: %d\n", omp_get_max_threads());
    printf("Schedule Type: STATIC (different chunk sizes)\n");
    printf("================================================================================\n\n");
    
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;
    
    double t_seq[RUNS], t_default[RUNS], t_1[RUNS], t_10[RUNS], t_50[RUNS], t_100[RUNS];
    
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    printf("\n");
    test_static_default(Arow, Acol, Aval, x, y, nrows, t_default);
    printf("\n");
    test_static_1(Arow, Acol, Aval, x, y, nrows, t_1);
    printf("\n");
    test_static_10(Arow, Acol, Aval, x, y, nrows, t_10);
    printf("\n");
    test_static_50(Arow, Acol, Aval, x, y, nrows, t_50);
    printf("\n");
    test_static_100(Arow, Acol, Aval, x, y, nrows, t_100);
    printf("\n");
    
    // Calculate averages and percentiles
    double avg_seq = 0, avg_default = 0, avg_1 = 0, avg_10 = 0, avg_50 = 0, avg_100 = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_default += t_default[i];
        avg_1 += t_1[i];
        avg_10 += t_10[i];
        avg_50 += t_50[i];
        avg_100 += t_100[i];
    }
    avg_seq /= RUNS;
    avg_default /= RUNS;
    avg_1 /= RUNS;
    avg_10 /= RUNS;
    avg_50 /= RUNS;
    avg_100 /= RUNS;
    
    double p90_seq = percentile90(t_seq, RUNS);
    double p90_default = percentile90(t_default, RUNS);
    double p90_1 = percentile90(t_1, RUNS);
    double p90_10 = percentile90(t_10, RUNS);
    double p90_50 = percentile90(t_50, RUNS);
    double p90_100 = percentile90(t_100, RUNS);
    
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Schedule Type                    | Average (ms) | 90th Perc (ms) | Speedup\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("SEQUENTIAL (baseline)            | %.6f       | %.6f         | 1.00x\n", 
           avg_seq, p90_seq);
    printf("schedule(static) - default       | %.6f       | %.6f         | %.2fx\n", 
           avg_default, p90_default, avg_seq / avg_default);
    printf("schedule(static, 1)              | %.6f       | %.6f         | %.2fx\n", 
           avg_1, p90_1, avg_seq / avg_1);
    printf("schedule(static, 10)             | %.6f       | %.6f         | %.2fx\n", 
           avg_10, p90_10, avg_seq / avg_10);
    printf("schedule(static, 50)             | %.6f       | %.6f         | %.2fx\n", 
           avg_50, p90_50, avg_seq / avg_50);
    printf("schedule(static, 100)            | %.6f       | %.6f         | %.2fx\n", 
           avg_100, p90_100, avg_seq / avg_100);
    printf("================================================================================\n\n");
    
    // Save results to file
    char filename[256];
    snprintf(filename, sizeof(filename), "../results/CLUSTER/scheduling_type/static/RESULTS_%s_STATIC_with_SEQ.txt", MATRIX_NAME);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule Type: STATIC (different chunk sizes) vs SEQUENTIAL\n");
        fprintf(f, "Matrix size: %d x %d, non_zero_val: %d\n", nrows, ncols, non_zero_val);
        //fprintf(f, "Threads: %d\n", omp_get_max_threads());
        fprintf(f, "Number of runs: %d\n\n", RUNS);
        
        fprintf(f, "SEQUENTIAL (no parallelization - baseline):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_seq, p90_seq);
        
        fprintf(f, "schedule(static) - default:\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_default[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_default, p90_default);
        
        fprintf(f, "schedule(static, 1):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_1[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_1, p90_1);
        
        fprintf(f, "schedule(static, 10):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_10[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_10, p90_10);
        
        fprintf(f, "schedule(static, 50):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_50[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_50, p90_50);
        
        fprintf(f, "schedule(static, 100):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_100[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_100, p90_100);
        
        fprintf(f, "SUMMARY:\n");
        fprintf(f, "SEQUENTIAL (baseline):           %.6f ms\n", avg_seq);
        fprintf(f, "schedule(static) - default:      %.6f ms (%.2fx speedup)\n", avg_default, avg_seq / avg_default);
        fprintf(f, "schedule(static, 1):             %.6f ms (%.2fx speedup)\n", avg_1, avg_seq / avg_1);
        fprintf(f, "schedule(static, 10):            %.6f ms (%.2fx speedup)\n", avg_10, avg_seq / avg_10);
        fprintf(f, "schedule(static, 50):            %.6f ms (%.2fx speedup)\n", avg_50, avg_seq / avg_50);
        fprintf(f, "schedule(static, 100):           %.6f ms (%.2fx speedup)\n", avg_100, avg_seq / avg_100);
        
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }
    
    free(x);
    free(y);
    return 0;
}