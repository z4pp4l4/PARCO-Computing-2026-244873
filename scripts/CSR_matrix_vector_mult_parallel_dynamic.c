#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>


//#include "bcsstk05_csr.h"
//#define MATRIX_NAME "bcsstk05"
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


// Dynamic schedule - chunk size 10
void test_dynamic_10(const int *Arow, const int *Acol, const double *Aval,
                     const double *x, double *y, int nrows, double *times,
                     int num_threads) {
    printf("schedule(dynamic, 10) with %d threads:\n", num_threads);
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        
        long start = get_time_in_nanosec();
        
        #pragma omp parallel for schedule(dynamic, 10) num_threads(num_threads)
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

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // ========================================================================
    // Parse command-line arguments for number of threads
    // ========================================================================
    int num_threads = omp_get_max_threads();  // Default: max available threads

    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) {
            fprintf(stderr, "Error: number of threads must be >= 1\n");
            fprintf(stderr, "Usage: %s [num_threads]\n", argv[0]);
            fprintf(stderr, "Example: %s 8\n", argv[0]);
            return 1;
        }
    }
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("SEQUENTIAL vs DYNAMIC SCHEDULING COMPARISON\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    printf("Number of threads: %d\n", num_threads);
    printf("Schedule Type: DYNAMIC (chunk size 10)\n");
    printf("================================================================================\n\n");
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;
    
    double t_seq[RUNS], t_10[RUNS];
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    printf("\n");
    test_dynamic_10(Arow, Acol, Aval, x, y, nrows, t_10, num_threads);
    printf("\n");
    
    double avg_seq = 0, avg_10 = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_10 += t_10[i];
    }
    avg_seq /= RUNS;
    avg_10 /= RUNS;
    
    double p90_seq = percentile90(t_seq, RUNS);
    double p90_10 = percentile90(t_10, RUNS);
    
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Schedule Type                    | Average (ms) | 90th Perc (ms) | Speedup\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("SEQUENTIAL (baseline)            | %.6f       | %.6f         | 1.00x\n",
           avg_seq, p90_seq);
    printf("schedule(dynamic, 10) [%d th]   | %.6f       | %.6f         | %.2fx\n",
           num_threads, avg_10, p90_10, avg_seq / avg_10);
    printf("================================================================================\n\n");
    
    // Save results to file with thread count in filename
    char filename[256];
    snprintf(filename, sizeof(filename), 
             "../results/CLUSTER/scheduling_type/dynamic/RESULTS_%s_DYNAMIC_chunk10_threads%d.txt", 
             MATRIX_NAME, num_threads);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule Type: DYNAMIC (chunk size 10) vs SEQUENTIAL\n");
        fprintf(f, "Matrix size: %d x %d, non_zero_val: %d\n", nrows, ncols, non_zero_val);
        fprintf(f, "Number of threads: %d\n", num_threads);
        fprintf(f, "Number of runs: %d\n\n", RUNS);
        
        fprintf(f, "SEQUENTIAL (no parallelization - baseline):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_seq, p90_seq);
        
        fprintf(f, "schedule(dynamic, 10):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_10[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_10, p90_10);
        fprintf(f, "SUMMARY:\n");
        fprintf(f, "SEQUENTIAL (baseline):           %.6f ms\n", avg_seq);
        fprintf(f, "schedule(dynamic, 10):           %.6f ms (%.2fx speedup)\n", avg_10, avg_seq / avg_10);
        
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }
    
    free(x);
    free(y);
    return 0;
}