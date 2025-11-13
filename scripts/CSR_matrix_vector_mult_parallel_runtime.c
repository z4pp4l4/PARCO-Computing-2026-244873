#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

// Choose one matrix

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
#define NUM_CHUNK_SIZES 8

// Different chunk sizes to test
int chunk_sizes[NUM_CHUNK_SIZES] = {1, 2, 5, 10, 20, 50, 100, 200};

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
// Utility: 90th percentile
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
// Sequential baseline (no parallelization)
// ============================================================================
void test_sequential(const int *Arow, const int *Acol, const double *Aval,
                     const double *x, double *y, int nrows, double *times) {
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
    }
}

// ============================================================================
// Parallel test with variable chunk size and schedule type
// ============================================================================
void test_parallel_chunks(const int *Arow, const int *Acol, const double *Aval,
                          const double *x, double *y, int nrows,
                          double *times, int num_threads,
                          omp_sched_t schedule_type, int chunk_size) {
    omp_set_schedule(schedule_type, chunk_size);
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        long start = get_time_in_nanosec();

        #pragma omp parallel for schedule(runtime) num_threads(num_threads)
        for (int i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (int j = Arow[i]; j < Arow[i + 1]; j++)
                sum += Aval[j] * x[Acol[j]];
            y[i] = sum;
        }

        long end = get_time_in_nanosec();
        times[r] = (end - start) / 1e6;
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
    srand(time(NULL));

    // --- Select schedule type
    char sched_input[32];
    printf("Enter OpenMP scheduling type (static/dynamic/guided/auto): ");
    scanf("%31s", sched_input);

    omp_sched_t sched_type = omp_sched_guided;
    if (strcmp(sched_input, "static") == 0)
        sched_type = omp_sched_static;
    else if (strcmp(sched_input, "dynamic") == 0)
        sched_type = omp_sched_dynamic;
    else if (strcmp(sched_input, "guided") == 0)
        sched_type = omp_sched_guided;
    else if (strcmp(sched_input, "auto") == 0)
        sched_type = omp_sched_auto;
    else
        printf("Unknown schedule '%s', using guided.\n", sched_input);

    // --- Get thread count
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) {
            fprintf(stderr, "Error: number of threads must be >= 1\n");
            return 1;
        }
    }
    omp_set_num_threads(num_threads);

    // --- Print header
    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("CHUNK SIZE PERFORMANCE ANALYSIS - schedule(%s,chunk_size)\n", sched_input);
    printf("Matrix: %s | Size: %d x %d | nnz = %d\n", MATRIX_NAME, nrows, ncols, non_zero_val);
    printf("Threads: %d\n", num_threads);
    printf("================================================================================\n\n");

    // --- Allocate
    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    // --- Baseline sequential
    double t_seq[RUNS];
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    double avg_seq = 0;
    for (int i = 0; i < RUNS; i++) avg_seq += t_seq[i];
    avg_seq /= RUNS;
    double p90_seq = percentile90(t_seq, RUNS);

    // --- Parallel runs for each chunk size
    double results[NUM_CHUNK_SIZES][RUNS];
    double avg[NUM_CHUNK_SIZES], p90[NUM_CHUNK_SIZES];

    for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
        int chunk = chunk_sizes[cs];
        printf("Testing schedule(%s,%d) with %d threads...\n", sched_input, chunk, num_threads);

        test_parallel_chunks(Arow, Acol, Aval, x, y, nrows,
                             results[cs], num_threads, sched_type, chunk);

        avg[cs] = 0;
        for (int i = 0; i < RUNS; i++) avg[cs] += results[cs][i];
        avg[cs] /= RUNS;

        double temp[RUNS];
        memcpy(temp, results[cs], RUNS * sizeof(double));
        p90[cs] = percentile90(temp, RUNS);

        printf("  Avg: %.6f ms | 90th Perc: %.6f ms | Speedup: %.2fx\n\n",
               avg[cs], p90[cs], avg_seq / avg[cs]);
    }

    // --- Find best chunk
    double best_avg = avg[0];
    int best_idx = 0;
    for (int i = 1; i < NUM_CHUNK_SIZES; i++) {
        if (avg[i] < best_avg) {
            best_avg = avg[i];
            best_idx = i;
        }
    }

    // --- Summary
    printf("================================================================================\n");
    printf("SUMMARY: CHUNK SIZE COMPARISON - schedule(%s,chunk_size)\n", sched_input);
    printf("================================================================================\n");
    printf("Chunk Size | Avg (ms) | 90th Perc (ms) | Speedup vs Seq\n");
    printf("--------------------------------------------------------\n");
    printf("Baseline   | %.6f | %.6f | 1.00x\n", avg_seq, p90_seq);
    printf("--------------------------------------------------------\n");
    for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
        printf("%8d  | %.6f | %.6f | %.2fx\n",
               chunk_sizes[cs], avg[cs], p90[cs], avg_seq / avg[cs]);
    }
    printf("================================================================================\n");
    printf("BEST CHUNK SIZE: %d (Avg: %.6f ms, Speedup: %.2fx)\n",
           chunk_sizes[best_idx], best_avg, avg_seq / best_avg);
    printf("================================================================================\n\n");

    // --- Save results
    char filename[256];
    snprintf(filename, sizeof(filename),
             "../results/CLUSTER/scheduling_type/runtime/RESULTS_%s_%s_CHUNK_ANALYSIS_threads%d.txt",
             MATRIX_NAME, sched_input, num_threads);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\nSchedule: %s\nThreads: %d\n\n", MATRIX_NAME, sched_input, num_threads);
        fprintf(f, "SEQUENTIAL baseline avg: %.6f ms | 90th: %.6f ms\n\n", avg_seq, p90_seq);
        for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
            fprintf(f, "Chunk %d: Avg %.6f ms | 90th %.6f ms | Speedup %.2fx\n",
                    chunk_sizes[cs], avg[cs], p90[cs], avg_seq / avg[cs]);
        }
        fprintf(f, "\nBest chunk size: %d (Avg %.6f ms, Speedup %.2fx)\n",
                chunk_sizes[best_idx], best_avg, avg_seq / best_avg);
        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}
