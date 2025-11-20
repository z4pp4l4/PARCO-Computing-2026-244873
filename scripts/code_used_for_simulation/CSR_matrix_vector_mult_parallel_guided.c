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
#define NUM_CHUNK_SIZES 8

// Different chunk sizes to test with guided schedule
int chunk_sizes[NUM_CHUNK_SIZES] = {1, 2, 5, 10, 20, 50, 100, 200};

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
    //printf("SEQUENTIAL (no parallelization):\n");
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
        //printf("  Run %2d: %.6f ms\n", r + 1, times[r]);
    }
}

// ============================================================================
// GUIDED schedule with variable chunk size
// ============================================================================
void test_guided_chunk(const int *Arow, const int *Acol, const double *Aval,
                       const double *x, double *y, int nrows, double *times,
                       int num_threads, int chunk_size) {
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        long start = get_time_in_nanosec();
        #pragma omp parallel for schedule(guided,chunk_size) num_threads(num_threads)
        for (int i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (int j = Arow[i]; j < Arow[i + 1]; j++)
                sum += Aval[j] * x[Acol[j]];
            y[i] = sum;
        }

        long end = get_time_in_nanosec();
        times[r] = (end - start) / 1e6;
        //printf("  Run %2d: %.6f ms\n", r + 1, times[r]);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {

    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) {
            fprintf(stderr, "Error: number of threads must be >= 1\n");
            fprintf(stderr, "Usage: %s [num_threads]\n", argv[0]);
            fprintf(stderr, "Example: %s 8\n", argv[0]);
            return 1;
        }
    }

    omp_set_num_threads(num_threads);
    srand(time(NULL));

    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("CHUNK SIZE PERFORMANCE ANALYSIS - schedule(guided, chunk_size)\n");
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, non_zero_val = %d\n", nrows, ncols, non_zero_val);
    printf("Number of threads: %d\n", num_threads);
    printf("================================================================================\n\n");

    double *x = malloc(ncols * sizeof(double));
    double *y = calloc(nrows, sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < ncols; i++)
        x[i] = ((double)rand() / RAND_MAX) * 10.0;

    // Test sequential baseline
    double t_seq[RUNS];
    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    printf("\n");

    // Calculate sequential statistics
    double avg_seq = 0;
    for (int i = 0; i < RUNS; i++)
        avg_seq += t_seq[i];
    avg_seq /= RUNS;
    double p90_seq = percentile90(t_seq, RUNS);

    // Store results for all chunk sizes
    double results[NUM_CHUNK_SIZES][RUNS];
    double avg[NUM_CHUNK_SIZES], p90[NUM_CHUNK_SIZES];

    // Test each chunk size
    for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
        int chunk = chunk_sizes[cs];
        printf("Testing schedule(guided, %d) with %d threads:\n", chunk, num_threads);

        test_guided_chunk(Arow, Acol, Aval, x, y, nrows, results[cs], num_threads, chunk);

        // Calculate statistics
        avg[cs] = 0;
        for (int i = 0; i < RUNS; i++)
            avg[cs] += results[cs][i];
        avg[cs] /= RUNS;

        double temp[RUNS];
        memcpy(temp, results[cs], RUNS * sizeof(double));
        p90[cs] = percentile90(temp, RUNS);

        printf("  Average: %.6f ms | 90th percentile: %.6f ms | Speedup: %.2fx\n\n",
               avg[cs], p90[cs], avg_seq / avg[cs]);
    }

    // Find best chunk size
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
    printf("SUMMARY: CHUNK SIZE COMPARISON - schedule(guided, chunk_size)\n");
    printf("================================================================================\n");
    printf("Chunk Size | Average (ms) | 90th Percentile (ms) | Speedup vs Sequential\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  BASELINE | %.6f       | %.6f               | 1.00x\n", avg_seq, p90_seq);
    printf("--------------------------------------------------------------------------------\n");

    for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
        printf("    %3d    | %.6f       | %.6f               | %.2fx\n",
               chunk_sizes[cs], avg[cs], p90[cs], avg_seq / avg[cs]);
    }
    printf("================================================================================\n\n");
    printf("BEST CHUNK SIZE: %d (Average: %.6f ms, Speedup: %.2fx)\n\n",
           chunk_sizes[best_idx], best_avg, avg_seq / best_avg);

    // --- Save results to file
    char filename[256];
    snprintf(filename, sizeof(filename),
             "../results/CLUSTER/scheduling_type/guided/RESULTS_%s_GUIDED_CHUNK_ANALYSIS_threads%d.txt",
             MATRIX_NAME, num_threads);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule Type: GUIDED - CHUNK SIZE ANALYSIS\n");
        fprintf(f, "Matrix size: %d x %d, non_zero_val: %d\n", nrows, ncols, non_zero_val);
        fprintf(f, "Number of threads: %d\n", num_threads);
        fprintf(f, "Number of runs per chunk size: %d\n\n", RUNS);

        fprintf(f, "SEQUENTIAL (baseline - no parallelization):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms\n\n", avg_seq, p90_seq);

        fprintf(f, "================================================================================\n");
        fprintf(f, "CHUNK SIZE ANALYSIS - schedule(guided, chunk_size)\n");
        fprintf(f, "================================================================================\n\n");

        for (int cs = 0; cs < NUM_CHUNK_SIZES; cs++) {
            fprintf(f, "CHUNK SIZE: %d\n", chunk_sizes[cs]);
            for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", results[cs][i]);
            fprintf(f, "Average: %.6f ms | 90th percentile: %.6f ms | Speedup: %.2fx\n\n",
                    avg[cs], p90[cs], avg_seq / avg[cs]);
        }

        fprintf(f, "================================================================================\n");
        fprintf(f, "BEST CHUNK SIZE: %d\n", chunk_sizes[best_idx]);
        fprintf(f, "Best Average: %.6f ms (Speedup: %.2fx)\n", best_avg, avg_seq / best_avg);

        fclose(f);
        printf("Results saved to: %s\n", filename);
    } else {
        perror("Error creating result file");
    }

    free(x);
    free(y);
    return 0;
}