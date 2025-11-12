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
// RUNTIME SpMV with user-selected scheduling
// ============================================================================
void test_runtime_schedule(const int *Arow, const int *Acol, const double *Aval,
                           const double *x, double *y, int nrows, double *times,
                           omp_sched_t schedule_type, int chunk_size) {
    printf("\nschedule(%s,%d):\n",
           (schedule_type == omp_sched_static)  ? "static" :
           (schedule_type == omp_sched_dynamic) ? "dynamic" :
           (schedule_type == omp_sched_guided)  ? "guided" : "auto",
           chunk_size);

    omp_set_schedule(schedule_type, chunk_size);
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        usleep(100);
        long start = get_time_in_nanosec();
        #pragma omp parallel for schedule(runtime)
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

    char sched_input[32];
    printf("Enter OpenMP scheduling type (static/dynamic/guided/auto): ");
    scanf("%31s", sched_input);

    omp_sched_t sched_type = omp_sched_guided; // default
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

    int chunk = 10;
    int num_threads = omp_get_max_threads();
    printf("================================================================================\n");
    printf("SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)\n");
    printf("SEQUENTIAL vs schedule(%s,%d) RUNTIME\n", sched_input, chunk);
    printf("Matrix: %s\n", MATRIX_NAME);
    printf("Matrix size: %d x %d, nnz = %d\n", nrows, ncols, non_zero_val);
    printf("Number of threads (from env): %d\n", num_threads);
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

    test_sequential(Arow, Acol, Aval, x, y, nrows, t_seq);
    test_runtime_schedule(Arow, Acol, Aval, x, y, nrows, t_runtime, sched_type, chunk);

    double avg_seq = 0, avg_rt = 0;
    for (int i = 0; i < RUNS; i++) {
        avg_seq += t_seq[i];
        avg_rt += t_runtime[i];
    }
    avg_seq /= RUNS;
    avg_rt /= RUNS;

    double p90_seq = percentile90(t_seq, RUNS);
    double p90_rt = percentile90(t_runtime, RUNS);

    printf("\n================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Mode                         | Avg (ms)  | 90th Perc (ms) | Speedup\n");
    printf("-----------------------------------------------------------------\n");
    printf("SEQUENTIAL                   | %.6f | %.6f | 1.00x\n", avg_seq, p90_seq);
    printf("schedule(%s,%d) [%d th] | %.6f | %.6f | %.2fx\n",
           sched_input, chunk, num_threads, avg_rt, p90_rt, avg_seq / avg_rt);
    printf("================================================================================\n\n");

    char filename[256];
    snprintf(filename, sizeof(filename),
             "../results/CLUSTER/scheduling_type/runtime/RESULTS_%s_RUNTIME_%s_chunk%d_threads%d.txt",
             MATRIX_NAME, sched_input, chunk, num_threads);

    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Matrix: %s\n", MATRIX_NAME);
        fprintf(f, "Schedule: %s, chunk=%d (runtime test)\n\n", sched_input, chunk);
        fprintf(f, "SEQUENTIAL times (ms):\n");
        for (int i = 0; i < RUNS; i++) fprintf(f, "%.6f\n", t_seq[i]);
        fprintf(f, "Average: %.6f | 90th: %.6f\n\n", avg_seq, p90_seq);
        fprintf(f, "%s times (ms):\n", sched_input);
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