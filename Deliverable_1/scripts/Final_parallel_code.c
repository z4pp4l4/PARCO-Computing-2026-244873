#define _POSIX_C_SOURCE 199309L 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h> 
#include <time.h>
#include <omp.h>
#include "Trefethen_2000_csr.h" 
#include "bcsstk05_csr.h" 
#include "bcsstm05_csr.h" 
#include "dataset20mfeatpixel_10NN_csr.h" 
#include "nemeth05_csr.h" 
#include "nemeth19_csr.h" 
#include "tols2000_csr.h"
#include "ww_36_pmec_36_csr.h"
#include "Journals_csr.h"
#include "breasttissue_10NN_csr.h"

const int *Arow_ptr = NULL; 
const int *Acol_ptr = NULL; 
const double *Aval_ptr = NULL; 
int NROWS = 0; int NCOLS = 0; int NNZ = 0; 
void select_matrix(char *name) {
    if (strcmp(name, "Trefethen_2000") == 0) {
        Arow_ptr = Trefethen_2000_Arow;
        Acol_ptr = Trefethen_2000_Acol;
        Aval_ptr = Trefethen_2000_Aval;
        NROWS = Trefethen_2000_nrows;
        NCOLS = Trefethen_2000_ncols;
        NNZ = Trefethen_2000_nnz;
        return;
    } else if (strcmp(name, "bcsstk05") == 0) {
        Arow_ptr = bcsstk05_Arow;
        Acol_ptr = bcsstk05_Acol;
        Aval_ptr = bcsstk05_Aval;
        NROWS = bcsstk05_nrows;
        NCOLS = bcsstk05_ncols;
        NNZ = bcsstk05_nnz;
        return;
    } else if (strcmp(name, "bcsstm05") == 0) {
        Arow_ptr = bcsstm05_Arow;
        Acol_ptr = bcsstm05_Acol;
        Aval_ptr = bcsstm05_Aval;
        NROWS = bcsstm05_nrows;
        NCOLS = bcsstm05_ncols;
        NNZ = bcsstm05_nnz;
        return;
    } else if (strcmp(name, "dataset20mfeatpixel_10NN") == 0) {
        Arow_ptr = dataset20mfeatpixel_10NN_Arow;
        Acol_ptr = dataset20mfeatpixel_10NN_Acol;
        Aval_ptr = dataset20mfeatpixel_10NN_Aval;
        NROWS = dataset20mfeatpixel_10NN_nrows;
        NCOLS = dataset20mfeatpixel_10NN_ncols;
        NNZ = dataset20mfeatpixel_10NN_nnz;
        return;
    } else if (strcmp(name, "nemeth05") == 0) {
        Arow_ptr = nemeth05_Arow;
        Acol_ptr = nemeth05_Acol;
        Aval_ptr = nemeth05_Aval;
        NROWS = nemeth05_nrows;
        NCOLS = nemeth05_ncols;
        NNZ = nemeth05_nnz;
        return;
    } else if (strcmp(name,"nemeth19") == 0) {
        Arow_ptr = nemeth19_Arow;
        Acol_ptr = nemeth19_Acol;
        Aval_ptr = nemeth19_Aval;
        NROWS = nemeth19_nrows;
        NCOLS = nemeth19_ncols;
        NNZ = nemeth19_nnz;
        return;
    } else if (strcmp(name,"tols2000") == 0) {
        Arow_ptr = tols2000_Arow;
        Acol_ptr = tols2000_Acol;
        Aval_ptr = tols2000_Aval;
        NROWS = tols2000_nrows;
        NCOLS = tols2000_ncols;
        NNZ = tols2000_nnz;
        return;
    } else if (strcmp(name,"ww_36_pmec_36") == 0) {
        Arow_ptr = ww_36_pmec_36_Arow;
        Acol_ptr = ww_36_pmec_36_Acol;
        Aval_ptr = ww_36_pmec_36_Aval;
        NROWS = ww_36_pmec_36_nrows;
        NCOLS = ww_36_pmec_36_ncols;
        NNZ = ww_36_pmec_36_nnz;
        return;
    } else if (strcmp(name,"Journals") == 0) {
        Arow_ptr = Journals_Arow;
        Acol_ptr = Journals_Acol;
        Aval_ptr = Journals_Aval;
        NROWS = Journals_nrows;
        NCOLS = Journals_ncols;
        NNZ = Journals_nnz;
        return;
    } else if (strcmp(name,"breasttissue_10NN") == 0) {
        Arow_ptr = breasttissue_10NN_Arow;
        Acol_ptr = breasttissue_10NN_Acol;
        Aval_ptr = breasttissue_10NN_Aval;
        NROWS = breasttissue_10NN_nrows;
        NCOLS = breasttissue_10NN_ncols;
        NNZ = breasttissue_10NN_nnz;
        return;
    }
    printf("Unknown matrix in input: '%s'\n", name);
    exit(1);
}
// Function used to get time in nanoseconds (for maximum precision) 
long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}


//function used to simulate a cache flush 
//used at each iteration for unbiased results
void flush_cache(){
    const size_t size = 512 * 1024 * 1024;
    char *buffer = malloc(size);
    if (!buffer) {
         return;
    }
    for (size_t i = 0; i < size; i += 64){
        buffer[i] = (char)(i & 0xFF);
    }

    volatile char sink = 0;
    for (size_t p = 0; p < 3; p++){
        for (size_t i = 0; i < size; i += 64){
            sink ^= buffer[i];
        }
    }
    free(buffer);
}

//as measurement metrc we use the 90th percentile as defined in specs
double percentile90(double *arr, int n){
    for (int i = 0; i < n - 1; i++){
        for (int j = i + 1; j < n; j++){
            if (arr[j] < arr[i]) {
                double tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp; 
            }
        } 
    }
    int idx = (int)(0.9 * n); 
    if (idx >= n) idx = n - 1;
    return arr[idx];
}

//Want to provide alway the sequential implementation of spmv for comparison with parallel versions
void sequential_SpMV(double *rand_arr, double *result_arr){
    //sequential implementation of SpMV
    for (int i = 0; i < NROWS; i++) {
        double sum = 0.0;
        for (int j = Arow_ptr[i]; j < Arow_ptr[i + 1]; j++)
            sum += Aval_ptr[j] * rand_arr[Acol_ptr[j]];
        result_arr[i] = sum;
    }
}
//parallel impl. with multiple scheduling strategies
void parallel_SpMV(double *x, double *result_arr, char *sched_name, int chunk_size, int t){   
    //first have to map the sched_name to the corresponding omp schedule
    int sched = -1;
    if (strcmp(sched_name, "static") == 0) 
        sched = 0;
    else if (strcmp(sched_name, "dynamic") ==0) 
        sched = 1;
    else if (strcmp(sched_name, "guided") == 0) 
        sched = 2;
    else if (strcmp(sched_name, "auto")== 0) 
        sched = 3;
    else {
        printf("Invalid schedule : '%s'\n", sched_name);
        exit(1);
    }
    switch (sched){
        case 0: 
            #pragma omp parallel for schedule(static, chunk_size) num_threads(t)
            for (int i = 0; i < NROWS; i++) {
                double sum = 0.0;
                for (int j = Arow_ptr[i]; j < Arow_ptr[i+1]; j++)
                    sum += Aval_ptr[j] * x[Acol_ptr[j]];
                result_arr[i] = sum;
            }
            break; 
        case 1: 
            #pragma omp parallel for schedule(dynamic,chunk_size) num_threads(t)
            for (int i = 0; i < NROWS; i++) {
                double sum = 0.0;
                for (int j = Arow_ptr[i]; j < Arow_ptr[i+1]; j++)
                    sum += Aval_ptr[j] * x[Acol_ptr[j]];
                result_arr[i] = sum; 
            }
            break;
        case 2: 
            #pragma omp parallel for schedule(guided,chunk_size) num_threads(t)
            for (int i = 0; i < NROWS; i++) {
                double sum = 0.0;
                for (int j = Arow_ptr[i]; j < Arow_ptr[i+1]; j++)
                    sum += Aval_ptr[j] * x[Acol_ptr[j]];
                result_arr[i] = sum;  
            }
            break; 
        
        case 3: 
            #pragma omp parallel for schedule(auto) num_threads(t)
            for (int i = 0; i < NROWS; i++) {
                double sum = 0.0; 
                for (int j = Arow_ptr[i]; j < Arow_ptr[i+1]; j++)
                    sum += Aval_ptr[j] *  x[Acol_ptr[j]];
                result_arr[i] = sum;
            }   
            break;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        printf("\nUSAGE:\n");
        printf("  %s MATRIX scheduling_type chunk_size num_threads\n", argv[0]);
        printf("\nEXAMPLE:\n");
        printf("  %s Trefethen_2000 static 10 8\n\n", argv[0]);
        return 1;
    }
    char *matrix_name =argv[1];
    char *sched_name= argv[2];
    int chunk = atoi(argv[3]);
    int threads= atoi(argv[4]);
    select_matrix(matrix_name); // for chosing the right matrix pointers 

    srand(time(NULL));
    double *x = malloc(NCOLS * sizeof(double));
    double *result_spmv =  malloc(NROWS * sizeof(double));
    if (!x || !result_spmv) {
        printf("Memory allocation failed\n");
        return 1; 
    }
    for (int i = 0; i < NCOLS; i++){
        x[i] = (double)rand() / RAND_MAX;
    }  

    const int RUNS = 15;
    double sequential_time[RUNS];
    double parallel_time[RUNS];
    //Running simulations for 15 times
    for (int r = 0; r < RUNS; r++) {
        flush_cache(); //done to avoid cache prefetching effect
        long long start = get_time_ns(); // Use long long
        sequential_SpMV(x, result_spmv); 
        long long end = get_time_ns();   // Use long long
        // Convert elapsed time (nanoseconds) to milliseconds
        sequential_time[r] = (double)(end - start) / 1000000.0; 
    }
    //runnin parallelized version
    for (int r = 0; r < RUNS; r++) {
        flush_cache();
        long long start = get_time_ns(); // Use long long
        parallel_SpMV(x, result_spmv, sched_name, chunk, threads);
        long long end = get_time_ns();   // Use long long
        // Convert elapsed time (nanoseconds) to milliseconds
        parallel_time[r] = (double)(end - start) / 1000000.0; 
    }
    double p90_seq = percentile90(sequential_time, RUNS);
    double p90_par = percentile90(parallel_time, RUNS);

    printf("\nRESULTS:\n");
    printf("SEQUENTIAL  :  90th = %.6f ms\n", p90_seq);
    printf("PARALLEL    :  90th = %.6f ms | speedup = %.2fx\n",
         p90_par, p90_seq / p90_par);

    printf("\nDONE.\n");

    free(x);
    free(result_spmv);

    return 0;
}
