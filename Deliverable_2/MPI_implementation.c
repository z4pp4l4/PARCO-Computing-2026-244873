#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define DEBUG 1
#if DEBUG
  #define DPRINTF(...) printf(__VA_ARGS__)
#else
  #define DPRINTF(...)
#endif

typedef struct {
    int row;
    int col;
    double val;
} COO_entry;

static int compare_coo_entries(const void *a, const void *b) {
    const COO_entry *ea = (const COO_entry *)a;
    const COO_entry *eb = (const COO_entry *)b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

static double env_double_or_neg(const char *name) {
    const char *s = getenv(name);
    if (!s || !*s) return -1.0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s) return -1.0;
    return v;
}

static double bytes_to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

/*
  In cyclic distribution across P ranks,
  rank r owns rows: r, r+P, r+2P, ...
  Returns how many such indices fit in [0, N)
*/
static int local_cyclic_size(int N, int stride, int offset) {
    if (offset >= N) return 0;
    return (N - offset + stride - 1) / stride;
}

static void read_header_mtx(
    MPI_File fh, int rank, int *rows, int *cols,
    int *nnz, int *is_pattern, int *is_symmetric, MPI_Offset *data_offset
) {
    int pattern = 0, symmetric = 0;

    if (rank == 0) {
        MPI_Offset off = 0;
        char line[1024];

        while (1) {
            int pos = 0;
            char ch;
            MPI_Status st;

            while (1) {
                MPI_File_read_at(fh, off, &ch, 1, MPI_CHAR, &st);
                off++;
                if (ch == '\n' || ch == '\r' || pos >= 1023) break;
                line[pos++] = ch;
            }
            line[pos] = '\0';
            if (pos == 0) continue;

            if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                char obj[32], fmt[32], field[32], symm[32];
                if (sscanf(line, "%%%%MatrixMarket %31s %31s %31s %31s",
                           obj, fmt, field, symm) == 4) {
                    if (strcmp(fmt, "coordinate") != 0) {
                        fprintf(stderr, "Only coordinate format supported\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    if (strcmp(field, "pattern") == 0) pattern = 1;
                    if (strcmp(symm, "symmetric") == 0) symmetric = 1;
                }
                break;
            }
        }

        while (1) {
            int pos = 0;
            char ch;
            MPI_Status st;
            while (1) {
                MPI_File_read_at(fh, off, &ch, 1, MPI_CHAR, &st);
                off++;
                if (ch == '\n' || ch == '\r' || pos >= 1023) break;
                line[pos++] = ch;
            }
            line[pos] = '\0';

            if (pos == 0 || line[0] == '%') continue;
            if (sscanf(line, "%d %d %d", rows, cols, nnz) == 3) {
                *data_offset = off;
                break;
            }
        }
    }

    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pattern, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *is_pattern   = pattern;
    *is_symmetric = symmetric;
}

static void read_and_distribute_1D(
    const char *filename, int rank, int size,
    int *rows, int *cols, int *global_nnz, int *is_pattern, int *is_symmetric,
    int *local_nnz, int **coo_r_local, int **coo_c_global, double **coo_v
) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    MPI_Offset data_offset = 0;
    read_header_mtx(fh, rank, rows, cols, global_nnz, is_pattern, is_symmetric, &data_offset);
    MPI_Offset data_size = file_size - data_offset;
    MPI_Offset chunk = data_size / size;

    MPI_Offset start = data_offset + rank * chunk;
    MPI_Offset end   = (rank == size - 1)
                        ? file_size
                        : data_offset + (rank + 1) * chunk;

    const MPI_Offset OVERLAP = 1 << 16;
    MPI_Offset read_end = end + OVERLAP;
    if (read_end > file_size) read_end = file_size;
    MPI_Offset read_size = read_end - start;

    char *buffer = malloc((size_t)read_size + 1);
    MPI_File_read_at_all(fh, start, buffer, (int)read_size, MPI_CHAR, MPI_STATUS_IGNORE);
    buffer[read_size] = '\0';

    char *p = buffer, *q = buffer + read_size;
    if (rank != 0) {
        while (p < q && *p != '\n') p++;
        if (p < q) p++;
    }
    if (rank != size - 1) {
        while (q > p && *(q - 1) != '\n') q--;
    }

    int cap = 4096, count = 0;
    int *lr = malloc(cap * sizeof(int));
    int *lc = malloc(cap * sizeof(int));
    double *lv = malloc(cap * sizeof(double));
    char *ptr = p;
    while (ptr < q && *ptr) {
        int r1, c1;
        double v = 1.0;
        int ok = 0;
        if (*is_pattern) {
            if (sscanf(ptr, "%d %d", &r1, &c1) == 2) 
                ok = 1;
        } else {
            if (sscanf(ptr, "%d %d %lf", &r1, &c1, &v) == 3) 
                ok = 1;
        }
        if (ok) {
            int i = r1 - 1;
            int j = c1 - 1;
            if ((i % size) == rank) {
                if (count == cap) {
                    cap *= 2;
                    lr = realloc(lr, cap * sizeof(int));
                    lc = realloc(lc, cap * sizeof(int));
                    lv = realloc(lv, cap * sizeof(double));
                }
                lr[count] = i / size;
                lc[count] = j;
                lv[count] = v;
                count++;
            }

            if (*is_symmetric && i != j && (j % size) == rank) {
                if (count == cap) {
                    cap *= 2;
                    lr = realloc(lr, cap * sizeof(int));
                    lc = realloc(lc, cap * sizeof(int));
                    lv = realloc(lv, cap * sizeof(double));
                }
                lr[count] = j / size;
                lc[count] = i;
                lv[count] = v;
                count++;
            }
        }
        while (ptr < q && *ptr != '\n') ptr++;
        if (ptr < q) ptr++;
    }

    free(buffer);
    MPI_File_close(&fh);
    *local_nnz = count;
    *coo_r_local = lr;
    *coo_c_global = lc;
    *coo_v = lv;
}

static void coo_to_csr(
    int local_nnz, int local_rows, const int *coo_r, const int *coo_c, const double *coo_v,
    int **row_ptr, int **col_idx, double **vals
) {
    COO_entry *entries = (COO_entry*)malloc((size_t)local_nnz * sizeof(COO_entry));
    if (!entries) {
        fprintf(stderr, "ERROR: COO_entry allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int k = 0; k < local_nnz; k++) {
        entries[k].row = coo_r[k];
        entries[k].col = coo_c[k];
        entries[k].val = coo_v[k];
    }

    qsort(entries, (size_t)local_nnz, sizeof(COO_entry), compare_coo_entries);
    *row_ptr = (int*)calloc((size_t)local_rows + 1, sizeof(int));
    *col_idx = (int*)malloc((size_t)local_nnz * sizeof(int));
    *vals    = (double*)malloc((size_t)local_nnz * sizeof(double));

    if (!(*row_ptr) || !(*col_idx) || !(*vals)) {
        fprintf(stderr, "ERROR: CSR allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int k = 0; k < local_nnz; k++) {
        int r = entries[k].row;
        if (r >= 0 && r < local_rows) (*row_ptr)[r + 1]++;
    }
    for (int r = 0; r < local_rows; r++){ 
        (*row_ptr)[r + 1] += (*row_ptr)[r];
    }
    int *cursor = (int*)malloc((size_t)local_rows * sizeof(int));
    memcpy(cursor, *row_ptr, (size_t)local_rows * sizeof(int));

    for (int k = 0; k < local_nnz; k++) {
        int r = entries[k].row;
        int dest = cursor[r]++;
        (*col_idx)[dest] = entries[k].col; // GLOBAL col
        (*vals)[dest]    = entries[k].val;
    }

    free(cursor);
    free(entries);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    omp_set_num_threads(1); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    if (argc < 2) {
        MPI_Finalize();
        return 0;
    }

    int rows = 0, cols = 0, global_nnz = 0;
    int local_nnz = 0;
    int *coo_r = NULL, *coo_c = NULL;
    double *coo_v = NULL;
    int is_pattern = 0, is_symmetric = 0;
    read_and_distribute_1D(argv[1], rank, size,&rows, &cols, &global_nnz, &is_pattern, &is_symmetric, &local_nnz, &coo_r, &coo_c, &coo_v);
    int local_num_rows = local_cyclic_size(rows, size, rank);

    if (rank == 0) {
        printf("[Rank 0] Matrix: %d x %d, nnz=%d | 1D cyclic rows over P=%d\n",rows, cols, global_nnz, size);
    }

    int sum_local_nnz = 0;
    MPI_Reduce(&local_nnz, &sum_local_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *row_ptr = NULL, *col_idx = NULL;
    double *vals = NULL;
    coo_to_csr(local_nnz, local_num_rows, coo_r, coo_c, coo_v, &row_ptr, &col_idx, &vals);

    long long local_flops = 2LL * (long long)local_nnz;
    long long global_flops = 0;
    MPI_Reduce(&local_flops, &global_flops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // NNZ stats
    int nnz_min = 0, nnz_max = 0;
    MPI_Allreduce(&local_nnz, &nnz_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_nnz, &nnz_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    long long local_nnz_ll = (long long)local_nnz;
    long long nnz_sum = 0;
    MPI_Allreduce(&local_nnz_ll, &nnz_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    double nnz_avg = (size > 0) ? ((double)nnz_sum / (double)size) : 0.0;

    // Allocate vectors
    double *x = (double*)malloc((size_t)cols * sizeof(double));
    double *y_local = (double*)calloc((size_t)local_num_rows, sizeof(double));
    if (!x || !y_local) {
        fprintf(stderr, "[Rank %d] ERROR: x/y allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // x generation on rank 0, broadcast to all
    if (rank == 0) {
        srand(12345);
        for (int j = 0; j < cols; j++) x[j] = (double)rand() / RAND_MAX;
    }

    double t_total_start = MPI_Wtime();
    double t_comp = 0.0, t_comm = 0.0;
    double t0 = MPI_Wtime();
    MPI_Bcast(x, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;

    // SpMV local
    double t1 = MPI_Wtime();
    double start_time = MPI_Wtime();
    #pragma omp parallel for
    for (int r = 0; r < local_num_rows; r++) {
        double acc = 0.0;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; k++) {
            int j = col_idx[k];
            acc += vals[k] * x[j];
        }
        y_local[r] = acc;
    }
    t_comp += MPI_Wtime() - t1;
    DPRINTF("[Rank %d] Local SpMV time: %.6f s\n", rank, MPI_Wtime() - start_time);
    double t_total = MPI_Wtime() - t_total_start;

    // Gather y to rank 0 (rank-ordered / cyclic order)
    int *recvcounts = NULL, *displs = NULL;
    double *y_gather = NULL;
    int sendcount = local_num_rows;

    if (rank == 0) {
        recvcounts = (int*)malloc((size_t)size * sizeof(int));
        displs     = (int*)malloc((size_t)size * sizeof(int));
        y_gather   = (double*)malloc((size_t)rows * sizeof(double));
    }
    t0 = MPI_Wtime();
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    t0 = MPI_Wtime();
    MPI_Gatherv(y_local, sendcount, MPI_DOUBLE, y_gather, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;

    // Worst-rank timings
    double t_total_max, t_comp_max, t_comm_max;
    MPI_Allreduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comp,  &t_comp_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comm,  &t_comm_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double comp_pct = 100.0 * t_comp_max / t_total_max;
    double comm_pct = 100.0 * t_comm_max / t_total_max;
    double gflops = (2.0 * (double)nnz_sum) / (t_total_max * 1e9);

    // baseline
    double T1 = env_double_or_neg("BASELINE_T1");
    double speedup = (T1 > 0) ? T1 / t_total_max : -1.0;
    double efficiency = (speedup > 0) ? speedup / size : -1.0;

    // Memory estimate (MiB)
    double mem_bytes =
        (double)local_nnz * sizeof(int)      +  // coo_r
        (double)local_nnz * sizeof(int)      +  // coo_c
        (double)local_nnz * sizeof(double)   +  // coo_v
        (double)(local_num_rows + 1) * sizeof(int) +
        (double)local_nnz * sizeof(int)      +
        (double)local_nnz * sizeof(double)   +
        (double)cols * sizeof(double)        +  // x full
        (double)local_num_rows * sizeof(double); // y_local

    double mem_mib = bytes_to_mib(mem_bytes);
    double mem_mib_max = 0.0;
    MPI_Allreduce(&mem_mib, &mem_mib_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n");
        printf("-----------------------------------------------------------------------------------------------\n");
        printf("  P |  Time(s) | Comp%% | Comm%% |  GFLOP/s | Speedup |  Eff%% | Mem(MiB) | NNZ min | NNZ avg | NNZ max\n");
        printf("-----------------------------------------------------------------------------------------------\n");
        printf("%3d | %8.4f | %5.1f | %5.1f | %9.3f | ", size, t_total_max, comp_pct, comm_pct, gflops);

        if (speedup < 0) printf("   N/A   |   N/A  | ");
        else printf("%7.2f | %6.2f | ", speedup, efficiency * 100.0);

        printf("%8.2f | %7d | %8.1f | %7d\n",mem_mib_max, nnz_min, nnz_avg, nnz_max);
        if (global_flops > 0) {
            printf("Global FLOPs per SpMV: %lld\n", global_flops);
        }

        // Reorder to true row order (since gathered by rank order)
        double *y_correct = (double*)malloc((size_t)rows * sizeof(double));
        int pos = 0;
        for (int rnk = 0; rnk < size; rnk++) {
            for (int li = 0; li < recvcounts[rnk]; li++) {
                int global_row = rnk + li * size; // cyclic rows
                y_correct[global_row] = y_gather[pos++];
            }
        }
        
        free(y_correct);
        free(y_gather);
        free(recvcounts);
        free(displs);
    }

    free(coo_r);
    free(coo_c);
    free(coo_v);
    free(row_ptr);
    free(col_idx);
    free(vals);
    free(x);
    free(y_local);

    MPI_Finalize();
    return 0;
}
