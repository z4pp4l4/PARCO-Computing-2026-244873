#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <time.h>


typedef struct {
    int row;
    int col;
    double val;
} COO_entry;

/* 
    Comparison function to sort COO entries by (row, col),
    required for correct CSR construction.
 */

static int compare_coo_entries(const void *a, const void *b) {
    const COO_entry *ea = (const COO_entry *)a;
    const COO_entry *eb = (const COO_entry *)b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

/* Read a double from environment variables.
   Used to optionally inject a sequential baseline (T1)
   for speedup and efficiency computation.
 */

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
static inline int owner_cyclic(int idx, int size) {
    return idx % size;
}

static inline int local_index_cyclic(int global_idx, int size) {
    // valid only if owner(global_idx)==rank
    return global_idx / size;
}

// deterministic "random" in [0,1) from global index (no communication needed)
static inline double x_value_from_index(int j) {
    uint32_t x = (uint32_t)j * 1103515245u + 12345u;
    x &= 0x7fffffff;
    return (double)x / (double)0x7fffffff;
}

static int cmp_int(const void *a, const void *b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

// binary search in sorted array
static int bsearch_int(const int *arr, int n, int key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo)/2;
        int v = arr[mid];
        if (v == key) return mid;
        if (v < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Parse Matrix Market header and broadcast metadata.
   Only rank 0 reads the header; all ranks receive matrix info.
 */
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

static void read_and_distribute_1D_root(
    const char *filename, int rank, int size,
    int *rows, int *cols, int *global_nnz,
    int *is_pattern, int *is_symmetric,
    int *local_nnz, int **coo_r_local,
    int **coo_c_global, double **coo_v
) {
    int *all_r = NULL, *all_c = NULL;
    double *all_v = NULL;

    int *send_counts = NULL;
    int *send_displs = NULL;

    if (rank == 0) {
        FILE *f = fopen(filename, "r");
        if (!f) {
            fprintf(stderr, "Cannot open file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[1024];
        *is_pattern = 0;
        *is_symmetric = 0;

        /* Parse Matrix Market header */
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '%') {
                if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                    char obj[32], fmt[32], field[32], symm[32];
                    sscanf(line, "%%%%MatrixMarket %31s %31s %31s %31s",
                           obj, fmt, field, symm);
                    if (strcmp(field, "pattern") == 0) *is_pattern = 1;
                    if (strcmp(symm, "symmetric") == 0) *is_symmetric = 1;
                }
                continue;
            }
            sscanf(line, "%d %d %d", rows, cols, global_nnz);
            break;
        }

        all_r = malloc(*global_nnz * sizeof(int));
        all_c = malloc(*global_nnz * sizeof(int));
        all_v = malloc(*global_nnz * sizeof(double));

        int count = 0;
        while (count < *global_nnz && fgets(line, sizeof(line), f)) {
            int r, c;
            double v = 1.0;
            if (*is_pattern) {
                sscanf(line, "%d %d", &r, &c);
            } else {
                sscanf(line, "%d %d %lf", &r, &c, &v);
            }
            all_r[count] = r - 1;
            all_c[count] = c - 1;
            all_v[count] = v;
            count++;

            if (*is_symmetric && r != c) {
                all_r[count] = c - 1;
                all_c[count] = r - 1;
                all_v[count] = v;
                count++;
            }
        }
        *global_nnz = count;
        fclose(f);

        send_counts = calloc(size, sizeof(int));
        for (int k = 0; k < *global_nnz; k++) {
            int owner = all_r[k] % size;
            send_counts[owner]++;
        }

        send_displs = calloc(size, sizeof(int));
        for (int r = 1; r < size; r++)
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    }

    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(is_pattern, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(is_symmetric, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(send_counts, 1, MPI_INT, local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *coo_r_local  = malloc(*local_nnz * sizeof(int));
    *coo_c_global = malloc(*local_nnz * sizeof(int));
    *coo_v        = malloc(*local_nnz * sizeof(double));

    int *tmp_r = NULL, *tmp_c = NULL;
    double *tmp_v = NULL;

    if (rank == 0) {
        tmp_r = malloc(*global_nnz * sizeof(int));
        tmp_c = malloc(*global_nnz * sizeof(int));
        tmp_v = malloc(*global_nnz * sizeof(double));

        int *pos = calloc(size, sizeof(int));
        for (int k = 0; k < *global_nnz; k++) {
            int owner = all_r[k] % size;
            int p = send_displs[owner] + pos[owner]++;
            tmp_r[p] = all_r[k];
            tmp_c[p] = all_c[k];
            tmp_v[p] = all_v[k];
        }
        free(pos);
    }

    MPI_Scatterv(tmp_r, send_counts, send_displs, MPI_INT,
                 *coo_r_local, *local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(tmp_c, send_counts, send_displs, MPI_INT,
                 *coo_c_global, *local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(tmp_v, send_counts, send_displs, MPI_DOUBLE,
                 *coo_v, *local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int k = 0; k < *local_nnz; k++)
        (*coo_r_local)[k] /= size;

    if (rank == 0) {
        free(all_r); free(all_c); free(all_v);
        free(tmp_r); free(tmp_c); free(tmp_v);
        free(send_counts); free(send_displs);
    }
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
    //omp_set_num_threads(1); 
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
    read_and_distribute_1D_root(argv[1], rank, size,&rows, &cols, &global_nnz, &is_pattern, &is_symmetric, &local_nnz, &coo_r, &coo_c, &coo_v);
    int local_num_rows = local_cyclic_size(rows, size, rank);

    double *y_local = (double*)calloc((size_t)local_num_rows, sizeof(double));
    if (!y_local) {
        fprintf(stderr, "[Rank %d] ERROR: y_local allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    if (rank == 0) {
        printf("[Rank 0] Matrix: %d x %d, nnz=%d | 1D cyclic rows over P=%d\n",rows, cols, global_nnz, size);
    }

    int sum_local_nnz = 0;
    MPI_Reduce(&local_nnz, &sum_local_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *row_ptr = NULL, *col_idx = NULL;
    double *vals = NULL;
    coo_to_csr(local_nnz, local_num_rows, coo_r, coo_c, coo_v, &row_ptr, &col_idx, &vals);

    // BUILD GHOST COLUMN LIST
    int *ghost_cols = NULL;
    int ghost_count = 0;

    {
        int *tmp = malloc((size_t)local_nnz * sizeof(int));
        int tmp_n = 0;

        for (int k = 0; k < local_nnz; k++) {
            int j = col_idx[k];
            if (owner_cyclic(j, size) != rank) {
                tmp[tmp_n++] = j;
            }
        }
        qsort(tmp, (size_t)tmp_n, sizeof(int), cmp_int);

        // unique
        int uniq = 0;
        for (int i = 0; i < tmp_n; i++) {
            if (i == 0 || tmp[i] != tmp[i-1]) {
                tmp[uniq++] = tmp[i];
            }
        }

        ghost_cols = malloc((size_t)uniq * sizeof(int));
        memcpy(ghost_cols, tmp, (size_t)uniq * sizeof(int));
        ghost_count = uniq;

        free(tmp);
    }



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


    // Load imbalance ratio (max/avg)
    double load_imbalance = (nnz_avg > 0) ? ((double)nnz_max / nnz_avg) : 1.0;

    // Ghost stats for communication volume analysis
    int ghost_min = 0, ghost_max = 0;
    long long ghost_count_ll = (long long)ghost_count;
    long long ghost_sum = 0;
    MPI_Allreduce(&ghost_count, &ghost_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ghost_count, &ghost_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&ghost_count_ll, &ghost_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    double ghost_avg = (size > 0) ? ((double)ghost_sum / (double)size) : 0.0;

    // Local x vector (cyclic ownership over columns)
    int x_local_n = local_cyclic_size(cols, size, rank);
    double *x_local = (double*)malloc((size_t)x_local_n * sizeof(double));
    if (!x_local) {
        fprintf(stderr, "[Rank %d] ERROR: x_local allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill local x deterministically (no comm)
    for (int li = 0; li < x_local_n; li++) {
        int global_j = rank + li * size;
        if (global_j < cols) {
            x_local[li] = x_value_from_index(global_j);
        }
    }

    double t_total_start = MPI_Wtime();
    double t_comp = 0.0, t_comm = 0.0;
    double t0 = MPI_Wtime();


    // HALO EXCHANGE for ghost x
    double *ghost_vals = NULL;

    int *send_counts = calloc(size, sizeof(int));
    int *recv_counts = calloc(size, sizeof(int));
    int *send_displs = calloc(size, sizeof(int));
    int *recv_displs = calloc(size, sizeof(int));

    // Count requests per owner
    for (int g = 0; g < ghost_count; g++) {
        int j = ghost_cols[g];
        send_counts[ owner_cyclic(j, size) ]++;
    }

    // Displacements
    for (int r = 1; r < size; r++)
        send_displs[r] = send_displs[r-1] + send_counts[r-1];

    // Pack indices
    int *send_idx = malloc(ghost_count * sizeof(int));
    int *pos = calloc(size, sizeof(int));
    for (int g = 0; g < ghost_count; g++) {
        int j = ghost_cols[g];
        int r = owner_cyclic(j, size);
        send_idx[ send_displs[r] + pos[r]++ ] = j;
    }
    free(pos);

   // Track Alltoall counts time
    double t_alltoall1 = MPI_Wtime();
    MPI_Alltoall(send_counts, 1, MPI_INT,
                recv_counts, 1, MPI_INT,
                MPI_COMM_WORLD);
    double t_exchange_counts = MPI_Wtime() - t_alltoall1;
    t_comm += t_exchange_counts;

    // Receive displs
    int tot_recv = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r] = tot_recv;
        tot_recv += recv_counts[r];
    }

    // Receive requested indices
    int *recv_idx = malloc(tot_recv * sizeof(int));

    // Track Alltoallv indices time
    t0 = MPI_Wtime();
    MPI_Alltoallv(send_idx, send_counts, send_displs, MPI_INT,
                recv_idx, recv_counts, recv_displs, MPI_INT,
                MPI_COMM_WORLD);
    double t_exchange_indices = MPI_Wtime() - t0;
    t_comm += t_exchange_indices;

    // Prepare values to send
    double *send_vals = malloc(tot_recv * sizeof(double));
    for (int i = 0; i < tot_recv; i++) {
        int j = recv_idx[i];
        send_vals[i] = x_local[ local_index_cyclic(j, size) ];
    }

    // Receive ghost values
    ghost_vals = malloc(ghost_count * sizeof(double));

    // Track Alltoallv ghost values time (main halo exchange)
    t0 = MPI_Wtime();
    MPI_Alltoallv(send_vals, recv_counts, recv_displs, MPI_DOUBLE, 
                ghost_vals, send_counts, send_displs, MPI_DOUBLE, 
                MPI_COMM_WORLD);
    double t_ghost_exchange = MPI_Wtime() - t0;
    t_comm += t_ghost_exchange;

    // Cleanup temp buffers
    free(send_idx);
    free(recv_idx);
    free(send_vals);
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);

    // SpMV local computation
    double start_time = MPI_Wtime();
    //#pragma omp parallel for
    for (int r = 0; r < local_num_rows; r++) {
        double acc = 0.0;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; k++) {
            int j = col_idx[k];
            double xj;

            if (owner_cyclic(j, size) == rank) {
                xj = x_local[ local_index_cyclic(j, size) ];
            } else {
                int idx = bsearch_int(ghost_cols, ghost_count, j);
                if (idx < 0) {
                    fprintf(stderr, "[Rank %d] ERROR: ghost column %d not found\n", rank, j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                xj = ghost_vals[idx];
            }
            acc += vals[k] * xj;
        }
        y_local[r] = acc;
    }
    double t_spmv = MPI_Wtime() - start_time;
    t_comp += t_spmv;
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
   // Track Gather counts time
    t0 = MPI_Wtime();
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double t_gather_counts = MPI_Wtime() - t0;
    t_comm += t_gather_counts;

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    // Track Gatherv result time
    t0 = MPI_Wtime();
    MPI_Gatherv(y_local, sendcount, MPI_DOUBLE, y_gather, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_gatherv = MPI_Wtime() - t0;
    t_comm += t_gatherv;

    // Collect timing statistics (max = worst case)
    double t_total_max, t_comp_max, t_comm_max;
    double t_ghost_max, t_spmv_max, t_gatherv_max;

    MPI_Allreduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comp,  &t_comp_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comm,  &t_comm_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_ghost_exchange, &t_ghost_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_spmv,  &t_spmv_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_gatherv, &t_gatherv_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
        (double)x_local_n * sizeof(double)   +  // x_local
        (double)ghost_count * sizeof(double) +  // ghost_vals
        (double)ghost_count * sizeof(int)    +  // ghost_cols
        (double)local_num_rows * sizeof(double); // y_local

    double mem_mib = bytes_to_mib(mem_bytes);
    double mem_mib_max = 0.0;
    double mem_mib_avg = 0.0;
    MPI_Allreduce(&mem_mib, &mem_mib_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&mem_mib, &mem_mib_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mem_mib_avg /= size;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n");
        printf("                    1D DISTRIBUTED SpMV - PERFORMANCE REPORT                   \n");
        printf("********************************************************************************\n\n");
        
        printf("MATRIX INFORMATION:\n");
        printf("  Matrix file:        %s\n", argv[1]);
        printf("  Dimensions:         %d x %d\n", rows, cols);
        printf("  Total NNZ:          %d\n", global_nnz);
        printf("  Matrix type:        %s, %s\n", 
            is_pattern ? "pattern" : "real", 
            is_symmetric ? "symmetric" : "general");
        printf("\n");
        
        printf("PARALLEL CONFIGURATION:\n");
        printf("  Total processes:    %d\n", size);
        printf("  Partitioning:       1D cyclic (row-wise, modulo)\n");
        printf("\n");
        
        printf("LOAD BALANCE:\n");
        printf("  NNZ per rank:       min=%d  avg=%.1f  max=%d\n", 
            nnz_min, nnz_avg, nnz_max);
        printf("  Imbalance ratio:    %.3f  (max/avg)\n", load_imbalance);
        printf("\n");
        
        printf("COMMUNICATION VOLUME:\n");
        printf("  Ghost cols/rank:    min=%d  avg=%.1f  max=%d\n",
            ghost_min, ghost_avg, ghost_max);
        printf("  Total ghost cols:   %lld\n", ghost_sum);
        double ghost_ratio = (nnz_sum > 0)
            ? 100.0 * ghost_sum / nnz_sum
            : 0.0;

        printf("  Ghost ratio:        %.2f%%  (ghost/total NNZ)\n", ghost_ratio);
        printf("\n");
        
        printf("MEMORY USAGE:\n");
        printf("  Per-rank memory:    avg=%.2f MiB  max=%.2f MiB\n", 
            mem_mib_avg, mem_mib_max);
        printf("  Total memory:       %.2f MiB  (estimated)\n", 
            mem_mib_avg * size);
        printf("\n");
        
        printf("PERFORMANCE METRICS:\n");
        printf("  Total time:         %.6f s\n", t_total_max);
        printf("  Computation time:   %.6f s  (%.1f%%)\n", t_comp_max, comp_pct);
        printf("    └─| Local SpMV:    %.6f s\n", t_spmv_max);
        printf("  Communication time: %.6f s  (%.1f%%)\n", t_comm_max, comm_pct);
        printf("    ├─| Ghost exchange: %.6f s\n", t_ghost_max);
        printf("    └─| Gatherv:       %.6f s\n", t_gatherv_max);
        printf("\n");
        
        printf("COMPUTATIONAL INTENSITY:\n");
        printf("  Total FLOPs:        %lld\n", global_flops);
        printf("  GFLOP/s:            %.3f\n", gflops);
        printf("\n");
        
        if (speedup > 0) {
            printf("SCALABILITY:\n");
            printf("  Baseline (T1):      %.6f s\n", T1);
            printf("  Speedup:            %.2fx\n", speedup);
            printf("  Efficiency:         %.2f%%\n", efficiency * 100.0);
            printf("\n");
        }
        
        printf("********************************************************************************\n");
        printf("SUMMARY TABLE:\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("  P | Time(s) | Comp%% | Comm%% | GFLOP/s | Speedup |  Eff%%  | Imbal | Ghost%%\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("%3d | %7.4f | %5.1f | %5.1f | %7.3f | ",
            size, t_total_max, comp_pct, comm_pct, gflops);
        
        if (speedup < 0) {
            printf("  N/A   |  N/A  | ");
        } else {
            printf("%7.2f | %5.1f | ", speedup, efficiency * 100.0);
        }
        
        printf("%.3f | %5.1f\n", load_imbalance, ghost_ratio);
        printf("--------------------------------------------------------------------------------\n");
        printf("\n");

        // Reorder to true row order
        double *y_correct = (double*)malloc((size_t)rows * sizeof(double));
        int pos = 0;
        for (int rnk = 0; rnk < size; rnk++) {
            for (int li = 0; li < recvcounts[rnk]; li++) {
                int global_row = rnk + li * size;
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
    free(x_local);
    free(y_local);
    free(ghost_cols);
    free(ghost_vals);


    MPI_Finalize();
    return 0;
}
