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

static int compare_coo_entries(const void *a, const void *b) {
    const COO_entry *ea = (const COO_entry *)a;
    const COO_entry *eb = (const COO_entry *)b;
    if (ea->row != eb->row)
        return ea->row - eb->row;
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
   BLOCK PARTITIONING: Processor pr owns rows [start_row, end_row)
*/
static void get_block_row_range(int N, int Px, int pr, int *start, int *end) {
    int base_size = N / Px;
    int remainder = N % Px;
    
    if (pr < remainder) {
        *start = pr * (base_size + 1);
        *end = *start + base_size + 1;
    } else {
        *start = pr * base_size + remainder;
        *end = *start + base_size;
    }
}

static void get_block_col_range(int N, int Py, int pc, int *start, int *end) {
    int base_size = N / Py;
    int remainder = N % Py;
    
    if (pc < remainder) {
        *start = pc * (base_size + 1);
        *end = *start + base_size + 1;
    } else {
        *start = pc * base_size + remainder;
        *end = *start + base_size;
    }
}

/* Returns the owner processor for a given global row in block distribution */
static int get_row_owner(int global_row, int N, int Px) {
    int base_size = N / Px;
    int remainder = N % Px;
    int threshold = remainder * (base_size + 1);
    
    if (global_row < threshold) {
        return global_row / (base_size + 1);
    } else {
        return remainder + (global_row - threshold) / base_size;
    }
}

/* Returns the owner processor for a given global col in block distribution */
static int get_col_owner(int global_col, int N, int Py) {
    int base_size = N / Py;
    int remainder = N % Py;
    int threshold = remainder * (base_size + 1);
    
    if (global_col < threshold) {
        return global_col / (base_size + 1);
    } else {
        return remainder + (global_col - threshold) / base_size;
    }
}

static void read_header_mtx(
    MPI_File fh, int rank, int *rows, int *cols, int *nnz, int *is_pattern, int *is_symmetric, MPI_Offset *data_offset
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
                off += 1;
                if (ch == '\n' || ch == '\r' || pos >= (int)sizeof(line) - 1) break;
                line[pos++] = ch;
            }
            line[pos] = '\0';

            if (pos == 0) continue;

            if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                char object[64], format[64], field[64], symmetry[64];
                int n = sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s",
                               object, format, field, symmetry);

                if (n >= 4) {
                    if (strcmp(format, "coordinate") != 0) {
                        fprintf(stderr, "[Rank 0] ERROR: only coordinate format supported.\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    if (strcmp(field, "pattern") == 0) pattern = 1;
                    if (strcmp(symmetry, "symmetric") == 0) symmetric = 1;
                } else {
                    fprintf(stderr, "[Rank 0] ERROR: cannot parse MatrixMarket banner: '%s'\n", line);
                    MPI_Abort(MPI_COMM_WORLD, 1);
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
                off += 1;
                if (ch == '\n' || ch == '\r' || pos >= (int)sizeof(line) - 1) break;
                line[pos++] = ch;
            }
            line[pos] = '\0';

            if (pos == 0) continue;
            if (line[0] == '%') continue;

            if (sscanf(line, "%d %d %d", rows, cols, nnz) == 3) {
                *data_offset = off;
                break;
            } else {
                fprintf(stderr, "[Rank 0] ERROR: cannot parse dimensions line: '%s'\n", line);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pattern, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *is_pattern = pattern;
    *is_symmetric = symmetric;
}

static void read_and_distribute_2D_block(
    const char *filename, int rank, int size, MPI_Comm grid_comm,
    int Px, int Py, int pr, int pc,
    int *rows, int *cols, int *global_nnz,
    int *is_pattern, int *is_symmetric,
    int *local_nnz,
    int **coo_r_local, int **coo_c_local, double **coo_v
) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);

    MPI_Offset data_offset;
    read_header_mtx(fh, rank, rows, cols, global_nnz, is_pattern, is_symmetric, &data_offset);

    MPI_Offset data_size = file_size - data_offset;
    MPI_Offset chunk = data_size / size;

    MPI_Offset start = data_offset + rank * chunk;
    MPI_Offset end   = (rank == size - 1)
                     ? file_size
                     : data_offset + (rank + 1) * chunk;

    const MPI_Offset OVERLAP = 4096;
    MPI_Offset read_end = (end + OVERLAP < file_size) ? end + OVERLAP : file_size;
    MPI_Offset read_size = read_end - start;

    char *buffer = malloc(read_size + 1);
    MPI_File_read_at_all(fh, start, buffer, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
    buffer[read_size] = '\0';

    MPI_File_close(&fh);

    char *p = buffer;
    char *q = buffer + read_size;

    if (rank != 0) {
        while (p < q && *p != '\n') p++;
        if (p < q) p++;
    }
    if (rank != size - 1) {
        while (q > p && *(q - 1) != '\n') q--;
    }

    /* PHASE A: parse all entries */
    int cap = 4096;
    int count = 0;
    COO_entry *parsed = malloc(cap * sizeof(COO_entry));

    char *ptr = p;
    while (ptr < q) {
        int r, c;
        double v = 1.0;

        char *p2, *p3;
        r = (int)strtol(ptr, &p2, 10);
        c = (int)strtol(p2, &p3, 10);

        if (p2 != ptr && p3 != p2) {
            if (!(*is_pattern))
                v = strtod(p3, NULL);

            if (count >= cap) {
                cap *= 2;
                parsed = realloc(parsed, cap * sizeof(COO_entry));
                if (!parsed) {
                    fprintf(stderr, "realloc failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            parsed[count++] = (COO_entry){r - 1, c - 1, v};

            if (*is_symmetric && r != c) {
                if (count >= cap) {
                    cap *= 2;
                    parsed = realloc(parsed, cap * sizeof(COO_entry));
                    if (!parsed) MPI_Abort(MPI_COMM_WORLD, 1);
                }
                parsed[count++] = (COO_entry){c - 1, r - 1, v};
            }
        }

        while (ptr < q && *ptr != '\n') ptr++;
        if (ptr < q) ptr++;
    }

    free(buffer);

    /* PHASE B: redistribute 2D BLOCK */
    int *send_cnt = calloc(size, sizeof(int));
    for (int k = 0; k < count; k++) {
        int i = parsed[k].row;
        int j = parsed[k].col;
        int owner_r = get_row_owner(i, *rows, Px);  // BLOCK
        int owner_c = get_col_owner(j, *cols, Py);  // BLOCK
        send_cnt[owner_r * Py + owner_c]++;
    }

    int *recv_cnt = calloc(size, sizeof(int));
    MPI_Alltoall(send_cnt, 1, MPI_INT,
                 recv_cnt, 1, MPI_INT, MPI_COMM_WORLD);

    int *sdisp = calloc(size, sizeof(int));
    int *rdisp = calloc(size, sizeof(int));
    for (int i = 1; i < size; i++) {
        sdisp[i] = sdisp[i-1] + send_cnt[i-1];
        rdisp[i] = rdisp[i-1] + recv_cnt[i-1];
    }

    int total_recv = rdisp[size-1] + recv_cnt[size-1];
    COO_entry *sendbuf = malloc(count * sizeof(COO_entry));
    COO_entry *local   = malloc(total_recv * sizeof(COO_entry));

    int *tmp = calloc(size, sizeof(int));
    for (int k = 0; k < count; k++) {
        int i = parsed[k].row;
        int j = parsed[k].col;
        int owner_r = get_row_owner(i, *rows, Px);  // BLOCK
        int owner_c = get_col_owner(j, *cols, Py);  // BLOCK
        int owner   = owner_r * Py + owner_c;

        sendbuf[sdisp[owner] + tmp[owner]++] = parsed[k];
    }
    
    for (int i = 0; i < size; i++) {
        send_cnt[i] *= sizeof(COO_entry);
        recv_cnt[i] *= sizeof(COO_entry);
        sdisp[i]    *= sizeof(COO_entry);
        rdisp[i]    *= sizeof(COO_entry);
    }

    MPI_Alltoallv(sendbuf, send_cnt, sdisp, MPI_BYTE, 
                  local,   recv_cnt, rdisp, MPI_BYTE, MPI_COMM_WORLD);

    *local_nnz = total_recv;
    long long check;
    long long l = *local_nnz;
    MPI_Allreduce(&l, &check, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        printf("CHECK NNZ: %lld (expected %d)\n", check, *global_nnz);

    *coo_r_local = malloc(total_recv * sizeof(int));
    *coo_c_local = malloc(total_recv * sizeof(int));
    *coo_v       = malloc(total_recv * sizeof(double));

    // Get this processor's row range for BLOCK partitioning
    int row_start, row_end;
    get_block_row_range(*rows, Px, pr, &row_start, &row_end);

    for (int k = 0; k < total_recv; k++) {
        int i = local[k].row;
        // Safety check
        if (i < row_start || i >= row_end) {
            fprintf(stderr,
                "[Rank %d] ERROR: wrong row ownership i=%d not in [%d,%d)\n",
                rank, i, row_start, row_end);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // GLOBAL → LOCAL row index
        local[k].row = i - row_start;
    }
    
    for (int k = 0; k < total_recv; k++) {
        (*coo_r_local)[k] = local[k].row;
        (*coo_c_local)[k] = local[k].col;
        (*coo_v)[k]       = local[k].val;
    }

    free(parsed);
    free(sendbuf);
    free(local);
    free(send_cnt); free(recv_cnt);
    free(sdisp); free(rdisp); free(tmp);
}

static void coo_to_csr(
    int local_nnz, int local_rows,
    const int *coo_r,const int *coo_c,const double *coo_v,
    int **row_ptr,int **col_idx,double **vals
) {
    COO_entry *entries = malloc(local_nnz * sizeof(COO_entry));
    if (!entries) {
        fprintf(stderr, "ERROR: COO_entry allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int k = 0; k < local_nnz; k++) {
        entries[k].row = coo_r[k];
        entries[k].col = coo_c[k];
        entries[k].val = coo_v[k];
    }
    
    qsort(entries, local_nnz, sizeof(COO_entry), compare_coo_entries);

    *row_ptr = calloc(local_rows + 1, sizeof(int));
    *col_idx = malloc(local_nnz * sizeof(int));
    *vals    = malloc(local_nnz * sizeof(double));

    if (!(*row_ptr) || !(*col_idx) || !(*vals)) {
        fprintf(stderr, "ERROR: CSR allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int k = 0; k < local_nnz; k++) {
        int r = entries[k].row;
        if (r >= 0 && r < local_rows)
            (*row_ptr)[r + 1]++;
    }
    for (int r = 0; r < local_rows; r++)
        (*row_ptr)[r + 1] += (*row_ptr)[r];

    int *cursor = malloc(local_rows * sizeof(int));
    memcpy(cursor, *row_ptr, local_rows * sizeof(int));

    for (int k = 0; k < local_nnz; k++) {
        int r = entries[k].row;
        int dest = cursor[r]++;
        (*col_idx)[dest] = entries[k].col;
        (*vals)[dest]    = entries[k].val;
    }

    free(cursor);
    free(entries);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s matrix.mtx\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0];
    int Py = dims[1];

    int periods[2] = {0, 0};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int pr = coords[0];
    int pc = coords[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, pr, pc, &row_comm);
    MPI_Comm_split(grid_comm, pc, pr, &col_comm);
    
    int rows = 0, cols = 0, global_nnz = 0;
    int local_nnz = 0;
    int *coo_r = NULL, *coo_c = NULL;
    double *coo_v = NULL;
    int is_pattern = 0, is_symmetric = 0;

    read_and_distribute_2D_block(
        argv[1], rank, size, grid_comm, Px, Py, pr, pc,
        &rows, &cols, &global_nnz, &is_pattern, &is_symmetric, 
        &local_nnz, &coo_r, &coo_c, &coo_v
    );

    if (rank == 0) {
        printf("[Rank 0] Matrix: %d x %d, nnz=%d | grid=%dx%d | pattern=%d symmetric=%d\n",
            rows, cols, global_nnz, Px, Py, is_pattern, is_symmetric);
        printf("[Rank 0] Partitioning: 2D BLOCK (contiguous blocks)\n");
    }

    // Calculate local dimensions for BLOCK partitioning
    int row_start, row_end, col_start, col_end;
    get_block_row_range(rows, Px, pr, &row_start, &row_end);
    get_block_col_range(cols, Py, pc, &col_start, &col_end);
    
    int local_num_rows = row_end - row_start;
    int local_num_cols = col_end - col_start;

    int sum_local_nnz = 0;
    MPI_Reduce(&local_nnz, &sum_local_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int *row_ptr = NULL, *col_idx = NULL; 
    double *vals = NULL;
    coo_to_csr(local_nnz, local_num_rows, coo_r, coo_c, coo_v, 
               &row_ptr, &col_idx, &vals);
    
    long long local_flops = 2LL * local_nnz;

    int nnz_min = 0, nnz_max = 0;
    MPI_Allreduce(&local_nnz, &nnz_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_nnz, &nnz_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    long long local_nnz_ll = (long long)local_nnz;
    long long nnz_sum = 0;
    MPI_Allreduce(&local_nnz_ll, &nnz_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double nnz_avg = (size > 0) ? ((double)nnz_sum / (double)size) : 0.0;
    double load_imbalance = (nnz_avg > 0) ? ((double)nnz_max / nnz_avg) : 1.0;

    double mem_bytes =
        (double)local_nnz * sizeof(int) +
        (double)local_nnz * sizeof(int) +
        (double)local_nnz * sizeof(double) +
        (double)(local_num_rows + 1) * sizeof(int) +
        (double)local_nnz * sizeof(int) +
        (double)local_nnz * sizeof(double) +
        (double)local_num_cols * sizeof(double) +
        (double)local_num_rows * sizeof(double) +
        (double)local_num_rows * sizeof(double);

    double mem_mib = bytes_to_mib(mem_bytes);
    double mem_mib_max = 0.0;
    MPI_Allreduce(&mem_mib, &mem_mib_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    double mem_mib_avg = 0.0;
    MPI_Allreduce(&mem_mib, &mem_mib_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mem_mib_avg /= size;

    int max_local_cols;
    MPI_Allreduce(&local_num_cols, &max_local_cols, 1, MPI_INT, MPI_MAX, row_comm);

    double *x_panel = (double*)malloc((size_t)max_local_cols * sizeof(double));
    double *x_local = malloc(local_num_cols * sizeof(double));
    double *y_partial = (double*)calloc((size_t)local_num_rows, sizeof(double));
    double *y_row_sum = (double*)calloc((size_t)local_num_rows, sizeof(double));

    if (!x_panel || !x_local || !y_partial || !y_row_sum) {
        fprintf(stderr, "[Rank %d] ERROR: x/y allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }   

    if (pr == 0) {
        srand(12345 + pc);
        for (int j = 0; j < local_num_cols; j++)
            x_local[j] = (double)rand() / RAND_MAX;
    }
    
    double t_bcast = 0.0, t_reduce = 0.0, t_spmv = 0.0;
    double t_total_start = MPI_Wtime();
    double t_comp = 0.0;
    double t_comm = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    memset(y_partial, 0, local_num_rows * sizeof(double));

    // SUMMA algorithm with BLOCK partitioning
    for (int pc_iter = 0; pc_iter < Py; pc_iter++) {
        int iter_col_start, iter_col_end;
        get_block_col_range(cols, Py, pc_iter, &iter_col_start, &iter_col_end);
        int panel_cols = iter_col_end - iter_col_start;

        if (pc == pc_iter) {
            memcpy(x_panel, x_local, panel_cols * sizeof(double));
        }

        double tb = MPI_Wtime();
        MPI_Bcast(x_panel, panel_cols, MPI_DOUBLE, pc_iter, row_comm);
        tb = MPI_Wtime() - tb;

        t_bcast += tb;
        t_comm  += tb;

        double tc = MPI_Wtime();
        #pragma omp parallel for
        for (int r = 0; r < local_num_rows; r++) {
            double acc = 0.0;
            for (int k = row_ptr[r]; k < row_ptr[r + 1]; k++) {
                int j = col_idx[k];  // global column index
                
                // Check if this column belongs to the current panel
                if (j >= iter_col_start && j < iter_col_end) {
                    int local_j = j - iter_col_start;
                    acc += vals[k] * x_panel[local_j];
                }
            }
            y_partial[r] += acc;
        }
        
        tc = MPI_Wtime() - tc;
        t_spmv += tc;
        t_comp += tc;
    }

    double tr = MPI_Wtime();
    MPI_Reduce(y_partial, y_row_sum, local_num_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    tr = MPI_Wtime() - tr;
    t_reduce += tr;
    t_comm   += tr;

    double global_time;
    double local_time = MPI_Wtime() - t0;
    MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    long long global_flops;
    MPI_Reduce(&local_flops, &global_flops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global FLOPs per SpMV: %lld\n", global_flops);
    }

    double t_total = MPI_Wtime() - t_total_start;

    double t_total_max, t_comp_max, t_comm_max;
    double t_bcast_max, t_reduce_max, t_spmv_max;

    MPI_Allreduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comp,  &t_comp_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comm,  &t_comm_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_bcast, &t_bcast_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_reduce, &t_reduce_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_spmv,  &t_spmv_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double comp_pct = 100.0 * t_comp_max / t_total_max;
    double comm_pct = 100.0 * t_comm_max / t_total_max;

    double gflops = (2.0 * nnz_sum) / (t_total_max * 1e9);

    double T1 = env_double_or_neg("BASELINE_T1");
    double speedup = (T1 > 0) ? T1 / t_total_max : -1.0;
    double efficiency = (speedup > 0) ? speedup / size : -1.0;

    if (rank == 0) {
        printf("\n");
        printf("                    2D DISTRIBUTED SpMV - PERFORMANCE REPORT                   \n");
        printf("*******************************************************************************\n\n");
        
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
        printf("  Process grid:       %d x %d (Px x Py)\n", Px, Py);
        printf("  Partitioning:       2D BLOCK (contiguous)\n");
        printf("  Topology:           Cartesian communicator\n");
        printf("\n");
        
        printf("LOAD BALANCE:\n");
        printf("  NNZ per rank:       min=%d  avg=%.1f  max=%d\n", 
            nnz_min, nnz_avg, nnz_max);
        printf("  Imbalance ratio:    %.3f  (max/avg)\n", load_imbalance);
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
        printf("    └─ Local SpMV:    %.6f s\n", t_spmv_max);
        printf("  Communication time: %.6f s  (%.1f%%)\n", t_comm_max, comm_pct);
        printf("    ├─ Bcast (x):     %.6f s\n", t_bcast_max);
        printf("    └─ Reduce (y):    %.6f s\n", t_reduce_max);
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
        
        printf("================================================================================\n");
        printf("SUMMARY TABLE:\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("  P | Grid  | Time(s) | Comp%% | Comm%% | GFLOP/s | Speedup |  Eff%%  | Imbal\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("%3d | %2dx%-2d | %7.4f | %5.1f | %5.1f | %7.3f | ",
            size, Px, Py, t_total_max, comp_pct, comm_pct, gflops);
        
        if (speedup < 0) {
            printf("  N/A   |  N/A  | ");
        } else {
            printf("%7.2f | %5.1f | ", speedup, efficiency * 100.0);
        }
        
        printf("%.3f\n", load_imbalance);
        printf("--------------------------------------------------------------------------------\n");
        printf("\n");
    }

    free(coo_r);
    free(coo_c);
    free(coo_v);
    free(row_ptr);
    free(col_idx);
    free(vals);
    free(x_local);
    free(x_panel);
    free(y_partial);
    free(y_row_sum);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}