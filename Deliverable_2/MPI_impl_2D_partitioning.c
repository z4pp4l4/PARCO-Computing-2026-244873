#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define DEBUG 1
#if DEBUG
  #define DPRINTF(...) printf(__VA_ARGS__)  // Debug print macro (prints if DEBUG=1)
#else
  #define DPRINTF(...)                       // Silent if DEBUG=0
#endif

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
   In cyclic distribution across P processors with stride P,
   a processor at offset gets elements: offset, offset+P, offset+2P, ...
   Returns how many such elements fit in range [0, N)
*/
static int local_cyclic_size(int N, int stride, int offset) {
    if (offset >= N) 
        return 0;
    return (N - offset + stride - 1) / stride;
}

static void read_header_mtx( MPI_File fh,int rank,int *rows,int *cols, int *nnz, MPI_Offset *data_offset) {
    if (rank == 0) {
        MPI_Offset off = 0;
        char line[1024];
        while (1) {
            // Read one line character-by-character
            int pos = 0;
            char ch;
            MPI_Status st;
            // Read until newline or end of buffer
            while (1) {
                MPI_File_read_at(fh, off, &ch, 1, MPI_CHAR, &st);
                off += 1;
                if (ch == '\n' || ch == '\r' || pos >= (int)sizeof(line) - 1) break;
                line[pos++] = ch;
            }
            line[pos] = '\0';

            // Skip empty lines
            if (pos == 0) {
                continue;
            }
            // Skip comment lines (start with '%')
            if (line[0] == '%') continue;

            // Parse "rows cols nnz" from first non-comment line
            if (sscanf(line, "%d %d %d", rows, cols, nnz) == 3) {
                *data_offset = off; // Remember file position after header
                break;
            } else {
                fprintf(stderr, "[Rank 0] ERROR: cannot parse dimensions line: '%s'\n", line);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast header info to all ranks
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

}
   
static void read_and_distribute_2D(
    const char *filename,int rank, int size, MPI_Comm grid_comm,
    int Px, int Py,           // Grid dimensions (rows x cols)
    int pr, int pc,           // This rank's position in grid
    int *rows, int *cols, int *global_nnz,
    int *local_nnz, int **coo_r_local, int **coo_c_local, double **coo_v
) {
    /* 
        READ MATRIX MARKET FILE (2D Process Grid Distribution)
        1. Divide file among processes (each reads a chunk)
        2. Each rank reads its chunk + overlap region for boundary lines
        3. Parse entries and distribute to 2D process grid:
            - Entry (i,j) goes to process where i%Px==pr and j%Py==pc
        4. Store as COO format (row, col, value) locally 
   */ 
    // Open file in read-only mode
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // Get file size in bytes
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    // Read header to get matrix dimensions and where data starts
    MPI_Offset data_offset = 0;
    read_header_mtx(fh, rank, rows, cols, global_nnz, &data_offset);
    
    // Calocal_colsulate how to split data among processes
    MPI_Offset data_size = file_size - data_offset;
    MPI_Offset chunk = data_size / size;  // Approximate chunk per process
    // Each rank reads from 'start' to 'end' (with overlap for boundary lines)
    MPI_Offset start = data_offset + rank * chunk;
    MPI_Offset end   = (rank == size - 1) ? file_size : data_offset + (rank + 1) * chunk;

    // Add overlap region so we don't miss entries at chunk boundaries
    const MPI_Offset OVERLAP = 1 << 16; // 64 KB
    MPI_Offset read_end = end + OVERLAP;
    if (read_end > file_size){ 
        read_end = file_size;
    }
    MPI_Offset read_size = read_end - start;

    // Allocate buffer and read chunk from file
    char *buffer = malloc((size_t)read_size + 1);
    MPI_File_read_at_all(fh, start, buffer, (int)read_size, MPI_CHAR, MPI_STATUS_IGNORE);
    buffer[read_size] = '\0';

    // Set up parsing boundaries (skip incomplete lines at edges)
    char *p = buffer;      // Parse start
    char *q = buffer + read_size;  // Parse end

    // If not rank 0, skip to next newline (skip partial line from previous rank)
    if (rank != 0) {
        while (p < q && *p != '\n') p++;
        if (p < q) p++;
    }
    // If not last rank, back up to previous newline (exclude partial line for next rank)
    if (rank != size - 1) {
        while (q > p && *(q - 1) != '\n') q--;
    }

    // Dynamic arrays to store local COO entries
    int cap = 4096;  // Initial capacity
    int *local_rows = malloc(cap * sizeof(int));      // Local rows
    int *local_cols = malloc(cap * sizeof(int));      // Local cols
    double *local_vals = malloc(cap * sizeof(double)); // Local values
    int count = 0;

    // Parse matrix entries from buffer
    char *ptr = p;
    while (ptr < q && *ptr) {
        int r1, c1;        // 1-indexed row and column from file
        double v;

        // Try to parse "row col value" from current line
        if (sscanf(ptr, "%d %d %lf", &r1, &c1, &v) == 3) {
            int i = r1 - 1;  // Convert to 0-indexed
            int j = c1 - 1;

            /* ---- Check if original entry (i,j) belongs to this process ---- */
            if (i % Px == pr && j % Py == pc) {
                // Expand arrays if needed
                if (count == cap) {
                    cap *= 2;
                    local_rows = realloc(local_rows, cap * sizeof(int));
                    local_cols = realloc(local_cols, cap * sizeof(int));
                    local_vals = realloc(local_vals, cap * sizeof(double));
                }
                // Store: convert global (i,j) to local (i/Px, j/Py) coordinates
                local_rows[count] = i / Px;
                local_cols[count] = j / Py;
                local_vals[count] = v;
                count++;
            }

        }

        // Move to next line
        while (ptr < q && *ptr != '\n'){ 
            ptr++;
        }
        if (*ptr == '\n'){
            ptr++;
        }
    }
    free(buffer);
    MPI_File_close(&fh);

    // Return local COO data
    *local_nnz = count;
    *coo_r_local = local_rows;
    *coo_c_local = local_cols;
    *coo_v = local_vals;
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
    //sort COO entries by (row, col) 
    qsort(entries, local_nnz, sizeof(COO_entry), compare_coo_entries);

    //CSR building
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

    // fill CSR col_idx and vals
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
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // This process's rank (0 to size-1)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total number of processes

    if (argc < 2) {
        if (rank == 0) {
            //printf("Usage: %s matrix.mtx\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    /* 
       CREATING 2D PROCESS GRID TOPOLOGY
    */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);  // Automatically factor 'size' into 2D grid
    int Px = dims[0];
    int Py = dims[1];

    // Create Cartesian communicator (allows neighbor-based communication)
    int periods[2] = {0, 0};  // No wraparound (non-periodic)
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    // Get this rank's coordinates in the grid
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int pr = coords[0];  // Row in grid
    int pc = coords[1];  // Column in grid
    

    //------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    DPRINTF("[Rank %d] Cordinates = (pr=%d, pc=%d) in grid %dx%d\n",
            rank, pr, pc, Px, Py);
    MPI_Barrier(MPI_COMM_WORLD);
    */
    //------------------------------------

    /* 
       row_comm: all processes with same pr (same row in grid)
       col_comm: all processes with same pc (same column in grid)
       Used for reduction operations along rows/columns.
    */
    MPI_Comm row_comm;
    //MPI_Comm col_comm;
    MPI_Comm_split(grid_comm, pr, pc, &row_comm); // Same row, ordered by column
    //MPI_Comm_split(grid_comm, pc, pr, &col_comm); // Same column, ordered by row

    // Variables to hold matrix data
    int rows = 0, cols = 0, global_nnz = 0;
    int local_nnz = 0;
    int *coo_r = NULL, *coo_c = NULL;
    double *coo_v = NULL;
    // Read matrix file and distribute to 2D grid
    read_and_distribute_2D(argv[1], rank, size,grid_comm, Px, Py, pr, pc, &rows, &cols, &global_nnz,&local_nnz,&coo_r, &coo_c, &coo_v);
    
    //------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    DPRINTF("[Rank %d] First local COO entries:\n", rank);
    for (int k = 0; k<10 && k < local_nnz; k++) {
        DPRINTF("  COO[%d] = (row=%d, col=%d, value=%.3f)\n",k, coo_r[k], coo_c[k], coo_v[k]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    */
    //------------------------------------

    // Calculate local matrix dimensions
    int local_num_rows = local_cyclic_size(rows, Px, pr);
    int local_num_cols = local_cyclic_size(cols, Py, pc);
    
    //------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    DPRINTF("[Rank %d] local_nnz=%d | local_rows=%d local_cols=%d\n", rank, local_nnz, local_num_rows, local_num_cols);
    MPI_Barrier(MPI_COMM_WORLD);
    */
    //------------------------------------

    if (rank == 0) {
        printf("[Rank 0] Matrix: %d x %d, nnz=%d | grid=%dx%d\n", rows, cols, global_nnz, Px, Py);
    }

    // Verify total non-zeros read (should match global_nnz if no truncation)
    int sum_local_nnz = 0;
    MPI_Reduce(&local_nnz, &sum_local_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    /*
    if (rank == 0) {
        printf("[Rank 0] Sum of local nnz (after 2D filter) = %d (can be <= global_nnz if file parsing cuts some boundary lines)\n",
               sum_local_nnz);
    } */

    // Convert from COO to CSR format (more efficient for SpMV)
    int *row_ptr = NULL, *col_idx = NULL; double *vals = NULL;
    coo_to_csr(local_nnz, local_num_rows, coo_r, coo_c, coo_v, &row_ptr, &col_idx, &vals);
    // For performance measurement (count FLOPs: each non-zero = 2 FLOPs in SpMV)
    long long local_flops = 2LL * local_nnz;   //CHECK FLOPS CALCULATION

    //  NNZ stats across ranks (min/avg/max) 
    int nnz_min = 0, nnz_max = 0;
    MPI_Allreduce(&local_nnz, &nnz_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_nnz, &nnz_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    long long local_nnz_ll = (long long)local_nnz;
    long long nnz_sum = 0;
    MPI_Allreduce(&local_nnz_ll, &nnz_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double nnz_avg = (size > 0) ? ((double)nnz_sum / (double)size) : 0.0;
    //  memory estimate per rank (MiB) 
    // NOTE: excludes MPI internals and temporary read buffer already freed.
    double mem_bytes =
        (double)local_nnz * sizeof(int)        // coo_r
        + (double)local_nnz * sizeof(int)        // coo_c
        + (double)local_nnz * sizeof(double)     // coo_v
        + (double)(local_num_rows + 1) * sizeof(int) // row_ptr
        + (double)local_nnz * sizeof(int)        // col_idx
        + (double)local_nnz * sizeof(double)     // vals
        + (double)local_num_cols * sizeof(double)    // x_local
        + (double)local_num_rows * sizeof(double)    // y_partial
        + (double)local_num_rows * sizeof(double);   // y_row_sum

    double mem_mib = bytes_to_mib(mem_bytes);
    double mem_mib_max = 0.0; // report worst rank
    MPI_Allreduce(&mem_mib, &mem_mib_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    // ------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    DPRINTF("[Rank %d] CSR row_ptr[0..5]: ", rank);
    for (int i = 0; i<5 && i <= local_num_rows; i++)
        DPRINTF("%d ", row_ptr[i]);
    DPRINTF("\n");
    MPI_Barrier(MPI_COMM_WORLD);
    */
    //------------------------------------
    

    double *x_local = (double*)malloc((size_t)local_num_cols * sizeof(double));
    double *y_partial = (double*)calloc((size_t)local_num_rows, sizeof(double));
    double *y_row_sum = (double*)calloc((size_t)local_num_rows, sizeof(double));
    /*
        y_partial and y_row_sum are both sized by local_num_rows (the local rows owned by the process) 
        and are initialized to zero, as they accumulate partial results from the SpMV computation and 
        row-wise reductions, respectively. This allocation ensures each MPI process manages only its 
        subset of the data, promoting scalability in distributed memory systems.
    */
    if (!x_local || !y_partial || !y_row_sum) {
        fprintf(stderr, "[Rank %d] ERROR: x/y allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }   
    
    //Only rank 0 generates random x's values
    // Each process computes values for global columns it owns
    double *x_global_values = (double*)malloc((size_t)cols * sizeof(double));
    
    // All processes will refer to the SAME global random vector
    // (they all use the same seed, so they compute identically)
    // RANDOM VECTOR GENERATION INSIDE EACH PROCESS
    if (rank == 0) {
        srand(12345);  // Seed again to ensure consistency
        for (int j = 0; j < cols; j++) {
            //srand((unsigned int)time(NULL)); //for different values each run
            x_global_values[j] = (double)rand() / RAND_MAX;  // Random in [0, 1)
        }
    }
    
    // Broadcast the complete x vector to all processes
    // PERFORMANCE TIMERS 
    double t_total_start = MPI_Wtime();
    double t_comp = 0.0;
    double t_comm = 0.0;

    //COMM: Bcast x vector 
    double t0 = MPI_Wtime();
    MPI_Bcast(x_global_values, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;
    // Now each process extracts its LOCAL columns from the global vector
    // Global column j belongs to process with pc = j % Py
    // Local index in that process is: j / Py
    for (int local_cols = 0; local_cols < local_num_cols; local_cols++) {
        int global_col = local_cols * Py + pc;  // Convert local index to global
        if (global_col < cols) {
            x_local[local_cols] = x_global_values[global_col];
        }
    }
    free(x_global_values);
    
    // ------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    // Debug: print first few x_local values
    DPRINTF("[Rank %d] First 5 x_local values: ", rank);
    for (int j = 0; j<5 && j < local_num_cols; j++) {
        DPRINTF("%.6f ", x_local[j]);
    }
    DPRINTF("\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    */
    //--------------------------------------

    // Compute local SpMV: y_partial[i] = sum of A[i,j]*x[j] for local columns j
    double start_time = MPI_Wtime();
    
    #pragma omp parallel for  // OpenMP parallelization (optional)
    for (int r = 0; r < local_num_rows; r++) {
        double acc = 0.0;
        // Sum all non-zeros in row r
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; k++) {
            int local_cols = col_idx[k]; // Local column index
            if (local_cols >= 0 && local_cols < local_num_cols) {
                acc += vals[k] * x_local[local_cols];
            }
        }
        y_partial[r] = acc;
    } 
    //---------------------------------------------------
    
    t_comp += MPI_Wtime() - start_time;
    DPRINTF("[Rank %d] Local SpMV time: %.6f s\n", rank, MPI_Wtime() - start_time);

    // Count total FLOPs globally
    long long global_flops;
    MPI_Reduce(&local_flops, &global_flops, 1,MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global FLOPs per SpMV: %lld\n", global_flops);
    }

    // ------------ DEBUG ------------------
    /*
    MPI_Barrier(MPI_COMM_WORLD);
    DPRINTF("[Rank %d] y_partial[0..4]: ", rank);
    for (int i = 0; i < 10 && local_num_rows; i++)
        DPRINTF("%.3f ", y_partial[i]);
    DPRINTF("\n");
    MPI_Barrier(MPI_COMM_WORLD);
    */
    //------------------------------------

    /* 
       Each process (pr, pc) computed partial sums using columns
       from processes (pr, 0), (pr, 1), ..., (pr, Py-1).
       Now sum across the row: reduce along row_comm so that
       processes in column 0 (pc=0) get the final result.
    */

    t0 = MPI_Wtime();
    //use MPI_Reduce to collect results from all processes in the same row
    MPI_Reduce(y_partial, y_row_sum, local_num_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    t_comm += MPI_Wtime() - t0;

    double *y_global = NULL;
    int *recvcounts = NULL;
    int *displacements = NULL;
    if (rank == 0) {
        y_global = malloc(rows * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }
    int local_rows = local_num_rows;
    // Only processes in column 0 (pc=0) have the reduced result
    int sendcount = (pc == 0) ? local_rows : 0;
    // Gather send counts from all processes
    //use MPI_Gather to know how much data to collect: ech process tells rank 0 how much data it's sending
    t0 = MPI_Wtime();
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT,0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;

    // calculate displacements for Gatherv
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < size; i++)
            displacements[i] = displacements[i - 1] + recvcounts[i - 1];
    }
    //  Collect results from all row roots to rank 0 with MPI_Gatherv
    /*
        MPI_Gatherv collects variable-sized chunks of data from each 
        participating process and assembles them into a contiguous buffer
        on the root process. Unlike its simpler counterpart, MPI_Gather, 
        which assumes equal data sizes from all processes, MPI_Gatherv 
        accommodates scenarios where different processes contribute differing
        amounts of data. 
    */
   t0 = MPI_Wtime();
    MPI_Gatherv(y_row_sum, sendcount, MPI_DOUBLE, y_global, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;
    //After this Gatherv, the final result vector y is fully assembled and has the order of the elements which follow the ranks.
    double t_total = MPI_Wtime() - t_total_start;

    //MAX among all ranks (worst-case performance)
    double t_total_max, t_comp_max, t_comm_max;
    MPI_Allreduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comp,  &t_comp_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_comm,  &t_comm_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // percentage calculations
    double comp_pct = 100.0 * t_comp_max / t_total_max;
    double comm_pct = 100.0 * t_comm_max / t_total_max;

    // GFLOP/s
    double gflops = (2.0 * nnz_sum) / (t_total_max * 1e9);

    // Speedup & efficiency (external baseline)
    double T1 = env_double_or_neg("BASELINE_T1");
    double speedup = (T1 > 0) ? T1 / t_total_max : -1.0;
    double efficiency = (speedup > 0) ? speedup / size : -1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n");
        printf("-----------------------------------------------------------------------------------------------\n");
        printf("  P |  Time(s) | Comp%% | Comm%% |  GFLOP/s | Speedup |  Eff%% | Mem(MiB) | NNZ min | NNZ avg | NNZ max\n");
        printf("-----------------------------------------------------------------------------------------------\n");
    }
    if (rank == 0) {
        printf(
            "%3d | %8.4f | %5.1f | %5.1f | %9.3f | ",
            size, t_total_max, comp_pct, comm_pct, gflops
        );

        if (speedup < 0) {
            printf("   N/A   |   N/A  | ");
        } else {
            printf("%7.2f | %6.2f | ", speedup, efficiency * 100.0);
        }

        printf(
            "%8.2f | %7d | %8.1f | %7d\n",
            mem_mib_max, nnz_min, nnz_avg, nnz_max
        );
    }


    // Rank 0 prints the final result
    if (rank == 0) {
        double *y_correct = malloc(rows * sizeof(double)); 
        // Printing the rank-ordered result
        /*
        printf("\nGlobal y[0..9] (rank - ordered): ");
        for (int i = 0; i < rows; i++)
            printf("%.2f ", y_global[i]);
        printf("\n");
        */
        // Reorder from rank-based to row-based ordering
        int pos = 0;
        for (int rank_idx = 0; rank_idx < size; rank_idx++) {
            if (recvcounts[rank_idx] > 0) {
                int pr = rank_idx % Px;
                for (int local_i = 0; local_i < recvcounts[rank_idx]; local_i++) {
                    int global_row = pr + local_i * Px;
                    y_correct[global_row] = y_global[pos++];
                }
            }
        }
        // Printing the correct index-ordered result
        /*
        printf("\nGlobal y[0..9] (index - ordered): ");
        for (int i = 0;  i < rows; i++)
            printf("%.2f ", y_correct[i]);
        printf("\n");
        */
        free(y_correct);
        free(y_global);
    }

    //Free all allocated memory
    free(coo_r);
    free(coo_c);
    free(coo_v);
    free(row_ptr);
    free(col_idx);
    free(vals);
    free(x_local);
    free(y_partial);
    free(y_row_sum);

    // Free communicators
    MPI_Comm_free(&row_comm);
    //MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}