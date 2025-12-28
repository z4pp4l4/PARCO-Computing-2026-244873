#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================
   Struttura COO
   ============================ */
typedef struct {
    int row;
    int col;
    double val;
} COO;


void read_matrix_market_mpiio(
    const char *filename,
    int *rows,
    int *cols,
    int *global_nnz,
    int rank,
    int size,
    int *local_nnz,
    COO **local_entries
) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename,
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);

    /* ============================
       1) Rank 0 reads header
       ============================ */
    MPI_Offset data_offset = 0;

    if (rank == 0) {
        char line[256];
        MPI_Status status;

        do {
            MPI_File_read(fh, line, 256, MPI_CHAR, &status);
            data_offset += strlen(line);
        } while (line[0] == '%');

        sscanf(line, "%d %d %d", rows, cols, global_nnz);
    }

    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    /* ============================
       2) Parallel chunk reading
       ============================ */
    MPI_Offset data_size = file_size - data_offset;
    MPI_Offset chunk = data_size / size;

    MPI_Offset start = data_offset + rank * chunk;
    MPI_Offset end   = (rank == size - 1)
                     ? file_size
                     : data_offset + (rank + 1) * chunk;

    MPI_Offset local_size = end - start;

    char *buffer = malloc(local_size + 1);

    MPI_File_read_at_all(fh, start,
                         buffer, local_size,
                         MPI_CHAR, MPI_STATUS_IGNORE);

    buffer[local_size] = '\0';

    /* ============================
       3) Line boundary alignment
       ============================ */
    char *ptr = buffer;
    if (rank != 0) {
        while (*ptr != '\n' && *ptr != '\0') ptr++;
        if (*ptr == '\n') ptr++;
    }

    /* ============================
       4) Local parsing
       ============================ */
    int capacity = 1024;
    *local_entries = malloc(capacity * sizeof(COO));
    *local_nnz = 0;

    while (ptr < buffer + local_size) {
        int r, c;
        double v;
        int read = sscanf(ptr, "%d %d %lf", &r, &c, &v);
        if (read == 3) {
            r--; c--;
            if (r % size == rank) {
                if (*local_nnz == capacity) {
                    capacity *= 2;
                    *local_entries = realloc(*local_entries,
                                             capacity * sizeof(COO));
                }
                (*local_entries)[*local_nnz].row = r;
                (*local_entries)[*local_nnz].col = c;
                (*local_entries)[*local_nnz].val = v;
                (*local_nnz)++;
            }
        }
        while (*ptr != '\n' && ptr < buffer + local_size) ptr++;
        ptr++;
    }

    free(buffer);
    MPI_File_close(&fh);
}


/* ============================
   COO → CSR locale
   ============================ */
void coo_to_csr(
    int local_nnz,
    int local_rows,
    int *coo_r,
    int *coo_c,
    double *coo_v,
    int **row_ptr,
    int **col_idx,
    double **vals
) {
    *row_ptr = calloc(local_rows + 1, sizeof(int));
    *col_idx = malloc(local_nnz * sizeof(int));
    *vals    = malloc(local_nnz * sizeof(double));

    for (int i = 0; i < local_nnz; i++)
        (*row_ptr)[coo_r[i] + 1]++;

    for (int i = 0; i < local_rows; i++)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    int *offset = calloc(local_rows, sizeof(int));

    for (int i = 0; i < local_nnz; i++) {
        int r = coo_r[i];
        int p = (*row_ptr)[r] + offset[r]++;
        (*col_idx)[p] = coo_c[i];
        (*vals)[p]    = coo_v[i];
    }

    free(offset);
}




int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            printf("Usage: %s matrix.mtx\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    /* ============================
       Lettura globale
       ============================ */
    int rows, cols, global_nnz;
    COO *entries = NULL;

    if (rank == 0) {
        read_matrix_market_mpiio(
            argv[1],
            &rows, &cols, &global_nnz,
            rank, size,
            &local_nnz,
            &local_entries
        );
        printf("[Rank 0] Matrix read: %d x %d, nnz=%d\n",
               rows, cols, global_nnz);
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ============================
       Conteggio NNZ per rank
       ============================ */
    int *nnz_per_rank = NULL;
    if (rank == 0) {
        nnz_per_rank = calloc(size, sizeof(int));
        for (int i = 0; i < global_nnz; i++) {
            int owner = entries[i].row % size;
            nnz_per_rank[owner]++;
        }
        printf("[Rank 0] NNZ per rank:\n", nnz_per_rank);
    }

    int local_nnz;
    MPI_Scatter(nnz_per_rank, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ============================
       Dimensioni locali
       ============================ */
    int local_num_rows = (rows + size - 1 - rank) / size;
    printf("[Rank %d] local_num_rows = %d\n", rank, local_num_rows);
    /* ============================
       DEBUG: distribuzione
       ============================ */
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Rank %d] local_nnz = %d, local_rows = %d\n",
           rank, local_nnz, local_num_rows);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================
       Allocazione COO locale
       ============================ */
    int    *local_rows = malloc(local_nnz * sizeof(int));
    int    *local_cols = malloc(local_nnz * sizeof(int));
    double *local_vals = malloc(local_nnz * sizeof(double));

    /* ============================
       Distribuzione dati
       ============================ */
    MPI_Datatype MPI_COO;
    MPI_Type_contiguous(2, MPI_INT, &MPI_COO);
    MPI_Type_commit(&MPI_COO);

    if (rank == 0) {
        int *offset = calloc(size, sizeof(int));

        for (int i = 0; i < global_nnz; i++) {
            int owner = entries[i].row % size;

            if (owner == 0) {
                int k = offset[0]++;
                local_rows[k] = entries[i].row / size;
                local_cols[k] = entries[i].col;
                local_vals[k] = entries[i].val;
            } else {
                MPI_Send(&entries[i], 1, MPI_COO,
                         owner, 0, MPI_COMM_WORLD);
            }
        }
        free(offset);
    } else {
        for (int i = 0; i < local_nnz; i++) {
            COO e;
            MPI_Recv(&e, 1, MPI_COO,
                     0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            local_rows[i] = e.row / size;
            local_cols[i] = e.col;
            local_vals[i] = e.val;
        }
    }

    MPI_Type_free(&MPI_COO);

    /* ============================
       COO → CSR
       ============================ */
    int *row_ptr, *col_idx;
    double *vals;

    coo_to_csr(local_nnz, local_num_rows,
               local_rows, local_cols, local_vals,
               &row_ptr, &col_idx, &vals);

    /* ============================
       DEBUG: CSR
       ============================ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n[Rank 0] CSR check (first rows):\n");
        for (int i = 0; i < 5 && i < local_num_rows; i++)
            printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================
       SpMV locale
       ============================ */
    double *x = malloc(cols * sizeof(double));
    double *y = calloc(local_num_rows, sizeof(double));

    for (int i = 0; i < cols; i++)
        x[i] = 1.0;

    for (int i = 0; i < local_num_rows; i++) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            y[i] += vals[k] * x[col_idx[k]];
        }
    }

    /* ============================
       DEBUG: output SpMV
       ============================ */
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Rank %d] y[0..4]: ", rank);
    for (int i = 0; i < 5 && i < local_num_rows; i++)
        printf("%.2f ", y[i]);
    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================
       Cleanup
       ============================ */
    free(local_rows);
    free(local_cols);
    free(local_vals);
    free(row_ptr);
    free(col_idx);
    free(vals);
    free(x);
    free(y);

    if (rank == 0) {
        free(entries);
        free(nnz_per_rank);
    }

    MPI_Finalize();
    return 0;
}
