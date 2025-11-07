
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h> 

typedef struct {
    int *Arow;
    int *Acol;
    double *Aval;
    int nrows;
    int ncols;
    int nnz;
} CSR;


CSR *read_matrix_market_to_CSR(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Cannot open file");
        exit(EXIT_FAILURE);
    }

    char line[256];
    int symmetric = 0;

    // ---- Step 1. Read header line ----
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Empty file.\n");
        exit(EXIT_FAILURE);
    }

    if (strstr(line, "symmetric") != NULL)
        symmetric = 1;
    else if (strstr(line, "general") != NULL)
        symmetric = 0;
    else
        printf("⚠️  Warning: symmetry not specified, assuming GENERAL.\n");

    // ---- Step 2. Skip comments ----
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Invalid Matrix Market file.\n");
            exit(EXIT_FAILURE);
        }
    } while (line[0] == '%');

    // ---- Step 3. Read size and nnz ----
    int rows, cols, nnz;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz) != 3) {
        fprintf(stderr, "Error reading matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int capacity = symmetric ? (nnz * 2) : nnz;
    int *coo_row = malloc(capacity * sizeof(int));
    int *coo_col = malloc(capacity * sizeof(int));
    double *coo_val = malloc(capacity * sizeof(double));

    // ---- Step 4. Read COO entries ----
    int count = 0;
    for (int i = 0; i < nnz; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
            fprintf(stderr, "Error reading matrix data at line %d.\n", i + 1);
            exit(EXIT_FAILURE);
        }
        r--; c--; // Convert to 0-based

        coo_row[count] = r;
        coo_col[count] = c;
        coo_val[count++] = v;

        // Mirror if symmetric and not on diagonal
        if (symmetric && r != c) {
            coo_row[count] = c;
            coo_col[count] = r;
            coo_val[count++] = v;
        }
    }
    fclose(f);
    nnz = count;

    // ---- Step 5. Build CSR ----
    CSR *csr = malloc(sizeof(CSR));
    csr->nrows = rows;
    csr->ncols = cols;
    csr->nnz = nnz;
    csr->Arow = calloc(rows + 1, sizeof(int));
    csr->Acol = malloc(nnz * sizeof(int));
    csr->Aval = malloc(nnz * sizeof(double));

    // Count nonzeros per row
    for (int i = 0; i < nnz; i++)
        csr->Arow[coo_row[i] + 1]++;

    // Prefix sum
    for (int i = 0; i < rows; i++)
        csr->Arow[i + 1] += csr->Arow[i];

    // Fill CSR arrays
    int *offset = calloc(rows, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int r = coo_row[i];
        int dest = csr->Arow[r] + offset[r];
        csr->Acol[dest] = coo_col[i];
        csr->Aval[dest] = coo_val[i];
        offset[r]++;
    }

    free(offset);
    free(coo_row);
    free(coo_col);
    free(coo_val);

    printf("✅ Loaded %s matrix (%s)\n", symmetric ? "SYMMETRIC" : "GENERAL", filename);
    printf("   → size: %dx%d, nnz = %d\n", rows, cols, nnz);

    return csr;
}


void export_CSR_to_header(CSR *csr, const char *basename) {
    char outpath[512];
    snprintf(outpath, sizeof(outpath), "%s_csr.h", basename);

    FILE *out = fopen(outpath, "w");
    if (!out) {
        perror("Cannot write output file");
        exit(EXIT_FAILURE);
    }

    // Sanitize guard name (add prefix to avoid starting with digit)
    char guard[256];
    snprintf(guard, sizeof(guard), "M_%s_CSR_H", basename);
    for (char *p = guard; *p; ++p)
        if (*p == '.') *p = '_';

    fprintf(out, "// Auto-generated CSR matrix header\n");
    fprintf(out, "#ifndef %s\n#define %s\n\n", guard, guard);

    fprintf(out, "static const int nrows = %d;\n", csr->nrows);
    fprintf(out, "static const int ncols = %d;\n", csr->ncols);
    fprintf(out, "static const int nnz = %d;\n\n", csr->nnz);

    fprintf(out, "static const int Arow[%d] = {", csr->nrows + 1);
    for (int i = 0; i <= csr->nrows; i++)
        fprintf(out, "%d%s", csr->Arow[i], (i == csr->nrows) ? "" : ", ");
    fprintf(out, "};\n\n");

    fprintf(out, "static const int Acol[%d] = {", csr->nnz);
    for (int i = 0; i < csr->nnz; i++)
        fprintf(out, "%d%s", csr->Acol[i], (i == csr->nnz - 1) ? "" : ", ");
    fprintf(out, "};\n\n");

    // Print Aval with scientific precision to preserve small values
    fprintf(out, "static const double Aval[%d] = {", csr->nnz);
    for (int i = 0; i < csr->nnz; i++)
        fprintf(out, "%.12e%s", csr->Aval[i], (i == csr->nnz - 1) ? "" : ", ");
    fprintf(out, "};\n\n");

    fprintf(out, "#endif // %s\n", guard);
    fclose(out);

    printf("💾 Exported CSR matrix to %s\n", outpath);
}

int main() {
    char filename[256];
    printf("Enter Matrix Market filename (in ../src): ");
    scanf("%255s", filename);

    // Build full path
    char fullpath[512];
    snprintf(fullpath, sizeof(fullpath), "../src/%s", filename);

    // Extract base name (without extension)
    char temp[512];
    strcpy(temp, filename);
    char *dot = strchr(temp, '.');
    if (dot) *dot = '\0';

    CSR *csr = read_matrix_market_to_CSR(fullpath);
    export_CSR_to_header(csr, temp);

    free(csr->Arow);
    free(csr->Acol);
    free(csr->Aval);
    free(csr);

    return 0;
}
