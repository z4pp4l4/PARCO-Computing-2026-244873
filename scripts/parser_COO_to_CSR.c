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
    int non_zero_val;
} CSR;

typedef struct {
    int row, col;
    double val;
} Coo_entry;

// Comparison function for qsort — sort by row, then by column
int compare_coo_entries(const void *a, const void *b) {
    Coo_entry *Entry_row = (Coo_entry *)a;
    Coo_entry *Entry_clm = (Coo_entry *)b;
    if (Entry_row->row != Entry_clm->row)
        return Entry_row->row - Entry_clm->row;
    return Entry_row->col - Entry_clm->col;
}

CSR *read_matrix_market_to_CSR(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Cannot open file");
        exit(EXIT_FAILURE);
    }

    char line[256];
    int symmetric = 0;

    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Empty file.\n");
        exit(EXIT_FAILURE);
    }

    if (strstr(line, "symmetric") != NULL)
        symmetric = 1;
    else if (strstr(line, "general") != NULL)
        symmetric = 0;
    else
        printf("symmetry not specified, assuming GENERAL.\n");

    
        do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Invalid Matrix Market file.\n");
            exit(EXIT_FAILURE);
        }
    } while (line[0] == '%');

       int rows, cols, non_zero_val;
    if (sscanf(line, "%d %d %d", &rows, &cols, &non_zero_val) != 3) {
        fprintf(stderr, "Error reading matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int capacity = symmetric ? (non_zero_val * 2) : non_zero_val;
    Coo_entry *entries = malloc(capacity * sizeof(Coo_entry));
    int count = 0;
    for (int i = 0; i < non_zero_val; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
            fprintf(stderr, "Error reading matrix data at line %d.\n", i + 1);
            exit(EXIT_FAILURE);
        }
        r--; c--; // Convert to 0-based indices

        entries[count++] = (Coo_entry){r, c, v};
        if (symmetric && r != c)
            entries[count++] = (Coo_entry){c, r, v};
    }
    fclose(f);
    non_zero_val = count;
    qsort(entries, non_zero_val, sizeof(Coo_entry), compare_coo_entries); //use quicksort to sort the entries

    // ---- Step 6. Build CSR ----
    CSR *csr = malloc(sizeof(CSR));
    csr->nrows = rows;
    csr->ncols = cols;
    csr->non_zero_val = non_zero_val;
    csr->Arow = calloc(rows + 1, sizeof(int));
    csr->Acol = malloc(non_zero_val * sizeof(int));
    csr->Aval = malloc(non_zero_val * sizeof(double));

    // Count nonzeros per row
    for (int i = 0; i < non_zero_val; i++)
        csr->Arow[entries[i].row + 1]++;

    // Prefix sum
    for (int i = 0; i < rows; i++)
        csr->Arow[i + 1] += csr->Arow[i];

    // Fill CSR arrays (already sorted)
    for (int i = 0; i < non_zero_val; i++) {
        int dest = csr->Arow[entries[i].row]++;
        csr->Acol[dest] = entries[i].col;
        csr->Aval[dest] = entries[i].val;
    }

    // Fix prefix-sum shifting (Arow got incremented in loop)
    for (int i = rows; i > 0; i--)
        csr->Arow[i] = csr->Arow[i - 1];
    csr->Arow[0] = 0;

    free(entries);

    printf("Loaded %s matrix (%s)\n", symmetric ? "SYMMETRIC" : "GENERAL", filename);
    printf("   → size: %dx%d, non_zero_val = %d\n", rows, cols, non_zero_val);

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

    // Sanitize guard name
    char guard[256];
    snprintf(guard, sizeof(guard), "M_%s_CSR_H", basename);
    for (char *p = guard; *p; ++p)
        if (*p == '.') *p = '_';

    fprintf(out, "// Auto-generated CSR matrix header\n");
    fprintf(out, "#ifndef %s\n#define %s\n\n", guard, guard);

    fprintf(out, "static const int nrows = %d;\n", csr->nrows);
    fprintf(out, "static const int ncols = %d;\n", csr->ncols);
    fprintf(out, "static const int non_zero_val = %d;\n\n", csr->non_zero_val);

    fprintf(out, "static const int Arow[%d] = {", csr->nrows + 1);
    for (int i = 0; i <= csr->nrows; i++)
        fprintf(out, "%d%s", csr->Arow[i], (i == csr->nrows) ? "" : ", ");
    fprintf(out, "};\n\n");

    fprintf(out, "static const int Acol[%d] = {", csr->non_zero_val);
    for (int i = 0; i < csr->non_zero_val; i++)
        fprintf(out, "%d%s", csr->Acol[i], (i == csr->non_zero_val - 1) ? "" : ", ");
    fprintf(out, "};\n\n");

    fprintf(out, "static const double Aval[%d] = {", csr->non_zero_val);
    for (int i = 0; i < csr->non_zero_val; i++)
        fprintf(out, "%.12e%s", csr->Aval[i], (i == csr->non_zero_val - 1) ? "" : ", ");
    fprintf(out, "};\n\n");

    fprintf(out, "#endif // %s\n", guard);
    fclose(out);

    printf("💾 Exported CSR matrix to %s\n", outpath);
}

int main() {
    char filename[256];
    printf("Enter Matrix Market filename (in ../src): ");
    scanf("%255s", filename);

    char fullpath[512];
    snprintf(fullpath, sizeof(fullpath), "../src/%s", filename);

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
