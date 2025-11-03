#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ROWS 10000
#define COLS 10000

typedef struct
{
    int *Arow;
    int *Acol;
    int *Aval;
    int non_zero_count;
} CSR;

int **generate_matrix(int rows, int cols, float sparsity){
    int **normal_matrix=malloc(rows*sizeof(int*));
    for(int i=0;i<rows ;i++){
        normal_matrix[i]=malloc(cols*sizeof(int));
        for(int j=0;j<cols;j++){
            float r = (float)rand() / RAND_MAX;
            if (r < sparsity)   // zero with probability "sparsity"
                normal_matrix[i][j] = 0;
            else
                normal_matrix[i][j] = rand() % 10;
        }
    }
    return normal_matrix;
}

CSR *matrix_to_CSR(int **normal_matrix, int rows, int cols){
    //CREATE CSR STRUCTURE
    CSR *csr_matrix=malloc(sizeof(CSR));
    //want to iterate through normal matrix and count non zero elements
    int non_zero_count=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(normal_matrix[i][j]!=0){
                non_zero_count++;
            }
        }
    }
    csr_matrix->non_zero_count=non_zero_count;
    csr_matrix->Aval=malloc(non_zero_count*sizeof(int));
    csr_matrix->Acol=malloc(non_zero_count*sizeof(int));
    csr_matrix->Arow=malloc((rows+1)*sizeof(int));
    //fill CSR structure
    if (non_zero_count==0){
        csr_matrix->Arow[0]=0;
        csr_matrix->Acol[0]=0;
        csr_matrix->Aval[0]=0;
        return csr_matrix;
    } else {
        int index=0;
        for(int i=0;i<rows;i++){
            csr_matrix->Arow[i]=index;
            for(int j=0;j<cols;j++){
                if(normal_matrix[i][j]!=0){
                    csr_matrix->Aval[index]=normal_matrix[i][j];
                    csr_matrix->Acol[index]=j;
                    index++;
                }
            } 
            //compressing row pointer with a prefix sum 
            csr_matrix->Arow[i+1]=index;
        }
        return csr_matrix;
    }

}
int* matrix_vector_mult(CSR *csr_matrix, int *vector, int rows){
    int *result=malloc(rows*sizeof(int));
    for(int i=0;i<rows;i++){
        result[i]=0;
        for(int j=csr_matrix->Arow[i];j<csr_matrix->Arow[i+1];j++){
            result[i]+=(csr_matrix->Aval[j])*(vector[csr_matrix->Acol[j]]);
        }
    }
    return result;
}

int main(){

    printf("Generating normal matrix...\n");
    printf("Enter sparsity degree of the matrix (0-1): ");
    float sparsity;
    scanf("%f",&sparsity);
    if (sparsity<0 || sparsity>1){
        printf("Invalid sparsity degree. Exiting...\n");
        return -1;
    }
    printf("Converting to CSR format...\n");

    
    int rows=ROWS;
    int cols=COLS;
    int **matrix=generate_matrix(rows,cols,sparsity);
    CSR *csr_matrix=matrix_to_CSR(matrix,rows,cols);

    int *vector=malloc(cols*sizeof(int));
    for(int i=0;i<cols;i++){
        vector[i]=rand()%10;
    }

    //perform matrix vector multiplication here
    int* result = matrix_vector_mult(csr_matrix,vector,rows);    
    
    //free normal matrix
    for(int i=0;i<rows;i++){
        free(matrix[i]);
    }   
    free(matrix);

    free(vector);
    free(result);
    free(csr_matrix->Arow);
    free(csr_matrix->Acol);
    free(csr_matrix->Aval);
    free(csr_matrix);

    return 0;
}