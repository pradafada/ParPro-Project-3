#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

double rand_double() {
    return (double)rand() / RAND_MAX;
}

void init_matrix(double *M, long size){
    for(long i=0;i<size;i++)
        M[i] = rand_double();
}

// Sequential GEMM (verification)
void sequential_gemm(double *A,double *B,double *C,int M,int N,int K){
    for(int i=0;i<M;i++)
        for(int k=0;k<N;k++){
            double a = A[i*N+k];
            for(int j=0;j<K;j++)
                C[i*K+j] += a * B[k*K+j];
        }
}

// Verify with epsilon tolerance
int verify(double *A,double *B,long size){
    for(long i=0;i<size;i++)
        if(fabs(A[i]-B[i]) > EPSILON)
            return 0;
    return 1;
}

int main(int argc,char *argv[])
{
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    srand(time(NULL)+rank);

    int tests[4] = {1024,2048,4096,8192};

    for(int t=0;t<4;t++)
    {
        int M = tests[t];
        int N = tests[t];
        int K = tests[t];

        int rows = M/size;

        double *A=NULL,*B=NULL,*C=NULL,*C_seq=NULL;

        if(rank==0)
        {
            A = malloc(sizeof(double)*M*N);
            B = malloc(sizeof(double)*N*K);
            C = calloc(M*K,sizeof(double));   // initialize to 0
            C_seq = calloc(M*K,sizeof(double));

            init_matrix(A,M*N);
            init_matrix(B,N*K);
        }
        else
        {
            B = malloc(sizeof(double)*N*K);
        }

        // Broadcast B to all processes once
        MPI_Bcast(B,N*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

        double *A_local = malloc(sizeof(double)*rows*N);
        double *C_local = calloc(rows*K,sizeof(double)); // initialize to 0

        // Scatter rows of A to all processes
        MPI_Scatter(A,rows*N,MPI_DOUBLE,
                    A_local,rows*N,MPI_DOUBLE,
                    0,MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD); // synchronize

        double start = MPI_Wtime();

        // Optimized local GEMM: i-k-j loop order for cache efficiency
        for(int i=0;i<rows;i++){
            for(int k=0;k<N;k++){
                double a = A_local[i*N+k]; // reuse A[i][k]
                for(int j=0;j<K;j++){
                    C_local[i*K+j] += a * B[k*K+j];
                }
            }
        }

        // Gather the local C blocks to root
        MPI_Gather(C_local,rows*K,MPI_DOUBLE,
                   C,rows*K,MPI_DOUBLE,
                   0,MPI_COMM_WORLD);

        if(rank==0)
        {
            double end = MPI_Wtime();
            printf("\nMatrix Size: %d x %d x %d\n",M,N,K);
            printf("MPI Collective Time (optimized): %f seconds\n",end-start);

            // Verification (not timed)
            sequential_gemm(A,B,C_seq,M,N,K);

            if(verify(C,C_seq,M*K))
                printf("Verification: YES\n");
            else
                printf("Verification: NO\n");
        }

        free(A_local);
        free(C_local);

        if(rank==0){
            free(A);
            free(B);
            free(C);
            free(C_seq);
        } else {
            free(B);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}