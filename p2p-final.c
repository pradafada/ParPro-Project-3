#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

double rand_double()
{
    return (double)rand() / RAND_MAX;
}

void init_matrix(double *M, long size)
{
    for(long i=0;i<size;i++)
        M[i] = rand_double();
}

void sequential_gemm(double *A,double *B,double *C,int M,int N,int K)
{
    for(int i=0;i<M;i++)
        for(int k=0;k<N;k++)
            for(int j=0;j<K;j++)
                C[i*K+j] += A[i*N+k] * B[k*K+j];
}

int verify(double *A,double *B,long size)
{
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
            C = malloc(sizeof(double)*M*K);
            C_seq = calloc(M*K,sizeof(double));

            init_matrix(A,M*N);
            init_matrix(B,N*K);
        }
        else
        {
            B = malloc(sizeof(double)*N*K);
        }

        /* Send B once to all processes */
        if(rank==0)
        {
            for(int p=1;p<size;p++)
                MPI_Send(B,N*K,MPI_DOUBLE,p,0,MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(B,N*K,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        double *A_local = malloc(sizeof(double)*rows*N);
        double *C_local = calloc(rows*K,sizeof(double));

        if(rank==0)
        {
            for(int p=1;p<size;p++)
                MPI_Send(A+p*rows*N,rows*N,MPI_DOUBLE,p,1,MPI_COMM_WORLD);

            for(int i=0;i<rows*N;i++)
                A_local[i] = A[i];
        }
        else
        {
            MPI_Recv(A_local,rows*N,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        /* Optimized multiplication */
        for(int i=0;i<rows;i++)
        {
            for(int k=0;k<N;k++)
            {
                double a = A_local[i*N+k];

                for(int j=0;j<K;j++)
                {
                    C_local[i*K+j] += a * B[k*K+j];
                }
            }
        }

        if(rank==0)
        {
            for(int i=0;i<rows*K;i++)
                C[i] = C_local[i];

            for(int p=1;p<size;p++)
                MPI_Recv(C+p*rows*K,rows*K,MPI_DOUBLE,p,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            double end = MPI_Wtime();

            printf("\nMatrix Size: %d x %d x %d\n",M,N,K);
            printf("MPI Point-to-Point Time: %f seconds\n",end-start);

            /* Verification (not timed) */
            sequential_gemm(A,B,C_seq,M,N,K);

            if(verify(C,C_seq,M*K))
                printf("Verification: YES\n");
            else
                printf("Verification: NO\n");
        }
        else
        {
            MPI_Send(C_local,rows*K,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
        }

        free(A_local);
        free(C_local);

        if(rank==0)
        {
            free(A);
            free(B);
            free(C);
            free(C_seq);
        }
        else
        {
            free(B);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}