#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

double rand_double() {
    return (double)rand() / RAND_MAX;
}

void sequential_gemm(double *A, double *B, double *C, int M, int N, int K) {
    for(int i=0;i<M;i++)
        for(int j=0;j<K;j++){
            C[i*K+j]=0;
            for(int k=0;k<N;k++)
                C[i*K+j]+=A[i*N+k]*B[k*K+j];
        }
}

int verify(double *C1,double *C2,int size){
    for(int i=0;i<size;i++)
        if(fabs(C1[i]-C2[i])>EPSILON)
            return 0;
    return 1;
}

/* POINT-TO-POINT VERSION */
void mpi_point_to_point(double *A,double *B,double *C,
                        int M,int N,int K,int rank,int size){

    int rows = M/size;
    int start = rank*rows;

    double *local_C = malloc(sizeof(double)*rows*K);

    for(int i=0;i<rows;i++)
        for(int j=0;j<K;j++){
            double sum=0;
            for(int k=0;k<N;k++)
                sum += A[(start+i)*N+k]*B[k*K+j];
            local_C[i*K+j]=sum;
        }

    if(rank==0){
        for(int i=0;i<rows*K;i++)
            C[i]=local_C[i];

        for(int p=1;p<size;p++)
            MPI_Recv(C+p*rows*K,rows*K,MPI_DOUBLE,p,0,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    else{
        MPI_Send(local_C,rows*K,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }

    free(local_C);
}

/* COLLECTIVE VERSION USING MPI_REDUCE */
void mpi_collective_reduce(double *A,double *B,double *C,
                           int M,int N,int K,int rank,int size){

    int rows = M/size;
    int start = rank*rows;

    double *C_partial = calloc(M*K,sizeof(double));

    /* Broadcast B to all processes */
    MPI_Bcast(B,N*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

    /* Each process computes its rows */
    for(int i=0;i<rows;i++)
        for(int j=0;j<K;j++){
            double sum=0;
            for(int k=0;k<N;k++)
                sum += A[(start+i)*N+k]*B[k*K+j];
            C_partial[(start+i)*K+j]=sum;
        }

    /* Reduce all partial buffers into final C */
    MPI_Reduce(C_partial,C,M*K,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    free(C_partial);
}

int main(int argc,char **argv){

    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int M,N,K;

    if(rank==0){
        printf("Enter M N K: \n");
        fflush(stdout);
        scanf("%d %d %d",&M,&N,&K);
    }

    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&K,1,MPI_INT,0,MPI_COMM_WORLD);

    double *A = malloc(sizeof(double)*M*N);
    double *B = malloc(sizeof(double)*N*K);
    double *C_ptp = malloc(sizeof(double)*M*K);
    double *C_reduce = malloc(sizeof(double)*M*K);
    double *C_seq = malloc(sizeof(double)*M*K);

    if(rank==0){
        srand(time(NULL));
        for(int i=0;i<M*N;i++) A[i]=rand_double();
        for(int i=0;i<N*K;i++) B[i]=rand_double();
    }

    MPI_Bcast(A,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);

    double start,end;

    /* POINT TO POINT */
    start=MPI_Wtime();
    mpi_point_to_point(A,B,C_ptp,M,N,K,rank,size);
    end=MPI_Wtime();

    if(rank==0)
        printf("MPI Point-to-Point Time: %f seconds\n",end-start);

    /* COLLECTIVE REDUCE */
    start=MPI_Wtime();
    mpi_collective_reduce(A,B,C_reduce,M,N,K,rank,size);
    end=MPI_Wtime();

    if(rank==0)
        printf("MPI Collective Time: %f seconds\n",end-start);

    if(rank==0){

        //sequential_gemm(A,B,C_seq,M,N,K);

        printf("Point-to-Point Correct: %s\n",
               verify(C_seq,C_ptp,M*K)?"Yes":"No");

        printf("Collective Correct: %s\n",
               verify(C_seq,C_reduce,M*K)?"Yes":"No");
    }

    free(A);
    free(B);
    free(C_ptp);
    free(C_reduce);
    free(C_seq);

    MPI_Finalize();
    return 0;
}