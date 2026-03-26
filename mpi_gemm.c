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
        for(int j=0;j<K;j++) {
            C[i*K+j]=0;
            for(int k=0;k<N;k++)
                C[i*K+j]+=A[i*N+k]*B[k*K+j];
        }
}

int verify(double *C1,double *C2,int size){
    for(int i=0;i<size;i++){
        if(fabs(C1[i]-C2[i])>EPSILON) return 0;
    }
    return 1;
}

int main(int argc,char *argv[]) {

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

    double *A=NULL,*B=NULL,*C=NULL,*C_seq=NULL;

    int rows=M/size;

    if(rank==0){
        A=(double*)malloc(M*N*sizeof(double));
        B=(double*)malloc(N*K*sizeof(double));
        C=(double*)malloc(M*K*sizeof(double));
        C_seq=(double*)malloc(M*K*sizeof(double));

        srand(time(NULL));
        for(int i=0;i<M*N;i++) A[i]=rand_double();
        for(int i=0;i<N*K;i++) B[i]=rand_double();
    }
    else{
        B=(double*)malloc(N*K*sizeof(double));
    }

    double *localA=(double*)malloc(rows*N*sizeof(double));
    double *localC=(double*)malloc(rows*K*sizeof(double));

    MPI_Bcast(B,N*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Scatter(A,rows*N,MPI_DOUBLE,localA,rows*N,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();

    for(int i=0;i<rows;i++)
        for(int j=0;j<K;j++){
            localC[i*K+j]=0;
            for(int k=0;k<N;k++)
                localC[i*K+j]+=localA[i*N+k]*B[k*K+j];
        }

    MPI_Gather(localC,rows*K,MPI_DOUBLE,C,rows*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

    double end=MPI_Wtime();

    if(rank==0){
        printf("MPI GEMM time: %f seconds\n",end-start);

        sequential_gemm(A,B,C_seq,M,N,K);

        if(verify(C,C_seq,M*K))
            printf("Verification PASSED\n");
        else
            printf("Verification FAILED\n");
    }

    MPI_Finalize();
}